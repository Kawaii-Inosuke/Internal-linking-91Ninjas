"""Unit tests for chunking (TRD §5 / §12).

These are pure-logic tests: no database and no Gemini API required.
"""
from __future__ import annotations

import pytest

from linker.chunking import (
    Chunk,
    chunk_content,
    embedding_input,
    estimate_tokens,
    split_blocks,
)


def test_empty_content_yields_no_chunks():
    assert chunk_content("") == []
    assert chunk_content("   \n  \n\t") == []


def test_split_blocks_strips_and_drops_empties():
    assert split_blocks("a\n\n  b  \n\nc\n") == ["a", "b", "c"]


def test_single_block_is_one_chunk_indexed_zero():
    chunks = chunk_content("Just one short paragraph.")
    assert chunks == [Chunk(index=0, content="Just one short paragraph.")]


def test_indices_are_sequential_and_ordered():
    # Blocks large enough that each pair (roughly) fills a chunk; verify ordering.
    blocks = [f"block-{i} " + "word " * 40 for i in range(6)]
    chunks = chunk_content("\n".join(blocks))
    assert [c.index for c in chunks] == list(range(len(chunks)))
    # Document order is preserved: block-0 appears before block-5 across chunks.
    joined = "\n".join(c.content for c in chunks)
    assert joined.index("block-0") < joined.index("block-5")


def test_chunks_do_not_exceed_max_tokens_when_blocks_fit():
    # Each block ~50 tokens (~200 chars); max_tokens=250 -> up to ~5 blocks/chunk.
    block = "word " * 50  # ~250 chars -> ~62 tokens
    content = "\n".join([block.strip()] * 10)
    chunks = chunk_content(content, max_tokens=250)
    assert len(chunks) > 1
    for chunk in chunks:
        assert estimate_tokens(chunk.content) <= 250


def test_blocks_are_never_split_across_chunks():
    blocks = [f"unique-block-{i} " + "filler " * 30 for i in range(8)]
    original = set(b.strip() for b in blocks)
    chunks = chunk_content("\n".join(blocks), max_tokens=120)
    # Every line in every chunk must be one of the original whole blocks.
    seen = []
    for chunk in chunks:
        for line in chunk.content.split("\n"):
            assert line in original
            seen.append(line)
    # And all blocks are present exactly once, in order.
    assert seen == [b.strip() for b in blocks]


def test_oversized_single_block_becomes_its_own_chunk():
    big = "x" * 4000  # ~1000 tokens, far over max_tokens
    chunks = chunk_content(f"small intro\n{big}\nsmall outro", max_tokens=250)
    contents = [c.content for c in chunks]
    assert big in contents  # kept whole, not split
    assert estimate_tokens(big) > 250


def test_embedding_input_prepends_title_but_content_excludes_it():
    title = "Cart Abandonment Guide"
    chunk_text = "Reducing cart abandonment improves revenue."
    enriched = embedding_input(title, chunk_text)
    assert enriched.startswith(title)
    assert chunk_text in enriched
    # Stored chunk content never contains the title (title only aids embedding).
    chunks = chunk_content(chunk_text)
    assert all(title not in c.content for c in chunks)


def test_embedding_input_without_title_returns_chunk_only():
    assert embedding_input(None, "body") == "body"
    assert embedding_input("   ", "body") == "body"


def test_estimate_tokens_is_positive_and_grows_with_length():
    assert estimate_tokens("") == 1  # floor of 1 for non-empty accounting
    assert estimate_tokens("a" * 400) == 100
    assert estimate_tokens("a" * 800) > estimate_tokens("a" * 400)


def test_invalid_max_tokens_rejected():
    with pytest.raises(ValueError):
        chunk_content("anything", max_tokens=0)
