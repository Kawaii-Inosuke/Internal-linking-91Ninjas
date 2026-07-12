"""Chunking for ingestion (TRD §5).

Blog ``content`` is newline-separated blocks (headings, paragraphs, bullets).
We greedily merge consecutive blocks into chunks of roughly 150-250 tokens,
never splitting mid-block and preserving document order. Whole 11K-char articles
are too diffuse to embed as one vector; single one-line bullets are too sparse.
"""
from __future__ import annotations

from dataclasses import dataclass

# ~4 characters per token is the standard rough proxy for English text and
# matches the TRD's own "150-250 tokens (~600-1200 chars)" ratio.
_CHARS_PER_TOKEN = 4
DEFAULT_MAX_TOKENS = 250


@dataclass(frozen=True)
class Chunk:
    """A single stored chunk in document order."""

    index: int
    content: str


def estimate_tokens(text: str) -> int:
    """Estimate the token count of ``text`` using a ~4-chars-per-token proxy."""
    return max(1, round(len(text) / _CHARS_PER_TOKEN))


def split_blocks(content: str) -> list[str]:
    """Split ``content`` into non-empty, stripped blocks on newline boundaries."""
    return [block.strip() for block in content.split("\n") if block.strip()]


def chunk_content(content: str, *, max_tokens: int = DEFAULT_MAX_TOKENS) -> list[Chunk]:
    """Chunk ``content`` into ordered :class:`Chunk` objects.

    Consecutive blocks are greedily merged until adding the next block would push
    the running chunk over ``max_tokens``; blocks are never split. A single block
    already larger than ``max_tokens`` becomes its own chunk. Empty/whitespace
    blocks are skipped, so empty content yields an empty list.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    blocks = split_blocks(content or "")
    chunks: list[Chunk] = []
    current: list[str] = []
    current_tokens = 0

    for block in blocks:
        block_tokens = estimate_tokens(block)
        if current and current_tokens + block_tokens > max_tokens:
            chunks.append(Chunk(index=len(chunks), content="\n".join(current)))
            current = [block]
            current_tokens = block_tokens
        else:
            current.append(block)
            current_tokens += block_tokens

    if current:
        chunks.append(Chunk(index=len(chunks), content="\n".join(current)))

    return chunks


def embedding_input(title: str | None, chunk_text: str) -> str:
    """Build the text sent to the embedder for a chunk.

    The page ``title`` is prepended for topical context (improves retrieval) but
    is NOT stored on the chunk (TRD §5). Falls back to the chunk text alone when
    no title is available.
    """
    clean_title = (title or "").strip()
    if not clean_title:
        return chunk_text
    return f"{clean_title}\n\n{chunk_text}"
