"""Unit tests for the local embedder (sentence-transformers).

These avoid the (heavy) real model download by injecting a fake
``sentence_transformers`` module. They pin the behaviour that matters for
correctness: the document-vs-query task split (bge query instruction prefix),
L2 normalization being requested, and dimension validation.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from linker.embeddings import (
    QUERY_INSTRUCTION,
    TASK_DOCUMENT,
    TASK_QUERY,
    LocalEmbedder,
)


class _FakeModel:
    """Records encode() inputs/kwargs and returns fixed-width unit rows."""

    def __init__(self, name: str, width: int = 4) -> None:
        self.name = name
        self.width = width
        self.calls: list[list[str]] = []
        self.last_kwargs: dict | None = None

    def encode(self, inputs, **kwargs):
        self.calls.append(list(inputs))
        self.last_kwargs = kwargs
        return np.ones((len(inputs), self.width), dtype=np.float64)


@pytest.fixture
def fake_st(monkeypatch):
    """Install a fake sentence_transformers module for the duration of a test."""
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _FakeModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_document_embedding_has_no_prefix(fake_st):
    emb = LocalEmbedder("fake", dim=4)
    emb.embed(["cart abandonment tips"], TASK_DOCUMENT)
    assert emb._model.calls[-1] == ["cart abandonment tips"]


def test_query_embedding_prepends_instruction(fake_st):
    emb = LocalEmbedder("fake", dim=4)
    emb.embed(["cart abandonment tips"], TASK_QUERY)
    assert emb._model.calls[-1] == [f"{QUERY_INSTRUCTION} cart abandonment tips"]


def test_normalization_is_requested(fake_st):
    emb = LocalEmbedder("fake", dim=4)
    emb.embed(["x"], TASK_DOCUMENT)
    assert emb._model.last_kwargs["normalize_embeddings"] is True


def test_batch_size_is_forwarded(fake_st):
    emb = LocalEmbedder("fake", dim=4, batch_size=16)
    emb.embed(["x"], TASK_DOCUMENT)
    assert emb._model.last_kwargs["batch_size"] == 16


def test_empty_input_returns_empty_without_calling_model(fake_st):
    emb = LocalEmbedder("fake", dim=4)
    assert emb.embed([], TASK_DOCUMENT) == []
    assert emb._model.calls == []


def test_vectors_are_float32(fake_st):
    emb = LocalEmbedder("fake", dim=4)
    out = emb.embed(["x", "y"], TASK_DOCUMENT)
    assert len(out) == 2
    assert all(v.dtype == np.float32 and v.shape == (4,) for v in out)


def test_unknown_task_type_rejected(fake_st):
    emb = LocalEmbedder("fake", dim=4)
    with pytest.raises(ValueError):
        emb.embed(["x"], "RETRIEVAL_DOCUMENT")  # old Gemini tag, no longer valid


def test_dimension_mismatch_rejected(fake_st):
    # Model yields width-4 rows but we expect 768: must fail fast.
    emb = LocalEmbedder("fake", dim=768)
    with pytest.raises(ValueError):
        emb.embed(["x"], TASK_DOCUMENT)


def test_smoke_test_returns_dim(fake_st):
    emb = LocalEmbedder("fake", dim=4)
    assert emb.smoke_test() == 4
