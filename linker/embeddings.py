"""Gemini embedding client (google-genai SDK).

Uses ``gemini-embedding-001`` at 768 dimensions with task-type-specific inputs
(``RETRIEVAL_DOCUMENT`` for stored chunks, ``RETRIEVAL_QUERY`` for queries). Sends
up to 100 texts per request and retries with exponential backoff on rate limits.
"""
from __future__ import annotations

import logging
import random
import time
from collections.abc import Sequence

import numpy as np
from google import genai
from google.genai import types

log = logging.getLogger(__name__)

# Gemini embedding requests accept at most 100 inputs; larger lists are batched.
MAX_BATCH = 100
TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_QUERY = "RETRIEVAL_QUERY"

_RATE_LIMIT_MARKERS = ("429", "resource_exhausted", "rate limit", "quota")


def _is_rate_limit(exc: Exception) -> bool:
    """Return True if ``exc`` looks like a retryable rate-limit / quota error."""
    if getattr(exc, "code", None) == 429:
        return True
    text = str(exc).lower()
    return any(marker in text for marker in _RATE_LIMIT_MARKERS)


class GeminiEmbedder:
    """Thin, batched wrapper over the google-genai embeddings endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str,
        dim: int,
        *,
        batch_size: int = MAX_BATCH,
        max_retries: int = 5,
        throttle_seconds: float = 0.0,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._dim = dim
        self._batch_size = max(1, min(batch_size, MAX_BATCH))
        self._max_retries = max_retries
        self._throttle = throttle_seconds

    @property
    def dim(self) -> int:
        """The configured output dimensionality."""
        return self._dim

    def embed(self, texts: Sequence[str], task_type: str) -> list[np.ndarray]:
        """Embed ``texts`` and return unit-normalized float32 vectors, in order.

        Batches into groups of ``batch_size`` (<= 100), throttling between batches
        and retrying each batch with backoff on rate-limit errors.
        """
        items = list(texts)
        out: list[np.ndarray] = []
        for start in range(0, len(items), self._batch_size):
            batch = items[start : start + self._batch_size]
            out.extend(self._embed_batch(batch, task_type))
            if self._throttle and start + self._batch_size < len(items):
                time.sleep(self._throttle)
        return out

    def _embed_batch(self, batch: Sequence[str], task_type: str) -> list[np.ndarray]:
        attempt = 0
        while True:
            try:
                resp = self._client.models.embed_content(
                    model=self._model,
                    contents=list(batch),
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self._dim,
                    ),
                )
                return [self._to_vector(item.values) for item in resp.embeddings]
            except Exception as exc:  # noqa: BLE001 - classify, then retry or re-raise
                attempt += 1
                if not _is_rate_limit(exc) or attempt > self._max_retries:
                    raise
                delay = min(60.0, 2.0**attempt) + random.uniform(0, 1)
                log.warning(
                    "Embedding rate-limited (attempt %d/%d); backing off %.1fs",
                    attempt,
                    self._max_retries,
                    delay,
                )
                time.sleep(delay)

    def _to_vector(self, values: Sequence[float]) -> np.ndarray:
        """Validate dimensionality and L2-normalize a single embedding."""
        vec = np.asarray(values, dtype=np.float32)
        if vec.shape[0] != self._dim:
            raise ValueError(
                f"Embedding dim {vec.shape[0]} != expected {self._dim}; "
                f"check EMBED_MODEL / EMBED_DIM."
            )
        # gemini-embedding-001 only normalizes its full 3072-d output; truncated
        # dimensions must be re-normalized so cosine similarity is well-behaved.
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec

    def smoke_test(self) -> int:
        """Embed a tiny probe to fail fast if the model/key/dim is wrong.

        Returns the observed dimension. This verifies at run time that the
        configured model name is currently valid (model names change — TRD §2).
        """
        vec = self.embed(["internal linking smoke test"], TASK_QUERY)[0]
        return int(vec.shape[0])
