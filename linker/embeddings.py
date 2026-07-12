"""Local embedding client (sentence-transformers).

Uses ``BAAI/bge-base-en-v1.5`` (768-dimensional) to embed text locally — no API
key, no rate limit, no network at inference time after the one-time model
download. The model is loaded once and reused for every batch.

bge draws a task distinction (mirroring the old Gemini
``RETRIEVAL_DOCUMENT`` / ``RETRIEVAL_QUERY`` split):

* **documents** (stored chunks) are embedded from their raw text, no prefix;
* **queries** (search text, used from M3) are prefixed with bge's recommended
  instruction so the query and passage vectors live in the same space.

Output vectors are L2-normalized (``normalize_embeddings=True``) so a cosine
distance search (``vector_cosine_ops``) is well-behaved.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

log = logging.getLogger(__name__)

# Task tags, mirroring the old Gemini RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY split.
TASK_DOCUMENT = "document"
TASK_QUERY = "query"

# bge-base-en-v1.5's recommended instruction for the *query* side of retrieval.
# Documents are embedded without any prefix; only queries get this.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages:"

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_BATCH_SIZE = 32


class LocalEmbedder:
    """Batched wrapper over a sentence-transformers model.

    The model is loaded once in ``__init__`` and reused across every ``embed``
    call; there is no per-batch reload and no rate limiting.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        dim: int = 768,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        # Imported lazily so config/chunking unit tests don't pay the (heavy)
        # torch import cost, and so a missing optional dep surfaces only when an
        # embedder is actually constructed.
        from sentence_transformers import SentenceTransformer

        self._model_name = model
        self._model = SentenceTransformer(model)
        self._dim = dim
        self._batch_size = max(1, batch_size)

    @property
    def dim(self) -> int:
        """The configured output dimensionality."""
        return self._dim

    def embed(self, texts: Sequence[str], task_type: str) -> list[np.ndarray]:
        """Embed ``texts`` and return unit-normalized float32 vectors, in order.

        ``task_type`` is :data:`TASK_DOCUMENT` for stored chunks (raw text) or
        :data:`TASK_QUERY` for search queries (bge query instruction prepended).
        sentence-transformers handles internal mini-batching via ``batch_size``.
        """
        items = list(texts)
        if not items:
            return []

        if task_type == TASK_QUERY:
            inputs = [f"{QUERY_INSTRUCTION} {text}" for text in items]
        elif task_type == TASK_DOCUMENT:
            inputs = items
        else:
            raise ValueError(
                f"Unknown task_type {task_type!r}; expected "
                f"{TASK_DOCUMENT!r} or {TASK_QUERY!r}."
            )

        matrix = self._model.encode(
            inputs,
            batch_size=self._batch_size,
            normalize_embeddings=True,  # L2-normalize so cosine distance is correct
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [self._to_vector(row) for row in matrix]

    def _to_vector(self, values: Sequence[float]) -> np.ndarray:
        """Validate dimensionality and return a float32 vector.

        The model already returns L2-normalized rows (``normalize_embeddings``),
        so this only enforces the expected dimension.
        """
        vec = np.asarray(values, dtype=np.float32)
        if vec.shape[0] != self._dim:
            raise ValueError(
                f"Embedding dim {vec.shape[0]} != expected {self._dim}; "
                f"check EMBED_MODEL / EMBED_DIM."
            )
        return vec

    def smoke_test(self) -> int:
        """Embed a tiny probe to fail fast if the model/dim is wrong.

        Returns the observed dimension. Also forces the (one-time) model download
        and load up front, before the full corpus is embedded.
        """
        vec = self.embed(["internal linking smoke test"], TASK_QUERY)[0]
        return int(vec.shape[0])
