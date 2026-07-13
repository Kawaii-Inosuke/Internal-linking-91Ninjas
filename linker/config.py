"""Configuration loading for the Internal Linking Assistant.

All secrets and tunables come from the environment (loaded from a local ``.env``
file via python-dotenv). Nothing is hardcoded. See ``.env.example``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Config:
    """Runtime configuration resolved from the environment.

    Only the values needed by the milestones implemented so far are loaded here;
    matching/LLM tunables (documented in ``.env.example``) are added as their
    milestones land.
    """

    # Kept for the M3 LLM relevance gate (Gemini Flash); NOT used for embeddings,
    # which run locally. Empty string when unset.
    gemini_api_key: str
    database_url: str
    embed_model: str
    embed_dim: int
    embed_batch_size: int
    # Max size of the ranked exact-keyword shortlist returned to the human
    # (TRD §6 Step A, M2 ranked-shortlist). See EXACT_MAX_RESULTS in .env.example.
    exact_max_results: int

    @classmethod
    def from_env(cls, *, load_dotenv_file: bool = True) -> "Config":
        """Build a :class:`Config` from environment variables.

        Raises :class:`ConfigError` with an actionable message if a required
        variable is missing or malformed. Only ``DATABASE_URL`` is required for
        M1 ingestion; ``GEMINI_API_KEY`` is optional here (embeddings are local)
        and is validated by the LLM gate that needs it in M3.
        """
        if load_dotenv_file:
            load_dotenv()

        if not os.getenv("DATABASE_URL"):
            raise ConfigError(
                "Missing required environment variable DATABASE_URL. "
                "Copy .env.example to .env and fill it in."
            )

        embed_dim = _int_env("EMBED_DIM", 768)
        if embed_dim <= 0:
            raise ConfigError(f"EMBED_DIM must be a positive integer, got {embed_dim}.")

        # Texts per sentence-transformers encode batch. A local model has no
        # request cap, so any positive value is fine; larger uses more memory.
        embed_batch_size = _int_env("EMBED_BATCH_SIZE", 32)
        if embed_batch_size < 1:
            raise ConfigError(
                f"EMBED_BATCH_SIZE must be a positive integer, got {embed_batch_size}."
            )

        # How many ranked candidates the exact-keyword pass returns. This is a
        # human decision-support shortlist, not an auto-linker, so the default is
        # small; the writer picks 1-3 from it.
        exact_max_results = _int_env("EXACT_MAX_RESULTS", 8)
        if exact_max_results < 1:
            raise ConfigError(
                f"EXACT_MAX_RESULTS must be a positive integer, got {exact_max_results}."
            )

        return cls(
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            database_url=os.environ["DATABASE_URL"],
            embed_model=os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5"),
            embed_dim=embed_dim,
            embed_batch_size=embed_batch_size,
            exact_max_results=exact_max_results,
        )


def _int_env(name: str, default: int) -> int:
    """Read an integer environment variable, raising :class:`ConfigError` if malformed."""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}.") from exc
