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

    gemini_api_key: str
    database_url: str
    embed_model: str
    embed_dim: int
    embed_batch_size: int
    embed_throttle_seconds: float

    @classmethod
    def from_env(cls, *, load_dotenv_file: bool = True) -> "Config":
        """Build a :class:`Config` from environment variables.

        Raises :class:`ConfigError` with an actionable message if a required
        variable is missing or malformed.
        """
        if load_dotenv_file:
            load_dotenv()

        missing = [key for key in ("GEMINI_API_KEY", "DATABASE_URL") if not os.getenv(key)]
        if missing:
            raise ConfigError(
                "Missing required environment variable(s): "
                + ", ".join(missing)
                + ". Copy .env.example to .env and fill them in."
            )

        embed_dim = _int_env("EMBED_DIM", 768)
        if embed_dim <= 0:
            raise ConfigError(f"EMBED_DIM must be a positive integer, got {embed_dim}.")

        # Gemini embedding requests accept at most 100 inputs per call.
        embed_batch_size = _int_env("EMBED_BATCH_SIZE", 100)
        if not 1 <= embed_batch_size <= 100:
            raise ConfigError(
                f"EMBED_BATCH_SIZE must be between 1 and 100, got {embed_batch_size}."
            )

        # Proactive pacing between embedding requests to respect free-tier RPM
        # (in addition to reactive 429 backoff); set 0 to disable.
        embed_throttle_seconds = _float_env("EMBED_THROTTLE_SECONDS", 0.5)
        if embed_throttle_seconds < 0:
            raise ConfigError(
                f"EMBED_THROTTLE_SECONDS must be >= 0, got {embed_throttle_seconds}."
            )

        return cls(
            gemini_api_key=os.environ["GEMINI_API_KEY"],
            database_url=os.environ["DATABASE_URL"],
            embed_model=os.getenv("EMBED_MODEL", "gemini-embedding-001"),
            embed_dim=embed_dim,
            embed_batch_size=embed_batch_size,
            embed_throttle_seconds=embed_throttle_seconds,
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


def _float_env(name: str, default: float) -> float:
    """Read a float environment variable, raising :class:`ConfigError` if malformed."""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}.") from exc
