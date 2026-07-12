"""Unit tests for environment configuration (TRD §10).

Pure-logic tests: no database and no embedding model required. ``load_dotenv_file``
is disabled so results depend only on the monkeypatched environment.
"""
from __future__ import annotations

import pytest

from linker.config import Config, ConfigError


@pytest.fixture
def base_env(monkeypatch):
    """Set the required variable; individual tests tweak the rest."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:5432/linker")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    for optional in ("EMBED_MODEL", "EMBED_DIM", "EMBED_BATCH_SIZE"):
        monkeypatch.delenv(optional, raising=False)


def _load() -> Config:
    return Config.from_env(load_dotenv_file=False)


def test_defaults(base_env):
    config = _load()
    assert config.embed_model == "BAAI/bge-base-en-v1.5"
    assert config.embed_dim == 768
    assert config.embed_batch_size == 32
    # Gemini key is optional (embeddings are local); absent -> empty string.
    assert config.gemini_api_key == ""


def test_gemini_key_kept_when_present(base_env, monkeypatch):
    # Embeddings don't use it, but the M3 LLM gate does, so it is still loaded.
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert _load().gemini_api_key == "test-key"


def test_missing_database_url_rejected(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(ConfigError) as exc:
        _load()
    assert "DATABASE_URL" in str(exc.value)


def test_missing_gemini_key_is_allowed(base_env, monkeypatch):
    # M1 ingestion is fully local; a missing Gemini key must not block it.
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert _load().gemini_api_key == ""


def test_invalid_embed_dim_rejected(base_env, monkeypatch):
    monkeypatch.setenv("EMBED_DIM", "not-a-number")
    with pytest.raises(ConfigError):
        _load()


def test_nonpositive_embed_dim_rejected(base_env, monkeypatch):
    monkeypatch.setenv("EMBED_DIM", "0")
    with pytest.raises(ConfigError):
        _load()


def test_nonpositive_batch_size_rejected(base_env, monkeypatch):
    monkeypatch.setenv("EMBED_BATCH_SIZE", "0")
    with pytest.raises(ConfigError):
        _load()


def test_custom_values_parsed(base_env, monkeypatch):
    monkeypatch.setenv("EMBED_MODEL", "BAAI/bge-large-en-v1.5")
    monkeypatch.setenv("EMBED_DIM", "1024")
    monkeypatch.setenv("EMBED_BATCH_SIZE", "64")
    config = _load()
    assert config.embed_model == "BAAI/bge-large-en-v1.5"
    assert config.embed_dim == 1024
    assert config.embed_batch_size == 64
