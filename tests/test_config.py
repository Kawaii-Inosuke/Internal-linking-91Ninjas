"""Unit tests for environment configuration (TRD §10).

Pure-logic tests: no database and no Gemini API required. ``load_dotenv_file`` is
disabled so results depend only on the monkeypatched environment.
"""
from __future__ import annotations

import pytest

from linker.config import Config, ConfigError


@pytest.fixture
def base_env(monkeypatch):
    """Set the two required variables; individual tests tweak the rest."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:5432/linker")
    for optional in ("EMBED_MODEL", "EMBED_DIM", "EMBED_BATCH_SIZE", "EMBED_THROTTLE_SECONDS"):
        monkeypatch.delenv(optional, raising=False)


def _load() -> Config:
    return Config.from_env(load_dotenv_file=False)


def test_defaults(base_env):
    config = _load()
    assert config.gemini_api_key == "test-key"
    assert config.embed_model == "gemini-embedding-001"
    assert config.embed_dim == 768
    assert config.embed_batch_size == 100
    assert config.embed_throttle_seconds == 0.5


def test_missing_required_lists_all(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(ConfigError) as exc:
        _load()
    assert "GEMINI_API_KEY" in str(exc.value)
    assert "DATABASE_URL" in str(exc.value)


def test_invalid_embed_dim_rejected(base_env, monkeypatch):
    monkeypatch.setenv("EMBED_DIM", "not-a-number")
    with pytest.raises(ConfigError):
        _load()


def test_nonpositive_embed_dim_rejected(base_env, monkeypatch):
    monkeypatch.setenv("EMBED_DIM", "0")
    with pytest.raises(ConfigError):
        _load()


def test_batch_size_out_of_range_rejected(base_env, monkeypatch):
    monkeypatch.setenv("EMBED_BATCH_SIZE", "101")
    with pytest.raises(ConfigError):
        _load()


def test_negative_throttle_rejected(base_env, monkeypatch):
    monkeypatch.setenv("EMBED_THROTTLE_SECONDS", "-1")
    with pytest.raises(ConfigError):
        _load()


def test_custom_values_parsed(base_env, monkeypatch):
    monkeypatch.setenv("EMBED_DIM", "1536")
    monkeypatch.setenv("EMBED_BATCH_SIZE", "50")
    monkeypatch.setenv("EMBED_THROTTLE_SECONDS", "0")
    config = _load()
    assert config.embed_dim == 1536
    assert config.embed_batch_size == 50
    assert config.embed_throttle_seconds == 0.0
