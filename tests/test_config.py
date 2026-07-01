"""Unit tests for AppSettings. No network access; .env isolated via _env_file=None."""

from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr, ValidationError

from ai_docs_agent.config import AppSettings, get_settings

_REQUIRED: dict[str, Any] = {
    "openai_api_key": "sk-test-openai",
    "pinecone_api_key": "pc-test-key",
}

# All environment variable names AppSettings binds to, used to isolate tests
# from a developer's ambient shell environment or a local .env file.
_ALL_SETTINGS_ENV_VARS = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_EMBEDDING_MODEL",
    "PINECONE_API_KEY",
    "PINECONE_INDEX_NAME",
    "PINECONE_CLOUD",
    "PINECONE_REGION",
    "PINECONE_DIMENSION",
    "PINECONE_METRIC",
    "PINECONE_CREATE_IF_MISSING",
    "PINECONE_SMOKE_NAMESPACE",
    "PINECONE_SMOKE_TIMEOUT_SECONDS",
    "PINECONE_SMOKE_POLL_INTERVAL_SECONDS",
    "URL_FETCH_TIMEOUT_SECONDS",
    "URL_MAX_RESPONSE_BYTES",
    "URL_MAX_REDIRECTS",
    "URL_MIN_TEXT_CHARS",
    "URL_USER_AGENT",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
)


def make_settings(**overrides: Any) -> AppSettings:
    return AppSettings(_env_file=None, **{**_REQUIRED, **overrides})


def test_defaults() -> None:
    settings = make_settings()

    assert settings.openai_base_url is None
    assert settings.openai_embedding_model == "text-embedding-3-small"
    assert settings.pinecone_index_name == "ai-docs-rag-agent"
    assert settings.pinecone_cloud == "aws"
    assert settings.pinecone_region == "us-east-1"
    assert settings.pinecone_dimension == 1536
    assert settings.pinecone_metric == "cosine"
    assert settings.pinecone_create_if_missing is False
    assert settings.pinecone_smoke_namespace == "__smoke_test__"
    assert settings.pinecone_smoke_timeout_seconds == 30
    assert settings.pinecone_smoke_poll_interval_seconds == 1
    assert settings.url_fetch_timeout_seconds == 15
    assert settings.url_max_response_bytes == 2_000_000
    assert settings.url_max_redirects == 5
    assert settings.url_min_text_chars == 200
    assert settings.url_user_agent == "ai-docs-rag-agent/0.1"
    assert settings.chunk_size == 1200
    assert settings.chunk_overlap == 200


def test_overrides_are_applied() -> None:
    settings = make_settings(
        pinecone_index_name="custom-index",
        pinecone_dimension=768,
        pinecone_create_if_missing=True,
    )

    assert settings.pinecone_index_name == "custom-index"
    assert settings.pinecone_dimension == 768
    assert settings.pinecone_create_if_missing is True


def test_secret_str_is_not_leaked_in_repr() -> None:
    settings = make_settings(openai_api_key="super-secret-value")

    assert isinstance(settings.openai_api_key, SecretStr)
    assert "super-secret-value" not in repr(settings)
    assert "super-secret-value" not in repr(settings.openai_api_key)


def test_rejects_non_positive_dimension() -> None:
    with pytest.raises(ValidationError):
        make_settings(pinecone_dimension=0)


def test_rejects_unsupported_metric() -> None:
    with pytest.raises(ValidationError):
        make_settings(pinecone_metric="euclidean")


def test_rejects_poll_interval_greater_than_timeout() -> None:
    with pytest.raises(ValidationError):
        make_settings(
            pinecone_smoke_timeout_seconds=5,
            pinecone_smoke_poll_interval_seconds=10,
        )


def test_blank_openai_base_url_is_normalized_to_none() -> None:
    settings = make_settings(openai_base_url="")

    assert settings.openai_base_url is None


def test_whitespace_only_openai_base_url_is_normalized_to_none() -> None:
    settings = make_settings(openai_base_url="   ")

    assert settings.openai_base_url is None


def test_non_empty_openai_base_url_is_preserved() -> None:
    settings = make_settings(openai_base_url="https://example.invalid/v1")

    assert settings.openai_base_url == "https://example.invalid/v1"


def test_rejects_non_positive_url_fetch_timeout() -> None:
    with pytest.raises(ValidationError):
        make_settings(url_fetch_timeout_seconds=0)


def test_rejects_non_positive_url_max_response_bytes() -> None:
    with pytest.raises(ValidationError):
        make_settings(url_max_response_bytes=0)


def test_rejects_negative_url_max_redirects() -> None:
    with pytest.raises(ValidationError):
        make_settings(url_max_redirects=-1)


def test_rejects_url_max_redirects_above_ten() -> None:
    with pytest.raises(ValidationError):
        make_settings(url_max_redirects=11)


def test_accepts_url_max_redirects_boundary_values() -> None:
    assert make_settings(url_max_redirects=0).url_max_redirects == 0
    assert make_settings(url_max_redirects=10).url_max_redirects == 10


def test_rejects_non_positive_url_min_text_chars() -> None:
    with pytest.raises(ValidationError):
        make_settings(url_min_text_chars=0)


def test_rejects_empty_url_user_agent() -> None:
    with pytest.raises(ValidationError):
        make_settings(url_user_agent="   ")


def test_rejects_non_positive_chunk_size() -> None:
    with pytest.raises(ValidationError):
        make_settings(chunk_size=0)


def test_rejects_negative_chunk_overlap() -> None:
    with pytest.raises(ValidationError):
        make_settings(chunk_overlap=-1)


def test_rejects_chunk_overlap_greater_than_or_equal_to_chunk_size() -> None:
    with pytest.raises(ValidationError):
        make_settings(chunk_size=100, chunk_overlap=100)


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in _ALL_SETTINGS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    monkeypatch.setenv("PINECONE_API_KEY", "pc-test-key")
    # get_settings() reads AppSettings() with its default env_file=".env"; running
    # from an empty tmp_path ensures no real .env is ever read, even if one is
    # later added to the project root.
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()

    try:
        first = get_settings()
        second = get_settings()
        assert first is second
    finally:
        get_settings.cache_clear()
