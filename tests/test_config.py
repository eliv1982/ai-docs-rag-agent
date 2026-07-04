"""Unit tests for AppSettings. No network access; .env isolated via _env_file=None."""

from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr, ValidationError

from ai_docs_agent.config import AppSettings, get_settings

_REQUIRED: dict[str, Any] = {
    "openai_api_key": "sk-test-openai",
    "pinecone_api_key": "pc-test-key",
    "openai_chat_model": "gpt-4o-mini",
    "telegram_bot_token": "test-telegram-token",
}

# All environment variable names AppSettings binds to, used to isolate tests
# from a developer's ambient shell environment or a local .env file.
_ALL_SETTINGS_ENV_VARS = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_EMBEDDING_MODEL",
    "OPENAI_CHAT_MODEL",
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
    "PINECONE_DOCUMENTS_NAMESPACE",
    "EMBEDDING_BATCH_SIZE",
    "PINECONE_UPSERT_BATCH_SIZE",
    "PINECONE_FETCH_BATCH_SIZE",
    "PINECONE_INDEX_VERIFY_TIMEOUT_SECONDS",
    "PINECONE_INDEX_VERIFY_POLL_INTERVAL_SECONDS",
    "PINECONE_REPLACE_OLD_SOURCE_VERSIONS",
    "RETRIEVAL_TOP_K",
    "PYPI_BASE_URL",
    "PYPI_TIMEOUT_SECONDS",
    "TELEGRAM_BOT_TOKEN",
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
    assert settings.pinecone_documents_namespace == "documentation"
    assert settings.embedding_batch_size == 64
    assert settings.pinecone_upsert_batch_size == 100
    assert settings.pinecone_fetch_batch_size == 500
    assert settings.pinecone_index_verify_timeout_seconds == 30
    assert settings.pinecone_index_verify_poll_interval_seconds == 1
    assert settings.pinecone_replace_old_source_versions is True
    assert settings.retrieval_top_k == 5
    assert settings.pypi_base_url == "https://pypi.org"
    assert settings.pypi_timeout_seconds == 10


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


def test_overrides_are_applied_for_indexing_settings() -> None:
    settings = make_settings(
        pinecone_documents_namespace="custom-namespace",
        embedding_batch_size=32,
        pinecone_upsert_batch_size=50,
        pinecone_fetch_batch_size=250,
        pinecone_index_verify_timeout_seconds=15,
        pinecone_index_verify_poll_interval_seconds=2,
        pinecone_replace_old_source_versions=False,
    )

    assert settings.pinecone_documents_namespace == "custom-namespace"
    assert settings.embedding_batch_size == 32
    assert settings.pinecone_upsert_batch_size == 50
    assert settings.pinecone_fetch_batch_size == 250
    assert settings.pinecone_index_verify_timeout_seconds == 15
    assert settings.pinecone_index_verify_poll_interval_seconds == 2
    assert settings.pinecone_replace_old_source_versions is False


def test_pinecone_documents_namespace_is_stripped() -> None:
    settings = make_settings(pinecone_documents_namespace="  documentation  ")

    assert settings.pinecone_documents_namespace == "documentation"


def test_rejects_empty_pinecone_documents_namespace() -> None:
    with pytest.raises(ValidationError):
        make_settings(pinecone_documents_namespace="   ")


def test_rejects_non_positive_embedding_batch_size() -> None:
    with pytest.raises(ValidationError):
        make_settings(embedding_batch_size=0)


def test_rejects_non_positive_pinecone_upsert_batch_size() -> None:
    with pytest.raises(ValidationError):
        make_settings(pinecone_upsert_batch_size=0)


def test_rejects_non_positive_pinecone_fetch_batch_size() -> None:
    with pytest.raises(ValidationError):
        make_settings(pinecone_fetch_batch_size=0)


def test_rejects_pinecone_fetch_batch_size_above_1000() -> None:
    with pytest.raises(ValidationError):
        make_settings(pinecone_fetch_batch_size=1001)


def test_accepts_pinecone_fetch_batch_size_boundary_value() -> None:
    settings = make_settings(pinecone_fetch_batch_size=1000)

    assert settings.pinecone_fetch_batch_size == 1000


def test_rejects_non_positive_index_verify_timeout() -> None:
    with pytest.raises(ValidationError):
        make_settings(pinecone_index_verify_timeout_seconds=0)


def test_rejects_non_positive_index_verify_poll_interval() -> None:
    with pytest.raises(ValidationError):
        make_settings(pinecone_index_verify_poll_interval_seconds=0)


def test_pypi_base_url_is_stripped_and_trailing_slash_is_removed() -> None:
    settings = make_settings(pypi_base_url="  https://pypi.org/  ")

    assert settings.pypi_base_url == "https://pypi.org"


def test_rejects_blank_pypi_base_url() -> None:
    with pytest.raises(ValidationError):
        make_settings(pypi_base_url="   ")


def test_rejects_pypi_base_url_with_non_http_scheme() -> None:
    with pytest.raises(ValidationError):
        make_settings(pypi_base_url="ftp://pypi.org")


def test_rejects_pypi_base_url_without_hostname() -> None:
    with pytest.raises(ValidationError):
        make_settings(pypi_base_url="https:///missing-host")


def test_rejects_pypi_base_url_with_credentials() -> None:
    with pytest.raises(ValidationError):
        make_settings(pypi_base_url="https://user:pass@pypi.org")


def test_rejects_pypi_base_url_with_query_or_fragment() -> None:
    with pytest.raises(ValidationError):
        make_settings(pypi_base_url="https://pypi.org?x=1")
    with pytest.raises(ValidationError):
        make_settings(pypi_base_url="https://pypi.org#frag")


def test_rejects_pypi_base_url_with_path() -> None:
    with pytest.raises(ValidationError):
        make_settings(pypi_base_url="https://pypi.org/simple")


def test_rejects_non_positive_pypi_timeout_seconds() -> None:
    with pytest.raises(ValidationError):
        make_settings(pypi_timeout_seconds=0)


def test_rejects_index_verify_poll_interval_greater_than_timeout() -> None:
    with pytest.raises(ValidationError):
        make_settings(
            pinecone_index_verify_timeout_seconds=5,
            pinecone_index_verify_poll_interval_seconds=10,
        )


def test_retrieval_top_k_override_is_applied() -> None:
    settings = make_settings(retrieval_top_k=10)

    assert settings.retrieval_top_k == 10


def test_retrieval_top_k_reads_from_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in _ALL_SETTINGS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    monkeypatch.setenv("PINECONE_API_KEY", "pc-test-key")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-telegram-token")
    monkeypatch.setenv("RETRIEVAL_TOP_K", "12")

    settings = AppSettings(_env_file=None)

    assert settings.retrieval_top_k == 12


def test_rejects_zero_retrieval_top_k() -> None:
    with pytest.raises(ValidationError):
        make_settings(retrieval_top_k=0)


def test_rejects_negative_retrieval_top_k() -> None:
    with pytest.raises(ValidationError):
        make_settings(retrieval_top_k=-1)


def test_rejects_retrieval_top_k_above_50() -> None:
    with pytest.raises(ValidationError):
        make_settings(retrieval_top_k=51)


def test_accepts_retrieval_top_k_boundary_values() -> None:
    assert make_settings(retrieval_top_k=1).retrieval_top_k == 1
    assert make_settings(retrieval_top_k=50).retrieval_top_k == 50


def test_openai_chat_model_override_is_applied() -> None:
    settings = make_settings(openai_chat_model="gpt-4.1-mini")

    assert settings.openai_chat_model == "gpt-4.1-mini"


def test_openai_chat_model_reads_from_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in _ALL_SETTINGS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    monkeypatch.setenv("PINECONE_API_KEY", "pc-test-key")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-4o")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-telegram-token")

    settings = AppSettings(_env_file=None)

    assert settings.openai_chat_model == "gpt-4o"


def test_rejects_missing_openai_chat_model(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ALL_SETTINGS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    monkeypatch.setenv("PINECONE_API_KEY", "pc-test-key")

    with pytest.raises(ValidationError):
        AppSettings(_env_file=None)


def test_rejects_blank_openai_chat_model() -> None:
    with pytest.raises(ValidationError):
        make_settings(openai_chat_model="   ")


def test_telegram_bot_token_override_is_applied() -> None:
    settings = make_settings(telegram_bot_token="custom-token-value")

    assert settings.telegram_bot_token.get_secret_value() == "custom-token-value"


def test_telegram_bot_token_is_secret_str_not_leaked_in_repr() -> None:
    settings = make_settings(telegram_bot_token="super-secret-telegram-token")

    assert isinstance(settings.telegram_bot_token, SecretStr)
    assert "super-secret-telegram-token" not in repr(settings.telegram_bot_token)
    assert "super-secret-telegram-token" not in str(settings)


def test_telegram_bot_token_reads_from_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in _ALL_SETTINGS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    monkeypatch.setenv("PINECONE_API_KEY", "pc-test-key")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "env-telegram-token")

    settings = AppSettings(_env_file=None)

    assert settings.telegram_bot_token.get_secret_value() == "env-telegram-token"


def test_rejects_missing_telegram_bot_token(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ALL_SETTINGS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    monkeypatch.setenv("PINECONE_API_KEY", "pc-test-key")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    with pytest.raises(ValidationError):
        AppSettings(_env_file=None)


def test_rejects_blank_telegram_bot_token() -> None:
    with pytest.raises(ValidationError):
        make_settings(telegram_bot_token="   ")


def test_other_validation_errors_do_not_leak_telegram_bot_token() -> None:
    with pytest.raises(ValidationError) as exc_info:
        make_settings(
            telegram_bot_token="super-secret-telegram-value", pinecone_metric="euclidean"
        )

    assert "super-secret-telegram-value" not in str(exc_info.value)


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in _ALL_SETTINGS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    monkeypatch.setenv("PINECONE_API_KEY", "pc-test-key")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-telegram-token")
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
