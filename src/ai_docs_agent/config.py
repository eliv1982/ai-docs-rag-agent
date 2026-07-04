"""Typed application configuration loaded from environment variables."""

from functools import lru_cache
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_SUPPORTED_METRICS = frozenset({"cosine"})
_SUPPORTED_HTTP_SCHEMES = frozenset({"http", "https"})


class AppSettings(BaseSettings):
    """OpenAI and Pinecone configuration for the AI Docs RAG Agent."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    openai_api_key: SecretStr = Field(validation_alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, validation_alias="OPENAI_BASE_URL")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", validation_alias="OPENAI_EMBEDDING_MODEL"
    )
    openai_chat_model: str = Field(validation_alias="OPENAI_CHAT_MODEL")

    pinecone_api_key: SecretStr = Field(validation_alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field(
        default="ai-docs-rag-agent", validation_alias="PINECONE_INDEX_NAME"
    )
    pinecone_cloud: str = Field(default="aws", validation_alias="PINECONE_CLOUD")
    pinecone_region: str = Field(default="us-east-1", validation_alias="PINECONE_REGION")
    pinecone_dimension: int = Field(default=1536, validation_alias="PINECONE_DIMENSION")
    pinecone_metric: str = Field(default="cosine", validation_alias="PINECONE_METRIC")
    pinecone_create_if_missing: bool = Field(
        default=False, validation_alias="PINECONE_CREATE_IF_MISSING"
    )

    pinecone_smoke_namespace: str = Field(
        default="__smoke_test__", validation_alias="PINECONE_SMOKE_NAMESPACE"
    )
    pinecone_smoke_timeout_seconds: float = Field(
        default=30, validation_alias="PINECONE_SMOKE_TIMEOUT_SECONDS"
    )
    pinecone_smoke_poll_interval_seconds: float = Field(
        default=1, validation_alias="PINECONE_SMOKE_POLL_INTERVAL_SECONDS"
    )

    url_fetch_timeout_seconds: float = Field(
        default=15, validation_alias="URL_FETCH_TIMEOUT_SECONDS"
    )
    url_max_response_bytes: int = Field(
        default=2_000_000, validation_alias="URL_MAX_RESPONSE_BYTES"
    )
    url_max_redirects: int = Field(default=5, validation_alias="URL_MAX_REDIRECTS")
    url_min_text_chars: int = Field(default=200, validation_alias="URL_MIN_TEXT_CHARS")
    url_user_agent: str = Field(
        default="ai-docs-rag-agent/0.1", validation_alias="URL_USER_AGENT"
    )
    chunk_size: int = Field(default=1200, validation_alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, validation_alias="CHUNK_OVERLAP")

    pinecone_documents_namespace: str = Field(
        default="documentation", validation_alias="PINECONE_DOCUMENTS_NAMESPACE"
    )
    embedding_batch_size: int = Field(default=64, validation_alias="EMBEDDING_BATCH_SIZE")
    pinecone_upsert_batch_size: int = Field(
        default=100, validation_alias="PINECONE_UPSERT_BATCH_SIZE"
    )
    pinecone_fetch_batch_size: int = Field(
        default=500, validation_alias="PINECONE_FETCH_BATCH_SIZE"
    )
    pinecone_index_verify_timeout_seconds: float = Field(
        default=30, validation_alias="PINECONE_INDEX_VERIFY_TIMEOUT_SECONDS"
    )
    pinecone_index_verify_poll_interval_seconds: float = Field(
        default=1, validation_alias="PINECONE_INDEX_VERIFY_POLL_INTERVAL_SECONDS"
    )
    pinecone_replace_old_source_versions: bool = Field(
        default=True, validation_alias="PINECONE_REPLACE_OLD_SOURCE_VERSIONS"
    )

    retrieval_top_k: int = Field(default=5, validation_alias="RETRIEVAL_TOP_K")
    pypi_base_url: str = Field(default="https://pypi.org", validation_alias="PYPI_BASE_URL")
    pypi_timeout_seconds: float = Field(default=10, validation_alias="PYPI_TIMEOUT_SECONDS")

    telegram_bot_token: SecretStr = Field(validation_alias="TELEGRAM_BOT_TOKEN")

    @field_validator("pinecone_dimension")
    @classmethod
    def _validate_dimension(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("pinecone_dimension must be positive.")
        return value

    @field_validator("pinecone_smoke_timeout_seconds")
    @classmethod
    def _validate_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("pinecone_smoke_timeout_seconds must be greater than zero.")
        return value

    @field_validator("pinecone_smoke_poll_interval_seconds")
    @classmethod
    def _validate_poll_interval(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("pinecone_smoke_poll_interval_seconds must be greater than zero.")
        return value

    @field_validator("pinecone_index_name", "openai_embedding_model", "openai_chat_model")
    @classmethod
    def _validate_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty.")
        return value

    @field_validator("openai_base_url")
    @classmethod
    def _normalize_base_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("pinecone_metric")
    @classmethod
    def _validate_metric(cls, value: str) -> str:
        if value not in _SUPPORTED_METRICS:
            raise ValueError(
                f"Unsupported pinecone_metric '{value}'. Supported values: "
                f"{sorted(_SUPPORTED_METRICS)}."
            )
        return value

    @field_validator("url_fetch_timeout_seconds")
    @classmethod
    def _validate_url_fetch_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("url_fetch_timeout_seconds must be greater than zero.")
        return value

    @field_validator("url_max_response_bytes")
    @classmethod
    def _validate_url_max_response_bytes(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("url_max_response_bytes must be greater than zero.")
        return value

    @field_validator("url_max_redirects")
    @classmethod
    def _validate_url_max_redirects(cls, value: int) -> int:
        if value < 0 or value > 10:
            raise ValueError("url_max_redirects must be between 0 and 10 inclusive.")
        return value

    @field_validator("url_min_text_chars")
    @classmethod
    def _validate_url_min_text_chars(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("url_min_text_chars must be greater than zero.")
        return value

    @field_validator("url_user_agent")
    @classmethod
    def _validate_url_user_agent(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("url_user_agent must not be empty.")
        return value

    @field_validator("chunk_size")
    @classmethod
    def _validate_chunk_size(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("chunk_size must be greater than zero.")
        return value

    @field_validator("chunk_overlap")
    @classmethod
    def _validate_chunk_overlap(cls, value: int) -> int:
        if value < 0:
            raise ValueError("chunk_overlap must be greater than or equal to zero.")
        return value

    @field_validator("pinecone_documents_namespace")
    @classmethod
    def _validate_pinecone_documents_namespace(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("pinecone_documents_namespace must not be empty.")
        return stripped

    @field_validator("embedding_batch_size")
    @classmethod
    def _validate_embedding_batch_size(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("embedding_batch_size must be greater than zero.")
        return value

    @field_validator("pinecone_upsert_batch_size")
    @classmethod
    def _validate_pinecone_upsert_batch_size(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("pinecone_upsert_batch_size must be greater than zero.")
        return value

    @field_validator("pinecone_fetch_batch_size")
    @classmethod
    def _validate_pinecone_fetch_batch_size(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("pinecone_fetch_batch_size must be greater than zero.")
        if value > 1000:
            raise ValueError("pinecone_fetch_batch_size must not exceed 1000.")
        return value

    @field_validator("pinecone_index_verify_timeout_seconds")
    @classmethod
    def _validate_index_verify_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("pinecone_index_verify_timeout_seconds must be greater than zero.")
        return value

    @field_validator("pinecone_index_verify_poll_interval_seconds")
    @classmethod
    def _validate_index_verify_poll_interval(cls, value: float) -> float:
        if value <= 0:
            raise ValueError(
                "pinecone_index_verify_poll_interval_seconds must be greater than zero."
            )
        return value

    @field_validator("retrieval_top_k")
    @classmethod
    def _validate_retrieval_top_k(cls, value: int) -> int:
        if value < 1 or value > 50:
            raise ValueError("retrieval_top_k must be between 1 and 50 inclusive.")
        return value

    @field_validator("pypi_base_url")
    @classmethod
    def _validate_pypi_base_url(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("pypi_base_url must not be empty.")

        parsed = urlsplit(stripped)
        if parsed.scheme not in _SUPPORTED_HTTP_SCHEMES:
            raise ValueError("pypi_base_url must use http or https.")
        if not parsed.hostname:
            raise ValueError("pypi_base_url must include a hostname.")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("pypi_base_url must not include embedded credentials.")
        if parsed.query or parsed.fragment:
            raise ValueError("pypi_base_url must not include query or fragment components.")
        if parsed.path not in ("", "/"):
            raise ValueError("pypi_base_url must not include a path component.")

        return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")

    @field_validator("pypi_timeout_seconds")
    @classmethod
    def _validate_pypi_timeout_seconds(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("pypi_timeout_seconds must be greater than zero.")
        return value

    @field_validator("telegram_bot_token")
    @classmethod
    def _validate_telegram_bot_token(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("telegram_bot_token must not be empty.")
        return value

    @model_validator(mode="after")
    def _validate_poll_interval_within_timeout(self) -> "AppSettings":
        if self.pinecone_smoke_poll_interval_seconds > self.pinecone_smoke_timeout_seconds:
            raise ValueError(
                "pinecone_smoke_poll_interval_seconds cannot exceed "
                "pinecone_smoke_timeout_seconds."
            )
        return self

    @model_validator(mode="after")
    def _validate_chunk_overlap_within_chunk_size(self) -> "AppSettings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size.")
        return self

    @model_validator(mode="after")
    def _validate_index_verify_poll_interval_within_timeout(self) -> "AppSettings":
        if (
            self.pinecone_index_verify_poll_interval_seconds
            > self.pinecone_index_verify_timeout_seconds
        ):
            raise ValueError(
                "pinecone_index_verify_poll_interval_seconds cannot exceed "
                "pinecone_index_verify_timeout_seconds."
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return a cached AppSettings instance loaded from the environment."""
    return AppSettings()
