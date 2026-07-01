"""Typed application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_SUPPORTED_METRICS = frozenset({"cosine"})


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

    @field_validator("pinecone_index_name", "openai_embedding_model")
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

    @model_validator(mode="after")
    def _validate_poll_interval_within_timeout(self) -> "AppSettings":
        if self.pinecone_smoke_poll_interval_seconds > self.pinecone_smoke_timeout_seconds:
            raise ValueError(
                "pinecone_smoke_poll_interval_seconds cannot exceed "
                "pinecone_smoke_timeout_seconds."
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return a cached AppSettings instance loaded from the environment."""
    return AppSettings()
