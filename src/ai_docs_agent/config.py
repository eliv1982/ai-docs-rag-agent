"""Typed application configuration loaded from environment variables.

Settings are split into small feature groups so each command validates only the
configuration it actually uses: a keyless command such as ``url_preview.py``
never needs an OpenAI, Pinecone, Telegram or memory secret. ``AppSettings`` is
the full composite used by the Telegram runtime (and by tests that need every
setting). User-facing entry points load their group through ``load_settings``,
which turns a pydantic ``ValidationError`` into a ``ConfigurationError`` whose
message names the offending variables but never echoes a supplied value.
"""

import re
from collections.abc import Sequence
from functools import lru_cache
from typing import TypeVar
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_SUPPORTED_METRICS = frozenset({"cosine"})
_SUPPORTED_HTTP_SCHEMES = frozenset({"http", "https"})
_NAMESPACE_PREFIX_PATTERN = re.compile(r"[A-Za-z0-9._-]+")
_VALUE_ERROR_PREFIX = "Value error, "


class ConfigurationError(Exception):
    """Raised when required configuration is missing or invalid.

    `problems` holds (variable name, reason) pairs. Neither the pairs nor the
    rendered message ever contain a configured value, so it is safe to print.
    """

    def __init__(self, problems: Sequence[tuple[str, str]]) -> None:
        self.problems = tuple(problems)
        lines = ["Invalid configuration:"]
        lines.extend(f"  - {name}: {reason}" for name, reason in self.problems)
        lines.append(
            "Set the variables above in the environment or in a .env file "
            "(see .env.example). Configured values are never printed."
        )
        super().__init__("\n".join(lines))

    @classmethod
    def from_validation_error(cls, exc: ValidationError) -> "ConfigurationError":
        """Redact a pydantic ValidationError down to variable names and reasons."""
        problems: list[tuple[str, str]] = []
        for error in exc.errors(include_input=False, include_url=False, include_context=False):
            name = ".".join(str(part) for part in error["loc"]) or "settings"
            if error["type"] == "missing":
                reason = "is required but not set"
            else:
                reason = error["msg"].removeprefix(_VALUE_ERROR_PREFIX)
            problems.append((name, reason))
        return cls(problems)


class _EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )


class UrlFetchSettings(_EnvSettings):
    """URL fetching and chunking configuration (no secrets)."""

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

    @model_validator(mode="after")
    def _validate_chunk_overlap_within_chunk_size(self) -> "UrlFetchSettings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size.")
        return self


class PyPISettings(_EnvSettings):
    """PyPI lookup configuration (no secrets)."""

    pypi_base_url: str = Field(default="https://pypi.org", validation_alias="PYPI_BASE_URL")
    pypi_timeout_seconds: float = Field(default=10, validation_alias="PYPI_TIMEOUT_SECONDS")

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


class VectorStoreSettings(_EnvSettings):
    """OpenAI embeddings plus Pinecone: everything retrieval and the smoke test need.

    Deliberately excludes the chat model, Telegram and user-memory settings.
    """

    openai_api_key: SecretStr = Field(validation_alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, validation_alias="OPENAI_BASE_URL")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", validation_alias="OPENAI_EMBEDDING_MODEL"
    )
    # Per-request timeout shared by every OpenAI client (chat model, embeddings,
    # direct grounded-answer client) so no call can hang indefinitely.
    openai_timeout_seconds: float = Field(
        default=60, validation_alias="OPENAI_TIMEOUT_SECONDS"
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

    pinecone_documents_namespace: str = Field(
        default="documentation", validation_alias="PINECONE_DOCUMENTS_NAMESPACE"
    )
    retrieval_top_k: int = Field(default=5, validation_alias="RETRIEVAL_TOP_K")

    @field_validator("openai_api_key", "pinecone_api_key")
    @classmethod
    def _validate_api_key_not_blank(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("must not be empty.")
        return value

    @field_validator("openai_timeout_seconds")
    @classmethod
    def _validate_openai_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("openai_timeout_seconds must be greater than zero.")
        return value

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
                f"Unsupported pinecone_metric. Supported values: {sorted(_SUPPORTED_METRICS)}."
            )
        return value

    @field_validator("pinecone_documents_namespace")
    @classmethod
    def _validate_pinecone_documents_namespace(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("pinecone_documents_namespace must not be empty.")
        return stripped

    @field_validator("retrieval_top_k")
    @classmethod
    def _validate_retrieval_top_k(cls, value: int) -> int:
        if value < 1 or value > 50:
            raise ValueError("retrieval_top_k must be between 1 and 50 inclusive.")
        return value

    @model_validator(mode="after")
    def _validate_poll_interval_within_timeout(self) -> "VectorStoreSettings":
        if self.pinecone_smoke_poll_interval_seconds > self.pinecone_smoke_timeout_seconds:
            raise ValueError(
                "pinecone_smoke_poll_interval_seconds cannot exceed "
                "pinecone_smoke_timeout_seconds."
            )
        return self


class AnswerSettings(VectorStoreSettings):
    """Grounded documentation answering: retrieval plus the OpenAI chat model."""

    openai_chat_model: str = Field(validation_alias="OPENAI_CHAT_MODEL")
    # Documentation relevance gate: retrieved chunks scoring below this cosine
    # similarity are discarded before answering. The default preserves the
    # historical value; it has not been re-calibrated.
    retrieval_score_threshold: float = Field(
        default=0.25, validation_alias="RETRIEVAL_SCORE_THRESHOLD"
    )

    @field_validator("openai_chat_model")
    @classmethod
    def _validate_chat_model_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty.")
        return value

    @field_validator("retrieval_score_threshold")
    @classmethod
    def _validate_retrieval_score_threshold(cls, value: float) -> float:
        # Written as a negated range check so NaN is rejected as well.
        if not 0.0 <= value <= 1.0:
            raise ValueError("retrieval_score_threshold must be between 0.0 and 1.0 inclusive.")
        return value


class AgentSettings(AnswerSettings, PyPISettings):
    """The tool-calling agent: documentation answering plus PyPI lookup."""


class IndexingSettings(VectorStoreSettings, UrlFetchSettings):
    """URL indexing: URL fetching/chunking plus embedding and Pinecone writes."""

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

    @model_validator(mode="after")
    def _validate_index_verify_poll_interval_within_timeout(self) -> "IndexingSettings":
        if (
            self.pinecone_index_verify_poll_interval_seconds
            > self.pinecone_index_verify_timeout_seconds
        ):
            raise ValueError(
                "pinecone_index_verify_poll_interval_seconds cannot exceed "
                "pinecone_index_verify_timeout_seconds."
            )
        return self


class UserMemorySettings(VectorStoreSettings):
    """Long-term user memory: the vector store plus the memory identity secret."""

    user_memory_hash_secret: SecretStr = Field(validation_alias="USER_MEMORY_HASH_SECRET")
    user_memory_namespace_prefix: str = Field(
        default="user-memory", validation_alias="USER_MEMORY_NAMESPACE_PREFIX"
    )
    user_memory_top_k: int = Field(default=5, validation_alias="USER_MEMORY_TOP_K")
    user_memory_score_threshold: float = Field(
        default=0.35, validation_alias="USER_MEMORY_SCORE_THRESHOLD"
    )
    user_memory_max_statement_length: int = Field(
        default=500, validation_alias="USER_MEMORY_MAX_STATEMENT_LENGTH"
    )

    @field_validator("user_memory_hash_secret")
    @classmethod
    def _validate_user_memory_hash_secret(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("user_memory_hash_secret must not be empty.")
        return value

    @field_validator("user_memory_namespace_prefix")
    @classmethod
    def _validate_user_memory_namespace_prefix(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("user_memory_namespace_prefix must not be empty.")
        if not _NAMESPACE_PREFIX_PATTERN.fullmatch(stripped):
            raise ValueError(
                "user_memory_namespace_prefix may contain only ASCII letters, digits, "
                "'.', '_' and '-'."
            )
        if len(stripped) > 40:
            raise ValueError("user_memory_namespace_prefix must not exceed 40 characters.")
        return stripped

    @field_validator("user_memory_top_k")
    @classmethod
    def _validate_user_memory_top_k(cls, value: int) -> int:
        if value < 1 or value > 50:
            raise ValueError("user_memory_top_k must be between 1 and 50 inclusive.")
        return value

    @field_validator("user_memory_score_threshold")
    @classmethod
    def _validate_user_memory_score_threshold(cls, value: float) -> float:
        if value < -1.0 or value > 1.0:
            raise ValueError(
                "user_memory_score_threshold must be between -1.0 and 1.0 inclusive "
                "(cosine similarity range)."
            )
        return value

    @field_validator("user_memory_max_statement_length")
    @classmethod
    def _validate_user_memory_max_statement_length(cls, value: int) -> int:
        if value < 1 or value > 4000:
            raise ValueError(
                "user_memory_max_statement_length must be between 1 and 4000 inclusive."
            )
        return value

    @model_validator(mode="after")
    def _validate_memory_namespace_prefix_not_documents_namespace(
        self,
    ) -> "UserMemorySettings":
        if self.user_memory_namespace_prefix == self.pinecone_documents_namespace:
            raise ValueError(
                "user_memory_namespace_prefix must differ from pinecone_documents_namespace."
            )
        return self


class AppSettings(AgentSettings, UserMemorySettings, IndexingSettings):
    """Full configuration: every feature plus the Telegram bot token.

    Used by the Telegram runtime, which needs the whole stack. Other entry
    points load the narrower group they actually use.
    """

    telegram_bot_token: SecretStr = Field(validation_alias="TELEGRAM_BOT_TOKEN")

    @field_validator("telegram_bot_token")
    @classmethod
    def _validate_telegram_bot_token(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("telegram_bot_token must not be empty.")
        return value


_SettingsT = TypeVar("_SettingsT", bound=_EnvSettings)


def load_settings(settings_cls: type[_SettingsT]) -> _SettingsT:
    """Load one settings group from the environment/.env with redacted failures.

    Raises ConfigurationError (never a raw pydantic ValidationError, whose text
    embeds the offending input values) when required configuration is missing
    or invalid.
    """
    try:
        return settings_cls()
    except ValidationError as exc:
        raise ConfigurationError.from_validation_error(exc) from None


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return a cached full AppSettings instance loaded from the environment."""
    return load_settings(AppSettings)
