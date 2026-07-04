"""Typed result and status models used across the application's domain services."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

DocumentationToolStatus = Literal[
    "success",
    "no_context",
    "retrieval_failure",
    "generation_failure",
]
PyPIToolStatus = Literal[
    "success",
    "invalid_package_name",
    "package_not_found",
    "timeout",
    "network_error",
    "malformed_response",
    "upstream_http_error",
]
AgentToolName = Literal["documentation_search", "pypi_lookup"]
AgentExecutionOutcome = Literal["success", "safe_fallback"]


class PineconeIndexStatus(BaseModel):
    """Snapshot of a Pinecone index's configuration and readiness."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    dimension: int
    metric: str
    host: str | None
    ready: bool


class PineconeSmokeTestResult(BaseModel):
    """Outcome of a single embed -> upsert -> query -> cleanup smoke test run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    index_name: str
    namespace: str
    dimension: int
    embedding_model: str
    record_id: str
    matched_id: str
    score: float
    cleanup_succeeded: bool
    elapsed_seconds: float


class FetchedPage(BaseModel):
    """A fetched documentation page with cleaned, normalized text (no raw HTML)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_url: str
    final_url: str
    title: str
    text: str
    content_hash: str


class DocumentChunk(BaseModel):
    """A single deterministic chunk of a fetched page's normalized text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    document_id: str
    source_url: str
    final_url: str
    title: str
    text: str
    chunk_index: int
    chunk_count: int
    content_hash: str

    @field_validator("text")
    @classmethod
    def _validate_text_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be empty.")
        return value

    @field_validator("chunk_index")
    @classmethod
    def _validate_chunk_index(cls, value: int) -> int:
        if value < 0:
            raise ValueError("chunk_index must be greater than or equal to zero.")
        return value

    @field_validator("chunk_count")
    @classmethod
    def _validate_chunk_count(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("chunk_count must be greater than zero.")
        return value

    @model_validator(mode="after")
    def _validate_chunk_index_within_count(self) -> "DocumentChunk":
        if self.chunk_index >= self.chunk_count:
            raise ValueError("chunk_index must be less than chunk_count.")
        return self

    def to_pinecone_metadata(self) -> dict[str, str | int]:
        """Return a flat metadata dict containing only Pinecone-safe scalar values."""
        return {
            "kind": "documentation_chunk",
            "text": self.text,
            "document_id": self.document_id,
            "source_url": self.source_url,
            "final_url": self.final_url,
            "title": self.title,
            "content_hash": self.content_hash,
            "chunk_index": self.chunk_index,
            "chunk_count": self.chunk_count,
        }


class DocumentIndexingResult(BaseModel):
    """The outcome of indexing one URL's chunks into Pinecone (embed -> upsert -> verify)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_url: str
    final_url: str
    document_id: str
    content_hash: str
    namespace: str
    chunk_count: int
    embedded_count: int
    upserted_count: int
    verified_count: int
    old_versions_cleanup_requested: bool
    old_versions_cleanup_succeeded: bool | None
    elapsed_seconds: float

    @field_validator("chunk_count")
    @classmethod
    def _validate_chunk_count(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("chunk_count must be greater than zero.")
        return value

    @field_validator("embedded_count", "upserted_count", "verified_count")
    @classmethod
    def _validate_counts_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("count fields must be greater than or equal to zero.")
        return value

    @field_validator("elapsed_seconds")
    @classmethod
    def _validate_elapsed_seconds(cls, value: float) -> float:
        if value < 0:
            raise ValueError("elapsed_seconds must be greater than or equal to zero.")
        return value

    @model_validator(mode="after")
    def _validate_counts_match_chunk_count(self) -> "DocumentIndexingResult":
        if self.embedded_count != self.chunk_count:
            raise ValueError("embedded_count must equal chunk_count.")
        if self.upserted_count != self.chunk_count:
            raise ValueError("upserted_count must equal chunk_count.")
        if self.verified_count != self.chunk_count:
            raise ValueError("verified_count must equal chunk_count.")
        return self

    @model_validator(mode="after")
    def _validate_cleanup_status_consistency(self) -> "DocumentIndexingResult":
        if not self.old_versions_cleanup_requested:
            if self.old_versions_cleanup_succeeded is not None:
                raise ValueError(
                    "old_versions_cleanup_succeeded must be None when cleanup was not requested."
                )
        elif self.old_versions_cleanup_succeeded is None:
            raise ValueError(
                "old_versions_cleanup_succeeded must be a bool when cleanup was requested."
            )
        return self


class UrlProcessingResult(BaseModel):
    """The full, internally-consistent outcome of processing one URL into chunks."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_url: str
    final_url: str
    title: str
    document_id: str
    content_hash: str
    text_char_count: int
    chunk_count: int
    chunks: tuple[DocumentChunk, ...]

    @field_validator("text_char_count")
    @classmethod
    def _validate_text_char_count(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("text_char_count must be greater than zero.")
        return value

    @field_validator("chunk_count")
    @classmethod
    def _validate_chunk_count(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("chunk_count must be greater than zero.")
        return value

    @model_validator(mode="after")
    def _validate_chunks_are_consistent(self) -> "UrlProcessingResult":
        if len(self.chunks) != self.chunk_count:
            raise ValueError("chunks length must equal chunk_count.")

        for expected_index, chunk in enumerate(self.chunks):
            if chunk.chunk_index != expected_index:
                raise ValueError("chunk indexes must be sequential starting at zero.")
            if (
                chunk.document_id != self.document_id
                or chunk.source_url != self.source_url
                or chunk.final_url != self.final_url
                or chunk.content_hash != self.content_hash
                or chunk.chunk_count != self.chunk_count
            ):
                raise ValueError(
                    "every chunk must share the result's document_id, source_url, "
                    "final_url, content_hash, and chunk_count."
                )

        return self


class PineconeQueryMatch(BaseModel):
    """A single low-level scored match returned by PineconeStore.query_similar."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    score: float
    metadata: dict[str, Any]


class RetrievedChunk(BaseModel):
    """A single retrieval result, decoded from a PineconeQueryMatch's metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str
    score: float
    document_id: str
    source_url: str
    final_url: str
    title: str
    content_hash: str
    chunk_index: int
    chunk_count: int
    text: str

    @field_validator("text")
    @classmethod
    def _validate_text_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be empty.")
        return value

    @field_validator("chunk_index")
    @classmethod
    def _validate_chunk_index(cls, value: int) -> int:
        if value < 0:
            raise ValueError("chunk_index must be greater than or equal to zero.")
        return value

    @field_validator("chunk_count")
    @classmethod
    def _validate_chunk_count(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("chunk_count must be greater than zero.")
        return value

    @model_validator(mode="after")
    def _validate_chunk_index_within_count(self) -> "RetrievedChunk":
        if self.chunk_index >= self.chunk_count:
            raise ValueError("chunk_index must be less than chunk_count.")
        return self


class RetrievalResult(BaseModel):
    """The full, internally-consistent outcome of a single retrieval search."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str
    namespace: str
    top_k: int
    matches: tuple[RetrievedChunk, ...]

    @field_validator("query")
    @classmethod
    def _validate_query_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank.")
        return value

    @field_validator("namespace")
    @classmethod
    def _validate_namespace_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("namespace must not be blank.")
        return value

    @field_validator("top_k")
    @classmethod
    def _validate_top_k_range(cls, value: int) -> int:
        if value < 1 or value > 50:
            raise ValueError("top_k must be between 1 and 50 inclusive.")
        return value

    @model_validator(mode="after")
    def _validate_matches_do_not_exceed_top_k(self) -> "RetrievalResult":
        if len(self.matches) > self.top_k:
            raise ValueError("matches length must not exceed top_k.")
        return self


class PyPIPackageInfo(BaseModel):
    """The typed result of one successful PyPI JSON API package lookup."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    package_name: str
    latest_version: str
    summary: str | None
    requires_python: str | None
    pypi_url: str
    project_url: str | None

    @field_validator("package_name", "latest_version", "pypi_url")
    @classmethod
    def _validate_required_non_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank.")
        return stripped

    @field_validator("summary", "requires_python", "project_url")
    @classmethod
    def _normalize_optional_string(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class AnswerSource(BaseModel):
    """A single document-level source backing a GroundedAnswerResult, derived from
    retrieved metadata (never model-generated)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    title: str
    url: str
    document_id: str
    chunk_index: int
    chunk_count: int

    @field_validator("title", "url", "document_id")
    @classmethod
    def _validate_non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank.")
        return value

    @field_validator("chunk_index")
    @classmethod
    def _validate_chunk_index(cls, value: int) -> int:
        if value < 0:
            raise ValueError("chunk_index must be greater than or equal to zero.")
        return value

    @field_validator("chunk_count")
    @classmethod
    def _validate_chunk_count(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("chunk_count must be greater than zero.")
        return value

    @model_validator(mode="after")
    def _validate_chunk_index_within_count(self) -> "AnswerSource":
        if self.chunk_index >= self.chunk_count:
            raise ValueError("chunk_index must be less than chunk_count.")
        return self


class GroundedAnswerResult(BaseModel):
    """The full, internally-consistent outcome of a single grounded RAG answer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question: str
    answer: str
    sources: tuple[AnswerSource, ...]
    retrieved_chunk_count: int

    @field_validator("question")
    @classmethod
    def _validate_question_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank.")
        return value

    @field_validator("answer")
    @classmethod
    def _validate_answer_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("answer must not be blank.")
        return value

    @field_validator("retrieved_chunk_count")
    @classmethod
    def _validate_retrieved_chunk_count(cls, value: int) -> int:
        if value < 0:
            raise ValueError("retrieved_chunk_count must be greater than or equal to zero.")
        return value

    @model_validator(mode="after")
    def _validate_sources_consistency(self) -> "GroundedAnswerResult":
        if self.retrieved_chunk_count == 0 and self.sources:
            raise ValueError("sources must be empty when retrieved_chunk_count is zero.")
        if self.retrieved_chunk_count > 0 and not self.sources:
            raise ValueError("sources must not be empty when retrieved_chunk_count is positive.")
        return self


class DocumentationSearchToolInput(BaseModel):
    """Structured input schema for the documentation LangChain tool."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question: str

    @field_validator("question")
    @classmethod
    def _validate_question_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be blank.")
        return stripped


class PyPILookupToolInput(BaseModel):
    """Structured input schema for the PyPI LangChain tool."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    package_name: str

    @field_validator("package_name")
    @classmethod
    def _validate_package_name_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("package_name must not be blank.")
        return stripped


class DocumentationToolResult(BaseModel):
    """Safe serialized output for the documentation-search LangChain tool."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: DocumentationToolStatus
    answer: str
    sources: tuple[AnswerSource, ...]
    context_found: bool

    @field_validator("answer")
    @classmethod
    def _validate_answer_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("answer must not be blank.")
        return stripped

    @model_validator(mode="after")
    def _validate_status_consistency(self) -> "DocumentationToolResult":
        if self.status == "success":
            if not self.context_found:
                raise ValueError("context_found must be true when status is success.")
            if not self.sources:
                raise ValueError("sources must not be empty when status is success.")
        else:
            if self.context_found:
                raise ValueError("context_found must be false for non-success statuses.")
            if self.sources:
                raise ValueError("sources must be empty for non-success statuses.")
        return self


class PyPIToolResult(BaseModel):
    """Safe serialized output for the PyPI LangChain tool."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: PyPIToolStatus
    package_name: str | None = None
    latest_version: str | None = None
    summary: str | None = None
    requires_python: str | None = None
    pypi_url: str | None = None
    project_url: str | None = None

    @field_validator(
        "package_name",
        "latest_version",
        "summary",
        "requires_python",
        "pypi_url",
        "project_url",
    )
    @classmethod
    def _normalize_optional_string(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @model_validator(mode="after")
    def _validate_status_consistency(self) -> "PyPIToolResult":
        if self.status == "success":
            for field_name in ("package_name", "latest_version", "pypi_url"):
                if getattr(self, field_name) is None:
                    raise ValueError(f"{field_name} is required when status is success.")
        return self


class LangChainAgentResult(BaseModel):
    """Typed result returned by the LangChain tool-calling agent layer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question: str
    answer: str
    sources: tuple[AnswerSource, ...]
    tools_used: tuple[AgentToolName, ...]
    tool_call_count: int
    used_no_tool: bool
    outcome: AgentExecutionOutcome
    failure_category: str | None = None

    @field_validator("question", "answer")
    @classmethod
    def _validate_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank.")
        return stripped

    @field_validator("tool_call_count")
    @classmethod
    def _validate_tool_call_count(cls, value: int) -> int:
        if value < 0:
            raise ValueError("tool_call_count must be greater than or equal to zero.")
        return value

    @field_validator("failure_category")
    @classmethod
    def _normalize_failure_category(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @model_validator(mode="after")
    def _validate_consistency(self) -> "LangChainAgentResult":
        if self.used_no_tool:
            if self.tools_used:
                raise ValueError("tools_used must be empty when used_no_tool is true.")
            if self.tool_call_count != 0:
                raise ValueError("tool_call_count must be zero when used_no_tool is true.")
        elif self.tool_call_count == 0:
            raise ValueError("tool_call_count must be positive when a tool was used.")

        if self.outcome == "success" and self.failure_category is not None:
            raise ValueError("failure_category must be null when outcome is success.")
        if self.outcome == "safe_fallback" and self.failure_category is None:
            raise ValueError("failure_category is required when outcome is safe_fallback.")
        return self


class ConversationMessage(BaseModel):
    """A single short-term conversation-memory message (user or assistant turn)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def _validate_content(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("content must not be blank.")
        return stripped
