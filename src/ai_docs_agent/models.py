"""Result and status models for Pinecone index operations and URL ingestion."""

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


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
