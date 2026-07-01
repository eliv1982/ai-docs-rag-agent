"""Result and status models for Pinecone index operations."""

from pydantic import BaseModel, ConfigDict


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
