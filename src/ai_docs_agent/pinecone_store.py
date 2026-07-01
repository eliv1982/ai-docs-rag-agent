"""Pinecone index management and an OpenAI-embedding integration smoke test."""

import time
import uuid
from collections.abc import Callable
from typing import Any, Protocol

from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

from ai_docs_agent.config import AppSettings
from ai_docs_agent.models import PineconeIndexStatus, PineconeSmokeTestResult

_SMOKE_TEST_TEXT = "AI Docs RAG Agent Pinecone integration smoke test."


class PineconeStoreError(Exception):
    """Base error for PineconeStore operations."""


class PineconeIndexNotFoundError(PineconeStoreError):
    """Raised when a required Pinecone index does not exist."""


class PineconeIndexConfigurationError(PineconeStoreError):
    """Raised when an existing Pinecone index's configuration does not match settings."""


class PineconeSmokeTestError(PineconeStoreError):
    """Raised when the embed -> upsert -> query smoke test pipeline fails."""


class EmbeddingsClient(Protocol):
    """Structural interface for the embedding client used by PineconeStore."""

    def embed_query(self, text: str) -> list[float]: ...


class PineconeClient(Protocol):
    """Structural interface for the subset of the Pinecone SDK PineconeStore uses."""

    def has_index(self, name: str) -> bool: ...

    def describe_index(self, name: str) -> Any: ...

    def create_index(
        self, *, name: str, dimension: int, metric: str, spec: Any
    ) -> None: ...

    def Index(self, name: str) -> Any: ...  # noqa: N802 - matches Pinecone SDK method name


class PineconeStore:
    """Manages a single Pinecone index and its OpenAI-embedding integration smoke test."""

    def __init__(
        self,
        settings: AppSettings,
        *,
        client: PineconeClient | None = None,
        embeddings: EmbeddingsClient | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._settings = settings
        self._client = client
        self._embeddings = embeddings
        self._clock = clock
        self._sleep = sleep

    @property
    def _pinecone_client(self) -> PineconeClient:
        if self._client is None:
            self._client = Pinecone(api_key=self._settings.pinecone_api_key.get_secret_value())
        return self._client

    @property
    def _embedding_client(self) -> EmbeddingsClient:
        if self._embeddings is None:
            kwargs: dict[str, Any] = {
                "model": self._settings.openai_embedding_model,
                "api_key": self._settings.openai_api_key.get_secret_value(),
            }
            if self._settings.openai_base_url is not None:
                kwargs["base_url"] = self._settings.openai_base_url
            self._embeddings = OpenAIEmbeddings(**kwargs)
        return self._embeddings

    def ensure_index(self) -> PineconeIndexStatus:
        """Ensure the configured Pinecone index exists, is valid, and is ready."""
        client = self._pinecone_client
        index_name = self._settings.pinecone_index_name

        if not self._index_exists(client, index_name):
            if not self._settings.pinecone_create_if_missing:
                raise PineconeIndexNotFoundError(
                    f"Pinecone index '{index_name}' does not exist and "
                    "pinecone_create_if_missing is false."
                )
            self._create_index(client, index_name)

        description = self._describe_index(client, index_name)
        self._validate_index_configuration(description)
        description = self._wait_until_ready(client, index_name, description)
        return self._to_index_status(description)

    def smoke_test(self) -> PineconeSmokeTestResult:
        """Run a live embed -> upsert -> query -> cleanup integration check."""
        status = self.ensure_index()
        record_id = f"smoke-{uuid.uuid4().hex}"
        namespace = self._settings.pinecone_smoke_namespace
        started_at = self._clock()
        index: Any = None
        upsert_attempted = False

        try:
            embedding = self._create_smoke_embedding(status.dimension)
            index = self._get_index_handle()
            upsert_attempted = True
            self._upsert_smoke_vector(index, record_id, embedding, namespace)
            matched_id, score = self._poll_for_match(index, embedding, record_id, namespace)
        except PineconeSmokeTestError:
            if upsert_attempted:
                self._cleanup(index, record_id, namespace)
            raise
        except Exception as exc:
            if upsert_attempted:
                self._cleanup(index, record_id, namespace)
            raise PineconeSmokeTestError("Pinecone smoke test failed.") from exc

        cleanup_succeeded = self._cleanup(index, record_id, namespace)
        elapsed = self._clock() - started_at

        return PineconeSmokeTestResult(
            index_name=self._settings.pinecone_index_name,
            namespace=namespace,
            dimension=status.dimension,
            embedding_model=self._settings.openai_embedding_model,
            record_id=record_id,
            matched_id=matched_id,
            score=float(score),
            cleanup_succeeded=cleanup_succeeded,
            elapsed_seconds=elapsed,
        )

    def _create_smoke_embedding(self, expected_dimension: int) -> list[float]:
        try:
            embedding = self._embedding_client.embed_query(_SMOKE_TEST_TEXT)
        except Exception as exc:
            raise PineconeSmokeTestError("Failed to create smoke-test embedding.") from exc

        if len(embedding) != expected_dimension:
            raise PineconeSmokeTestError(
                f"Embedding dimension {len(embedding)} does not match configured "
                f"index dimension {expected_dimension}."
            )
        return embedding

    def _get_index_handle(self) -> Any:
        index_name = self._settings.pinecone_index_name
        try:
            return self._pinecone_client.Index(index_name)
        except Exception as exc:
            raise PineconeSmokeTestError(
                f"Failed to obtain a Pinecone index handle for '{index_name}'."
            ) from exc

    def _index_exists(self, client: PineconeClient, index_name: str) -> bool:
        try:
            return client.has_index(index_name)
        except Exception as exc:
            raise PineconeStoreError(
                f"Failed to check existence of Pinecone index '{index_name}'."
            ) from exc

    def _create_index(self, client: PineconeClient, index_name: str) -> None:
        try:
            client.create_index(
                name=index_name,
                dimension=self._settings.pinecone_dimension,
                metric=self._settings.pinecone_metric,
                spec=ServerlessSpec(
                    cloud=self._settings.pinecone_cloud,
                    region=self._settings.pinecone_region,
                ),
            )
        except Exception as exc:
            raise PineconeStoreError(f"Failed to create Pinecone index '{index_name}'.") from exc

    def _wait_until_ready(
        self, client: PineconeClient, index_name: str, description: Any
    ) -> Any:
        """Bounded-poll (re-describing as needed) until the index is ready and has a host.

        Applies equally to a just-created index and to a pre-existing index that
        was found via has_index() but has not finished provisioning yet.
        """
        if self._is_ready(description):
            return description

        deadline = self._clock() + self._settings.pinecone_smoke_timeout_seconds
        while True:
            if self._clock() >= deadline:
                raise PineconeStoreError(
                    f"Timed out waiting for Pinecone index '{index_name}' to become ready."
                )
            self._sleep(self._settings.pinecone_smoke_poll_interval_seconds)
            description = self._describe_index(client, index_name)
            if self._is_ready(description):
                return description

    @staticmethod
    def _is_ready(description: Any) -> bool:
        return bool(description.status.ready) and bool(getattr(description, "host", None))

    def _describe_index(self, client: PineconeClient, index_name: str) -> Any:
        try:
            return client.describe_index(index_name)
        except Exception as exc:
            raise PineconeStoreError(f"Failed to describe Pinecone index '{index_name}'.") from exc

    def _validate_index_configuration(self, description: Any) -> None:
        expected_dimension = self._settings.pinecone_dimension
        expected_metric = self._settings.pinecone_metric
        if description.dimension != expected_dimension:
            raise PineconeIndexConfigurationError(
                f"Pinecone index dimension mismatch: expected {expected_dimension}, "
                f"got {description.dimension}."
            )
        if description.metric != expected_metric:
            raise PineconeIndexConfigurationError(
                f"Pinecone index metric mismatch: expected '{expected_metric}', "
                f"got '{description.metric}'."
            )

    def _to_index_status(self, description: Any) -> PineconeIndexStatus:
        return PineconeIndexStatus(
            name=self._settings.pinecone_index_name,
            dimension=description.dimension,
            metric=description.metric,
            host=getattr(description, "host", None),
            ready=bool(description.status.ready),
        )

    def _upsert_smoke_vector(
        self, index: Any, record_id: str, embedding: list[float], namespace: str
    ) -> None:
        try:
            index.upsert(
                vectors=[
                    {
                        "id": record_id,
                        "values": embedding,
                        "metadata": {
                            "kind": "integration_smoke_test",
                            "text": _SMOKE_TEST_TEXT,
                            "embedding_model": self._settings.openai_embedding_model,
                        },
                    }
                ],
                namespace=namespace,
            )
        except Exception as exc:
            raise PineconeSmokeTestError("Failed to upsert smoke-test vector.") from exc

    def _poll_for_match(
        self, index: Any, embedding: list[float], record_id: str, namespace: str
    ) -> tuple[str, float]:
        deadline = self._clock() + self._settings.pinecone_smoke_timeout_seconds
        while True:
            match = self._query_for_match(index, embedding, namespace)
            if match is not None and match[0] == record_id:
                return match
            if self._clock() >= deadline:
                raise PineconeSmokeTestError(
                    f"Timed out waiting for smoke-test vector '{record_id}' to appear "
                    "in query results."
                )
            self._sleep(self._settings.pinecone_smoke_poll_interval_seconds)

    def _query_for_match(
        self, index: Any, embedding: list[float], namespace: str
    ) -> tuple[str, float] | None:
        try:
            response = index.query(
                vector=embedding,
                top_k=1,
                include_metadata=True,
                namespace=namespace,
            )
        except Exception as exc:
            raise PineconeSmokeTestError("Failed to query Pinecone for smoke-test vector.") from exc

        matches = getattr(response, "matches", None)
        if not matches:
            return None
        top_match = matches[0]
        return top_match.id, float(top_match.score)

    def _cleanup(self, index: Any, record_id: str, namespace: str) -> bool:
        # Deliberately swallowed: cleanup failure must not mask the primary
        # smoke-test outcome, so it is surfaced via cleanup_succeeded instead.
        try:
            index.delete(ids=[record_id], namespace=namespace)
        except Exception:
            return False
        return True
