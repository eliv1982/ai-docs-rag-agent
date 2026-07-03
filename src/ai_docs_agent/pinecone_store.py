"""Pinecone index management and an OpenAI-embedding integration smoke test."""

import math
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

from ai_docs_agent.config import AppSettings
from ai_docs_agent.models import PineconeIndexStatus, PineconeQueryMatch, PineconeSmokeTestResult

_SMOKE_TEST_TEXT = "AI Docs RAG Agent Pinecone integration smoke test."


class PineconeStoreError(Exception):
    """Base error for PineconeStore operations."""


class PineconeIndexNotFoundError(PineconeStoreError):
    """Raised when a required Pinecone index does not exist."""


class PineconeIndexConfigurationError(PineconeStoreError):
    """Raised when an existing Pinecone index's configuration does not match settings."""


class PineconeSmokeTestError(PineconeStoreError):
    """Raised when the embed -> upsert -> query smoke test pipeline fails."""


class PineconeEmbeddingError(PineconeStoreError):
    """Raised when creating or validating document embeddings fails."""


class PineconeUpsertError(PineconeStoreError):
    """Raised when upserting vectors into Pinecone fails."""


class PineconeFetchError(PineconeStoreError):
    """Raised when fetching vectors from Pinecone fails."""


class PineconeDeleteError(PineconeStoreError):
    """Raised when deleting vectors from Pinecone by metadata filter fails."""


class PineconeQueryError(PineconeStoreError):
    """Raised when querying Pinecone for similar vectors fails or returns a malformed response."""


class EmbeddingsClient(Protocol):
    """Structural interface for the embedding client used by PineconeStore."""

    def embed_query(self, text: str) -> list[float]: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


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
        self._ready_index_handle: Any | None = None

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

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Create OpenAI embeddings for a batch of document chunk texts, preserving order."""
        if not texts:
            return []

        try:
            raw_embeddings = self._embedding_client.embed_documents(list(texts))
        except Exception as exc:
            raise PineconeEmbeddingError("Failed to create document embeddings.") from exc

        if len(raw_embeddings) != len(texts):
            raise PineconeEmbeddingError(
                f"Received {len(raw_embeddings)} embedding(s) for {len(texts)} text(s)."
            )

        expected_dimension = self._settings.pinecone_dimension
        return [
            self._validate_embedding_values(embedding, expected_dimension)
            for embedding in raw_embeddings
        ]

    @staticmethod
    def _validate_embedding_values(
        embedding: Sequence[Any], expected_dimension: int
    ) -> list[float]:
        if len(embedding) != expected_dimension:
            raise PineconeEmbeddingError(
                f"Embedding dimension {len(embedding)} does not match configured "
                f"index dimension {expected_dimension}."
            )

        values: list[float] = []
        for raw_value in embedding:
            if isinstance(raw_value, bool):
                raise PineconeEmbeddingError(
                    "Embedding contains a boolean value, which is not a valid "
                    "embedding component."
                )
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise PineconeEmbeddingError(
                    "Embedding contains a non-numeric value."
                ) from exc
            if math.isnan(value) or math.isinf(value):
                raise PineconeEmbeddingError("Embedding contains a NaN or infinite value.")
            values.append(value)
        return values

    def embed_query(self, text: str) -> list[float]:
        """Create an OpenAI embedding for a single query string."""
        if not text.strip():
            raise PineconeEmbeddingError("Query text must not be blank.")

        try:
            embedding = self._embedding_client.embed_query(text)
        except Exception as exc:
            raise PineconeEmbeddingError("Failed to create query embedding.") from exc

        return self._validate_embedding_values(embedding, self._settings.pinecone_dimension)

    def upsert_vectors(
        self, vectors: Sequence[dict[str, object]], *, namespace: str
    ) -> int:
        """Upsert a batch of already-embedded vectors and return the confirmed count."""
        if not vectors:
            return 0

        index = self._get_ready_index_handle()
        try:
            response = index.upsert(vectors=list(vectors), namespace=namespace)
        except Exception as exc:
            self._invalidate_ready_index_handle()
            raise PineconeUpsertError("Failed to upsert vectors into Pinecone.") from exc

        return self._parse_upserted_count(response)

    @staticmethod
    def _parse_upserted_count(response: Any) -> int:
        try:
            if isinstance(response, Mapping):
                raw_count = response["upserted_count"]
            else:
                raw_count = response.upserted_count
        except (KeyError, AttributeError) as exc:
            raise PineconeUpsertError(
                "Pinecone upsert response did not include an 'upserted_count' field."
            ) from exc

        if raw_count is None or isinstance(raw_count, bool) or not isinstance(raw_count, int):
            raise PineconeUpsertError(
                "Pinecone upsert response 'upserted_count' must be a non-negative integer."
            )
        if raw_count < 0:
            raise PineconeUpsertError(
                "Pinecone upsert response 'upserted_count' must be a non-negative integer."
            )
        return raw_count

    def fetch_existing_ids(self, ids: Sequence[str], *, namespace: str) -> set[str]:
        """Return the subset of `ids` that already exist in the configured index/namespace."""
        if not ids:
            return set()

        index = self._get_ready_index_handle()
        try:
            response = index.fetch(ids=list(ids), namespace=namespace)
        except Exception as exc:
            self._invalidate_ready_index_handle()
            raise PineconeFetchError("Failed to fetch vectors from Pinecone.") from exc

        return self._extract_fetched_ids(response)

    @staticmethod
    def _extract_fetched_ids(response: Any) -> set[str]:
        try:
            if isinstance(response, Mapping):
                vectors = response["vectors"]
            else:
                vectors = response.vectors
        except (KeyError, AttributeError) as exc:
            raise PineconeFetchError(
                "Pinecone fetch response did not include a 'vectors' field."
            ) from exc

        if vectors is None:
            raise PineconeFetchError("Pinecone fetch response 'vectors' field was None.")
        if not isinstance(vectors, Mapping):
            raise PineconeFetchError(
                "Pinecone fetch response 'vectors' field must be a mapping of ID to vector."
            )

        ids: set[str] = set()
        for key in vectors:
            if not isinstance(key, str):
                raise PineconeFetchError(
                    "Pinecone fetch response 'vectors' mapping contained a non-string ID key."
                )
            ids.add(key)
        return ids

    def delete_vectors_by_filter(
        self, metadata_filter: dict[str, object], *, namespace: str
    ) -> None:
        """Delete every vector in `namespace` that matches a metadata filter."""
        index = self._get_ready_index_handle()
        try:
            index.delete(filter=dict(metadata_filter), namespace=namespace)
        except Exception as exc:
            self._invalidate_ready_index_handle()
            raise PineconeDeleteError(
                "Failed to delete vectors from Pinecone by filter."
            ) from exc

    def query_similar(
        self,
        vector: Sequence[float],
        *,
        namespace: str,
        top_k: int,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[PineconeQueryMatch]:
        """Query the configured index/namespace for the vectors most similar to `vector`."""
        validated_vector = self._validate_embedding_values(
            vector, self._settings.pinecone_dimension
        )
        if not namespace.strip():
            raise PineconeQueryError("Query namespace must not be blank.")
        if top_k < 1 or top_k > 50:
            raise PineconeQueryError("top_k must be between 1 and 50 inclusive.")

        index = self._get_ready_index_handle()
        try:
            response = index.query(
                vector=validated_vector,
                namespace=namespace,
                top_k=top_k,
                filter=metadata_filter,
                include_metadata=True,
                include_values=False,
            )
        except Exception as exc:
            self._invalidate_ready_index_handle()
            raise PineconeQueryError("Failed to query Pinecone for similar vectors.") from exc

        return self._extract_query_matches(response)

    @staticmethod
    def _extract_query_matches(response: Any) -> list[PineconeQueryMatch]:
        try:
            if isinstance(response, Mapping):
                raw_matches = response["matches"]
            else:
                raw_matches = response.matches
        except (KeyError, AttributeError) as exc:
            raise PineconeQueryError(
                "Pinecone query response did not include a 'matches' field."
            ) from exc

        if isinstance(raw_matches, str | bytes) or not isinstance(raw_matches, Sequence):
            raise PineconeQueryError(
                "Pinecone query response 'matches' field must be a sequence of matches."
            )

        return [PineconeStore._parse_query_match(raw_match) for raw_match in raw_matches]

    @staticmethod
    def _parse_query_match(raw_match: Any) -> PineconeQueryMatch:
        match_id = PineconeStore._extract_match_field(raw_match, "id")
        if not isinstance(match_id, str) or not match_id.strip():
            raise PineconeQueryError("Pinecone query match had a missing or blank 'id'.")

        score = PineconeStore._extract_match_field(raw_match, "score")
        if isinstance(score, bool) or not isinstance(score, int | float):
            raise PineconeQueryError("Pinecone query match 'score' must be a numeric value.")
        score_value = float(score)
        if math.isnan(score_value) or math.isinf(score_value):
            raise PineconeQueryError("Pinecone query match 'score' must be finite.")

        metadata = PineconeStore._extract_match_field(raw_match, "metadata")
        if metadata is None or not isinstance(metadata, Mapping):
            raise PineconeQueryError("Pinecone query match 'metadata' must be a mapping.")

        return PineconeQueryMatch(id=match_id, score=score_value, metadata=dict(metadata))

    @staticmethod
    def _extract_match_field(raw_match: Any, field: str) -> Any:
        try:
            if isinstance(raw_match, Mapping):
                return raw_match[field]
            return getattr(raw_match, field)
        except (KeyError, AttributeError) as exc:
            raise PineconeQueryError(
                f"Pinecone query match did not include a '{field}' field."
            ) from exc

    def _get_ready_index_handle(self) -> Any:
        """Return a cached, ready Pinecone index handle for this store instance.

        Control-plane readiness (`ensure_index()`) is checked at most once per
        store instance for the data-plane methods (upsert/fetch/delete); the
        handle is re-fetched only after a data-plane call fails, since that may
        indicate the cached handle (e.g. its host) is no longer valid.
        """
        if self._ready_index_handle is None:
            self.ensure_index()
            self._ready_index_handle = self._get_index_handle()
        return self._ready_index_handle

    def _invalidate_ready_index_handle(self) -> None:
        self._ready_index_handle = None

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
