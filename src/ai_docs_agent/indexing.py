"""Orchestrates URL ingestion, embedding, Pinecone upsert, verification, and cleanup.

Pipeline: URL -> UrlIngestionService.process_url -> DocumentChunk[] -> batch OpenAI
embeddings -> batch Pinecone upsert -> bounded, cumulative fetch-based verification ->
deletion of obsolete records for the same source_url in the same namespace: other
document versions (different document_id) and stale chunks of the current document
that fall outside its current chunk_count (e.g. after a chunking-config change that
shrinks the chunk layout for unchanged content).
"""

import time
from collections.abc import Callable

from ai_docs_agent.config import AppSettings
from ai_docs_agent.models import DocumentIndexingResult, UrlProcessingResult
from ai_docs_agent.pinecone_store import PineconeStore, PineconeStoreError
from ai_docs_agent.url_ingestion import UrlIngestionService


class DocumentIndexingError(Exception):
    """Base class for domain errors raised while indexing a URL into Pinecone."""


class DocumentEmbeddingError(DocumentIndexingError):
    """Raised when creating embeddings for one or more document chunks fails."""


class DocumentUpsertError(DocumentIndexingError):
    """Raised when writing embedded chunks to Pinecone fails partway through."""


class DocumentVerificationError(DocumentIndexingError):
    """Raised when bounded post-upsert verification cannot confirm all chunk IDs."""


class DocumentCleanupError(DocumentIndexingError):
    """Raised when deleting stale versions of a page after a successful upsert fails."""


class DocumentIndexingService:
    """Orchestrates UrlIngestionService and PineconeStore into a single indexing pipeline."""

    def __init__(
        self,
        settings: AppSettings,
        *,
        url_ingestion_service: UrlIngestionService | None = None,
        pinecone_store: PineconeStore | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._settings = settings
        self._url_ingestion_service = url_ingestion_service or UrlIngestionService(settings)
        self._pinecone_store = pinecone_store or PineconeStore(settings)
        self._clock = clock
        self._sleep = sleep

    def index_url(self, url: str, *, namespace: str | None = None) -> DocumentIndexingResult:
        """Fetch, chunk, embed, upsert, verify, and (optionally) clean up one URL."""
        resolved_namespace = self._resolve_namespace(namespace)
        started_at = self._clock()

        processing = self._url_ingestion_service.process_url(url)

        embeddings = self._embed_chunks(processing)
        upserted_count = self._upsert_chunks(processing, embeddings, resolved_namespace)
        verified_count = self._verify_chunks(processing, resolved_namespace)

        cleanup_requested = self._settings.pinecone_replace_old_source_versions
        cleanup_succeeded: bool | None = None
        if cleanup_requested:
            cleanup_succeeded = self._cleanup_old_versions(processing, resolved_namespace)

        elapsed_seconds = self._clock() - started_at
        return DocumentIndexingResult(
            source_url=processing.source_url,
            final_url=processing.final_url,
            document_id=processing.document_id,
            content_hash=processing.content_hash,
            namespace=resolved_namespace,
            chunk_count=processing.chunk_count,
            embedded_count=len(embeddings),
            upserted_count=upserted_count,
            verified_count=verified_count,
            old_versions_cleanup_requested=cleanup_requested,
            old_versions_cleanup_succeeded=cleanup_succeeded,
            elapsed_seconds=elapsed_seconds,
        )

    def _resolve_namespace(self, namespace: str | None) -> str:
        candidate = (
            namespace.strip()
            if namespace is not None
            else self._settings.pinecone_documents_namespace
        )
        if not candidate:
            raise DocumentIndexingError("Resolved Pinecone namespace must not be empty.")
        return candidate

    def _embed_chunks(self, processing: UrlProcessingResult) -> list[list[float]]:
        texts = [chunk.text for chunk in processing.chunks]
        batch_size = self._settings.embedding_batch_size

        embeddings: list[list[float]] = []
        try:
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                embeddings.extend(self._pinecone_store.embed_documents(batch))
        except PineconeStoreError as exc:
            raise DocumentEmbeddingError(
                f"Failed to create embeddings for document '{processing.document_id}'."
            ) from exc

        if len(embeddings) != len(texts):
            raise DocumentEmbeddingError(
                f"Embedded {len(embeddings)} of {len(texts)} chunk(s) for document "
                f"'{processing.document_id}'."
            )
        return embeddings

    def _upsert_chunks(
        self,
        processing: UrlProcessingResult,
        embeddings: list[list[float]],
        namespace: str,
    ) -> int:
        records = [
            {"id": chunk.id, "values": embedding, "metadata": chunk.to_pinecone_metadata()}
            for chunk, embedding in zip(processing.chunks, embeddings, strict=True)
        ]
        batch_size = self._settings.pinecone_upsert_batch_size

        upserted_total = 0
        try:
            for start in range(0, len(records), batch_size):
                batch = records[start : start + batch_size]
                upserted_total += self._pinecone_store.upsert_vectors(
                    batch, namespace=namespace
                )
        except PineconeStoreError as exc:
            raise DocumentUpsertError(
                f"Failed to upsert chunks for document '{processing.document_id}' after "
                f"{upserted_total} of {len(records)} record(s) were confirmed written."
            ) from exc

        if upserted_total != len(records):
            raise DocumentUpsertError(
                f"Upserted {upserted_total} of {len(records)} chunk(s) for document "
                f"'{processing.document_id}'."
            )
        return upserted_total

    def _verify_chunks(self, processing: UrlProcessingResult, namespace: str) -> int:
        expected_ids = [chunk.id for chunk in processing.chunks]
        expected_id_set = set(expected_ids)
        if len(expected_id_set) != len(expected_ids):
            raise DocumentVerificationError(
                f"Duplicate chunk IDs detected for document '{processing.document_id}'."
            )

        fetch_batch_size = self._settings.pinecone_fetch_batch_size
        deadline = self._clock() + self._settings.pinecone_index_verify_timeout_seconds

        confirmed_ids: set[str] = set()
        while True:
            round_ids = self._fetch_all_ids(expected_ids, fetch_batch_size, namespace, processing)
            confirmed_ids |= round_ids & expected_id_set

            if confirmed_ids == expected_id_set:
                return len(confirmed_ids)

            if self._clock() >= deadline:
                raise DocumentVerificationError(
                    f"Verification timed out for document '{processing.document_id}' in "
                    f"namespace '{namespace}': found {len(confirmed_ids)} of "
                    f"{len(expected_ids)} expected chunk(s)."
                )
            self._sleep(self._settings.pinecone_index_verify_poll_interval_seconds)

    def _fetch_all_ids(
        self,
        expected_ids: list[str],
        fetch_batch_size: int,
        namespace: str,
        processing: UrlProcessingResult,
    ) -> set[str]:
        found_ids: set[str] = set()
        try:
            for start in range(0, len(expected_ids), fetch_batch_size):
                batch = expected_ids[start : start + fetch_batch_size]
                found_ids |= self._pinecone_store.fetch_existing_ids(batch, namespace=namespace)
        except PineconeStoreError as exc:
            raise DocumentVerificationError(
                f"Failed to verify chunks for document '{processing.document_id}' in "
                f"namespace '{namespace}'."
            ) from exc
        return found_ids

    def _cleanup_old_versions(self, processing: UrlProcessingResult, namespace: str) -> bool:
        metadata_filter = {
            "$or": [
                {
                    "$and": [
                        {"source_url": {"$eq": processing.source_url}},
                        {"document_id": {"$ne": processing.document_id}},
                    ]
                },
                {
                    "$and": [
                        {"source_url": {"$eq": processing.source_url}},
                        {"document_id": {"$eq": processing.document_id}},
                        {"chunk_index": {"$gte": processing.chunk_count}},
                    ]
                },
            ]
        }
        try:
            self._pinecone_store.delete_vectors_by_filter(metadata_filter, namespace=namespace)
        except PineconeStoreError:
            return False
        return True
