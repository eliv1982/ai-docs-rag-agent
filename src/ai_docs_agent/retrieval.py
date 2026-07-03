"""Typed, read-only semantic search over indexed documentation chunks in Pinecone.

Pipeline: query text -> validation -> OpenAI query embedding -> Pinecone similarity
query (always scoped to kind == "documentation_chunk") -> strict per-match metadata
decoding into RetrievedChunk -> RetrievalResult, preserving Pinecone's own match
order. No generation, prompt construction, citation formatting, reranking, dedup,
or retrieval-time cleanup is performed here.
"""

from typing import Any

from pydantic import ValidationError

from ai_docs_agent.config import AppSettings
from ai_docs_agent.models import PineconeQueryMatch, RetrievalResult, RetrievedChunk
from ai_docs_agent.pinecone_store import PineconeStore, PineconeStoreError

_DOCUMENTATION_CHUNK_FILTER: dict[str, Any] = {"kind": {"$eq": "documentation_chunk"}}
_EXPECTED_KIND = "documentation_chunk"
_REQUIRED_METADATA_KEYS = (
    "kind",
    "text",
    "document_id",
    "source_url",
    "final_url",
    "title",
    "content_hash",
    "chunk_index",
    "chunk_count",
)


class RetrievalError(Exception):
    """Base class for domain errors raised while retrieving chunks from Pinecone."""


class QueryEmbeddingError(RetrievalError):
    """Raised when creating an embedding for the query text fails."""


class QueryExecutionError(RetrievalError):
    """Raised when querying Pinecone for similar chunks fails."""


class MalformedRetrievalMetadataError(RetrievalError):
    """Raised when a Pinecone match's metadata does not match the expected chunk shape."""


class RetrievalService:
    """Embeds a query and retrieves the most similar indexed documentation chunks."""

    def __init__(
        self,
        settings: AppSettings,
        *,
        pinecone_store: PineconeStore | None = None,
    ) -> None:
        self._settings = settings
        self._pinecone_store = pinecone_store or PineconeStore(settings)

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        namespace: str | None = None,
    ) -> RetrievalResult:
        """Embed `query` and return the most similar indexed documentation chunks."""
        resolved_query = self._resolve_query(query)
        resolved_top_k = self._resolve_top_k(top_k)
        resolved_namespace = self._resolve_namespace(namespace)

        try:
            vector = self._pinecone_store.embed_query(resolved_query)
        except PineconeStoreError as exc:
            raise QueryEmbeddingError(
                "Failed to create an embedding for the retrieval query."
            ) from exc

        try:
            matches = self._pinecone_store.query_similar(
                vector,
                namespace=resolved_namespace,
                top_k=resolved_top_k,
                metadata_filter=_DOCUMENTATION_CHUNK_FILTER,
            )
        except PineconeStoreError as exc:
            raise QueryExecutionError(
                f"Failed to query Pinecone namespace '{resolved_namespace}' for similar chunks."
            ) from exc

        retrieved_chunks = tuple(self._decode_match(match) for match in matches)

        return RetrievalResult(
            query=resolved_query,
            namespace=resolved_namespace,
            top_k=resolved_top_k,
            matches=retrieved_chunks,
        )

    def _resolve_query(self, query: str) -> str:
        stripped = query.strip()
        if not stripped:
            raise RetrievalError("Query must not be blank.")
        return stripped

    def _resolve_top_k(self, top_k: int | None) -> int:
        resolved = top_k if top_k is not None else self._settings.retrieval_top_k
        if resolved < 1 or resolved > 50:
            raise RetrievalError("top_k must be between 1 and 50 inclusive.")
        return resolved

    def _resolve_namespace(self, namespace: str | None) -> str:
        candidate = (
            namespace.strip()
            if namespace is not None
            else self._settings.pinecone_documents_namespace
        )
        if not candidate:
            raise RetrievalError("Resolved Pinecone namespace must not be empty.")
        return candidate

    def _decode_match(self, match: PineconeQueryMatch) -> RetrievedChunk:
        metadata = match.metadata
        for key in _REQUIRED_METADATA_KEYS:
            if key not in metadata:
                raise MalformedRetrievalMetadataError(
                    f"Pinecone match '{match.id}' metadata is missing required key '{key}'."
                )

        kind = metadata["kind"]
        if kind != _EXPECTED_KIND:
            raise MalformedRetrievalMetadataError(
                f"Pinecone match '{match.id}' metadata has unexpected kind '{kind}'."
            )

        document_id = self._require_non_blank_string(metadata, "document_id", match.id)
        source_url = self._require_non_blank_string(metadata, "source_url", match.id)
        final_url = self._require_non_blank_string(metadata, "final_url", match.id)
        title = self._require_non_blank_string(metadata, "title", match.id)
        content_hash = self._require_non_blank_string(metadata, "content_hash", match.id)
        text = self._require_non_blank_string(metadata, "text", match.id)

        chunk_index = self._require_plain_int(metadata, "chunk_index", match.id)
        chunk_count = self._require_plain_int(metadata, "chunk_count", match.id)
        if chunk_count <= 0:
            raise MalformedRetrievalMetadataError(
                f"Pinecone match '{match.id}' metadata 'chunk_count' must be positive."
            )
        if not 0 <= chunk_index < chunk_count:
            raise MalformedRetrievalMetadataError(
                f"Pinecone match '{match.id}' metadata 'chunk_index' {chunk_index} is out of "
                f"bounds for chunk_count {chunk_count}."
            )

        try:
            return RetrievedChunk(
                chunk_id=match.id,
                score=match.score,
                document_id=document_id,
                source_url=source_url,
                final_url=final_url,
                title=title,
                content_hash=content_hash,
                chunk_index=chunk_index,
                chunk_count=chunk_count,
                text=text,
            )
        except ValidationError as exc:
            raise MalformedRetrievalMetadataError(
                f"Pinecone match '{match.id}' metadata failed chunk validation."
            ) from exc

    @staticmethod
    def _require_non_blank_string(metadata: dict[str, Any], key: str, match_id: str) -> str:
        value = metadata[key]
        if not isinstance(value, str) or not value.strip():
            raise MalformedRetrievalMetadataError(
                f"Pinecone match '{match_id}' metadata '{key}' must be a non-blank string."
            )
        return value

    @staticmethod
    def _require_plain_int(metadata: dict[str, Any], key: str, match_id: str) -> int:
        value = metadata[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise MalformedRetrievalMetadataError(
                f"Pinecone match '{match_id}' metadata '{key}' must be an integer."
            )
        return value
