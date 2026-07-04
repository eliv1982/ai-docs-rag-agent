"""Unit tests for RetrievalService. Uses a fake PineconeStore only; no real network access."""

from typing import Any

import pytest

from ai_docs_agent.config import AppSettings
from ai_docs_agent.models import PineconeQueryMatch, RetrievalResult
from ai_docs_agent.pinecone_store import PineconeEmbeddingError, PineconeQueryError
from ai_docs_agent.retrieval import (
    MalformedRetrievalMetadataError,
    QueryEmbeddingError,
    QueryExecutionError,
    RetrievalError,
    RetrievalService,
)

_REQUIRED: dict[str, Any] = {
    "openai_api_key": "sk-test-openai",
    "pinecone_api_key": "pc-test-key",
    "openai_chat_model": "gpt-4o-mini",
    "telegram_bot_token": "test-telegram-token",
    "user_memory_hash_secret": "unit-test-user-memory-secret",
}


def make_settings(**overrides: Any) -> AppSettings:
    return AppSettings(_env_file=None, **{**_REQUIRED, **overrides})


def make_metadata(**overrides: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "kind": "documentation_chunk",
        "text": "Some chunk text.",
        "document_id": "doc-abc123",
        "source_url": "https://docs.example.com/page",
        "final_url": "https://docs.example.com/page",
        "title": "Example Page",
        "content_hash": "hash-value",
        "chunk_index": 0,
        "chunk_count": 1,
    }
    return {**defaults, **overrides}


def make_match(**overrides: Any) -> PineconeQueryMatch:
    metadata_overrides = overrides.pop("metadata_overrides", {})
    defaults: dict[str, Any] = {
        "id": "doc-abc123-chunk-0000",
        "score": 0.9,
        "metadata": make_metadata(**metadata_overrides),
    }
    return PineconeQueryMatch(**{**defaults, **overrides})


class FakePineconeStore:
    """Fake low-level PineconeStore for RetrievalService orchestration tests."""

    def __init__(
        self,
        *,
        vector: list[float] | None = None,
        embed_error: Exception | None = None,
        matches: list[PineconeQueryMatch] | None = None,
        query_error: Exception | None = None,
    ) -> None:
        self.embed_calls: list[str] = []
        self.query_calls: list[dict[str, Any]] = []
        self._vector = vector if vector is not None else [0.1, 0.2, 0.3]
        self._embed_error = embed_error
        self._matches = matches if matches is not None else []
        self._query_error = query_error

    def embed_query(self, text: str) -> list[float]:
        self.embed_calls.append(text)
        if self._embed_error is not None:
            raise self._embed_error
        return list(self._vector)

    def query_similar(
        self,
        vector: list[float],
        *,
        namespace: str,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[PineconeQueryMatch]:
        self.query_calls.append(
            {
                "vector": vector,
                "namespace": namespace,
                "top_k": top_k,
                "metadata_filter": metadata_filter,
            }
        )
        if self._query_error is not None:
            raise self._query_error
        return list(self._matches)


def make_service(
    *, settings: AppSettings | None = None, store: FakePineconeStore | None = None
) -> tuple[RetrievalService, FakePineconeStore]:
    settings = settings or make_settings()
    store = store or FakePineconeStore()
    service = RetrievalService(settings, pinecone_store=store)
    return service, store


def _assert_malformed(metadata: dict[str, Any]) -> None:
    match = PineconeQueryMatch(id="a", score=0.9, metadata=metadata)
    store = FakePineconeStore(matches=[match])
    service, _store = make_service(store=store)

    with pytest.raises(MalformedRetrievalMetadataError):
        service.search("query")


# --- happy path / resolution ---------------------------------------------------


def test_search_happy_path_round_trips_metadata() -> None:
    match = make_match()
    store = FakePineconeStore(matches=[match])
    service, _store = make_service(store=store)

    result = service.search("how do I configure the client?")

    assert isinstance(result, RetrievalResult)
    assert result.query == "how do I configure the client?"
    assert len(result.matches) == 1
    chunk = result.matches[0]
    assert chunk.chunk_id == match.id
    assert chunk.score == match.score
    assert chunk.document_id == match.metadata["document_id"]
    assert chunk.source_url == match.metadata["source_url"]
    assert chunk.final_url == match.metadata["final_url"]
    assert chunk.title == match.metadata["title"]
    assert chunk.content_hash == match.metadata["content_hash"]
    assert chunk.chunk_index == match.metadata["chunk_index"]
    assert chunk.chunk_count == match.metadata["chunk_count"]
    assert chunk.text == match.metadata["text"]


def test_search_normalizes_query_whitespace() -> None:
    store = FakePineconeStore(matches=[])
    service, _store = make_service(store=store)

    result = service.search("  how do I configure the client?  ")

    assert result.query == "how do I configure the client?"
    assert store.embed_calls == ["how do I configure the client?"]


def test_search_uses_default_top_k_from_settings() -> None:
    settings = make_settings(retrieval_top_k=7)
    store = FakePineconeStore(matches=[])
    service, _store = make_service(settings=settings, store=store)

    result = service.search("query")

    assert result.top_k == 7
    assert store.query_calls[0]["top_k"] == 7


def test_search_uses_overridden_top_k() -> None:
    settings = make_settings(retrieval_top_k=7)
    store = FakePineconeStore(matches=[])
    service, _store = make_service(settings=settings, store=store)

    result = service.search("query", top_k=3)

    assert result.top_k == 3
    assert store.query_calls[0]["top_k"] == 3


def test_search_uses_default_namespace_from_settings() -> None:
    settings = make_settings(pinecone_documents_namespace="docs-default")
    store = FakePineconeStore(matches=[])
    service, _store = make_service(settings=settings, store=store)

    result = service.search("query")

    assert result.namespace == "docs-default"
    assert store.query_calls[0]["namespace"] == "docs-default"


def test_search_uses_overridden_namespace() -> None:
    store = FakePineconeStore(matches=[])
    service, _store = make_service(store=store)

    result = service.search("query", namespace="  custom-ns  ")

    assert result.namespace == "custom-ns"
    assert store.query_calls[0]["namespace"] == "custom-ns"


# --- validation before any external call ----------------------------------------


def test_search_rejects_blank_query_before_any_call() -> None:
    store = FakePineconeStore(matches=[])
    service, _store = make_service(store=store)

    with pytest.raises(RetrievalError):
        service.search("   ")

    assert store.embed_calls == []
    assert store.query_calls == []


def test_search_rejects_top_k_below_one_before_any_call() -> None:
    store = FakePineconeStore(matches=[])
    service, _store = make_service(store=store)

    with pytest.raises(RetrievalError):
        service.search("query", top_k=0)

    assert store.embed_calls == []
    assert store.query_calls == []


def test_search_rejects_top_k_above_50_before_any_call() -> None:
    store = FakePineconeStore(matches=[])
    service, _store = make_service(store=store)

    with pytest.raises(RetrievalError):
        service.search("query", top_k=51)

    assert store.embed_calls == []
    assert store.query_calls == []


def test_search_rejects_blank_namespace_before_any_call() -> None:
    store = FakePineconeStore(matches=[])
    service, _store = make_service(store=store)

    with pytest.raises(RetrievalError):
        service.search("query", namespace="   ")

    assert store.embed_calls == []
    assert store.query_calls == []


# --- mandatory filter / failure mapping -----------------------------------------


def test_search_applies_mandatory_documentation_chunk_filter() -> None:
    store = FakePineconeStore(matches=[])
    service, _store = make_service(store=store)

    service.search("query")

    assert store.query_calls[0]["metadata_filter"] == {"kind": {"$eq": "documentation_chunk"}}


def test_search_embedding_failure_prevents_query() -> None:
    store = FakePineconeStore(embed_error=PineconeEmbeddingError("boom"))
    service, _store = make_service(store=store)

    with pytest.raises(QueryEmbeddingError) as exc_info:
        service.search("query")

    assert isinstance(exc_info.value.__cause__, PineconeEmbeddingError)
    assert store.query_calls == []


def test_search_query_failure_is_wrapped() -> None:
    store = FakePineconeStore(query_error=PineconeQueryError("boom"))
    service, _store = make_service(store=store)

    with pytest.raises(QueryExecutionError) as exc_info:
        service.search("query")

    assert isinstance(exc_info.value.__cause__, PineconeQueryError)


def test_search_empty_result_is_successful() -> None:
    store = FakePineconeStore(matches=[])
    service, _store = make_service(store=store)

    result = service.search("query")

    assert result.matches == ()


def test_search_preserves_exact_match_order() -> None:
    matches = [make_match(id="c"), make_match(id="a"), make_match(id="b")]
    store = FakePineconeStore(matches=matches)
    service, _store = make_service(store=store)

    result = service.search("query")

    assert [chunk.chunk_id for chunk in result.matches] == ["c", "a", "b"]


# --- malformed metadata ----------------------------------------------------------


def test_search_raises_for_missing_kind_key() -> None:
    metadata = make_metadata()
    del metadata["kind"]
    _assert_malformed(metadata)


def test_search_raises_for_missing_text_key() -> None:
    metadata = make_metadata()
    del metadata["text"]
    _assert_malformed(metadata)


def test_search_raises_for_missing_document_id_key() -> None:
    metadata = make_metadata()
    del metadata["document_id"]
    _assert_malformed(metadata)


def test_search_raises_for_missing_source_url_key() -> None:
    metadata = make_metadata()
    del metadata["source_url"]
    _assert_malformed(metadata)


def test_search_raises_for_missing_final_url_key() -> None:
    metadata = make_metadata()
    del metadata["final_url"]
    _assert_malformed(metadata)


def test_search_raises_for_missing_title_key() -> None:
    metadata = make_metadata()
    del metadata["title"]
    _assert_malformed(metadata)


def test_search_raises_for_missing_content_hash_key() -> None:
    metadata = make_metadata()
    del metadata["content_hash"]
    _assert_malformed(metadata)


def test_search_raises_for_missing_chunk_index_key() -> None:
    metadata = make_metadata()
    del metadata["chunk_index"]
    _assert_malformed(metadata)


def test_search_raises_for_missing_chunk_count_key() -> None:
    metadata = make_metadata()
    del metadata["chunk_count"]
    _assert_malformed(metadata)


def test_search_raises_for_wrong_kind_value() -> None:
    _assert_malformed(make_metadata(kind="integration_smoke_test"))


def test_search_raises_for_non_string_document_id() -> None:
    _assert_malformed(make_metadata(document_id=123))


def test_search_raises_for_blank_title() -> None:
    _assert_malformed(make_metadata(title="   "))


def test_search_raises_for_non_string_text() -> None:
    _assert_malformed(make_metadata(text=12345))


def test_search_raises_for_boolean_chunk_index() -> None:
    _assert_malformed(make_metadata(chunk_index=True))


def test_search_raises_for_boolean_chunk_count() -> None:
    _assert_malformed(make_metadata(chunk_count=True))


def test_search_raises_for_non_integer_chunk_index() -> None:
    _assert_malformed(make_metadata(chunk_index="0"))


def test_search_raises_for_non_integer_chunk_count() -> None:
    _assert_malformed(make_metadata(chunk_count="1"))


def test_search_raises_for_chunk_index_equal_to_chunk_count() -> None:
    _assert_malformed(make_metadata(chunk_index=1, chunk_count=1))


def test_search_raises_for_negative_chunk_index() -> None:
    _assert_malformed(make_metadata(chunk_index=-1, chunk_count=1))


def test_search_raises_for_non_positive_chunk_count() -> None:
    _assert_malformed(make_metadata(chunk_count=0, chunk_index=0))


def test_search_tolerates_additional_unrelated_metadata_keys() -> None:
    metadata = make_metadata(embedding_model="text-embedding-3-small", extra_flag=True)
    match = PineconeQueryMatch(id="a", score=0.9, metadata=metadata)
    store = FakePineconeStore(matches=[match])
    service, _store = make_service(store=store)

    result = service.search("query")

    assert len(result.matches) == 1
    assert result.matches[0].document_id == metadata["document_id"]


def test_search_malformed_metadata_raises_domain_exception_not_raw_error() -> None:
    metadata = make_metadata()
    del metadata["chunk_index"]
    match = PineconeQueryMatch(id="a", score=0.9, metadata=metadata)
    store = FakePineconeStore(matches=[match])
    service, _store = make_service(store=store)

    # pytest.raises(MalformedRetrievalMetadataError) below would let a raw
    # KeyError/TypeError/pydantic ValidationError propagate uncaught (failing
    # this test) instead of silently matching, so this also proves those
    # implementation exceptions never leak past the service boundary.
    with pytest.raises(MalformedRetrievalMetadataError):
        service.search("query")
