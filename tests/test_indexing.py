"""Unit tests for DocumentIndexingService. Uses fakes only; no real network access."""

from typing import Any

import pytest

from ai_docs_agent.config import AppSettings
from ai_docs_agent.indexing import (
    DocumentEmbeddingError,
    DocumentIndexingError,
    DocumentIndexingService,
    DocumentUpsertError,
    DocumentVerificationError,
)
from ai_docs_agent.models import DocumentChunk, DocumentIndexingResult, UrlProcessingResult
from ai_docs_agent.pinecone_store import (
    PineconeEmbeddingError,
    PineconeStoreError,
    PineconeUpsertError,
)

_REQUIRED: dict[str, Any] = {
    "openai_api_key": "sk-test-openai",
    "pinecone_api_key": "pc-test-key",
    "openai_chat_model": "gpt-4o-mini",
    "telegram_bot_token": "test-telegram-token",
}


def make_settings(**overrides: Any) -> AppSettings:
    return AppSettings(_env_file=None, **{**_REQUIRED, **overrides})


def make_chunk(
    *,
    document_id: str,
    chunk_index: int,
    chunk_count: int,
    source_url: str = "https://docs.example.com/page",
    content_hash: str = "hash-1",
) -> DocumentChunk:
    return DocumentChunk(
        id=f"{document_id}-chunk-{chunk_index:04d}",
        document_id=document_id,
        source_url=source_url,
        final_url=source_url,
        title="Example Page",
        text=f"chunk text {chunk_index}",
        chunk_index=chunk_index,
        chunk_count=chunk_count,
        content_hash=content_hash,
    )


def make_processing(
    *,
    chunk_count: int = 2,
    document_id: str = "doc-abc",
    content_hash: str = "hash-1",
    source_url: str = "https://docs.example.com/page",
) -> UrlProcessingResult:
    chunks = tuple(
        make_chunk(
            document_id=document_id,
            chunk_index=index,
            chunk_count=chunk_count,
            source_url=source_url,
            content_hash=content_hash,
        )
        for index in range(chunk_count)
    )
    return UrlProcessingResult(
        source_url=source_url,
        final_url=source_url,
        title="Example Page",
        document_id=document_id,
        content_hash=content_hash,
        text_char_count=200,
        chunk_count=chunk_count,
        chunks=chunks,
    )


def _evaluate_pinecone_filter(filter_expr: dict[str, Any], metadata: dict[str, Any]) -> bool:
    """Minimal local evaluator for the $and/$or/$eq/$ne/$gte subset used by cleanup filters."""
    if "$and" in filter_expr:
        return all(_evaluate_pinecone_filter(clause, metadata) for clause in filter_expr["$and"])
    if "$or" in filter_expr:
        return any(_evaluate_pinecone_filter(clause, metadata) for clause in filter_expr["$or"])
    ((field, condition),) = filter_expr.items()
    ((operator, value),) = condition.items()
    if operator == "$eq":
        return metadata.get(field) == value
    if operator == "$ne":
        return metadata.get(field) != value
    if operator == "$gte":
        return metadata.get(field) >= value
    raise ValueError(f"Unsupported operator: {operator}")


class ManualClock:
    """A manually-advanced fake clock/sleep pair so tests run instantly."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def sleep(self, seconds: float) -> None:
        self.advance(seconds)


def fail_sleep(seconds: float) -> None:
    raise AssertionError("sleep should not be called on this path")


class FakeUrlIngestionService:
    def __init__(
        self, *, result: UrlProcessingResult | None = None, error: Exception | None = None
    ) -> None:
        self._result = result
        self._error = error
        self.calls: list[str] = []

    def process_url(self, url: str) -> UrlProcessingResult:
        self.calls.append(url)
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


class FakePineconeStore:
    """Fake low-level PineconeStore for orchestration-level tests, fully scriptable."""

    def __init__(
        self,
        *,
        embed_dimension: int = 3,
        embed_batches: list[list[list[float]]] | None = None,
        embed_error: Exception | None = None,
        upsert_counts: list[int] | None = None,
        upsert_error_on_batch: int | None = None,
        upsert_error: Exception | None = None,
        fetch_results: list[set[str]] | None = None,
        fetch_response_fn: Any = None,
        fetch_error: Exception | None = None,
        delete_error: Exception | None = None,
    ) -> None:
        self.embed_calls: list[list[str]] = []
        self.upsert_calls: list[tuple[list[dict[str, Any]], str]] = []
        self.fetch_calls: list[tuple[list[str], str]] = []
        self.delete_calls: list[tuple[dict[str, Any], str]] = []
        self.call_order: list[str] = []

        self._embed_dimension = embed_dimension
        self._embed_batches = embed_batches
        self._embed_error = embed_error
        self._embed_counter = 0

        self._upsert_counts = upsert_counts
        self._upsert_error_on_batch = upsert_error_on_batch
        self._upsert_error = upsert_error
        self._upsert_batch_index = 0

        self._fetch_results = fetch_results
        self._fetch_response_fn = fetch_response_fn
        self._fetch_error = fetch_error
        self._fetch_call_index = 0

        self._delete_error = delete_error

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.call_order.append("embed")
        self.embed_calls.append(list(texts))
        if self._embed_error is not None:
            raise self._embed_error
        if self._embed_batches is not None:
            return self._embed_batches[len(self.embed_calls) - 1]
        result: list[list[float]] = []
        for _ in texts:
            result.append([float(self._embed_counter)] * self._embed_dimension)
            self._embed_counter += 1
        return result

    def upsert_vectors(self, vectors: list[dict[str, Any]], *, namespace: str) -> int:
        self.call_order.append("upsert")
        batch_index = self._upsert_batch_index
        self._upsert_batch_index += 1
        self.upsert_calls.append((list(vectors), namespace))
        if self._upsert_error is not None and batch_index == (self._upsert_error_on_batch or 0):
            raise self._upsert_error
        if self._upsert_counts is not None:
            return self._upsert_counts[batch_index]
        return len(vectors)

    def fetch_existing_ids(self, ids: list[str], *, namespace: str) -> set[str]:
        self.call_order.append("fetch")
        self.fetch_calls.append((list(ids), namespace))
        if self._fetch_error is not None:
            raise self._fetch_error
        if self._fetch_response_fn is not None:
            return self._fetch_response_fn(ids)
        if self._fetch_results is not None:
            call_index = min(self._fetch_call_index, len(self._fetch_results) - 1)
            self._fetch_call_index += 1
            return self._fetch_results[call_index] & set(ids)
        return set(ids)

    def delete_vectors_by_filter(
        self, metadata_filter: dict[str, Any], *, namespace: str
    ) -> None:
        self.call_order.append("delete")
        self.delete_calls.append((metadata_filter, namespace))
        if self._delete_error is not None:
            raise self._delete_error


def make_service(
    *,
    settings: AppSettings | None = None,
    url_service: FakeUrlIngestionService | None = None,
    store: FakePineconeStore | None = None,
    clock: ManualClock | None = None,
    sleep: Any = fail_sleep,
) -> tuple[DocumentIndexingService, FakeUrlIngestionService, FakePineconeStore]:
    settings = settings or make_settings()
    url_service = url_service or FakeUrlIngestionService(result=make_processing())
    store = store or FakePineconeStore()
    clock = clock or ManualClock()
    service = DocumentIndexingService(
        settings,
        url_ingestion_service=url_service,
        pinecone_store=store,
        clock=clock,
        sleep=sleep,
    )
    return service, url_service, store


# --- happy path / namespace resolution ---------------------------------------


def test_index_url_happy_path_returns_consistent_result() -> None:
    processing = make_processing(chunk_count=3)
    service, url_service, store = make_service(
        url_service=FakeUrlIngestionService(result=processing)
    )

    result = service.index_url("https://docs.example.com/page")

    assert isinstance(result, DocumentIndexingResult)
    assert result.source_url == processing.source_url
    assert result.final_url == processing.final_url
    assert result.document_id == processing.document_id
    assert result.content_hash == processing.content_hash
    assert result.chunk_count == 3
    assert result.embedded_count == 3
    assert result.upserted_count == 3
    assert result.verified_count == 3
    assert url_service.calls == ["https://docs.example.com/page"]


def test_explicit_namespace_is_stripped_and_used() -> None:
    service, _url_service, store = make_service()

    result = service.index_url("https://docs.example.com/page", namespace="  custom-ns  ")

    assert result.namespace == "custom-ns"
    assert all(namespace == "custom-ns" for _, namespace in store.upsert_calls)


def test_default_namespace_uses_settings_value() -> None:
    settings = make_settings(pinecone_documents_namespace="docs-default")
    service, _url_service, _store = make_service(settings=settings)

    result = service.index_url("https://docs.example.com/page")

    assert result.namespace == "docs-default"


def test_blank_explicit_namespace_rejected_before_url_ingestion() -> None:
    service, url_service, store = make_service()

    with pytest.raises(DocumentIndexingError):
        service.index_url("https://docs.example.com/page", namespace="   ")

    assert url_service.calls == []
    assert store.embed_calls == []


# --- embedding -----------------------------------------------------------------


def test_embedding_batches_split_according_to_configured_size() -> None:
    processing = make_processing(chunk_count=5)
    settings = make_settings(embedding_batch_size=2)
    service, _url_service, store = make_service(
        settings=settings, url_service=FakeUrlIngestionService(result=processing)
    )

    service.index_url("https://docs.example.com/page")

    assert [len(batch) for batch in store.embed_calls] == [2, 2, 1]


def test_embedding_failure_prevents_upsert_and_delete() -> None:
    store = FakePineconeStore(embed_error=PineconeEmbeddingError("boom"))
    service, _url_service, _store = make_service(store=store)

    with pytest.raises(DocumentEmbeddingError) as exc_info:
        service.index_url("https://docs.example.com/page")

    assert isinstance(exc_info.value.__cause__, PineconeEmbeddingError)
    assert store.upsert_calls == []
    assert store.delete_calls == []


def test_embed_count_mismatch_without_exception_raises_embedding_error() -> None:
    processing = make_processing(chunk_count=2)
    store = FakePineconeStore(embed_batches=[[[0.1, 0.2, 0.3]]])
    service, _url_service, _store = make_service(
        url_service=FakeUrlIngestionService(result=processing), store=store
    )

    with pytest.raises(DocumentEmbeddingError):
        service.index_url("https://docs.example.com/page")

    assert store.upsert_calls == []


# --- upsert ----------------------------------------------------------------------


def test_upsert_batches_preserve_chunk_and_embedding_alignment() -> None:
    processing = make_processing(chunk_count=5)
    settings = make_settings(embedding_batch_size=2, pinecone_upsert_batch_size=3)
    service, _url_service, store = make_service(
        settings=settings, url_service=FakeUrlIngestionService(result=processing)
    )

    service.index_url("https://docs.example.com/page")

    assert [len(batch) for batch, _ in store.upsert_calls] == [3, 2]

    all_records = [record for batch, _ in store.upsert_calls for record in batch]
    for index, (record, chunk) in enumerate(zip(all_records, processing.chunks, strict=True)):
        assert record["id"] == chunk.id
        assert record["metadata"] == chunk.to_pinecone_metadata()
        assert record["values"] == [float(index)] * 3


def test_upsert_partial_batch_failure_raises_and_skips_cleanup() -> None:
    processing = make_processing(chunk_count=5)
    settings = make_settings(pinecone_upsert_batch_size=2)
    store = FakePineconeStore(
        upsert_error_on_batch=1, upsert_error=PineconeUpsertError("boom")
    )
    service, _url_service, _store = make_service(
        settings=settings, url_service=FakeUrlIngestionService(result=processing), store=store
    )

    with pytest.raises(DocumentUpsertError) as exc_info:
        service.index_url("https://docs.example.com/page")

    assert isinstance(exc_info.value.__cause__, PineconeUpsertError)
    assert len(store.upsert_calls) == 2
    assert store.delete_calls == []
    assert store.fetch_calls == []


def test_upsert_count_mismatch_without_exception_still_raises() -> None:
    processing = make_processing(chunk_count=3)
    settings = make_settings(pinecone_upsert_batch_size=10)
    store = FakePineconeStore(upsert_counts=[2])
    service, _url_service, _store = make_service(
        settings=settings, url_service=FakeUrlIngestionService(result=processing), store=store
    )

    with pytest.raises(DocumentUpsertError):
        service.index_url("https://docs.example.com/page")

    assert store.delete_calls == []


# --- verification ----------------------------------------------------------------


def test_verification_immediate_success_does_not_sleep() -> None:
    processing = make_processing(chunk_count=2)
    service, _url_service, _store = make_service(
        url_service=FakeUrlIngestionService(result=processing)
    )

    result = service.index_url("https://docs.example.com/page")

    assert result.verified_count == 2


def test_verification_polls_until_all_ids_found() -> None:
    processing = make_processing(chunk_count=2)
    expected_ids = {chunk.id for chunk in processing.chunks}
    store = FakePineconeStore(fetch_results=[set(), expected_ids])
    settings = make_settings(
        pinecone_index_verify_timeout_seconds=10,
        pinecone_index_verify_poll_interval_seconds=1,
    )
    clock = ManualClock()
    service, _url_service, _store = make_service(
        settings=settings,
        url_service=FakeUrlIngestionService(result=processing),
        store=store,
        clock=clock,
        sleep=clock.sleep,
    )

    result = service.index_url("https://docs.example.com/page")

    assert result.verified_count == 2
    assert len(store.fetch_calls) == 2


def test_verification_accumulates_confirmed_ids_across_polls_that_each_see_only_one_id() -> None:
    # Regression: poll 1 sees only A, poll 2 sees only B (not A again). A naive
    # "found_ids reset every round" implementation would never confirm both IDs
    # at once and would eventually time out despite both being visible.
    processing = make_processing(chunk_count=2)
    chunk_a, chunk_b = processing.chunks
    store = FakePineconeStore(fetch_results=[{chunk_a.id}, {chunk_b.id}])
    settings = make_settings(
        pinecone_index_verify_timeout_seconds=10,
        pinecone_index_verify_poll_interval_seconds=1,
    )
    clock = ManualClock()
    service, _url_service, _store = make_service(
        settings=settings,
        url_service=FakeUrlIngestionService(result=processing),
        store=store,
        clock=clock,
        sleep=clock.sleep,
    )

    result = service.index_url("https://docs.example.com/page")

    assert result.verified_count == 2
    assert len(store.fetch_calls) == 2


def test_unexpected_id_does_not_increase_verified_count() -> None:
    processing = make_processing(chunk_count=2)

    def fetch_with_extra(ids: list[str]) -> set[str]:
        return set(ids) | {"unexpected-id-not-a-chunk"}

    store = FakePineconeStore(fetch_response_fn=fetch_with_extra)
    service, _url_service, _store = make_service(
        url_service=FakeUrlIngestionService(result=processing), store=store
    )

    result = service.index_url("https://docs.example.com/page")

    assert result.verified_count == 2


def test_duplicate_expected_chunk_ids_are_rejected() -> None:
    processing = make_processing(chunk_count=2)
    tampered_processing = UrlProcessingResult.model_construct(
        source_url=processing.source_url,
        final_url=processing.final_url,
        title=processing.title,
        document_id=processing.document_id,
        content_hash=processing.content_hash,
        text_char_count=processing.text_char_count,
        chunk_count=2,
        chunks=(processing.chunks[0], processing.chunks[0]),
    )
    store = FakePineconeStore()
    service, _url_service, _store = make_service(
        url_service=FakeUrlIngestionService(result=tampered_processing), store=store
    )

    with pytest.raises(DocumentVerificationError):
        service.index_url("https://docs.example.com/page")

    assert store.fetch_calls == []


def test_verification_timeout_message_uses_cumulative_count_not_last_round_only() -> None:
    processing = make_processing(chunk_count=2, document_id="doc-verify")
    chunk_a, _chunk_b = processing.chunks
    store = FakePineconeStore(fetch_results=[{chunk_a.id}, set()])
    settings = make_settings(
        pinecone_index_verify_timeout_seconds=2,
        pinecone_index_verify_poll_interval_seconds=1,
    )
    clock = ManualClock()
    service, _url_service, _store = make_service(
        settings=settings,
        url_service=FakeUrlIngestionService(result=processing),
        store=store,
        clock=clock,
        sleep=clock.sleep,
    )

    with pytest.raises(DocumentVerificationError) as exc_info:
        service.index_url("https://docs.example.com/page")

    message = str(exc_info.value)
    assert "found 1 of 2" in message


def test_fetch_batching_and_cumulative_polling_work_together() -> None:
    processing = make_processing(chunk_count=4)
    chunk_0, chunk_1, chunk_2, chunk_3 = processing.chunks
    store = FakePineconeStore(
        fetch_results=[{chunk_0.id}, set(), {chunk_1.id}, {chunk_2.id, chunk_3.id}]
    )
    settings = make_settings(
        pinecone_fetch_batch_size=2,
        pinecone_index_verify_timeout_seconds=10,
        pinecone_index_verify_poll_interval_seconds=1,
    )
    clock = ManualClock()
    service, _url_service, _store = make_service(
        settings=settings,
        url_service=FakeUrlIngestionService(result=processing),
        store=store,
        clock=clock,
        sleep=clock.sleep,
    )

    result = service.index_url("https://docs.example.com/page")

    assert result.verified_count == 4
    assert len(store.fetch_calls) == 4


def test_verification_timeout_raises_and_skips_cleanup() -> None:
    processing = make_processing(chunk_count=1)
    store = FakePineconeStore(fetch_results=[set()])
    settings = make_settings(
        pinecone_index_verify_timeout_seconds=2,
        pinecone_index_verify_poll_interval_seconds=1,
    )
    clock = ManualClock()
    service, _url_service, _store = make_service(
        settings=settings,
        url_service=FakeUrlIngestionService(result=processing),
        store=store,
        clock=clock,
        sleep=clock.sleep,
    )

    with pytest.raises(DocumentVerificationError):
        service.index_url("https://docs.example.com/page")

    assert store.delete_calls == []


def test_verification_timeout_message_contains_only_counts_namespace_and_document_id() -> None:
    processing = make_processing(chunk_count=2, document_id="doc-verify")
    store = FakePineconeStore(fetch_results=[set()])
    settings = make_settings(
        pinecone_index_verify_timeout_seconds=1,
        pinecone_index_verify_poll_interval_seconds=1,
        openai_api_key="sk-super-secret",
        pinecone_api_key="pc-super-secret",
    )
    clock = ManualClock()
    service, _url_service, _store = make_service(
        settings=settings,
        url_service=FakeUrlIngestionService(result=processing),
        store=store,
        clock=clock,
        sleep=clock.sleep,
    )

    with pytest.raises(DocumentVerificationError) as exc_info:
        service.index_url("https://docs.example.com/page")

    message = str(exc_info.value)
    assert "doc-verify" in message
    assert "documentation" in message
    assert "0" in message
    assert "2" in message
    assert "sk-super-secret" not in message
    assert "pc-super-secret" not in message


def test_verification_splits_ids_into_configured_fetch_batches() -> None:
    processing = make_processing(chunk_count=5)
    settings = make_settings(pinecone_fetch_batch_size=2)
    service, _url_service, store = make_service(
        settings=settings, url_service=FakeUrlIngestionService(result=processing)
    )

    result = service.index_url("https://docs.example.com/page")

    assert result.verified_count == 5
    assert [len(ids) for ids, _ in store.fetch_calls] == [2, 2, 1]


# --- cleanup -----------------------------------------------------------------------


def test_cleanup_happens_only_after_verification_completes() -> None:
    service, _url_service, store = make_service()

    service.index_url("https://docs.example.com/page")

    assert store.call_order.index("delete") > store.call_order.index("fetch")


def test_cleanup_disabled_skips_delete_and_reports_not_requested() -> None:
    settings = make_settings(pinecone_replace_old_source_versions=False)
    service, _url_service, store = make_service(settings=settings)

    result = service.index_url("https://docs.example.com/page")

    assert result.old_versions_cleanup_requested is False
    assert result.old_versions_cleanup_succeeded is None
    assert store.delete_calls == []


def test_cleanup_success_reports_true() -> None:
    service, _url_service, _store = make_service()

    result = service.index_url("https://docs.example.com/page")

    assert result.old_versions_cleanup_requested is True
    assert result.old_versions_cleanup_succeeded is True


def test_cleanup_failure_returns_result_with_false_status_not_raise() -> None:
    processing = make_processing(chunk_count=1)
    store = FakePineconeStore(delete_error=PineconeStoreError("boom"))
    service, _url_service, _store = make_service(
        url_service=FakeUrlIngestionService(result=processing), store=store
    )

    result = service.index_url("https://docs.example.com/page")

    assert result.upserted_count == processing.chunk_count
    assert result.verified_count == processing.chunk_count
    assert result.old_versions_cleanup_requested is True
    assert result.old_versions_cleanup_succeeded is False


def test_cleanup_filter_has_exact_structure() -> None:
    processing = make_processing(
        source_url="https://docs.example.com/page",
        document_id="doc-current",
        chunk_count=3,
    )
    service, _url_service, store = make_service(
        url_service=FakeUrlIngestionService(result=processing)
    )

    service.index_url("https://docs.example.com/page")

    metadata_filter, namespace = store.delete_calls[0]
    assert metadata_filter == {
        "$or": [
            {
                "$and": [
                    {"source_url": {"$eq": "https://docs.example.com/page"}},
                    {"document_id": {"$ne": "doc-current"}},
                ]
            },
            {
                "$and": [
                    {"source_url": {"$eq": "https://docs.example.com/page"}},
                    {"document_id": {"$eq": "doc-current"}},
                    {"chunk_index": {"$gte": 3}},
                ]
            },
        ]
    }
    assert namespace == "documentation"


def test_cleanup_filter_matches_old_document_version_of_same_source_url() -> None:
    processing = make_processing(
        source_url="https://docs.example.com/page", document_id="doc-current", chunk_count=3
    )
    service, _url_service, store = make_service(
        url_service=FakeUrlIngestionService(result=processing)
    )

    service.index_url("https://docs.example.com/page")

    metadata_filter, _namespace = store.delete_calls[0]
    old_version_chunk = {
        "source_url": "https://docs.example.com/page",
        "document_id": "doc-old",
        "chunk_index": 0,
    }
    assert _evaluate_pinecone_filter(metadata_filter, old_version_chunk) is True


def test_cleanup_filter_does_not_match_current_document_chunks_below_chunk_count() -> None:
    processing = make_processing(
        source_url="https://docs.example.com/page", document_id="doc-current", chunk_count=3
    )
    service, _url_service, store = make_service(
        url_service=FakeUrlIngestionService(result=processing)
    )

    service.index_url("https://docs.example.com/page")

    metadata_filter, _namespace = store.delete_calls[0]
    for index in range(3):
        current_chunk = {
            "source_url": "https://docs.example.com/page",
            "document_id": "doc-current",
            "chunk_index": index,
        }
        assert _evaluate_pinecone_filter(metadata_filter, current_chunk) is False


def test_cleanup_filter_matches_stale_chunk_index_at_or_above_chunk_count() -> None:
    processing = make_processing(
        source_url="https://docs.example.com/page", document_id="doc-current", chunk_count=3
    )
    service, _url_service, store = make_service(
        url_service=FakeUrlIngestionService(result=processing)
    )

    service.index_url("https://docs.example.com/page")

    metadata_filter, _namespace = store.delete_calls[0]
    stale_chunk = {
        "source_url": "https://docs.example.com/page",
        "document_id": "doc-current",
        "chunk_index": 3,
    }
    assert _evaluate_pinecone_filter(metadata_filter, stale_chunk) is True


def test_cleanup_filter_does_not_match_a_different_source_url() -> None:
    processing = make_processing(
        source_url="https://docs.example.com/page", document_id="doc-current", chunk_count=3
    )
    service, _url_service, store = make_service(
        url_service=FakeUrlIngestionService(result=processing)
    )

    service.index_url("https://docs.example.com/page")

    metadata_filter, _namespace = store.delete_calls[0]
    other_source_chunk = {
        "source_url": "https://docs.example.com/other-page",
        "document_id": "doc-other",
        "chunk_index": 0,
    }
    assert _evaluate_pinecone_filter(metadata_filter, other_source_chunk) is False


def test_cleanup_filter_covers_same_hash_shrink_scenario_without_orphan_chunks() -> None:
    # Same page, same content_hash, but chunk_count shrank (e.g. a chunking-config
    # change) from 5 to 4: chunk_index 4 from the old layout shares document_id AND
    # content_hash with the current version, so a content_hash-only filter would
    # miss it. The chunk_index >= chunk_count clause must still catch it.
    processing = make_processing(
        source_url="https://docs.example.com/page",
        document_id="doc-same",
        content_hash="hash-same",
        chunk_count=4,
    )
    service, _url_service, store = make_service(
        url_service=FakeUrlIngestionService(result=processing)
    )

    service.index_url("https://docs.example.com/page")

    metadata_filter, _namespace = store.delete_calls[0]
    orphan_chunk_from_old_layout = {
        "source_url": "https://docs.example.com/page",
        "document_id": "doc-same",
        "chunk_index": 4,
    }
    assert _evaluate_pinecone_filter(metadata_filter, orphan_chunk_from_old_layout) is True


# --- idempotency / determinism ------------------------------------------------------


def test_repeated_indexing_of_same_page_upserts_same_ids() -> None:
    processing = make_processing(chunk_count=2, document_id="doc-same", content_hash="hash-same")

    service_one, _url_one, store_one = make_service(
        url_service=FakeUrlIngestionService(result=processing)
    )
    service_one.index_url("https://docs.example.com/page")
    first_ids = [record["id"] for batch, _ in store_one.upsert_calls for record in batch]

    service_two, _url_two, store_two = make_service(
        url_service=FakeUrlIngestionService(result=processing)
    )
    service_two.index_url("https://docs.example.com/page")
    second_ids = [record["id"] for batch, _ in store_two.upsert_calls for record in batch]

    assert first_ids == second_ids


def test_changed_content_produces_different_document_and_chunk_ids() -> None:
    processing_v1 = make_processing(document_id="doc-v1", content_hash="hash-v1", chunk_count=1)
    processing_v2 = make_processing(document_id="doc-v2", content_hash="hash-v2", chunk_count=1)

    service_one, _url_one, _store_one = make_service(
        url_service=FakeUrlIngestionService(result=processing_v1)
    )
    result_one = service_one.index_url("https://docs.example.com/page")

    service_two, _url_two, _store_two = make_service(
        url_service=FakeUrlIngestionService(result=processing_v2)
    )
    result_two = service_two.index_url("https://docs.example.com/page")

    assert result_one.document_id != result_two.document_id
    ids_one = {chunk.id for chunk in processing_v1.chunks}
    ids_two = {chunk.id for chunk in processing_v2.chunks}
    assert ids_one.isdisjoint(ids_two)


# --- timing --------------------------------------------------------------------------


def test_elapsed_seconds_reflects_fake_clock_duration() -> None:
    clock = ManualClock()
    processing = make_processing(chunk_count=1)
    store = FakePineconeStore()

    original_embed_documents = store.embed_documents

    def embed_documents_with_clock_advance(texts: list[str]) -> list[list[float]]:
        clock.advance(2.5)
        return original_embed_documents(texts)

    store.embed_documents = embed_documents_with_clock_advance  # type: ignore[method-assign]

    service, _url_service, _store = make_service(
        url_service=FakeUrlIngestionService(result=processing),
        store=store,
        clock=clock,
        sleep=clock.sleep,
    )

    result = service.index_url("https://docs.example.com/page")

    assert result.elapsed_seconds == pytest.approx(2.5)
