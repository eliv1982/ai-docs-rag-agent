"""Unit tests for the URL-ingestion domain models. No network access."""

from typing import Any

import pytest
from pydantic import ValidationError

from ai_docs_agent.models import (
    AnswerSource,
    DocumentChunk,
    DocumentIndexingResult,
    GroundedAnswerResult,
    PineconeQueryMatch,
    RetrievalResult,
    RetrievedChunk,
    UrlProcessingResult,
)


def make_chunk(**overrides: Any) -> DocumentChunk:
    defaults: dict[str, Any] = {
        "id": "doc-abc123-chunk-0000",
        "document_id": "doc-abc123",
        "source_url": "https://docs.example.com/page",
        "final_url": "https://docs.example.com/page",
        "title": "Example Page",
        "text": "Some chunk text.",
        "chunk_index": 0,
        "chunk_count": 1,
        "content_hash": "hash-value",
    }
    return DocumentChunk(**{**defaults, **overrides})


def make_result(**overrides: Any) -> UrlProcessingResult:
    chunk = overrides.pop("chunk", make_chunk())
    defaults: dict[str, Any] = {
        "source_url": "https://docs.example.com/page",
        "final_url": "https://docs.example.com/page",
        "title": "Example Page",
        "document_id": "doc-abc123",
        "content_hash": "hash-value",
        "text_char_count": 16,
        "chunk_count": 1,
        "chunks": (chunk,),
    }
    return UrlProcessingResult(**{**defaults, **overrides})


# --- DocumentChunk -----------------------------------------------------------


def test_document_chunk_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        make_chunk(text="   ")


def test_document_chunk_rejects_negative_chunk_index() -> None:
    with pytest.raises(ValidationError):
        make_chunk(chunk_index=-1)


def test_document_chunk_rejects_non_positive_chunk_count() -> None:
    with pytest.raises(ValidationError):
        make_chunk(chunk_count=0)


def test_document_chunk_rejects_index_not_less_than_count() -> None:
    with pytest.raises(ValidationError):
        make_chunk(chunk_index=2, chunk_count=2)


def test_document_chunk_is_frozen() -> None:
    chunk = make_chunk()
    with pytest.raises(ValidationError):
        chunk.text = "mutated"  # type: ignore[misc]


def test_document_chunk_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        make_chunk(unexpected="field")


def test_to_pinecone_metadata_has_exact_flat_fields() -> None:
    chunk = make_chunk()

    metadata = chunk.to_pinecone_metadata()

    assert set(metadata.keys()) == {
        "kind",
        "text",
        "document_id",
        "source_url",
        "final_url",
        "title",
        "content_hash",
        "chunk_index",
        "chunk_count",
    }
    assert metadata["kind"] == "documentation_chunk"
    for value in metadata.values():
        assert isinstance(value, str | int)


# --- UrlProcessingResult ------------------------------------------------------


def test_url_processing_result_accepts_consistent_chunks() -> None:
    result = make_result()

    assert result.chunk_count == 1
    assert result.chunks[0].document_id == result.document_id


def test_url_processing_result_rejects_non_positive_text_char_count() -> None:
    with pytest.raises(ValidationError):
        make_result(text_char_count=0)


def test_url_processing_result_rejects_non_positive_chunk_count() -> None:
    with pytest.raises(ValidationError):
        make_result(chunk_count=0)


def test_url_processing_result_rejects_chunk_count_mismatch() -> None:
    with pytest.raises(ValidationError):
        make_result(chunk_count=2)


def test_url_processing_result_rejects_non_sequential_chunk_indexes() -> None:
    bad_chunk = make_chunk(chunk_index=1, chunk_count=2)
    with pytest.raises(ValidationError):
        make_result(chunk_count=1, chunks=(bad_chunk,))


def test_url_processing_result_rejects_chunk_with_different_document_id() -> None:
    mismatched_chunk = make_chunk(document_id="doc-different")
    with pytest.raises(ValidationError):
        make_result(chunks=(mismatched_chunk,))


def test_url_processing_result_rejects_chunk_with_different_source_url() -> None:
    mismatched_chunk = make_chunk(source_url="https://docs.example.com/other-source")
    with pytest.raises(ValidationError):
        make_result(chunks=(mismatched_chunk,))


def test_url_processing_result_rejects_chunk_with_different_final_url() -> None:
    mismatched_chunk = make_chunk(final_url="https://docs.example.com/other-final")
    with pytest.raises(ValidationError):
        make_result(chunks=(mismatched_chunk,))


def test_url_processing_result_rejects_chunk_with_different_content_hash() -> None:
    mismatched_chunk = make_chunk(content_hash="different-hash")
    with pytest.raises(ValidationError):
        make_result(chunks=(mismatched_chunk,))


def test_url_processing_result_rejects_chunk_with_different_chunk_count() -> None:
    mismatched_chunk = make_chunk(chunk_count=3, chunk_index=0)
    with pytest.raises(ValidationError):
        make_result(chunk_count=1, chunks=(mismatched_chunk,))


def test_url_processing_result_is_frozen() -> None:
    result = make_result()
    with pytest.raises(ValidationError):
        result.title = "mutated"  # type: ignore[misc]


def test_url_processing_result_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        make_result(unexpected="field")


# --- DocumentIndexingResult ---------------------------------------------------


def make_indexing_result(**overrides: Any) -> DocumentIndexingResult:
    defaults: dict[str, Any] = {
        "source_url": "https://docs.example.com/page",
        "final_url": "https://docs.example.com/page",
        "document_id": "doc-abc123",
        "content_hash": "hash-value",
        "namespace": "documentation",
        "chunk_count": 3,
        "embedded_count": 3,
        "upserted_count": 3,
        "verified_count": 3,
        "old_versions_cleanup_requested": True,
        "old_versions_cleanup_succeeded": True,
        "elapsed_seconds": 1.5,
    }
    return DocumentIndexingResult(**{**defaults, **overrides})


def test_document_indexing_result_accepts_consistent_values() -> None:
    result = make_indexing_result()

    assert result.chunk_count == 3
    assert result.old_versions_cleanup_succeeded is True


def test_document_indexing_result_rejects_non_positive_chunk_count() -> None:
    with pytest.raises(ValidationError):
        make_indexing_result(chunk_count=0)


def test_document_indexing_result_rejects_negative_embedded_count() -> None:
    with pytest.raises(ValidationError):
        make_indexing_result(embedded_count=-1)


def test_document_indexing_result_rejects_negative_upserted_count() -> None:
    with pytest.raises(ValidationError):
        make_indexing_result(upserted_count=-1)


def test_document_indexing_result_rejects_negative_verified_count() -> None:
    with pytest.raises(ValidationError):
        make_indexing_result(verified_count=-1)


def test_document_indexing_result_rejects_embedded_count_mismatch() -> None:
    with pytest.raises(ValidationError):
        make_indexing_result(embedded_count=2)


def test_document_indexing_result_rejects_upserted_count_mismatch() -> None:
    with pytest.raises(ValidationError):
        make_indexing_result(upserted_count=2)


def test_document_indexing_result_rejects_verified_count_mismatch() -> None:
    with pytest.raises(ValidationError):
        make_indexing_result(verified_count=2)


def test_document_indexing_result_rejects_negative_elapsed_seconds() -> None:
    with pytest.raises(ValidationError):
        make_indexing_result(elapsed_seconds=-0.1)


def test_document_indexing_result_allows_zero_elapsed_seconds() -> None:
    result = make_indexing_result(elapsed_seconds=0)

    assert result.elapsed_seconds == 0


def test_document_indexing_result_cleanup_not_requested_forbids_bool_status() -> None:
    with pytest.raises(ValidationError):
        make_indexing_result(
            old_versions_cleanup_requested=False, old_versions_cleanup_succeeded=True
        )


def test_document_indexing_result_cleanup_not_requested_allows_none_status() -> None:
    result = make_indexing_result(
        old_versions_cleanup_requested=False, old_versions_cleanup_succeeded=None
    )

    assert result.old_versions_cleanup_succeeded is None


def test_document_indexing_result_cleanup_requested_forbids_none_status() -> None:
    with pytest.raises(ValidationError):
        make_indexing_result(
            old_versions_cleanup_requested=True, old_versions_cleanup_succeeded=None
        )


def test_document_indexing_result_cleanup_requested_allows_false_status() -> None:
    result = make_indexing_result(
        old_versions_cleanup_requested=True, old_versions_cleanup_succeeded=False
    )

    assert result.old_versions_cleanup_succeeded is False


def test_document_indexing_result_is_frozen() -> None:
    result = make_indexing_result()
    with pytest.raises(ValidationError):
        result.document_id = "mutated"  # type: ignore[misc]


def test_document_indexing_result_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        make_indexing_result(unexpected="field")


# --- PineconeQueryMatch --------------------------------------------------------


def make_query_match(**overrides: Any) -> PineconeQueryMatch:
    defaults: dict[str, Any] = {
        "id": "doc-abc123-chunk-0000",
        "score": 0.87,
        "metadata": {"text": "Some chunk text.", "document_id": "doc-abc123"},
    }
    return PineconeQueryMatch(**{**defaults, **overrides})


def test_pinecone_query_match_is_frozen() -> None:
    match = make_query_match()
    with pytest.raises(ValidationError):
        match.score = 0.5  # type: ignore[misc]


def test_pinecone_query_match_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        make_query_match(unexpected="field")


# --- RetrievedChunk --------------------------------------------------------


def make_retrieved_chunk(**overrides: Any) -> RetrievedChunk:
    defaults: dict[str, Any] = {
        "chunk_id": "doc-abc123-chunk-0000",
        "score": 0.87,
        "document_id": "doc-abc123",
        "source_url": "https://docs.example.com/page",
        "final_url": "https://docs.example.com/page",
        "title": "Example Page",
        "content_hash": "hash-value",
        "chunk_index": 0,
        "chunk_count": 1,
        "text": "Some chunk text.",
    }
    return RetrievedChunk(**{**defaults, **overrides})


def test_retrieved_chunk_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        make_retrieved_chunk(text="   ")


def test_retrieved_chunk_rejects_negative_chunk_index() -> None:
    with pytest.raises(ValidationError):
        make_retrieved_chunk(chunk_index=-1)


def test_retrieved_chunk_rejects_non_positive_chunk_count() -> None:
    with pytest.raises(ValidationError):
        make_retrieved_chunk(chunk_count=0)


def test_retrieved_chunk_rejects_index_not_less_than_count() -> None:
    with pytest.raises(ValidationError):
        make_retrieved_chunk(chunk_index=2, chunk_count=2)


def test_retrieved_chunk_is_frozen() -> None:
    chunk = make_retrieved_chunk()
    with pytest.raises(ValidationError):
        chunk.text = "mutated"  # type: ignore[misc]


def test_retrieved_chunk_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        make_retrieved_chunk(unexpected="field")


# --- RetrievalResult --------------------------------------------------------


def make_retrieval_result(**overrides: Any) -> RetrievalResult:
    chunks = overrides.pop("matches", (make_retrieved_chunk(),))
    defaults: dict[str, Any] = {
        "query": "how do I configure the client?",
        "namespace": "documentation",
        "top_k": 5,
        "matches": chunks,
    }
    return RetrievalResult(**{**defaults, **overrides})


def test_retrieval_result_accepts_empty_matches() -> None:
    result = make_retrieval_result(matches=())

    assert result.matches == ()


def test_retrieval_result_accepts_fewer_matches_than_top_k() -> None:
    result = make_retrieval_result(top_k=5, matches=(make_retrieved_chunk(),))

    assert len(result.matches) == 1
    assert result.top_k == 5


def test_retrieval_result_rejects_more_matches_than_top_k() -> None:
    two_chunks = (
        make_retrieved_chunk(chunk_id="chunk-a"),
        make_retrieved_chunk(chunk_id="chunk-b"),
    )
    with pytest.raises(ValidationError):
        make_retrieval_result(top_k=1, matches=two_chunks)


def test_retrieval_result_rejects_blank_query() -> None:
    with pytest.raises(ValidationError):
        make_retrieval_result(query="   ")


def test_retrieval_result_rejects_blank_namespace() -> None:
    with pytest.raises(ValidationError):
        make_retrieval_result(namespace="   ")


def test_retrieval_result_rejects_top_k_below_one() -> None:
    with pytest.raises(ValidationError):
        make_retrieval_result(top_k=0, matches=())


def test_retrieval_result_rejects_top_k_above_50() -> None:
    with pytest.raises(ValidationError):
        make_retrieval_result(top_k=51, matches=())


def test_retrieval_result_accepts_top_k_boundary_values() -> None:
    assert make_retrieval_result(top_k=1, matches=()).top_k == 1
    assert make_retrieval_result(top_k=50, matches=()).top_k == 50


def test_retrieval_result_is_frozen() -> None:
    result = make_retrieval_result()
    with pytest.raises(ValidationError):
        result.query = "mutated"  # type: ignore[misc]


def test_retrieval_result_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        make_retrieval_result(unexpected="field")


# --- AnswerSource --------------------------------------------------------


def make_answer_source(**overrides: Any) -> AnswerSource:
    defaults: dict[str, Any] = {
        "title": "Example Page",
        "url": "https://docs.example.com/page",
        "document_id": "doc-abc123",
        "chunk_index": 0,
        "chunk_count": 1,
    }
    return AnswerSource(**{**defaults, **overrides})


def test_answer_source_accepts_valid_values() -> None:
    source = make_answer_source()

    assert source.title == "Example Page"
    assert source.url == "https://docs.example.com/page"


def test_answer_source_is_frozen() -> None:
    source = make_answer_source()
    with pytest.raises(ValidationError):
        source.url = "https://mutated.example.com"  # type: ignore[misc]


def test_answer_source_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        make_answer_source(unexpected="field")


def test_answer_source_rejects_blank_title() -> None:
    with pytest.raises(ValidationError):
        make_answer_source(title="   ")


def test_answer_source_rejects_blank_url() -> None:
    with pytest.raises(ValidationError):
        make_answer_source(url="   ")


def test_answer_source_rejects_negative_chunk_index() -> None:
    with pytest.raises(ValidationError):
        make_answer_source(chunk_index=-1)


def test_answer_source_rejects_non_positive_chunk_count() -> None:
    with pytest.raises(ValidationError):
        make_answer_source(chunk_count=0)


def test_answer_source_rejects_index_not_less_than_count() -> None:
    with pytest.raises(ValidationError):
        make_answer_source(chunk_index=2, chunk_count=2)


# --- GroundedAnswerResult --------------------------------------------------------


def make_grounded_answer_result(**overrides: Any) -> GroundedAnswerResult:
    sources = overrides.pop("sources", (make_answer_source(),))
    defaults: dict[str, Any] = {
        "question": "how do I configure the client?",
        "answer": "Set the API key via the OPENAI_API_KEY environment variable.",
        "sources": sources,
        "retrieved_chunk_count": 1,
    }
    return GroundedAnswerResult(**{**defaults, **overrides})


def test_grounded_answer_result_accepts_valid_values() -> None:
    result = make_grounded_answer_result()

    assert result.retrieved_chunk_count == 1
    assert len(result.sources) == 1


def test_grounded_answer_result_allows_empty_sources_for_zero_context_fallback() -> None:
    result = make_grounded_answer_result(
        answer="В базе знаний не найдено достаточно информации для ответа на этот вопрос.",
        sources=(),
        retrieved_chunk_count=0,
    )

    assert result.sources == ()
    assert result.retrieved_chunk_count == 0


def test_grounded_answer_result_rejects_empty_sources_with_positive_chunk_count() -> None:
    with pytest.raises(ValidationError):
        make_grounded_answer_result(sources=(), retrieved_chunk_count=1)


def test_grounded_answer_result_rejects_nonempty_sources_with_zero_chunk_count() -> None:
    with pytest.raises(ValidationError):
        make_grounded_answer_result(retrieved_chunk_count=0)


def test_grounded_answer_result_rejects_blank_question() -> None:
    with pytest.raises(ValidationError):
        make_grounded_answer_result(question="   ")


def test_grounded_answer_result_rejects_blank_answer() -> None:
    with pytest.raises(ValidationError):
        make_grounded_answer_result(answer="   ")


def test_grounded_answer_result_rejects_negative_retrieved_chunk_count() -> None:
    with pytest.raises(ValidationError):
        make_grounded_answer_result(retrieved_chunk_count=-1, sources=())


def test_grounded_answer_result_is_frozen() -> None:
    result = make_grounded_answer_result()
    with pytest.raises(ValidationError):
        result.answer = "mutated"  # type: ignore[misc]


def test_grounded_answer_result_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        make_grounded_answer_result(unexpected="field")
