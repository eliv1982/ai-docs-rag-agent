"""Unit tests for the URL-ingestion domain models. No network access."""

from typing import Any

import pytest
from pydantic import ValidationError

from ai_docs_agent.models import DocumentChunk, UrlProcessingResult


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
