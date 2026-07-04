"""Unit tests for UserMemoryService. Uses a fake PineconeStore only; no real network access."""

import logging
from datetime import UTC, datetime
from typing import Any

import pytest

from ai_docs_agent.config import AppSettings
from ai_docs_agent.models import PineconeQueryMatch, UserMemoryRecallResult, UserMemoryWriteResult
from ai_docs_agent.pinecone_store import (
    PineconeEmbeddingError,
    PineconeFetchError,
    PineconeQueryError,
    PineconeUpsertError,
)
from ai_docs_agent.user_memory import (
    InvalidMemoryStatementError,
    InvalidUserIdentifierError,
    MalformedMemoryRecordError,
    MemoryEmbeddingError,
    MemoryIdentityConfigurationError,
    MemoryRecallError,
    MemoryStorageError,
    UserMemoryService,
    normalize_statement,
    parse_remember_command,
    statement_content_hash,
)

_REQUIRED: dict[str, Any] = {
    "openai_api_key": "sk-test-openai",
    "pinecone_api_key": "pc-test-key",
    "openai_chat_model": "gpt-4o-mini",
    "telegram_bot_token": "test-telegram-token",
    "user_memory_hash_secret": "unit-test-user-memory-secret",
}

_RAW_USER_ID = "123456789"
_OTHER_RAW_USER_ID = "987654321"
_STATEMENT = "В примерах я предпочитаю httpx."
_QUERY = "Какую HTTP-библиотеку я предпочитаю?"


def make_settings(**overrides: Any) -> AppSettings:
    return AppSettings(_env_file=None, **{**_REQUIRED, **overrides})


def make_memory_metadata(**overrides: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "kind": "user_memory",
        "text": _STATEMENT,
        "content_hash": statement_content_hash(_STATEMENT),
        "schema_version": 1,
        "created_at": "2026-07-04T00:00:00Z",
    }
    return {**defaults, **overrides}


def make_memory_match(**overrides: Any) -> PineconeQueryMatch:
    metadata_overrides = overrides.pop("metadata_overrides", {})
    defaults: dict[str, Any] = {
        "id": "memory-" + statement_content_hash(_STATEMENT)[:32],
        "score": 0.9,
        "metadata": make_memory_metadata(**metadata_overrides),
    }
    return PineconeQueryMatch(**{**defaults, **overrides})


class FakePineconeStore:
    """Fake low-level PineconeStore for UserMemoryService orchestration tests.

    Keeps per-namespace records in a plain dict so dedup/isolation behavior can
    be asserted without any network access.
    """

    def __init__(
        self,
        *,
        vector: list[float] | None = None,
        embed_error: Exception | None = None,
        fetch_error: Exception | None = None,
        upsert_error: Exception | None = None,
        upsert_count: int | None = None,
        matches: list[PineconeQueryMatch] | None = None,
        query_error: Exception | None = None,
    ) -> None:
        self.embed_documents_calls: list[list[str]] = []
        self.embed_query_calls: list[str] = []
        self.fetch_calls: list[dict[str, Any]] = []
        self.upsert_calls: list[dict[str, Any]] = []
        self.query_calls: list[dict[str, Any]] = []
        self.records: dict[str, dict[str, dict[str, Any]]] = {}
        self._vector = vector if vector is not None else [0.1, 0.2, 0.3]
        self._embed_error = embed_error
        self._fetch_error = fetch_error
        self._upsert_error = upsert_error
        self._upsert_count = upsert_count
        self._matches = matches if matches is not None else []
        self._query_error = query_error

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embed_documents_calls.append(list(texts))
        if self._embed_error is not None:
            raise self._embed_error
        return [list(self._vector) for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        self.embed_query_calls.append(text)
        if self._embed_error is not None:
            raise self._embed_error
        return list(self._vector)

    def fetch_existing_ids(self, ids: list[str], *, namespace: str) -> set[str]:
        self.fetch_calls.append({"ids": list(ids), "namespace": namespace})
        if self._fetch_error is not None:
            raise self._fetch_error
        existing = self.records.get(namespace, {})
        return {record_id for record_id in ids if record_id in existing}

    def upsert_vectors(self, vectors: list[dict[str, Any]], *, namespace: str) -> int:
        self.upsert_calls.append({"vectors": list(vectors), "namespace": namespace})
        if self._upsert_error is not None:
            raise self._upsert_error
        for vector in vectors:
            self.records.setdefault(namespace, {})[vector["id"]] = vector
        if self._upsert_count is not None:
            return self._upsert_count
        return len(vectors)

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
    *,
    settings: AppSettings | None = None,
    store: FakePineconeStore | None = None,
) -> tuple[UserMemoryService, FakePineconeStore]:
    settings = settings or make_settings()
    store = store or FakePineconeStore()
    service = UserMemoryService(
        settings,
        pinecone_store=store,  # type: ignore[arg-type]
        clock=lambda: 0.0,
        wall_clock=lambda: 1_780_000_000.0,
    )
    return service, store


# --- identity and namespaces ----------------------------------------------------


def test_same_user_and_secret_produce_stable_namespace() -> None:
    service, store = make_service()

    service.remember(_RAW_USER_ID, _STATEMENT)
    service.recall(_RAW_USER_ID, _QUERY)

    assert store.upsert_calls[0]["namespace"] == store.query_calls[0]["namespace"]

    other_service, other_store = make_service()
    other_service.remember(_RAW_USER_ID, _STATEMENT)
    assert other_store.upsert_calls[0]["namespace"] == store.upsert_calls[0]["namespace"]


def test_different_users_receive_different_namespaces() -> None:
    service, store = make_service()

    service.remember(_RAW_USER_ID, _STATEMENT)
    service.remember(_OTHER_RAW_USER_ID, _STATEMENT)

    namespaces = {call["namespace"] for call in store.upsert_calls}
    assert len(namespaces) == 2


def test_changing_secret_changes_namespace() -> None:
    service_a, store_a = make_service(
        settings=make_settings(user_memory_hash_secret="secret-one")
    )
    service_b, store_b = make_service(
        settings=make_settings(user_memory_hash_secret="secret-two")
    )

    service_a.remember(_RAW_USER_ID, _STATEMENT)
    service_b.remember(_RAW_USER_ID, _STATEMENT)

    assert store_a.upsert_calls[0]["namespace"] != store_b.upsert_calls[0]["namespace"]


def test_namespace_is_prefixed_bounded_and_contains_no_raw_identifier() -> None:
    service, store = make_service()

    service.remember(_RAW_USER_ID, _STATEMENT)

    namespace = store.upsert_calls[0]["namespace"]
    assert namespace.startswith("user-memory-")
    assert _RAW_USER_ID not in namespace
    digest_part = namespace.removeprefix("user-memory-")
    assert len(digest_part) == 32
    assert all(ch in "0123456789abcdef" for ch in digest_part)


def test_result_identity_digest_is_short_and_not_raw_identifier() -> None:
    service, _store = make_service()

    result = service.remember(_RAW_USER_ID, _STATEMENT)

    assert len(result.identity_digest) == 12
    assert _RAW_USER_ID not in result.identity_digest


def test_empty_user_identifier_is_rejected() -> None:
    service, store = make_service()

    with pytest.raises(InvalidUserIdentifierError):
        service.remember("   ", _STATEMENT)
    with pytest.raises(InvalidUserIdentifierError):
        service.recall("", _QUERY)

    assert store.upsert_calls == []
    assert store.query_calls == []


def test_oversized_and_control_character_user_identifiers_are_rejected() -> None:
    service, _store = make_service()

    with pytest.raises(InvalidUserIdentifierError):
        service.remember("x" * 257, _STATEMENT)
    with pytest.raises(InvalidUserIdentifierError):
        service.remember("user\x00id", _STATEMENT)


def test_blank_hash_secret_is_rejected_safely() -> None:
    # AppSettings itself rejects a blank secret; model_construct bypasses that
    # validation to prove the service still refuses to derive an identity.
    settings = make_settings()
    broken = settings.model_copy(update={"user_memory_hash_secret": _BlankSecret()})
    store = FakePineconeStore()
    service = UserMemoryService(broken, pinecone_store=store)  # type: ignore[arg-type]

    with pytest.raises(MemoryIdentityConfigurationError):
        service.remember(_RAW_USER_ID, _STATEMENT)

    assert store.upsert_calls == []


class _BlankSecret:
    @staticmethod
    def get_secret_value() -> str:
        return "   "


# --- explicit-write command parsing ----------------------------------------------


def test_explicit_remember_command_is_recognized() -> None:
    assert (
        parse_remember_command("Запомни: в примерах я предпочитаю httpx.")
        == "в примерах я предпочитаю httpx."
    )
    assert parse_remember_command("  запомни:   текст  ") == "текст"
    assert parse_remember_command("Remember: I prefer httpx.") == "I prefer httpx."


def test_ordinary_conversation_does_not_trigger_storage() -> None:
    for message in (
        "Я использую httpx.",
        "Расскажи про httpx.",
        "Мне нравится requests?",
        "Как настроить клиент?",
        "запомнить это не команда",
        "Запомни без двоеточия",
    ):
        assert parse_remember_command(message) is None, message


def test_command_prefix_only_yields_empty_statement_which_is_rejected() -> None:
    statement = parse_remember_command("Запомни:   ")
    assert statement == ""

    service, store = make_service()
    with pytest.raises(InvalidMemoryStatementError):
        service.remember(_RAW_USER_ID, statement)
    assert store.embed_documents_calls == []


# --- statement validation ---------------------------------------------------------


def test_empty_statement_is_rejected_before_embedding() -> None:
    service, store = make_service()

    with pytest.raises(InvalidMemoryStatementError):
        service.remember(_RAW_USER_ID, "   \n  ")

    assert store.embed_documents_calls == []
    assert store.upsert_calls == []


def test_oversized_statement_is_rejected_before_embedding() -> None:
    settings = make_settings(user_memory_max_statement_length=50)
    service, store = make_service(settings=settings)

    with pytest.raises(InvalidMemoryStatementError):
        service.remember(_RAW_USER_ID, "x" * 51)

    assert store.embed_documents_calls == []


def test_control_character_statement_is_rejected_before_embedding() -> None:
    service, store = make_service()

    with pytest.raises(InvalidMemoryStatementError):
        service.remember(_RAW_USER_ID, "текст с \x00 нулевым байтом")

    assert store.embed_documents_calls == []


def test_unicode_russian_statement_is_preserved_verbatim() -> None:
    service, store = make_service()

    service.remember(_RAW_USER_ID, f"  {_STATEMENT}  ")

    metadata = store.upsert_calls[0]["vectors"][0]["metadata"]
    assert metadata["text"] == _STATEMENT


# --- deduplication ---------------------------------------------------------------


def test_content_hash_and_record_id_are_deterministic() -> None:
    assert statement_content_hash(_STATEMENT) == statement_content_hash(_STATEMENT)
    assert normalize_statement("  В ПРИМЕРАХ   я предпочитаю httpx.  ") == normalize_statement(
        "в примерах я предпочитаю httpx."
    )

    service, store = make_service()
    result = service.remember(_RAW_USER_ID, _STATEMENT)
    expected_id = "memory-" + statement_content_hash(_STATEMENT)[:32]
    assert result.memory_id == expected_id
    assert store.upsert_calls[0]["vectors"][0]["id"] == expected_id


def test_repeated_identical_remember_is_deduplicated() -> None:
    service, store = make_service()

    first = service.remember(_RAW_USER_ID, _STATEMENT)
    second = service.remember(_RAW_USER_ID, _STATEMENT)

    assert isinstance(first, UserMemoryWriteResult)
    assert first.status == "created"
    assert second.status == "duplicate"
    assert second.memory_id == first.memory_id


def test_duplicate_remember_does_not_add_a_second_record_or_embed_again() -> None:
    service, store = make_service()

    service.remember(_RAW_USER_ID, _STATEMENT)
    service.remember(_RAW_USER_ID, "  в примерах Я  предпочитаю httpx.  ")

    namespace = store.upsert_calls[0]["namespace"]
    assert len(store.records[namespace]) == 1
    assert len(store.upsert_calls) == 1
    assert len(store.embed_documents_calls) == 1


def test_same_statement_for_different_users_stays_isolated() -> None:
    service, store = make_service()

    first = service.remember(_RAW_USER_ID, _STATEMENT)
    second = service.remember(_OTHER_RAW_USER_ID, _STATEMENT)

    assert first.status == "created"
    assert second.status == "created"
    assert first.memory_id == second.memory_id  # same content hash...
    assert len(store.records) == 2  # ...but two distinct namespaces


# --- namespace discipline ---------------------------------------------------------


def test_upsert_uses_only_the_derived_user_namespace() -> None:
    settings = make_settings()
    service, store = make_service(settings=settings)

    service.remember(_RAW_USER_ID, _STATEMENT)

    assert len(store.upsert_calls) == 1
    namespace = store.upsert_calls[0]["namespace"]
    assert namespace.startswith("user-memory-")
    assert namespace != settings.pinecone_documents_namespace
    assert store.fetch_calls[0]["namespace"] == namespace


def test_recall_uses_only_the_derived_user_namespace() -> None:
    settings = make_settings()
    service, store = make_service(settings=settings)

    service.recall(_RAW_USER_ID, _QUERY)

    assert len(store.query_calls) == 1
    namespace = store.query_calls[0]["namespace"]
    assert namespace.startswith("user-memory-")
    assert namespace != settings.pinecone_documents_namespace


def test_documentation_namespace_is_never_touched() -> None:
    settings = make_settings(pinecone_documents_namespace="documentation")
    service, store = make_service(settings=settings)

    service.remember(_RAW_USER_ID, _STATEMENT)
    service.recall(_RAW_USER_ID, _QUERY)

    all_namespaces = [
        *(call["namespace"] for call in store.fetch_calls),
        *(call["namespace"] for call in store.upsert_calls),
        *(call["namespace"] for call in store.query_calls),
    ]
    assert all_namespaces
    assert "documentation" not in all_namespaces


# --- metadata --------------------------------------------------------------------


def test_metadata_is_flat_and_contains_no_raw_user_identifier() -> None:
    service, store = make_service()

    service.remember(_RAW_USER_ID, _STATEMENT)

    vector = store.upsert_calls[0]["vectors"][0]
    metadata = vector["metadata"]
    assert set(metadata) == {"kind", "text", "content_hash", "schema_version", "created_at"}
    assert metadata["kind"] == "user_memory"
    assert metadata["schema_version"] == 1
    expected_created_at = datetime.fromtimestamp(1_780_000_000.0, tz=UTC).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    assert metadata["created_at"] == expected_created_at
    for value in metadata.values():
        assert isinstance(value, str | int)
        if isinstance(value, str):
            assert _RAW_USER_ID not in value
    assert _RAW_USER_ID not in vector["id"]


# --- recall behavior ---------------------------------------------------------------


def test_recall_query_is_embedded_exactly_once() -> None:
    service, store = make_service()

    service.recall(_RAW_USER_ID, _QUERY)

    assert store.embed_query_calls == [_QUERY]
    assert store.embed_documents_calls == []


def test_recall_applies_configured_top_k_and_kind_filter() -> None:
    settings = make_settings(user_memory_top_k=7)
    service, store = make_service(settings=settings)

    service.recall(_RAW_USER_ID, _QUERY)

    call = store.query_calls[0]
    assert call["top_k"] == 7
    assert call["metadata_filter"] == {"kind": {"$eq": "user_memory"}}


def test_recall_excludes_below_threshold_candidates() -> None:
    settings = make_settings(user_memory_score_threshold=0.5)
    matches = [
        make_memory_match(id="memory-a" * 4, score=0.9, metadata_overrides={"text": "выше"}),
        make_memory_match(id="memory-b" * 4, score=0.49, metadata_overrides={"text": "ниже"}),
    ]
    store = FakePineconeStore(matches=matches)
    service, _store = make_service(settings=settings, store=store)

    result = service.recall(_RAW_USER_ID, _QUERY)

    assert result.found is True
    assert result.raw_candidate_count == 2
    assert [match.text for match in result.matches] == ["выше"]
    assert result.threshold == 0.5


def test_recall_matches_stay_ordered_by_score_descending() -> None:
    matches = [
        make_memory_match(id="memory-low", score=0.5, metadata_overrides={"text": "низкий"}),
        make_memory_match(id="memory-high", score=0.9, metadata_overrides={"text": "высокий"}),
        make_memory_match(id="memory-mid", score=0.7, metadata_overrides={"text": "средний"}),
    ]
    store = FakePineconeStore(matches=matches)
    service, _store = make_service(store=store)

    result = service.recall(_RAW_USER_ID, _QUERY)

    assert [match.score for match in result.matches] == [0.9, 0.7, 0.5]


def test_recall_no_match_result_is_safe_and_typed() -> None:
    settings = make_settings(user_memory_score_threshold=0.95)
    store = FakePineconeStore(matches=[make_memory_match(score=0.2)])
    service, _store = make_service(settings=settings, store=store)

    result = service.recall(_RAW_USER_ID, _QUERY)

    assert isinstance(result, UserMemoryRecallResult)
    assert result.found is False
    assert result.matches == ()
    assert result.raw_candidate_count == 1
    assert result.threshold == 0.95
    assert _RAW_USER_ID not in result.identity_digest


def test_recall_blank_query_is_rejected_before_embedding() -> None:
    service, store = make_service()

    with pytest.raises(InvalidMemoryStatementError):
        service.recall(_RAW_USER_ID, "   ")

    assert store.embed_query_calls == []
    assert store.query_calls == []


# --- failure mapping ----------------------------------------------------------------


def test_embedding_failure_is_mapped_safely_on_remember() -> None:
    store = FakePineconeStore(embed_error=PineconeEmbeddingError("embed boom"))
    service, _store = make_service(store=store)

    with pytest.raises(MemoryEmbeddingError):
        service.remember(_RAW_USER_ID, _STATEMENT)

    assert store.upsert_calls == []


def test_embedding_failure_is_mapped_safely_on_recall() -> None:
    store = FakePineconeStore(embed_error=PineconeEmbeddingError("embed boom"))
    service, _store = make_service(store=store)

    with pytest.raises(MemoryEmbeddingError):
        service.recall(_RAW_USER_ID, _QUERY)


def test_storage_failure_is_mapped_safely() -> None:
    store = FakePineconeStore(upsert_error=PineconeUpsertError("upsert boom"))
    service, _store = make_service(store=store)

    with pytest.raises(MemoryStorageError):
        service.remember(_RAW_USER_ID, _STATEMENT)


def test_dedup_fetch_failure_is_mapped_safely() -> None:
    store = FakePineconeStore(fetch_error=PineconeFetchError("fetch boom"))
    service, _store = make_service(store=store)

    with pytest.raises(MemoryStorageError):
        service.remember(_RAW_USER_ID, _STATEMENT)

    assert store.embed_documents_calls == []


def test_query_failure_is_mapped_safely() -> None:
    store = FakePineconeStore(query_error=PineconeQueryError("query boom"))
    service, _store = make_service(store=store)

    with pytest.raises(MemoryRecallError):
        service.recall(_RAW_USER_ID, _QUERY)


def test_malformed_upsert_response_count_is_mapped_safely() -> None:
    store = FakePineconeStore(upsert_count=0)
    service, _store = make_service(store=store)

    with pytest.raises(MemoryStorageError):
        service.remember(_RAW_USER_ID, _STATEMENT)


def test_malformed_recall_metadata_is_mapped_safely() -> None:
    for metadata in (
        make_memory_metadata(kind="documentation_chunk"),
        make_memory_metadata(text="   "),
        make_memory_metadata(text=123),
        make_memory_metadata(content_hash=""),
        {"unexpected": "shape"},
    ):
        store = FakePineconeStore(
            matches=[PineconeQueryMatch(id="memory-x", score=0.9, metadata=metadata)]
        )
        service, _store = make_service(store=store)

        with pytest.raises(MalformedMemoryRecordError):
            service.recall(_RAW_USER_ID, _QUERY)


# --- observability -----------------------------------------------------------------


def test_logs_contain_no_identifier_text_query_vector_or_secret(
    caplog: pytest.LogCaptureFixture,
) -> None:
    service, store = make_service()

    with caplog.at_level(logging.DEBUG, logger="ai_docs_agent.user_memory"):
        service.remember(_RAW_USER_ID, _STATEMENT)
        service.remember(_RAW_USER_ID, _STATEMENT)
        service.recall(_RAW_USER_ID, _QUERY)
        store._query_error = PineconeQueryError("query boom")
        with pytest.raises(MemoryRecallError):
            service.recall(_RAW_USER_ID, _QUERY)

    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert log_text  # remember/recall logging is actually exercised
    assert _RAW_USER_ID not in log_text
    assert _STATEMENT not in log_text
    assert "httpx" not in log_text
    assert _QUERY not in log_text
    assert "unit-test-user-memory-secret" not in log_text
    assert "0.1, 0.2" not in log_text
    namespace = store.upsert_calls[0]["namespace"]
    full_digest = namespace.removeprefix("user-memory-")
    assert full_digest not in log_text  # only the shortened digest may appear


def test_log_records_include_safe_operational_facts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    service, _store = make_service()

    with caplog.at_level(logging.INFO, logger="ai_docs_agent.user_memory"):
        service.remember(_RAW_USER_ID, _STATEMENT)
        service.recall(_RAW_USER_ID, _QUERY)

    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "status=created" in log_text
    assert f"statement_length={len(_STATEMENT)}" in log_text
    assert "raw_count=0" in log_text
    assert "accepted_count=0" in log_text


# --- import safety ------------------------------------------------------------------


def test_importing_the_module_performs_no_network_request() -> None:
    # Re-execute the module source under a throwaway name (instead of reloading
    # the canonical module, which would break exception-class identity for other
    # tests). Import-time network access would require a settings/client object,
    # none of which may exist at module scope.
    import importlib.util
    from pathlib import Path

    module_path = (
        Path(__file__).resolve().parents[1] / "src" / "ai_docs_agent" / "user_memory.py"
    )
    spec = importlib.util.spec_from_file_location("user_memory_import_check", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert not hasattr(module, "service")
    assert not hasattr(module, "store")
    assert not hasattr(module, "settings")
