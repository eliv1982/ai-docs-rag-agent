"""Unit tests for PineconeStore. Uses fakes only; no real network access."""

from types import SimpleNamespace
from typing import Any

import pytest

from ai_docs_agent.config import AppSettings
from ai_docs_agent.models import PineconeQueryMatch, PineconeSmokeTestResult
from ai_docs_agent.pinecone_store import (
    PineconeDeleteError,
    PineconeEmbeddingError,
    PineconeFetchError,
    PineconeIndexConfigurationError,
    PineconeIndexNotFoundError,
    PineconeQueryError,
    PineconeSmokeTestError,
    PineconeStore,
    PineconeStoreError,
    PineconeUpsertError,
)

_AUTO = object()

_REQUIRED: dict[str, Any] = {
    "openai_api_key": "sk-test-openai",
    "pinecone_api_key": "pc-test-key",
    "openai_chat_model": "gpt-4o-mini",
}


def make_settings(**overrides: Any) -> AppSettings:
    return AppSettings(_env_file=None, **{**_REQUIRED, **overrides})


def make_description(
    *,
    dimension: int = 1536,
    metric: str = "cosine",
    ready: bool = True,
    host: str | None = "fake-host",
) -> SimpleNamespace:
    return SimpleNamespace(
        dimension=dimension,
        metric=metric,
        host=host,
        status=SimpleNamespace(ready=ready),
    )


class FakeClock:
    """Manually advanced fake clock/sleep pair so tests run instantly."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class FakeEmbeddings:
    def __init__(self, vector: list[float]) -> None:
        self._vector = vector
        self.calls: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.calls.append(text)
        return list(self._vector)


class RaisingEmbeddings:
    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("embedding backend unavailable")


class FakeIndexHandle:
    """Fake Pinecone Index handle that becomes query-visible after N queries."""

    def __init__(
        self,
        *,
        visible_after: int = 0,
        upsert_response: Any = _AUTO,
        fetch_response: Any = _AUTO,
        query_response: Any = _AUTO,
    ) -> None:
        self.upserted: list[dict[str, Any]] = []
        self.deleted: list[dict[str, Any]] = []
        self.fetch_calls: list[dict[str, Any]] = []
        self.similar_query_calls: list[dict[str, Any]] = []
        self._visible_after = visible_after
        self.query_calls = 0
        self.existing_ids: set[str] = set()
        self._upsert_response = upsert_response
        self._fetch_response = fetch_response
        self._query_response = query_response

    def upsert(self, *, vectors: list[dict[str, Any]], namespace: str) -> Any:
        self.upserted.append({"vectors": vectors, "namespace": namespace})
        for vector in vectors:
            self.existing_ids.add(vector["id"])
        if self._upsert_response is _AUTO:
            return SimpleNamespace(upserted_count=len(vectors))
        return self._upsert_response

    def query(
        self,
        *,
        vector: list[float],
        top_k: int,
        include_metadata: bool,
        namespace: str,
        include_values: bool = False,
        filter: dict[str, Any] | None = None,  # noqa: A002 - matches Pinecone SDK kwarg
    ):
        self.query_calls += 1
        self.similar_query_calls.append(
            {
                "vector": vector,
                "top_k": top_k,
                "include_metadata": include_metadata,
                "include_values": include_values,
                "namespace": namespace,
                "filter": filter,
            }
        )
        if self._query_response is not _AUTO:
            return self._query_response
        if not self.upserted or self.query_calls <= self._visible_after:
            return SimpleNamespace(matches=[])
        record = self.upserted[-1]["vectors"][0]
        return SimpleNamespace(matches=[SimpleNamespace(id=record["id"], score=0.987)])

    def fetch(self, *, ids: list[str], namespace: str) -> Any:
        self.fetch_calls.append({"ids": list(ids), "namespace": namespace})
        if self._fetch_response is not _AUTO:
            return self._fetch_response
        found = {
            vec_id: SimpleNamespace(id=vec_id) for vec_id in ids if vec_id in self.existing_ids
        }
        return SimpleNamespace(vectors=found)

    def delete(
        self,
        *,
        ids: list[str] | None = None,
        filter: dict[str, Any] | None = None,  # noqa: A002 - matches Pinecone SDK kwarg
        namespace: str,
    ) -> None:
        entry: dict[str, Any] = {"namespace": namespace}
        if ids is not None:
            entry["ids"] = ids
            for vec_id in ids:
                self.existing_ids.discard(vec_id)
        if filter is not None:
            entry["filter"] = filter
        self.deleted.append(entry)


class FailingDeleteIndexHandle(FakeIndexHandle):
    def delete(
        self,
        *,
        ids: list[str] | None = None,
        filter: dict[str, Any] | None = None,  # noqa: A002 - matches Pinecone SDK kwarg
        namespace: str,
    ) -> None:
        raise RuntimeError("delete backend unavailable")


class RaisingQueryIndexHandle(FakeIndexHandle):
    def query(self, **kwargs: Any):
        raise RuntimeError("query backend unavailable")


class RaisingUpsertIndexHandle(FakeIndexHandle):
    def upsert(self, *, vectors: list[dict[str, Any]], namespace: str) -> Any:
        raise RuntimeError("upsert backend unavailable")


class RaisingFetchIndexHandle(FakeIndexHandle):
    def fetch(self, *, ids: list[str], namespace: str) -> Any:
        raise RuntimeError("fetch backend unavailable")


class FlakyQueryIndexHandle(FakeIndexHandle):
    """Raises on its Nth query call (1-indexed); succeeds on every other call."""

    def __init__(self, *, fail_on_call: int) -> None:
        super().__init__()
        self._fail_on_call = fail_on_call
        self._flaky_calls = 0

    def query(self, **kwargs: Any) -> Any:
        self._flaky_calls += 1
        if self._flaky_calls == self._fail_on_call:
            raise RuntimeError("transient query failure")
        return super().query(**kwargs)


class FakeDocumentEmbeddings:
    """Fake OpenAIEmbeddings-like client for embed_documents batches."""

    def __init__(self, embeddings_by_text: dict[str, list[float]] | None = None) -> None:
        self._embeddings_by_text = embeddings_by_text
        self.embed_documents_calls: list[list[str]] = []

    def embed_query(self, text: str) -> list[float]:
        raise AssertionError("embed_query must not be used for document chunks")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embed_documents_calls.append(list(texts))
        if self._embeddings_by_text is not None:
            return [self._embeddings_by_text[text] for text in texts]
        return [[float(len(text))] * 1536 for text in texts]


class ScriptedDocumentEmbeddings:
    """Returns a fixed, preconfigured embed_documents result regardless of input."""

    def __init__(self, embeddings: list[list[Any]]) -> None:
        self._embeddings = embeddings
        self.embed_documents_calls: list[list[str]] = []

    def embed_query(self, text: str) -> list[float]:
        raise AssertionError("embed_query must not be used for document chunks")

    def embed_documents(self, texts: list[str]) -> list[list[Any]]:
        self.embed_documents_calls.append(list(texts))
        return self._embeddings


class RaisingDocumentEmbeddings:
    def embed_query(self, text: str) -> list[float]:
        raise AssertionError("embed_query must not be used for document chunks")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding backend unavailable")


class ScriptedQueryEmbeddings:
    """Returns a fixed, preconfigured embed_query result regardless of input text."""

    def __init__(self, vector: list[Any]) -> None:
        self._vector = vector
        self.calls: list[str] = []

    def embed_query(self, text: str) -> list[Any]:
        self.calls.append(text)
        return list(self._vector)


class FakePineconeClient:
    def __init__(
        self,
        *,
        existing: bool,
        descriptions: list[Any] | None = None,
        visible_after: int = 0,
    ) -> None:
        self.existing = existing
        self._descriptions = descriptions or []
        self._describe_calls = 0
        self.created: dict[str, Any] | None = None
        self.index_handle: FakeIndexHandle = FakeIndexHandle(visible_after=visible_after)
        self.has_index_call_count = 0
        self.describe_index_call_count = 0
        self.index_call_count = 0

    def has_index(self, name: str) -> bool:
        self.has_index_call_count += 1
        return self.existing

    def describe_index(self, name: str) -> Any:
        if not self._descriptions:
            raise AssertionError("describe_index called with no descriptions configured")
        index = min(self._describe_calls, len(self._descriptions) - 1)
        self._describe_calls += 1
        self.describe_index_call_count += 1
        return self._descriptions[index]

    def create_index(self, *, name: str, dimension: int, metric: str, spec: Any) -> None:
        self.created = {
            "name": name,
            "dimension": dimension,
            "metric": metric,
            "cloud": spec.cloud,
            "region": spec.region,
        }
        self.existing = True

    def Index(self, name: str) -> FakeIndexHandle:  # noqa: N802 - matches Pinecone SDK
        self.index_call_count += 1
        return self.index_handle


# --- ensure_index ---------------------------------------------------------


def test_ensure_index_missing_and_create_disabled_raises() -> None:
    client = FakePineconeClient(existing=False)
    settings = make_settings(pinecone_create_if_missing=False)
    store = PineconeStore(settings, client=client)

    with pytest.raises(PineconeIndexNotFoundError):
        store.ensure_index()


def test_ensure_index_creates_when_missing_and_polls_until_ready() -> None:
    clock = FakeClock()
    descriptions = [
        make_description(ready=False),
        make_description(ready=False),
        make_description(ready=True),
    ]
    client = FakePineconeClient(existing=False, descriptions=descriptions)
    settings = make_settings(
        pinecone_create_if_missing=True,
        pinecone_smoke_timeout_seconds=10,
        pinecone_smoke_poll_interval_seconds=1,
    )
    store = PineconeStore(settings, client=client, clock=clock, sleep=clock.sleep)

    status = store.ensure_index()

    assert client.created == {
        "name": settings.pinecone_index_name,
        "dimension": settings.pinecone_dimension,
        "metric": settings.pinecone_metric,
        "cloud": settings.pinecone_cloud,
        "region": settings.pinecone_region,
    }
    assert status.ready is True
    assert status.name == settings.pinecone_index_name


def test_ensure_index_existing_is_not_recreated() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    status = store.ensure_index()

    assert client.created is None
    assert status.dimension == settings.pinecone_dimension


def test_ensure_index_dimension_mismatch_raises() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description(dimension=768)])
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    with pytest.raises(PineconeIndexConfigurationError):
        store.ensure_index()


def test_ensure_index_metric_mismatch_raises() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description(metric="euclidean")])
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    with pytest.raises(PineconeIndexConfigurationError):
        store.ensure_index()


def test_ensure_index_create_times_out_when_never_ready() -> None:
    clock = FakeClock()
    client = FakePineconeClient(existing=False, descriptions=[make_description(ready=False)])
    settings = make_settings(
        pinecone_create_if_missing=True,
        pinecone_smoke_timeout_seconds=3,
        pinecone_smoke_poll_interval_seconds=1,
    )
    store = PineconeStore(settings, client=client, clock=clock, sleep=clock.sleep)

    with pytest.raises(PineconeStoreError):
        store.ensure_index()


def test_ensure_index_existing_waits_for_ready_and_host() -> None:
    clock = FakeClock()
    descriptions = [
        make_description(ready=False, host=None),
        make_description(ready=False, host=None),
        make_description(ready=True, host="real-host"),
    ]
    client = FakePineconeClient(existing=True, descriptions=descriptions)
    settings = make_settings(
        pinecone_smoke_timeout_seconds=10, pinecone_smoke_poll_interval_seconds=1
    )
    store = PineconeStore(settings, client=client, clock=clock, sleep=clock.sleep)

    status = store.ensure_index()

    assert status.ready is True
    assert status.host == "real-host"
    assert client.created is None


def test_ensure_index_existing_times_out_when_never_ready() -> None:
    clock = FakeClock()
    client = FakePineconeClient(
        existing=True, descriptions=[make_description(ready=False, host=None)]
    )
    settings = make_settings(
        pinecone_smoke_timeout_seconds=3, pinecone_smoke_poll_interval_seconds=1
    )
    store = PineconeStore(settings, client=client, clock=clock, sleep=clock.sleep)

    with pytest.raises(PineconeStoreError):
        store.ensure_index()

    assert client.created is None


# --- smoke_test ------------------------------------------------------------


def test_smoke_test_success() -> None:
    clock = FakeClock()
    embeddings = FakeEmbeddings([0.1] * 1536)
    client = FakePineconeClient(
        existing=True, descriptions=[make_description(dimension=1536, metric="cosine")]
    )
    settings = make_settings()
    store = PineconeStore(
        settings, client=client, embeddings=embeddings, clock=clock, sleep=clock.sleep
    )

    result = store.smoke_test()

    assert isinstance(result, PineconeSmokeTestResult)
    assert result.matched_id == result.record_id
    assert result.cleanup_succeeded is True
    assert result.dimension == 1536
    assert result.index_name == settings.pinecone_index_name
    assert result.namespace == settings.pinecone_smoke_namespace
    assert result.embedding_model == settings.openai_embedding_model
    assert embeddings.calls == ["AI Docs RAG Agent Pinecone integration smoke test."]
    assert client.index_handle.deleted == [
        {"ids": [result.record_id], "namespace": settings.pinecone_smoke_namespace}
    ]


def test_smoke_test_polls_until_match_is_visible() -> None:
    clock = FakeClock()
    embeddings = FakeEmbeddings([0.2] * 1536)
    client = FakePineconeClient(
        existing=True,
        descriptions=[make_description(dimension=1536, metric="cosine")],
        visible_after=2,
    )
    settings = make_settings(
        pinecone_smoke_timeout_seconds=10, pinecone_smoke_poll_interval_seconds=1
    )
    store = PineconeStore(
        settings, client=client, embeddings=embeddings, clock=clock, sleep=clock.sleep
    )

    result = store.smoke_test()

    assert result.matched_id == result.record_id
    assert client.index_handle.query_calls >= 3


def test_smoke_test_times_out_when_match_never_appears() -> None:
    clock = FakeClock()
    embeddings = FakeEmbeddings([0.3] * 1536)
    client = FakePineconeClient(
        existing=True,
        descriptions=[make_description(dimension=1536, metric="cosine")],
        visible_after=999,
    )
    settings = make_settings(
        pinecone_smoke_timeout_seconds=3, pinecone_smoke_poll_interval_seconds=1
    )
    store = PineconeStore(
        settings, client=client, embeddings=embeddings, clock=clock, sleep=clock.sleep
    )

    with pytest.raises(PineconeSmokeTestError):
        store.smoke_test()

    assert client.index_handle.deleted, "cleanup must be attempted after a timeout failure"


def test_smoke_test_reports_cleanup_failure_without_hiding_it() -> None:
    clock = FakeClock()
    embeddings = FakeEmbeddings([0.4] * 1536)
    client = FakePineconeClient(
        existing=True, descriptions=[make_description(dimension=1536, metric="cosine")]
    )
    client.index_handle = FailingDeleteIndexHandle()
    settings = make_settings()
    store = PineconeStore(
        settings, client=client, embeddings=embeddings, clock=clock, sleep=clock.sleep
    )

    result = store.smoke_test()

    assert result.cleanup_succeeded is False


def test_smoke_test_attempts_cleanup_when_query_raises() -> None:
    clock = FakeClock()
    embeddings = FakeEmbeddings([0.5] * 1536)
    client = FakePineconeClient(
        existing=True, descriptions=[make_description(dimension=1536, metric="cosine")]
    )
    client.index_handle = RaisingQueryIndexHandle()
    settings = make_settings()
    store = PineconeStore(
        settings, client=client, embeddings=embeddings, clock=clock, sleep=clock.sleep
    )

    with pytest.raises(PineconeSmokeTestError) as exc_info:
        store.smoke_test()

    assert exc_info.value.__cause__ is not None
    assert client.index_handle.deleted, "cleanup must be attempted even after a hard failure"


def test_smoke_test_wraps_embedding_failure_and_skips_cleanup_before_upsert() -> None:
    clock = FakeClock()
    client = FakePineconeClient(
        existing=True, descriptions=[make_description(dimension=1536, metric="cosine")]
    )
    settings = make_settings(
        openai_api_key="sk-super-secret", pinecone_api_key="pc-super-secret"
    )
    store = PineconeStore(
        settings, client=client, embeddings=RaisingEmbeddings(), clock=clock, sleep=clock.sleep
    )

    with pytest.raises(PineconeSmokeTestError) as exc_info:
        store.smoke_test()

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert client.index_handle.deleted == [], "cleanup must not run before upsert is attempted"
    message = str(exc_info.value)
    assert "sk-super-secret" not in message
    assert "pc-super-secret" not in message


def test_smoke_test_rejects_embedding_with_wrong_dimension() -> None:
    clock = FakeClock()
    embeddings = FakeEmbeddings([0.1] * 10)
    client = FakePineconeClient(
        existing=True, descriptions=[make_description(dimension=1536, metric="cosine")]
    )
    settings = make_settings()
    store = PineconeStore(
        settings, client=client, embeddings=embeddings, clock=clock, sleep=clock.sleep
    )

    with pytest.raises(PineconeSmokeTestError):
        store.smoke_test()


def test_index_not_found_error_does_not_leak_secrets() -> None:
    client = FakePineconeClient(existing=False)
    settings = make_settings(
        openai_api_key="sk-super-secret", pinecone_api_key="pc-super-secret"
    )
    store = PineconeStore(settings, client=client)

    with pytest.raises(PineconeIndexNotFoundError) as exc_info:
        store.ensure_index()

    message = str(exc_info.value)
    assert "sk-super-secret" not in message
    assert "pc-super-secret" not in message


# --- embed_documents --------------------------------------------------------


def test_embed_documents_single_batch_preserves_order_and_values() -> None:
    embeddings_by_text = {
        "alpha": [0.1] * 1536,
        "beta": [0.2] * 1536,
    }
    embeddings = FakeDocumentEmbeddings(embeddings_by_text)
    settings = make_settings()
    store = PineconeStore(settings, embeddings=embeddings)

    result = store.embed_documents(["alpha", "beta"])

    assert result == [embeddings_by_text["alpha"], embeddings_by_text["beta"]]
    assert embeddings.embed_documents_calls == [["alpha", "beta"]]


def test_embed_documents_empty_input_returns_empty_and_skips_call() -> None:
    embeddings = FakeDocumentEmbeddings()
    settings = make_settings()
    store = PineconeStore(settings, embeddings=embeddings)

    result = store.embed_documents([])

    assert result == []
    assert embeddings.embed_documents_calls == []


def test_embed_documents_rejects_count_mismatch() -> None:
    embeddings = ScriptedDocumentEmbeddings([[0.1] * 1536])
    settings = make_settings()
    store = PineconeStore(settings, embeddings=embeddings)

    with pytest.raises(PineconeEmbeddingError):
        store.embed_documents(["alpha", "beta"])


def test_embed_documents_rejects_dimension_mismatch() -> None:
    embeddings = ScriptedDocumentEmbeddings([[0.1] * 10])
    settings = make_settings(pinecone_dimension=1536)
    store = PineconeStore(settings, embeddings=embeddings)

    with pytest.raises(PineconeEmbeddingError):
        store.embed_documents(["alpha"])


def test_embed_documents_rejects_non_numeric_value() -> None:
    embeddings = ScriptedDocumentEmbeddings([[0.1] * 1535 + ["not-a-number"]])
    settings = make_settings(pinecone_dimension=1536)
    store = PineconeStore(settings, embeddings=embeddings)

    with pytest.raises(PineconeEmbeddingError):
        store.embed_documents(["alpha"])


def test_embed_documents_rejects_nan_value() -> None:
    embeddings = ScriptedDocumentEmbeddings([[0.1] * 1535 + [float("nan")]])
    settings = make_settings(pinecone_dimension=1536)
    store = PineconeStore(settings, embeddings=embeddings)

    with pytest.raises(PineconeEmbeddingError):
        store.embed_documents(["alpha"])


def test_embed_documents_rejects_infinite_value() -> None:
    embeddings = ScriptedDocumentEmbeddings([[0.1] * 1535 + [float("inf")]])
    settings = make_settings(pinecone_dimension=1536)
    store = PineconeStore(settings, embeddings=embeddings)

    with pytest.raises(PineconeEmbeddingError):
        store.embed_documents(["alpha"])


def test_embed_documents_rejects_boolean_true_value() -> None:
    embeddings = ScriptedDocumentEmbeddings([[0.1] * 1535 + [True]])
    settings = make_settings(pinecone_dimension=1536)
    store = PineconeStore(settings, embeddings=embeddings)

    with pytest.raises(PineconeEmbeddingError):
        store.embed_documents(["alpha"])


def test_embed_documents_rejects_boolean_false_value() -> None:
    embeddings = ScriptedDocumentEmbeddings([[0.1] * 1535 + [False]])
    settings = make_settings(pinecone_dimension=1536)
    store = PineconeStore(settings, embeddings=embeddings)

    with pytest.raises(PineconeEmbeddingError):
        store.embed_documents(["alpha"])


def test_embed_documents_accepts_plain_int_and_float_values() -> None:
    embeddings = ScriptedDocumentEmbeddings([[0, 1, 2]])
    settings = make_settings(pinecone_dimension=3)
    store = PineconeStore(settings, embeddings=embeddings)

    result = store.embed_documents(["alpha"])

    assert result == [[0.0, 1.0, 2.0]]


def test_embed_documents_wraps_openai_failure_with_cause() -> None:
    settings = make_settings()
    store = PineconeStore(settings, embeddings=RaisingDocumentEmbeddings())

    with pytest.raises(PineconeEmbeddingError) as exc_info:
        store.embed_documents(["alpha"])

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_embed_documents_error_does_not_leak_secrets() -> None:
    settings = make_settings(
        openai_api_key="sk-super-secret", pinecone_api_key="pc-super-secret"
    )
    store = PineconeStore(settings, embeddings=RaisingDocumentEmbeddings())

    with pytest.raises(PineconeEmbeddingError) as exc_info:
        store.embed_documents(["alpha"])

    message = str(exc_info.value)
    assert "sk-super-secret" not in message
    assert "pc-super-secret" not in message


# --- upsert_vectors ----------------------------------------------------------


def _make_record(record_id: str = "doc-1-chunk-0000") -> dict[str, Any]:
    return {
        "id": record_id,
        "values": [0.1, 0.2, 0.3],
        "metadata": {"kind": "documentation_chunk", "text": "hello"},
    }


def test_upsert_vectors_sends_exact_ids_metadata_values_and_namespace() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    settings = make_settings()
    store = PineconeStore(settings, client=client)
    record = _make_record()

    upserted = store.upsert_vectors([record], namespace="documentation")

    assert upserted == 1
    assert client.index_handle.upserted == [
        {"vectors": [record], "namespace": "documentation"}
    ]


def test_upsert_vectors_calls_ensure_index_first() -> None:
    client = FakePineconeClient(existing=False)
    settings = make_settings(pinecone_create_if_missing=False)
    store = PineconeStore(settings, client=client)

    with pytest.raises(PineconeIndexNotFoundError):
        store.upsert_vectors([_make_record()], namespace="documentation")

    assert client.index_handle.upserted == []


def test_upsert_vectors_empty_input_returns_zero_and_skips_network() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    upserted = store.upsert_vectors([], namespace="documentation")

    assert upserted == 0
    assert client.index_handle.upserted == []


def test_upsert_vectors_reads_count_from_object_response() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    client.index_handle = FakeIndexHandle(upsert_response=SimpleNamespace(upserted_count=1))
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    upserted = store.upsert_vectors(
        [_make_record(), _make_record("doc-1-chunk-0001")], namespace="documentation"
    )

    assert upserted == 1


def test_upsert_vectors_reads_count_from_mapping_response() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    client.index_handle = FakeIndexHandle(upsert_response={"upserted_count": 2})
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    upserted = store.upsert_vectors([_make_record()], namespace="documentation")

    assert upserted == 2


def test_upsert_vectors_returns_count_even_when_higher_than_batch_size() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    client.index_handle = FakeIndexHandle(upsert_response={"upserted_count": 5})
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    upserted = store.upsert_vectors([_make_record()], namespace="documentation")

    assert upserted == 5


def test_upsert_vectors_returns_count_even_when_lower_than_batch_size() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    client.index_handle = FakeIndexHandle(upsert_response={"upserted_count": 0})
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    upserted = store.upsert_vectors(
        [_make_record(), _make_record("doc-1-chunk-0001")], namespace="documentation"
    )

    assert upserted == 0


def test_upsert_vectors_rejects_missing_upserted_count_field() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    client.index_handle = FakeIndexHandle(upsert_response=SimpleNamespace())
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    with pytest.raises(PineconeUpsertError):
        store.upsert_vectors([_make_record()], namespace="documentation")


def test_upsert_vectors_rejects_none_upserted_count() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    client.index_handle = FakeIndexHandle(upsert_response={"upserted_count": None})
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    with pytest.raises(PineconeUpsertError):
        store.upsert_vectors([_make_record()], namespace="documentation")


def test_upsert_vectors_rejects_boolean_upserted_count() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    client.index_handle = FakeIndexHandle(upsert_response={"upserted_count": True})
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    with pytest.raises(PineconeUpsertError):
        store.upsert_vectors([_make_record()], namespace="documentation")


def test_upsert_vectors_rejects_string_upserted_count() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    client.index_handle = FakeIndexHandle(upsert_response={"upserted_count": "1"})
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    with pytest.raises(PineconeUpsertError):
        store.upsert_vectors([_make_record()], namespace="documentation")


def test_upsert_vectors_rejects_float_upserted_count() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    client.index_handle = FakeIndexHandle(upsert_response={"upserted_count": 1.0})
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    with pytest.raises(PineconeUpsertError):
        store.upsert_vectors([_make_record()], namespace="documentation")


def test_upsert_vectors_rejects_negative_upserted_count() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    client.index_handle = FakeIndexHandle(upsert_response={"upserted_count": -1})
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    with pytest.raises(PineconeUpsertError):
        store.upsert_vectors([_make_record()], namespace="documentation")


def test_upsert_vectors_strict_parsing_error_does_not_leak_secrets() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    client.index_handle = FakeIndexHandle(upsert_response=SimpleNamespace())
    settings = make_settings(
        openai_api_key="sk-super-secret", pinecone_api_key="pc-super-secret"
    )
    store = PineconeStore(settings, client=client)

    with pytest.raises(PineconeUpsertError) as exc_info:
        store.upsert_vectors([_make_record()], namespace="documentation")

    message = str(exc_info.value)
    assert "sk-super-secret" not in message
    assert "pc-super-secret" not in message


def test_upsert_vectors_wraps_sdk_error_with_cause() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    client.index_handle = RaisingUpsertIndexHandle()
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    with pytest.raises(PineconeUpsertError) as exc_info:
        store.upsert_vectors([_make_record()], namespace="documentation")

    assert isinstance(exc_info.value.__cause__, RuntimeError)


# --- fetch_existing_ids -------------------------------------------------------


def test_fetch_existing_ids_returns_exact_found_ids() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    settings = make_settings()
    store = PineconeStore(settings, client=client)
    store.upsert_vectors([_make_record("a"), _make_record("b")], namespace="documentation")

    found = store.fetch_existing_ids(["a", "b", "c"], namespace="documentation")

    assert found == {"a", "b"}
    assert client.index_handle.fetch_calls == [
        {"ids": ["a", "b", "c"], "namespace": "documentation"}
    ]


def test_fetch_existing_ids_empty_input_returns_empty_and_skips_network() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    found = store.fetch_existing_ids([], namespace="documentation")

    assert found == set()
    assert client.index_handle.fetch_calls == []


def test_fetch_existing_ids_calls_ensure_index_first() -> None:
    client = FakePineconeClient(existing=False)
    settings = make_settings(pinecone_create_if_missing=False)
    store = PineconeStore(settings, client=client)

    with pytest.raises(PineconeIndexNotFoundError):
        store.fetch_existing_ids(["a"], namespace="documentation")


def test_fetch_existing_ids_wraps_sdk_error_with_cause() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    client.index_handle = RaisingFetchIndexHandle()
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    with pytest.raises(PineconeFetchError) as exc_info:
        store.fetch_existing_ids(["a"], namespace="documentation")

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_fetch_existing_ids_supports_object_vectors_mapping() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    client.index_handle = FakeIndexHandle(
        fetch_response=SimpleNamespace(vectors={"a": SimpleNamespace(id="a")})
    )
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    found = store.fetch_existing_ids(["a", "b"], namespace="documentation")

    assert found == {"a"}


def test_fetch_existing_ids_supports_mapping_response() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    client.index_handle = FakeIndexHandle(fetch_response={"vectors": {"a": {"id": "a"}}})
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    found = store.fetch_existing_ids(["a"], namespace="documentation")

    assert found == {"a"}


def test_fetch_existing_ids_accepts_empty_vectors_mapping() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    client.index_handle = FakeIndexHandle(fetch_response=SimpleNamespace(vectors={}))
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    found = store.fetch_existing_ids(["a"], namespace="documentation")

    assert found == set()


def test_fetch_existing_ids_rejects_missing_vectors_field() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    client.index_handle = FakeIndexHandle(fetch_response=SimpleNamespace())
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    with pytest.raises(PineconeFetchError):
        store.fetch_existing_ids(["a"], namespace="documentation")


def test_fetch_existing_ids_rejects_none_vectors_field() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    client.index_handle = FakeIndexHandle(fetch_response=SimpleNamespace(vectors=None))
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    with pytest.raises(PineconeFetchError):
        store.fetch_existing_ids(["a"], namespace="documentation")


def test_fetch_existing_ids_rejects_list_vectors_field() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    client.index_handle = FakeIndexHandle(
        fetch_response=SimpleNamespace(vectors=[SimpleNamespace(id="a")])
    )
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    with pytest.raises(PineconeFetchError):
        store.fetch_existing_ids(["a"], namespace="documentation")


def test_fetch_existing_ids_rejects_non_string_key() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    client.index_handle = FakeIndexHandle(
        fetch_response=SimpleNamespace(vectors={1: SimpleNamespace(id=1)})
    )
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    with pytest.raises(PineconeFetchError):
        store.fetch_existing_ids(["a"], namespace="documentation")


def test_fetch_existing_ids_strict_parsing_error_does_not_leak_secrets() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    client.index_handle = FakeIndexHandle(fetch_response=SimpleNamespace())
    settings = make_settings(
        openai_api_key="sk-super-secret", pinecone_api_key="pc-super-secret"
    )
    store = PineconeStore(settings, client=client)

    with pytest.raises(PineconeFetchError) as exc_info:
        store.fetch_existing_ids(["a"], namespace="documentation")

    message = str(exc_info.value)
    assert "sk-super-secret" not in message
    assert "pc-super-secret" not in message


# --- delete_vectors_by_filter -------------------------------------------------


def test_delete_vectors_by_filter_uses_exact_filter_and_namespace() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    settings = make_settings()
    store = PineconeStore(settings, client=client)
    metadata_filter = {
        "$and": [
            {"source_url": {"$eq": "https://docs.example.com/page"}},
            {"content_hash": {"$ne": "hash-value"}},
        ]
    }

    store.delete_vectors_by_filter(metadata_filter, namespace="documentation")

    assert client.index_handle.deleted == [
        {"filter": metadata_filter, "namespace": "documentation"}
    ]


def test_delete_vectors_by_filter_does_not_pass_ids() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    store.delete_vectors_by_filter({"source_url": {"$eq": "x"}}, namespace="documentation")

    assert "ids" not in client.index_handle.deleted[0]


def test_delete_vectors_by_filter_calls_ensure_index_first() -> None:
    client = FakePineconeClient(existing=False)
    settings = make_settings(pinecone_create_if_missing=False)
    store = PineconeStore(settings, client=client)

    with pytest.raises(PineconeIndexNotFoundError):
        store.delete_vectors_by_filter({"source_url": {"$eq": "x"}}, namespace="documentation")


def test_delete_vectors_by_filter_wraps_sdk_error_with_cause() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    client.index_handle = FailingDeleteIndexHandle()
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    with pytest.raises(PineconeDeleteError) as exc_info:
        store.delete_vectors_by_filter({"source_url": {"$eq": "x"}}, namespace="documentation")

    assert isinstance(exc_info.value.__cause__, RuntimeError)


# --- cached ready index handle -------------------------------------------------


def test_data_plane_calls_share_one_cached_ready_handle() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    store.upsert_vectors([_make_record("a")], namespace="documentation")
    store.upsert_vectors([_make_record("b")], namespace="documentation")
    store.fetch_existing_ids(["a"], namespace="documentation")
    store.delete_vectors_by_filter({"source_url": {"$eq": "x"}}, namespace="documentation")

    assert client.has_index_call_count == 1
    assert client.describe_index_call_count == 1
    assert client.index_call_count == 1


def test_embed_documents_never_touches_pinecone_client() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    embeddings = FakeDocumentEmbeddings()
    settings = make_settings()
    store = PineconeStore(settings, client=client, embeddings=embeddings)

    store.embed_documents(["alpha", "beta"])

    assert client.has_index_call_count == 0
    assert client.describe_index_call_count == 0
    assert client.index_call_count == 0


class FlakyUpsertIndexHandle(FakeIndexHandle):
    """Raises on its Nth upsert call (1-indexed); succeeds on every other call."""

    def __init__(self, *, fail_on_call: int) -> None:
        super().__init__()
        self._fail_on_call = fail_on_call
        self._calls = 0

    def upsert(self, *, vectors: list[dict[str, Any]], namespace: str) -> Any:
        self._calls += 1
        if self._calls == self._fail_on_call:
            raise RuntimeError("transient upsert failure")
        return super().upsert(vectors=vectors, namespace=namespace)


def test_data_plane_failure_invalidates_cache_and_next_call_recovers() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    client.index_handle = FlakyUpsertIndexHandle(fail_on_call=2)
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    store.upsert_vectors([_make_record("a")], namespace="documentation")
    assert client.index_call_count == 1

    with pytest.raises(PineconeUpsertError):
        store.upsert_vectors([_make_record("b")], namespace="documentation")

    store.upsert_vectors([_make_record("c")], namespace="documentation")

    assert client.index_call_count == 2


def test_separate_store_instances_do_not_share_cached_handle() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    settings = make_settings()
    store_one = PineconeStore(settings, client=client)
    store_two = PineconeStore(settings, client=client)

    store_one.upsert_vectors([_make_record("a")], namespace="documentation")
    store_two.upsert_vectors([_make_record("b")], namespace="documentation")

    assert client.index_call_count == 2


# --- embed_query ---------------------------------------------------------------


def test_embed_query_success_returns_validated_vector() -> None:
    vector = [0.1] * 1536
    embeddings = FakeEmbeddings(vector)
    settings = make_settings()
    store = PineconeStore(settings, embeddings=embeddings)

    result = store.embed_query("how do I configure the client?")

    assert result == vector
    assert embeddings.calls == ["how do I configure the client?"]


def test_embed_query_rejects_whitespace_only_text_before_client_call() -> None:
    embeddings = ScriptedQueryEmbeddings([0.1] * 1536)
    settings = make_settings()
    store = PineconeStore(settings, embeddings=embeddings)

    with pytest.raises(PineconeEmbeddingError):
        store.embed_query("   ")

    assert embeddings.calls == []


def test_embed_query_rejects_dimension_mismatch() -> None:
    embeddings = ScriptedQueryEmbeddings([0.1] * 10)
    settings = make_settings(pinecone_dimension=1536)
    store = PineconeStore(settings, embeddings=embeddings)

    with pytest.raises(PineconeEmbeddingError):
        store.embed_query("query text")


def test_embed_query_rejects_non_numeric_value() -> None:
    embeddings = ScriptedQueryEmbeddings([0.1] * 1535 + ["not-a-number"])
    settings = make_settings(pinecone_dimension=1536)
    store = PineconeStore(settings, embeddings=embeddings)

    with pytest.raises(PineconeEmbeddingError):
        store.embed_query("query text")


def test_embed_query_rejects_nan_value() -> None:
    embeddings = ScriptedQueryEmbeddings([0.1] * 1535 + [float("nan")])
    settings = make_settings(pinecone_dimension=1536)
    store = PineconeStore(settings, embeddings=embeddings)

    with pytest.raises(PineconeEmbeddingError):
        store.embed_query("query text")


def test_embed_query_rejects_positive_infinite_value() -> None:
    embeddings = ScriptedQueryEmbeddings([0.1] * 1535 + [float("inf")])
    settings = make_settings(pinecone_dimension=1536)
    store = PineconeStore(settings, embeddings=embeddings)

    with pytest.raises(PineconeEmbeddingError):
        store.embed_query("query text")


def test_embed_query_rejects_negative_infinite_value() -> None:
    embeddings = ScriptedQueryEmbeddings([0.1] * 1535 + [float("-inf")])
    settings = make_settings(pinecone_dimension=1536)
    store = PineconeStore(settings, embeddings=embeddings)

    with pytest.raises(PineconeEmbeddingError):
        store.embed_query("query text")


def test_embed_query_rejects_boolean_value() -> None:
    embeddings = ScriptedQueryEmbeddings([0.1] * 1535 + [True])
    settings = make_settings(pinecone_dimension=1536)
    store = PineconeStore(settings, embeddings=embeddings)

    with pytest.raises(PineconeEmbeddingError):
        store.embed_query("query text")


def test_embed_query_wraps_client_failure_with_cause() -> None:
    settings = make_settings()
    store = PineconeStore(settings, embeddings=RaisingEmbeddings())

    with pytest.raises(PineconeEmbeddingError) as exc_info:
        store.embed_query("query text")

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_embed_query_error_does_not_leak_secrets() -> None:
    settings = make_settings(
        openai_api_key="sk-super-secret", pinecone_api_key="pc-super-secret"
    )
    store = PineconeStore(settings, embeddings=RaisingEmbeddings())

    with pytest.raises(PineconeEmbeddingError) as exc_info:
        store.embed_query("query text")

    message = str(exc_info.value)
    assert "sk-super-secret" not in message
    assert "pc-super-secret" not in message


# --- query_similar ---------------------------------------------------------------


def _make_object_match(
    *,
    match_id: str = "doc-1-chunk-0000",
    score: float = 0.9,
    metadata: dict[str, Any] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=match_id,
        score=score,
        metadata=metadata if metadata is not None else {"text": "hello", "document_id": "doc-1"},
    )


def test_query_similar_sends_exact_sdk_arguments() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    client.index_handle = FakeIndexHandle(query_response=SimpleNamespace(matches=[]))
    settings = make_settings()
    store = PineconeStore(settings, client=client)
    vector = [0.1] * 1536
    metadata_filter = {"kind": {"$eq": "documentation_chunk"}}

    store.query_similar(
        vector, namespace="documentation", top_k=5, metadata_filter=metadata_filter
    )

    assert client.index_handle.similar_query_calls == [
        {
            "vector": vector,
            "top_k": 5,
            "include_metadata": True,
            "include_values": False,
            "namespace": "documentation",
            "filter": metadata_filter,
        }
    ]


def test_query_similar_parses_object_style_response() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    match = _make_object_match(match_id="a", score=0.9, metadata={"text": "hi"})
    client.index_handle = FakeIndexHandle(query_response=SimpleNamespace(matches=[match]))
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    results = store.query_similar([0.1] * 1536, namespace="documentation", top_k=5)

    assert results == [PineconeQueryMatch(id="a", score=0.9, metadata={"text": "hi"})]


def test_query_similar_parses_mapping_style_response() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    client.index_handle = FakeIndexHandle(
        query_response={"matches": [{"id": "a", "score": 0.9, "metadata": {"text": "hi"}}]}
    )
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    results = store.query_similar([0.1] * 1536, namespace="documentation", top_k=5)

    assert results == [PineconeQueryMatch(id="a", score=0.9, metadata={"text": "hi"})]


def test_query_similar_empty_matches_returns_empty_list() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    client.index_handle = FakeIndexHandle(query_response=SimpleNamespace(matches=[]))
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    results = store.query_similar([0.1] * 1536, namespace="documentation", top_k=5)

    assert results == []


def test_query_similar_preserves_match_order() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    matches = [
        _make_object_match(match_id="c", score=0.5),
        _make_object_match(match_id="a", score=0.9),
        _make_object_match(match_id="b", score=0.7),
    ]
    client.index_handle = FakeIndexHandle(query_response=SimpleNamespace(matches=matches))
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    results = store.query_similar([0.1] * 1536, namespace="documentation", top_k=5)

    assert [match.id for match in results] == ["c", "a", "b"]


def test_query_similar_reuses_cached_ready_handle() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    client.index_handle = FakeIndexHandle(query_response=SimpleNamespace(matches=[]))
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    store.query_similar([0.1] * 1536, namespace="documentation", top_k=5)
    store.query_similar([0.1] * 1536, namespace="documentation", top_k=5)

    assert client.has_index_call_count == 1
    assert client.describe_index_call_count == 1
    assert client.index_call_count == 1


def test_query_similar_rejects_invalid_vector_before_network() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    settings = make_settings(pinecone_dimension=1536)
    store = PineconeStore(settings, client=client)

    with pytest.raises(PineconeEmbeddingError):
        store.query_similar([0.1] * 10, namespace="documentation", top_k=5)

    assert client.index_call_count == 0


def test_query_similar_rejects_blank_namespace_before_network() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    with pytest.raises(PineconeQueryError):
        store.query_similar([0.1] * 1536, namespace="   ", top_k=5)

    assert client.index_call_count == 0


def test_query_similar_rejects_top_k_below_one_before_network() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    with pytest.raises(PineconeQueryError):
        store.query_similar([0.1] * 1536, namespace="documentation", top_k=0)

    assert client.index_call_count == 0


def test_query_similar_rejects_top_k_above_50_before_network() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    with pytest.raises(PineconeQueryError):
        store.query_similar([0.1] * 1536, namespace="documentation", top_k=51)

    assert client.index_call_count == 0


def test_query_similar_rejects_missing_matches_field() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    client.index_handle = FakeIndexHandle(query_response=SimpleNamespace())
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    with pytest.raises(PineconeQueryError):
        store.query_similar([0.1] * 1536, namespace="documentation", top_k=5)


def test_query_similar_rejects_non_sequence_matches_container() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    client.index_handle = FakeIndexHandle(query_response=SimpleNamespace(matches="not-a-list"))
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    with pytest.raises(PineconeQueryError):
        store.query_similar([0.1] * 1536, namespace="documentation", top_k=5)


def test_query_similar_rejects_missing_id() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    bad_match = SimpleNamespace(score=0.9, metadata={"text": "hi"})
    client.index_handle = FakeIndexHandle(query_response=SimpleNamespace(matches=[bad_match]))
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    with pytest.raises(PineconeQueryError):
        store.query_similar([0.1] * 1536, namespace="documentation", top_k=5)


def test_query_similar_rejects_blank_id() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    bad_match = SimpleNamespace(id="   ", score=0.9, metadata={"text": "hi"})
    client.index_handle = FakeIndexHandle(query_response=SimpleNamespace(matches=[bad_match]))
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    with pytest.raises(PineconeQueryError):
        store.query_similar([0.1] * 1536, namespace="documentation", top_k=5)


def test_query_similar_rejects_missing_score() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    bad_match = SimpleNamespace(id="a", metadata={"text": "hi"})
    client.index_handle = FakeIndexHandle(query_response=SimpleNamespace(matches=[bad_match]))
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    with pytest.raises(PineconeQueryError):
        store.query_similar([0.1] * 1536, namespace="documentation", top_k=5)


def test_query_similar_rejects_non_numeric_score() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    bad_match = SimpleNamespace(id="a", score="high", metadata={"text": "hi"})
    client.index_handle = FakeIndexHandle(query_response=SimpleNamespace(matches=[bad_match]))
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    with pytest.raises(PineconeQueryError):
        store.query_similar([0.1] * 1536, namespace="documentation", top_k=5)


def test_query_similar_rejects_boolean_score() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    bad_match = SimpleNamespace(id="a", score=True, metadata={"text": "hi"})
    client.index_handle = FakeIndexHandle(query_response=SimpleNamespace(matches=[bad_match]))
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    with pytest.raises(PineconeQueryError):
        store.query_similar([0.1] * 1536, namespace="documentation", top_k=5)


def test_query_similar_rejects_nan_score() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    bad_match = SimpleNamespace(id="a", score=float("nan"), metadata={"text": "hi"})
    client.index_handle = FakeIndexHandle(query_response=SimpleNamespace(matches=[bad_match]))
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    with pytest.raises(PineconeQueryError):
        store.query_similar([0.1] * 1536, namespace="documentation", top_k=5)


def test_query_similar_rejects_infinite_score() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    bad_match = SimpleNamespace(id="a", score=float("inf"), metadata={"text": "hi"})
    client.index_handle = FakeIndexHandle(query_response=SimpleNamespace(matches=[bad_match]))
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    with pytest.raises(PineconeQueryError):
        store.query_similar([0.1] * 1536, namespace="documentation", top_k=5)


def test_query_similar_rejects_missing_metadata() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    bad_match = SimpleNamespace(id="a", score=0.9)
    client.index_handle = FakeIndexHandle(query_response=SimpleNamespace(matches=[bad_match]))
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    with pytest.raises(PineconeQueryError):
        store.query_similar([0.1] * 1536, namespace="documentation", top_k=5)


def test_query_similar_rejects_none_metadata() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    bad_match = SimpleNamespace(id="a", score=0.9, metadata=None)
    client.index_handle = FakeIndexHandle(query_response=SimpleNamespace(matches=[bad_match]))
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    with pytest.raises(PineconeQueryError):
        store.query_similar([0.1] * 1536, namespace="documentation", top_k=5)


def test_query_similar_rejects_non_mapping_metadata() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    bad_match = SimpleNamespace(id="a", score=0.9, metadata="not-a-mapping")
    client.index_handle = FakeIndexHandle(query_response=SimpleNamespace(matches=[bad_match]))
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    with pytest.raises(PineconeQueryError):
        store.query_similar([0.1] * 1536, namespace="documentation", top_k=5)


def test_query_similar_wraps_sdk_error_with_cause() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    client.index_handle = RaisingQueryIndexHandle()
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    with pytest.raises(PineconeQueryError) as exc_info:
        store.query_similar([0.1] * 1536, namespace="documentation", top_k=5)

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_query_similar_invalidates_cache_on_sdk_failure_and_recovers() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    client.index_handle = FlakyQueryIndexHandle(fail_on_call=2)
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    store.query_similar([0.1] * 1536, namespace="documentation", top_k=5)
    assert client.index_call_count == 1

    with pytest.raises(PineconeQueryError):
        store.query_similar([0.1] * 1536, namespace="documentation", top_k=5)

    store.query_similar([0.1] * 1536, namespace="documentation", top_k=5)

    assert client.index_call_count == 2


class MalformedThenHealthyQueryIndexHandle(FakeIndexHandle):
    """Returns a malformed response on its first query call, then healthy
    matches afterwards; used to prove that a response-parsing failure (as
    opposed to an SDK exception) does not invalidate the cached index handle."""

    def __init__(self) -> None:
        super().__init__()
        self._calls = 0

    def query(self, **kwargs: Any) -> Any:
        self._calls += 1
        if self._calls == 1:
            return SimpleNamespace()  # missing the required 'matches' field
        return SimpleNamespace(matches=[_make_object_match(match_id="a", score=0.9)])


def test_query_similar_malformed_response_preserves_cached_handle_and_recovers() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    client.index_handle = MalformedThenHealthyQueryIndexHandle()
    settings = make_settings()
    store = PineconeStore(settings, client=client)

    with pytest.raises(PineconeQueryError):
        store.query_similar([0.1] * 1536, namespace="documentation", top_k=5)

    assert client.index_call_count == 1

    results = store.query_similar([0.1] * 1536, namespace="documentation", top_k=5)

    expected_metadata = {"text": "hello", "document_id": "doc-1"}
    assert results == [PineconeQueryMatch(id="a", score=0.9, metadata=expected_metadata)]
    assert client.index_call_count == 1


def test_query_similar_error_does_not_leak_secrets() -> None:
    client = FakePineconeClient(existing=True, descriptions=[make_description()])
    client.index_handle = RaisingQueryIndexHandle()
    settings = make_settings(
        openai_api_key="sk-super-secret", pinecone_api_key="pc-super-secret"
    )
    store = PineconeStore(settings, client=client)

    with pytest.raises(PineconeQueryError) as exc_info:
        store.query_similar([0.1] * 1536, namespace="documentation", top_k=5)

    message = str(exc_info.value)
    assert "sk-super-secret" not in message
    assert "pc-super-secret" not in message
