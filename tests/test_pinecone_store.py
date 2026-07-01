"""Unit tests for PineconeStore. Uses fakes only; no real network access."""

from types import SimpleNamespace
from typing import Any

import pytest

from ai_docs_agent.config import AppSettings
from ai_docs_agent.models import PineconeSmokeTestResult
from ai_docs_agent.pinecone_store import (
    PineconeIndexConfigurationError,
    PineconeIndexNotFoundError,
    PineconeSmokeTestError,
    PineconeStore,
    PineconeStoreError,
)

_REQUIRED: dict[str, Any] = {
    "openai_api_key": "sk-test-openai",
    "pinecone_api_key": "pc-test-key",
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

    def __init__(self, *, visible_after: int = 0) -> None:
        self.upserted: list[dict[str, Any]] = []
        self.deleted: list[dict[str, Any]] = []
        self._visible_after = visible_after
        self.query_calls = 0

    def upsert(self, *, vectors: list[dict[str, Any]], namespace: str) -> None:
        self.upserted.append({"vectors": vectors, "namespace": namespace})

    def query(self, *, vector: list[float], top_k: int, include_metadata: bool, namespace: str):
        self.query_calls += 1
        if not self.upserted or self.query_calls <= self._visible_after:
            return SimpleNamespace(matches=[])
        record = self.upserted[-1]["vectors"][0]
        return SimpleNamespace(matches=[SimpleNamespace(id=record["id"], score=0.987)])

    def delete(self, *, ids: list[str], namespace: str) -> None:
        self.deleted.append({"ids": ids, "namespace": namespace})


class FailingDeleteIndexHandle(FakeIndexHandle):
    def delete(self, *, ids: list[str], namespace: str) -> None:
        raise RuntimeError("delete backend unavailable")


class RaisingQueryIndexHandle(FakeIndexHandle):
    def query(self, **kwargs: Any):
        raise RuntimeError("query backend unavailable")


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

    def has_index(self, name: str) -> bool:
        return self.existing

    def describe_index(self, name: str) -> Any:
        if not self._descriptions:
            raise AssertionError("describe_index called with no descriptions configured")
        index = min(self._describe_calls, len(self._descriptions) - 1)
        self._describe_calls += 1
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
