"""Unit tests for the search_query script's formatting and CLI wiring.

Uses an injected fake RetrievalService so no network, OpenAI, or Pinecone calls
occur. format_search_report is a pure function; main() is exercised only with
the fake service injected, never the real one.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from ai_docs_agent.models import RetrievalResult, RetrievedChunk
from ai_docs_agent.retrieval import RetrievalError

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "search_query.py"


def _load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("search_query_script", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_chunk(**overrides: Any) -> RetrievedChunk:
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


def _make_result(**overrides: Any) -> RetrievalResult:
    matches = overrides.pop("matches", (_make_chunk(),))
    defaults: dict[str, Any] = {
        "query": "how do I configure the client?",
        "namespace": "documentation",
        "top_k": 5,
        "matches": matches,
    }
    return RetrievalResult(**{**defaults, **overrides})


class FakeService:
    """Stands in for RetrievalService: no network, OpenAI, or Pinecone calls."""

    def __init__(
        self,
        *,
        result: RetrievalResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self._result = result
        self._error = error
        self.calls: list[tuple[str, int | None, str | None]] = []

    def search(
        self, query: str, *, top_k: int | None = None, namespace: str | None = None
    ) -> RetrievalResult:
        self.calls.append((query, top_k, namespace))
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


# --- format_search_report -------------------------------------------------------


def test_format_search_report_success_content() -> None:
    module = _load_script_module()
    result = _make_result()

    lines = module.format_search_report(result)

    assert lines[0] == "Retrieval search OK"
    joined = "\n".join(lines)
    assert result.query in joined
    assert result.namespace in joined
    assert str(result.top_k) in joined
    chunk = result.matches[0]
    assert chunk.title in joined
    assert chunk.source_url in joined
    assert chunk.final_url in joined
    assert chunk.document_id in joined
    assert str(chunk.chunk_index) in joined
    assert str(chunk.chunk_count) in joined
    assert chunk.text in joined


def test_format_search_report_multiple_results_preserve_order() -> None:
    module = _load_script_module()
    chunks = (
        _make_chunk(chunk_id="c", document_id="doc-c"),
        _make_chunk(chunk_id="a", document_id="doc-a"),
        _make_chunk(chunk_id="b", document_id="doc-b"),
    )
    result = _make_result(matches=chunks, top_k=3)

    lines = module.format_search_report(result)
    joined = "\n".join(lines)

    assert joined.index("doc-c") < joined.index("doc-a") < joined.index("doc-b")
    assert "#1" in joined
    assert "#2" in joined
    assert "#3" in joined


def test_format_search_report_empty_matches() -> None:
    module = _load_script_module()
    result = _make_result(matches=())

    lines = module.format_search_report(result)
    joined = "\n".join(lines)

    assert "Retrieval search OK" in lines[0]
    assert "0" in joined
    assert "no matches" in joined.lower()


def test_format_search_report_is_pure_and_needs_no_network() -> None:
    module = _load_script_module()
    result = _make_result()

    first = module.format_search_report(result)
    second = module.format_search_report(result)

    assert first == second


# --- main() ----------------------------------------------------------------------


def test_main_success_prints_report_and_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_script_module()
    fake_service = FakeService(result=_make_result())

    exit_code = module.main(["how do I configure the client?"], service=fake_service)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Retrieval search OK" in captured.out
    assert fake_service.calls == [("how do I configure the client?", None, None)]


def test_main_forwards_query_argument() -> None:
    module = _load_script_module()
    fake_service = FakeService(result=_make_result())

    module.main(["some query text"], service=fake_service)

    assert fake_service.calls[0][0] == "some query text"


def test_main_forwards_top_k_override() -> None:
    module = _load_script_module()
    fake_service = FakeService(result=_make_result(top_k=3))

    module.main(["query", "--top-k", "3"], service=fake_service)

    assert fake_service.calls == [("query", 3, None)]


def test_main_forwards_namespace_override() -> None:
    module = _load_script_module()
    fake_service = FakeService(result=_make_result(namespace="custom-ns"))

    module.main(["query", "--namespace", "custom-ns"], service=fake_service)

    assert fake_service.calls == [("query", None, "custom-ns")]


def test_main_without_overrides_passes_none() -> None:
    module = _load_script_module()
    fake_service = FakeService(result=_make_result())

    module.main(["query"], service=fake_service)

    assert fake_service.calls == [("query", None, None)]


def test_main_domain_error_returns_nonzero_exit(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_script_module()
    fake_service = FakeService(error=RetrievalError("query must not be blank"))

    exit_code = module.main(["   "], service=fake_service)

    captured = capsys.readouterr()
    assert exit_code != 0
    assert "FAILED" in captured.out


def test_main_unexpected_error_returns_nonzero_exit_without_leaking_message(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script_module()
    fake_service = FakeService(error=RuntimeError("super-secret-internal-detail"))

    exit_code = module.main(["query"], service=fake_service)

    captured = capsys.readouterr()
    assert exit_code != 0
    assert "Retrieval search FAILED: unexpected internal error" in captured.out
    assert "super-secret-internal-detail" not in captured.out
    assert "Traceback" not in captured.out


def test_main_output_contains_no_secrets(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_script_module()
    fake_service = FakeService(result=_make_result())

    module.main(["query"], service=fake_service)

    captured = capsys.readouterr()
    assert "sk-" not in captured.out
    assert "pc-" not in captured.out
    assert "OPENAI_API_KEY" not in captured.out
    assert "PINECONE_API_KEY" not in captured.out


# --- stream hardening (stdout/stderr reconfigure) ---------------------------------


class WriteOnlyStream:
    """Stream stand-in with no `reconfigure` attribute at all."""

    def __init__(self) -> None:
        self.written: list[str] = []

    def write(self, text: str) -> int:
        self.written.append(text)
        return len(text)

    def flush(self) -> None:
        return None


class NonCallableReconfigureStream(WriteOnlyStream):
    """Stream stand-in whose `reconfigure` attribute exists but is not callable."""

    reconfigure = "not-a-method"


class RaisingReconfigureStream(WriteOnlyStream):
    """Stream stand-in whose `reconfigure()` raises, as some streams may."""

    def reconfigure(self, **kwargs: object) -> None:
        raise ValueError("reconfigure not supported")


class NarrowCodepageStream(WriteOnlyStream):
    """Emulates a real narrow-codepage console stream.

    Encodes every write through a real ascii codec, so it raises
    UnicodeEncodeError for non-ASCII text exactly like a narrow console would,
    until reconfigure(errors="replace") has been applied.
    """

    def __init__(self) -> None:
        super().__init__()
        self._errors = "strict"

    def reconfigure(self, *, errors: str) -> None:
        self._errors = errors

    def write(self, text: str) -> int:
        encoded = text.encode("ascii", errors=self._errors)
        self.written.append(encoded.decode("ascii", errors="replace"))
        return len(text)


def test_main_success_without_stdout_reconfigure_attribute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module()
    fake_stdout = WriteOnlyStream()
    monkeypatch.setattr(sys, "stdout", fake_stdout)
    fake_service = FakeService(result=_make_result())

    exit_code = module.main(["query"], service=fake_service)

    assert exit_code == 0
    assert any("Retrieval search OK" in chunk for chunk in fake_stdout.written)


def test_main_unrecognized_argument_without_stderr_reconfigure_attribute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module()
    fake_stderr = WriteOnlyStream()
    monkeypatch.setattr(sys, "stderr", fake_stderr)

    with pytest.raises(SystemExit) as exc_info:
        module.main(["query", "--unknown-flag"])

    assert exc_info.value.code == 2
    assert any("unrecognized arguments" in chunk for chunk in fake_stderr.written)


def test_main_success_with_non_callable_stdout_reconfigure_attribute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module()
    fake_stdout = NonCallableReconfigureStream()
    monkeypatch.setattr(sys, "stdout", fake_stdout)
    fake_service = FakeService(result=_make_result())

    exit_code = module.main(["query"], service=fake_service)

    assert exit_code == 0
    assert any("Retrieval search OK" in chunk for chunk in fake_stdout.written)


def test_main_success_when_stdout_reconfigure_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module()
    fake_stdout = RaisingReconfigureStream()
    monkeypatch.setattr(sys, "stdout", fake_stdout)
    fake_service = FakeService(result=_make_result())

    exit_code = module.main(["query"], service=fake_service)

    assert exit_code == 0
    assert any("Retrieval search OK" in chunk for chunk in fake_stdout.written)


def test_main_unrecognized_argument_when_stderr_reconfigure_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module()
    fake_stderr = RaisingReconfigureStream()
    monkeypatch.setattr(sys, "stderr", fake_stderr)

    with pytest.raises(SystemExit) as exc_info:
        module.main(["query", "--unknown-flag"])

    assert exc_info.value.code == 2
    assert any("unrecognized arguments" in chunk for chunk in fake_stderr.written)


def test_main_unrecognized_argument_with_unicode_on_narrow_stderr_does_not_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reproduces the audited failure: a narrow-codepage stderr must not raise
    UnicodeEncodeError (nor leak a traceback) when argparse writes an
    'unrecognized arguments' message containing non-ASCII text."""
    module = _load_script_module()
    fake_stderr = NarrowCodepageStream()
    monkeypatch.setattr(sys, "stderr", fake_stderr)

    with pytest.raises(SystemExit) as exc_info:
        module.main(["some query", "--unknown-flag-日本語"])

    assert exc_info.value.code == 2
    written = "".join(fake_stderr.written)
    assert "Traceback" not in written
    assert "unrecognized arguments" in written
