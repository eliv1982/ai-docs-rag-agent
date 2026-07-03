"""Unit tests for the ask_docs script's formatting and CLI wiring.

Uses an injected fake DocumentationAnswerService so no network, OpenAI, or
Pinecone calls occur. format_answer_report is a pure function; main() is
exercised only with the fake service injected, never the real one.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from ai_docs_agent.agent import AnswerServiceError
from ai_docs_agent.models import AnswerSource, GroundedAnswerResult

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "ask_docs.py"


def _load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ask_docs_script", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_source(**overrides: Any) -> AnswerSource:
    defaults: dict[str, Any] = {
        "title": "Example Page",
        "url": "https://docs.example.com/page",
        "document_id": "doc-abc123",
        "chunk_index": 0,
        "chunk_count": 1,
    }
    return AnswerSource(**{**defaults, **overrides})


def _make_result(**overrides: Any) -> GroundedAnswerResult:
    sources = overrides.pop("sources", (_make_source(),))
    defaults: dict[str, Any] = {
        "question": "how do I configure the client?",
        "answer": "Set the API key via the documented environment variable.",
        "sources": sources,
        "retrieved_chunk_count": 1,
    }
    return GroundedAnswerResult(**{**defaults, **overrides})


class FakeService:
    """Stands in for DocumentationAnswerService: no network, OpenAI, or Pinecone calls."""

    def __init__(
        self,
        *,
        result: GroundedAnswerResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self._result = result
        self._error = error
        self.calls: list[tuple[str, int | None, str | None]] = []

    def answer(
        self, question: str, *, top_k: int | None = None, namespace: str | None = None
    ) -> GroundedAnswerResult:
        self.calls.append((question, top_k, namespace))
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


# --- format_answer_report ----------------------------------------------------------


def test_format_answer_report_success_content() -> None:
    module = _load_script_module()
    result = _make_result()

    lines = module.format_answer_report(result)

    assert lines[0] == "Answer:"
    joined = "\n".join(lines)
    assert result.answer in joined
    assert "Sources:" in joined
    assert "1. Example Page — https://docs.example.com/page" in joined


def test_format_answer_report_preserves_source_order() -> None:
    module = _load_script_module()
    sources = (
        _make_source(title="Page A", url="https://docs.example.com/a"),
        _make_source(title="Page B", url="https://docs.example.com/b"),
    )
    result = _make_result(sources=sources, retrieved_chunk_count=2)

    lines = module.format_answer_report(result)
    joined = "\n".join(lines)

    assert joined.index("1. Page A") < joined.index("2. Page B")


def test_format_answer_report_empty_sources() -> None:
    module = _load_script_module()
    result = _make_result(
        answer="В базе знаний не найдено достаточно информации для ответа на этот вопрос.",
        sources=(),
        retrieved_chunk_count=0,
    )

    lines = module.format_answer_report(result)
    joined = "\n".join(lines)

    assert "Sources: none" in joined
    assert "Sources:\n" not in joined + "\n"


# --- main() --------------------------------------------------------------------------


def test_main_success_prints_report_and_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_script_module()
    fake_service = FakeService(result=_make_result())

    exit_code = module.main(["how do I configure the client?"], service=fake_service)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Answer:" in captured.out
    assert fake_service.calls == [("how do I configure the client?", None, None)]


def test_main_forwards_top_k_and_namespace() -> None:
    module = _load_script_module()
    fake_service = FakeService(result=_make_result())

    module.main(["query", "--top-k", "3", "--namespace", "custom-ns"], service=fake_service)

    assert fake_service.calls == [("query", 3, "custom-ns")]


def test_main_without_overrides_passes_none() -> None:
    module = _load_script_module()
    fake_service = FakeService(result=_make_result())

    module.main(["query"], service=fake_service)

    assert fake_service.calls == [("query", None, None)]


def test_main_no_context_fallback_output(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_script_module()
    fallback_result = _make_result(
        answer="В базе знаний не найдено достаточно информации для ответа на этот вопрос.",
        sources=(),
        retrieved_chunk_count=0,
    )
    fake_service = FakeService(result=fallback_result)

    exit_code = module.main(["query"], service=fake_service)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Sources: none" in captured.out


def test_main_domain_error_returns_nonzero_exit(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_script_module()
    fake_service = FakeService(error=AnswerServiceError("query must not be blank"))

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
    assert "Answer FAILED: unexpected internal error" in captured.out
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


# --- stream hardening (reused from scripts/search_query.py) ------------------------


class NarrowCodepageStream:
    """Emulates a real narrow-codepage console stream.

    Encodes every write through a real ascii codec, so it raises
    UnicodeEncodeError for non-ASCII text exactly like a narrow console would,
    until reconfigure(errors="replace") has been applied.
    """

    def __init__(self) -> None:
        self.written: list[str] = []
        self._errors = "strict"

    def reconfigure(self, *, errors: str) -> None:
        self._errors = errors

    def write(self, text: str) -> int:
        encoded = text.encode("ascii", errors=self._errors)
        self.written.append(encoded.decode("ascii", errors="replace"))
        return len(text)

    def flush(self) -> None:
        return None


def test_main_unicode_answer_on_narrow_stdout_does_not_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module()
    fake_stdout = NarrowCodepageStream()
    monkeypatch.setattr(sys, "stdout", fake_stdout)
    result = _make_result(answer="Ответ содержит непередаваемые символы: 日本語.")
    fake_service = FakeService(result=result)

    exit_code = module.main(["query"], service=fake_service)

    assert exit_code == 0
    written = "".join(fake_stdout.written)
    assert "Traceback" not in written
