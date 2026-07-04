"""Unit tests for the ask_agent script's formatting and CLI wiring.

Uses an injected fake LangChainToolCallingAgent so no network, OpenAI, Pinecone, or PyPI
calls occur.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from ai_docs_agent.models import AnswerSource, LangChainAgentResult

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "ask_agent.py"


def _load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ask_agent_script", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_source(**overrides: Any) -> AnswerSource:
    defaults: dict[str, Any] = {
        "title": "PyPI package page: httpx",
        "url": "https://pypi.org/project/httpx/",
        "document_id": "pypi:httpx",
        "chunk_index": 0,
        "chunk_count": 1,
    }
    return AnswerSource(**{**defaults, **overrides})


def _make_result(**overrides: Any) -> LangChainAgentResult:
    sources = overrides.pop("sources", (_make_source(),))
    defaults: dict[str, Any] = {
        "question": "Какая последняя версия пакета httpx на PyPI?",
        "answer": "Последняя версия пакета httpx на PyPI: 9.9.9.",
        "sources": sources,
        "tools_used": ("pypi_lookup",),
        "tool_call_count": 1,
        "used_no_tool": False,
        "outcome": "success",
        "failure_category": None,
    }
    return LangChainAgentResult(**{**defaults, **overrides})


class FakeService:
    """Stands in for LangChainToolCallingAgent: no external calls occur."""

    def __init__(
        self,
        *,
        result: LangChainAgentResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self._result = result
        self._error = error
        self.calls: list[str] = []

    def answer(self, question: str) -> LangChainAgentResult:
        self.calls.append(question)
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


def test_format_agent_report_success_content() -> None:
    module = _load_script_module()
    result = _make_result()

    lines = module.format_agent_report(result)

    joined = "\n".join(lines)
    assert lines[0] == "Answer:"
    assert result.answer in joined
    assert "Sources:" in joined
    assert "Tools used: pypi_lookup" in joined


def test_format_agent_report_safe_fallback_includes_outcome() -> None:
    module = _load_script_module()
    result = _make_result(
        answer="Пакет не найден на PyPI.",
        sources=(),
        outcome="safe_fallback",
        failure_category="package_not_found",
    )

    lines = module.format_agent_report(result)
    joined = "\n".join(lines)

    assert "Sources: none" in joined
    assert "Outcome: safe_fallback (package_not_found)" in joined


def test_main_success_prints_report_and_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_script_module()
    fake_service = FakeService(result=_make_result())

    exit_code = module.main(["Какая последняя версия пакета httpx на PyPI?"], service=fake_service)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Answer:" in captured.out
    assert "Tools used: pypi_lookup" in captured.out
    assert fake_service.calls == ["Какая последняя версия пакета httpx на PyPI?"]


def test_main_safe_fallback_still_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_script_module()
    fake_service = FakeService(
        result=_make_result(
            answer="Пакет не найден на PyPI.",
            sources=(),
            outcome="safe_fallback",
            failure_category="package_not_found",
        )
    )

    exit_code = module.main(["missing-package"], service=fake_service)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "package_not_found" in captured.out


def test_main_orchestration_error_returns_nonzero_exit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script_module()
    fake_service = FakeService(error=module.LangChainAgentExecutionError("internal detail"))

    exit_code = module.main(["query"], service=fake_service)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Agent FAILED: unexpected orchestration error" in captured.out
    assert "internal detail" not in captured.out


def test_main_unexpected_error_returns_nonzero_exit_without_leaking_message(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script_module()
    fake_service = FakeService(error=RuntimeError("super-secret-startup-detail"))

    exit_code = module.main(["query"], service=fake_service)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Agent FAILED: unexpected startup error" in captured.out
    assert "super-secret-startup-detail" not in captured.out
    assert "Traceback" not in captured.out


def test_main_output_contains_no_secrets(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_script_module()
    fake_service = FakeService(result=_make_result())

    module.main(["query"], service=fake_service)

    captured = capsys.readouterr()
    assert "sk-" not in captured.out
    assert "pc-" not in captured.out
    assert "OPENAI_API_KEY" not in captured.out


class NarrowCodepageStream:
    """Emulates a narrow-codepage console stream."""

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


def test_main_unicode_output_on_narrow_stdout_does_not_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module()
    fake_stdout = NarrowCodepageStream()
    monkeypatch.setattr(sys, "stdout", fake_stdout)
    fake_service = FakeService(
        result=_make_result(answer="Ответ содержит непередаваемые символы: 日本語.")
    )

    exit_code = module.main(["query"], service=fake_service)

    assert exit_code == 0
    written = "".join(fake_stdout.written)
    assert "Traceback" not in written
    assert "Answer:" in written
