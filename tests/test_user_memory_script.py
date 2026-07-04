"""Unit tests for the user_memory script's formatting and CLI wiring.

Uses an injected fake UserMemoryService so no real OpenAI/Pinecone calls occur.
"""

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from ai_docs_agent.models import UserMemoryMatch, UserMemoryRecallResult, UserMemoryWriteResult
from ai_docs_agent.user_memory import InvalidMemoryStatementError, MemoryStorageError

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "user_memory.py"


def _load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("user_memory_script", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_write_result(**overrides: Any) -> UserMemoryWriteResult:
    defaults: dict[str, Any] = {
        "status": "created",
        "memory_id": "memory-0123456789abcdef0123456789abcdef",
        "identity_digest": "abc123def456",
    }
    return UserMemoryWriteResult(**{**defaults, **overrides})


def _make_recall_result(**overrides: Any) -> UserMemoryRecallResult:
    defaults: dict[str, Any] = {
        "matches": (
            UserMemoryMatch(
                memory_id="memory-0123456789abcdef0123456789abcdef",
                text="В примерах я предпочитаю httpx.",
                score=0.8123,
                memory_kind="user_memory",
            ),
        ),
        "found": True,
        "threshold": 0.35,
        "top_k": 5,
        "raw_candidate_count": 2,
        "identity_digest": "abc123def456",
    }
    return UserMemoryRecallResult(**{**defaults, **overrides})


class FakeService:
    """Stands in for UserMemoryService: no OpenAI/Pinecone calls are performed."""

    def __init__(
        self,
        *,
        write_result: UserMemoryWriteResult | None = None,
        recall_result: UserMemoryRecallResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self._write_result = write_result
        self._recall_result = recall_result
        self._error = error
        self.remember_calls: list[tuple[str, str]] = []
        self.recall_calls: list[tuple[str, str]] = []

    def remember(self, user_identifier: str, statement: str) -> UserMemoryWriteResult:
        self.remember_calls.append((user_identifier, statement))
        if self._error is not None:
            raise self._error
        assert self._write_result is not None
        return self._write_result

    def recall(self, user_identifier: str, query: str) -> UserMemoryRecallResult:
        self.recall_calls.append((user_identifier, query))
        if self._error is not None:
            raise self._error
        assert self._recall_result is not None
        return self._recall_result


def test_format_write_report_created() -> None:
    module = _load_script_module()

    lines = module.format_write_report(_make_write_result())

    assert lines == [
        "User memory remember: created",
        "Memory ID: memory-0123456789abcdef0123456789abcdef",
        "Identity digest: abc123def456",
    ]


def test_format_write_report_duplicate() -> None:
    module = _load_script_module()

    lines = module.format_write_report(_make_write_result(status="duplicate"))

    assert lines[0] == "User memory remember: duplicate"


def test_format_recall_report_with_matches() -> None:
    module = _load_script_module()

    lines = module.format_recall_report(_make_recall_result())

    assert lines == [
        "User memory recall: 1 of 2 candidate(s) accepted (threshold 0.35, top_k 5)",
        "Identity digest: abc123def456",
        "1. [0.8123] В примерах я предпочитаю httpx.",
    ]


def test_format_recall_report_empty_is_safe() -> None:
    module = _load_script_module()
    result = _make_recall_result(matches=(), found=False, raw_candidate_count=0)

    lines = module.format_recall_report(result)

    assert lines == [
        "User memory recall: 0 of 0 candidate(s) accepted (threshold 0.35, top_k 5)",
        "Identity digest: abc123def456",
        "No stored memories matched the query.",
    ]


def test_main_remember_success_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_script_module()
    fake_service = FakeService(write_result=_make_write_result())

    exit_code = module.main(
        ["remember", "stage4h-demo-user", "В примерах я предпочитаю httpx."],
        service=fake_service,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "User memory remember: created" in captured.out
    assert fake_service.remember_calls == [
        ("stage4h-demo-user", "В примерах я предпочитаю httpx.")
    ]


def test_main_remember_duplicate_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_script_module()
    fake_service = FakeService(write_result=_make_write_result(status="duplicate"))

    exit_code = module.main(
        ["remember", "stage4h-demo-user", "В примерах я предпочитаю httpx."],
        service=fake_service,
    )

    assert exit_code == 0
    assert "duplicate" in capsys.readouterr().out


def test_main_recall_success_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_script_module()
    fake_service = FakeService(recall_result=_make_recall_result())

    exit_code = module.main(
        ["recall", "stage4h-demo-user", "Какую HTTP-библиотеку я предпочитаю?"],
        service=fake_service,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "В примерах я предпочитаю httpx." in captured.out
    assert fake_service.recall_calls == [
        ("stage4h-demo-user", "Какую HTTP-библиотеку я предпочитаю?")
    ]


def test_main_recall_safe_empty_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_script_module()
    fake_service = FakeService(
        recall_result=_make_recall_result(matches=(), found=False, raw_candidate_count=0)
    )

    exit_code = module.main(
        ["recall", "stage4h-demo-user", "Какую HTTP-библиотеку я предпочитаю?"],
        service=fake_service,
    )

    assert exit_code == 0
    assert "No stored memories matched the query." in capsys.readouterr().out


def test_main_domain_error_returns_one_without_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script_module()
    fake_service = FakeService(
        error=InvalidMemoryStatementError("Memory statement must not be blank.")
    )

    exit_code = module.main(["remember", "stage4h-demo-user", "   "], service=fake_service)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "User memory FAILED: Memory statement must not be blank." in captured.out
    assert "Traceback" not in captured.out


def test_main_infrastructure_error_returns_one(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_script_module()
    fake_service = FakeService(error=MemoryStorageError("Failed to store the memory record."))

    exit_code = module.main(
        ["remember", "stage4h-demo-user", "statement"], service=fake_service
    )

    assert exit_code == 1
    assert "User memory FAILED:" in capsys.readouterr().out


def test_main_unexpected_error_returns_one_without_leaking_details(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script_module()
    fake_service = FakeService(error=RuntimeError("super-secret-internal-detail"))

    exit_code = module.main(
        ["remember", "stage4h-demo-user", "statement"], service=fake_service
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "User memory FAILED: unexpected internal error" in captured.out
    assert "super-secret-internal-detail" not in captured.out
    assert "Traceback" not in captured.out


def test_main_output_contains_no_raw_namespace_or_secrets(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script_module()
    fake_service = FakeService(write_result=_make_write_result())

    module.main(
        ["remember", "stage4h-demo-user", "В примерах я предпочитаю httpx."],
        service=fake_service,
    )

    captured = capsys.readouterr()
    assert "user-memory-" not in captured.out  # namespace is never printed
    assert "USER_MEMORY_HASH_SECRET" not in captured.out
    assert "sk-" not in captured.out
    assert "pc-" not in captured.out


def test_import_performs_no_work() -> None:
    module = _load_script_module()

    # Importing defines only functions; no settings/service is constructed.
    assert not hasattr(module, "service")
    assert not hasattr(module, "settings")
