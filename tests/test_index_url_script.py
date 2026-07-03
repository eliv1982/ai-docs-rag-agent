"""Unit tests for the index_url script's formatting and CLI wiring.

Uses an injected fake DocumentIndexingService so no network, DNS, OpenAI, or
Pinecone calls occur. format_index_report is a pure function; main() is
exercised only with the fake service injected, never the real one.
"""

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from ai_docs_agent.indexing import DocumentIndexingError
from ai_docs_agent.models import DocumentIndexingResult
from ai_docs_agent.url_ingestion import InvalidUrlError

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "index_url.py"


def _load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("index_url_script", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_result(**overrides: Any) -> DocumentIndexingResult:
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
        "elapsed_seconds": 1.23,
    }
    return DocumentIndexingResult(**{**defaults, **overrides})


class FakeService:
    """Stands in for DocumentIndexingService: no network, DNS, OpenAI, or Pinecone calls."""

    def __init__(
        self,
        *,
        result: DocumentIndexingResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self._result = result
        self._error = error
        self.calls: list[tuple[str, str | None]] = []

    def index_url(self, url: str, *, namespace: str | None = None) -> DocumentIndexingResult:
        self.calls.append((url, namespace))
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


# --- format_index_report -------------------------------------------------------


def test_format_index_report_success_content() -> None:
    module = _load_script_module()
    result = _make_result()

    lines, exit_code = module.format_index_report(result)

    assert exit_code == 0
    assert lines[0] == "URL indexing OK"
    joined = "\n".join(lines)
    assert result.source_url in joined
    assert result.final_url in joined
    assert result.document_id in joined
    assert result.content_hash in joined
    assert result.namespace in joined
    assert str(result.chunk_count) in joined
    assert str(result.embedded_count) in joined
    assert str(result.upserted_count) in joined
    assert str(result.verified_count) in joined


def test_format_index_report_cleanup_not_requested_is_success() -> None:
    module = _load_script_module()
    result = _make_result(
        old_versions_cleanup_requested=False, old_versions_cleanup_succeeded=None
    )

    lines, exit_code = module.format_index_report(result)

    assert exit_code == 0
    assert "not requested" in "\n".join(lines)


def test_format_index_report_cleanup_success_is_ok_with_zero_exit_code() -> None:
    module = _load_script_module()
    result = _make_result(
        old_versions_cleanup_requested=True, old_versions_cleanup_succeeded=True
    )

    lines, exit_code = module.format_index_report(result)

    assert exit_code == 0
    assert "URL indexing OK" in lines[0]


def test_format_index_report_cleanup_failure_returns_exit_code_two() -> None:
    module = _load_script_module()
    result = _make_result(
        old_versions_cleanup_requested=True, old_versions_cleanup_succeeded=False
    )

    lines, exit_code = module.format_index_report(result)

    assert exit_code == 2
    assert not any("indexing OK" in line for line in lines)
    assert any("cleanup" in line.lower() for line in lines)


# --- main() ----------------------------------------------------------------------


def test_main_success_prints_report_and_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_script_module()
    fake_service = FakeService(result=_make_result())

    exit_code = module.main(["https://docs.example.com/page"], service=fake_service)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "URL indexing OK" in captured.out
    assert fake_service.calls == [("https://docs.example.com/page", None)]


def test_main_passes_explicit_namespace_argument() -> None:
    module = _load_script_module()
    fake_service = FakeService(result=_make_result(namespace="custom-ns"))

    module.main(
        ["https://docs.example.com/page", "--namespace", "custom-ns"], service=fake_service
    )

    assert fake_service.calls == [("https://docs.example.com/page", "custom-ns")]


def test_main_cleanup_failure_returns_exit_code_two(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_script_module()
    fake_service = FakeService(
        result=_make_result(
            old_versions_cleanup_requested=True, old_versions_cleanup_succeeded=False
        )
    )

    exit_code = module.main(["https://docs.example.com/page"], service=fake_service)

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "FAILED" in captured.out


def test_main_domain_error_returns_exit_code_one(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_script_module()
    fake_service = FakeService(error=DocumentIndexingError("boom"))

    exit_code = module.main(["https://docs.example.com/page"], service=fake_service)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "FAILED" in captured.out


def test_main_url_ingestion_error_returns_exit_code_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script_module()
    fake_service = FakeService(error=InvalidUrlError("bad url"))

    exit_code = module.main(["not-a-url"], service=fake_service)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "FAILED" in captured.out


def test_main_unexpected_error_returns_exit_one_without_leaking_message(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script_module()
    fake_service = FakeService(error=RuntimeError("super-secret-internal-detail"))

    exit_code = module.main(["https://docs.example.com/page"], service=fake_service)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "URL indexing FAILED: unexpected internal error" in captured.out
    assert "super-secret-internal-detail" not in captured.out
    assert "Traceback" not in captured.out


def test_main_output_contains_no_secrets(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_script_module()
    fake_service = FakeService(result=_make_result())

    module.main(["https://docs.example.com/page"], service=fake_service)

    captured = capsys.readouterr()
    assert "sk-" not in captured.out
    assert "pc-" not in captured.out
    assert "OPENAI_API_KEY" not in captured.out
    assert "PINECONE_API_KEY" not in captured.out


def test_format_index_report_is_pure_and_needs_no_network() -> None:
    module = _load_script_module()
    result = _make_result()

    first = module.format_index_report(result)
    second = module.format_index_report(result)

    assert first == second
