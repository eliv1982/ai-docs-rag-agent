"""Unit tests for the url_preview script's formatting and CLI wiring.

Uses an injected fake UrlIngestionService so no network, DNS, OpenAI, or
Pinecone calls occur. format_preview_report is a pure function; main() is
exercised only with the fake service injected, never the real one.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from ai_docs_agent.models import DocumentChunk, UrlProcessingResult
from ai_docs_agent.url_ingestion import ContentExtractionError, InvalidUrlError

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "url_preview.py"


def _load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("url_preview_script", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_result(*, text_char_count: int = 600) -> UrlProcessingResult:
    chunk = DocumentChunk(
        id="doc-abc123-chunk-0000",
        document_id="doc-abc123",
        source_url="https://docs.example.com/page",
        final_url="https://docs.example.com/page",
        title="Example Page",
        text="A" * text_char_count,
        chunk_index=0,
        chunk_count=1,
        content_hash="deadbeef",
    )
    return UrlProcessingResult(
        source_url="https://docs.example.com/page",
        final_url="https://docs.example.com/page",
        title="Example Page",
        document_id="doc-abc123",
        content_hash="deadbeef",
        text_char_count=text_char_count,
        chunk_count=1,
        chunks=(chunk,),
    )


class FakeService:
    """Stands in for UrlIngestionService: no network, DNS, OpenAI, or Pinecone calls."""

    def __init__(
        self,
        *,
        result: UrlProcessingResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self._result = result
        self._error = error
        self.calls: list[str] = []

    def process_url(self, url: str) -> UrlProcessingResult:
        self.calls.append(url)
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


def test_format_preview_report_success_content() -> None:
    module = _load_script_module()
    result = _make_result()

    lines = module.format_preview_report(result)

    assert lines[0] == "URL processing OK"
    joined = "\n".join(lines)
    assert result.source_url in joined
    assert result.final_url in joined
    assert result.title in joined
    assert result.document_id in joined
    assert result.content_hash in joined
    assert str(result.text_char_count) in joined
    assert str(result.chunk_count) in joined
    assert result.chunks[0].id in joined


def test_format_preview_report_truncates_preview_to_500_chars() -> None:
    module = _load_script_module()
    result = _make_result(text_char_count=600)

    lines = module.format_preview_report(result)

    preview_line = lines[-1].strip()
    assert len(preview_line) == 500


def test_format_preview_report_short_chunk_is_not_padded() -> None:
    module = _load_script_module()
    result = _make_result(text_char_count=10)

    lines = module.format_preview_report(result)

    preview_line = lines[-1].strip()
    assert len(preview_line) == 10


def test_main_success_prints_report_and_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_script_module()
    fake_service = FakeService(result=_make_result())

    exit_code = module.main(["https://docs.example.com/page"], service=fake_service)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "URL processing OK" in captured.out
    assert fake_service.calls == ["https://docs.example.com/page"]


def test_main_domain_failure_returns_nonzero_exit(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_script_module()
    fake_service = FakeService(error=InvalidUrlError("bad url"))

    exit_code = module.main(["not-a-url"], service=fake_service)

    captured = capsys.readouterr()
    assert exit_code != 0
    assert "FAILED" in captured.out


def test_main_content_extraction_error_returns_nonzero_exit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script_module()
    fake_service = FakeService(
        error=ContentExtractionError(
            "Response declared an unsupported or unknown charset 'bogus-codec'."
        )
    )

    exit_code = module.main(["https://docs.example.com/page"], service=fake_service)

    captured = capsys.readouterr()
    assert exit_code != 0
    assert "FAILED" in captured.out
    assert "Traceback" not in captured.out
    assert "<html" not in captured.out.lower()


def test_main_unexpected_error_returns_nonzero_exit_without_leaking_message(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script_module()
    fake_service = FakeService(error=RuntimeError("super-secret-internal-detail"))

    exit_code = module.main(["https://docs.example.com/page"], service=fake_service)

    captured = capsys.readouterr()
    assert exit_code != 0
    assert "URL preview FAILED: unexpected internal error" in captured.out
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


def test_format_preview_report_is_pure_and_needs_no_network() -> None:
    module = _load_script_module()
    result = _make_result()

    first = module.format_preview_report(result)
    second = module.format_preview_report(result)

    assert first == second
