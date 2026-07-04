"""Unit tests for the pypi_lookup script's formatting and CLI wiring.

Uses an injected fake PyPILookupService so no real HTTP calls occur.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from ai_docs_agent.models import PyPIPackageInfo
from ai_docs_agent.pypi import InvalidPackageNameError

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "pypi_lookup.py"


def _load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("pypi_lookup_script", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_result(**overrides: Any) -> PyPIPackageInfo:
    defaults: dict[str, Any] = {
        "package_name": "httpx",
        "latest_version": "9.9.9",
        "summary": "HTTP client for Python.",
        "requires_python": ">=3.8",
        "pypi_url": "https://pypi.org/project/httpx/",
        "project_url": "https://www.python-httpx.org/",
    }
    return PyPIPackageInfo(**{**defaults, **overrides})


class FakeService:
    """Stands in for PyPILookupService: no network calls are performed."""

    def __init__(
        self,
        *,
        result: PyPIPackageInfo | None = None,
        error: Exception | None = None,
    ) -> None:
        self._result = result
        self._error = error
        self.calls: list[str] = []

    def lookup(self, package_name: str) -> PyPIPackageInfo:
        self.calls.append(package_name)
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


def test_format_pypi_report_success_content() -> None:
    module = _load_script_module()
    result = _make_result()

    lines = module.format_pypi_report(result)

    assert lines == [
        "Package: httpx",
        "Latest version: 9.9.9",
        "Summary: HTTP client for Python.",
        "Requires Python: >=3.8",
        "PyPI URL: https://pypi.org/project/httpx/",
        "Project URL: https://www.python-httpx.org/",
    ]


def test_format_pypi_report_uses_stable_placeholders_for_missing_optional_fields() -> None:
    module = _load_script_module()
    result = _make_result(summary=None, requires_python=None, project_url=None)

    lines = module.format_pypi_report(result)

    assert "Summary: not specified" in lines
    assert "Requires Python: not specified" in lines
    assert "Project URL: not specified" in lines


def test_main_success_prints_report_and_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_script_module()
    fake_service = FakeService(result=_make_result())

    exit_code = module.main(["httpx"], service=fake_service)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Package: httpx" in captured.out
    assert fake_service.calls == ["httpx"]


def test_main_forwards_package_name_argument() -> None:
    module = _load_script_module()
    fake_service = FakeService(result=_make_result())

    module.main(["typing-extensions"], service=fake_service)

    assert fake_service.calls == ["typing-extensions"]


def test_main_domain_error_returns_nonzero_exit(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_script_module()
    fake_service = FakeService(error=InvalidPackageNameError("Package name must not be blank."))

    exit_code = module.main(["   "], service=fake_service)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "PyPI lookup FAILED: Package name must not be blank." in captured.out


def test_main_unexpected_error_returns_nonzero_exit_without_leaking_message(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script_module()
    fake_service = FakeService(error=RuntimeError("super-secret-internal-detail"))

    exit_code = module.main(["httpx"], service=fake_service)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "PyPI lookup FAILED: unexpected internal error" in captured.out
    assert "super-secret-internal-detail" not in captured.out
    assert "Traceback" not in captured.out


def test_main_output_contains_no_secrets(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_script_module()
    fake_service = FakeService(result=_make_result())

    module.main(["httpx"], service=fake_service)

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
    fake_service = FakeService(result=_make_result(summary="Русский текст и 日本語."))

    exit_code = module.main(["httpx"], service=fake_service)

    assert exit_code == 0
    written = "".join(fake_stdout.written)
    assert "Traceback" not in written
    assert "Summary:" in written
