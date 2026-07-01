"""Unit tests for the pinecone_smoke_test script's report formatting.

Loads the script by file path (it is not part of the installed package) and
calls only its pure `format_smoke_test_report` function; `main()` is never
invoked, so no settings are loaded and no network calls occur.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

from ai_docs_agent.models import PineconeSmokeTestResult

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "pinecone_smoke_test.py"


def _load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("pinecone_smoke_test_script", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_result(*, cleanup_succeeded: bool) -> PineconeSmokeTestResult:
    return PineconeSmokeTestResult(
        index_name="ai-docs-rag-agent",
        namespace="__smoke_test__",
        dimension=3,
        embedding_model="text-embedding-3-small",
        record_id="smoke-abc123",
        matched_id="smoke-abc123",
        score=0.99,
        cleanup_succeeded=cleanup_succeeded,
        elapsed_seconds=0.42,
    )


def test_format_report_success_is_ok_with_zero_exit_code() -> None:
    module = _load_script_module()

    lines, exit_code = module.format_smoke_test_report(_make_result(cleanup_succeeded=True))

    assert exit_code == 0
    assert "Pinecone smoke test OK" in lines[0]


def test_format_report_cleanup_failure_is_not_ok_with_nonzero_exit_code() -> None:
    module = _load_script_module()

    lines, exit_code = module.format_smoke_test_report(_make_result(cleanup_succeeded=False))

    assert exit_code != 0
    assert not any("OK" in line for line in lines)
    assert any("cleanup" in line.lower() for line in lines)
