"""Unit tests for the run_telegram_bot script's startup wiring.

Uses an injected fake application factory so no network, Telegram, OpenAI, or
Pinecone calls occur. main() is exercised only with fakes injected, never
with the real build_application()/get_settings().
"""

import importlib.util
import logging
from pathlib import Path
from types import ModuleType

import pytest
from pydantic import ValidationError
from telegram.error import InvalidToken

from ai_docs_agent.telegram_bot import TelegramStartupSummary

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_telegram_bot.py"


def _load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_telegram_bot_script", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeApplication:
    def __init__(
        self,
        *,
        run_error: Exception | None = None,
        bot_data: dict[str, object] | None = None,
    ) -> None:
        self._run_error = run_error
        self.run_polling_calls = 0
        self.bot_data = bot_data or {
            "telegram_startup_summary": TelegramStartupSummary(
                pinecone_index_name="docs-index",
                pinecone_namespace="documentation-live-check",
                embedding_model="text-embedding-3-small",
                retrieval_top_k=5,
                score_threshold=0.25,
            )
        }

    def run_polling(self) -> None:
        self.run_polling_calls += 1
        if self._run_error is not None:
            raise self._run_error


def test_main_calls_run_polling_on_success() -> None:
    module = _load_script_module()
    application = FakeApplication()

    exit_code = module.main(application_factory=lambda: application)

    assert exit_code == 0
    assert application.run_polling_calls == 1


def test_main_logs_safe_startup_summary(
    caplog: pytest.LogCaptureFixture,
) -> None:
    module = _load_script_module()
    application = FakeApplication()

    with caplog.at_level(logging.INFO):
        exit_code = module.main(application_factory=lambda: application)

    assert exit_code == 0
    assert (
        "Telegram bot starting index=docs-index namespace=documentation-live-check"
        in caplog.text
    )
    assert "embedding_model=text-embedding-3-small" in caplog.text
    assert "top_k=5" in caplog.text
    assert "score_threshold=0.25" in caplog.text
    assert "super-secret" not in caplog.text
    assert "test-telegram-token" not in caplog.text


def test_configure_logging_preserves_ai_docs_info_and_suppresses_pinecone_info(
    caplog: pytest.LogCaptureFixture,
) -> None:
    module = _load_script_module()
    ai_docs_logger = logging.getLogger("ai_docs_agent.test_logger")
    pinecone_logger = logging.getLogger("pinecone.index")
    pinecone_parent_logger = logging.getLogger("pinecone")
    original_ai_docs_level = ai_docs_logger.level
    original_pinecone_level = pinecone_logger.level
    original_pinecone_parent_level = pinecone_parent_logger.level

    try:
        ai_docs_logger.setLevel(logging.NOTSET)
        pinecone_logger.setLevel(logging.NOTSET)
        pinecone_parent_logger.setLevel(logging.NOTSET)

        module._configure_logging()

        with caplog.at_level(logging.INFO):
            ai_docs_logger.info("safe ai_docs_agent info message")
            pinecone_logger.info(
                "Upserting 1 vectors into namespace 'user-memory-full-digest-secret'"
            )
            pinecone_logger.info(
                "Connecting to Pinecone index host example-index-host-1234.svc.us-east1.pinecone.io"
            )
            pinecone_logger.warning("pinecone warning still visible")
            pinecone_logger.error("pinecone error still visible")

        assert "safe ai_docs_agent info message" in caplog.text
        assert "user-memory-full-digest-secret" not in caplog.text
        assert "example-index-host-1234.svc.us-east1.pinecone.io" not in caplog.text
        assert "pinecone warning still visible" in caplog.text
        assert "pinecone error still visible" in caplog.text
    finally:
        ai_docs_logger.setLevel(original_ai_docs_level)
        pinecone_logger.setLevel(original_pinecone_level)
        pinecone_parent_logger.setLevel(original_pinecone_parent_level)


def test_main_returns_nonzero_when_build_raises_validation_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script_module()

    def failing_factory() -> None:
        raise ValidationError.from_exception_data("AppSettings", [])

    exit_code = module.main(application_factory=failing_factory)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Telegram bot FAILED to start" in captured.out


def test_main_does_not_call_run_polling_when_build_fails() -> None:
    module = _load_script_module()
    application = FakeApplication()

    def failing_factory() -> FakeApplication:
        raise ValidationError.from_exception_data("AppSettings", [])

    module.main(application_factory=failing_factory)

    assert application.run_polling_calls == 0


def test_main_returns_nonzero_on_run_polling_failure(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_script_module()
    application = FakeApplication(run_error=InvalidToken("invalid token format"))

    exit_code = module.main(application_factory=lambda: application)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Telegram bot FAILED to start" in captured.out
    assert "Traceback" not in captured.out


def test_main_returns_success_and_logs_clean_stop_on_keyboard_interrupt(
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    module = _load_script_module()
    application = FakeApplication(run_error=KeyboardInterrupt())

    with caplog.at_level(logging.INFO):
        exit_code = module.main(application_factory=lambda: application)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Telegram bot stopped" in caplog.text
    assert "FAILED" not in captured.out
    assert "Traceback" not in captured.out


def test_main_unexpected_build_error_returns_nonzero_without_leaking_details(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script_module()

    def failing_factory() -> None:
        raise RuntimeError("super-secret-internal-detail")

    exit_code = module.main(application_factory=failing_factory)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Telegram bot FAILED to start: unexpected internal error" in captured.out
    assert "super-secret-internal-detail" not in captured.out
    assert "Traceback" not in captured.out


def test_main_error_output_does_not_leak_token(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_script_module()
    application = FakeApplication(run_error=InvalidToken("token 123456:super-secret-token"))

    module.main(application_factory=lambda: application)

    captured = capsys.readouterr()
    assert "super-secret-token" not in captured.out


def test_main_normal_polling_completion_logs_clean_stop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    module = _load_script_module()
    application = FakeApplication()

    with caplog.at_level(logging.INFO):
        exit_code = module.main(application_factory=lambda: application)

    assert exit_code == 0
    assert caplog.text.count("Telegram bot stopped") == 1


def test_import_performs_no_work() -> None:
    module = _load_script_module()

    assert not hasattr(module, "application")
