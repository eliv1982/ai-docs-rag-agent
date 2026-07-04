"""Start the Telegram bot: build the application and run long polling.

Performs real network calls to the Telegram Bot API and, per incoming user
message, to OpenAI and Pinecone through the existing RAG/memory stack. Blocks,
running polling until interrupted (e.g. Ctrl+C). Not part of the automated
test suite; contains no business logic of its own -- it only wires together
ai_docs_agent.config.get_settings() and ai_docs_agent.telegram_bot.build_application().
"""

import logging
import sys
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError
from telegram.error import TelegramError

from ai_docs_agent.config import get_settings
from ai_docs_agent.observability import log_exception_safely
from ai_docs_agent.telegram_bot import (
    _STARTUP_SUMMARY_BOT_DATA_KEY,
    TelegramStartupSummary,
    build_application,
)

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def _log_startup_summary(application: Any) -> None:
    bot_data = getattr(application, "bot_data", {})
    summary = bot_data.get(_STARTUP_SUMMARY_BOT_DATA_KEY)
    if not isinstance(summary, TelegramStartupSummary):
        logger.info("Telegram bot starting")
        return

    logger.info(
        "Telegram bot starting index=%s namespace=%s embedding_model=%s top_k=%d "
        "score_threshold=%.2f",
        summary.pinecone_index_name,
        summary.pinecone_namespace,
        summary.embedding_model,
        summary.retrieval_top_k,
        summary.score_threshold,
    )


def main(*, application_factory: Callable[[], Any] | None = None) -> int:
    _configure_logging()
    factory = application_factory or (lambda: build_application(get_settings()))

    try:
        application = factory()
    except ValidationError as exc:
        logger.error("Telegram bot failed to start due to invalid configuration.")
        print(f"Telegram bot FAILED to start: {exc}")
        return 1
    except Exception as exc:
        log_exception_safely(
            logger,
            "Telegram bot failed to start due to unexpected internal error",
            exc=exc,
        )
        print("Telegram bot FAILED to start: unexpected internal error")
        return 1

    _log_startup_summary(application)

    try:
        application.run_polling()
    except KeyboardInterrupt:
        logger.info("Telegram bot stopped")
        return 0
    except TelegramError as exc:
        # Telegram-side error messages are outside our control and must never
        # be echoed verbatim, in case they ever embed the configured token.
        logger.error("Telegram bot failed due to Telegram API error type=%s.", type(exc).__name__)
        print("Telegram bot FAILED to start: Telegram API error.")
        return 1
    except Exception as exc:
        log_exception_safely(
            logger,
            "Telegram bot failed during polling",
            exc=exc,
        )
        print("Telegram bot FAILED to start: unexpected internal error")
        return 1

    logger.info("Telegram bot stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
