"""Telegram interface over the final integrated agent flow (Stage 4I).

Pipeline: Telegram text message -> str(chat_id) as session_id ->
IntegratedConversationAgentService.handle_message() -> either the deterministic
explicit "Запомни: ..." persistent-memory path or autonomous LangChain tool
selection (documentation_search / pypi_lookup / request-scoped
user_memory_recall) -> deterministic answer + source-list text -> Telegram
response, split into <=4000-character parts on newline boundaries where
practical. /start sends a short introduction; /reset clears only the current
chat's short-term history and never deletes explicitly saved persistent
preferences. Only question and answer text ever pass through the short-term
memory layer -- no Telegram message objects, usernames, or other metadata are
stored, and Pinecone namespaces/record IDs/identity digests/top_k/model names
are never exposed to chat users. No URL indexing or admin controls are
implemented here.

Importing this module builds nothing and makes no network calls: settings,
the integrated service graph, and the Telegram Application are all
constructed lazily by build_application()/run_bot(), which are the only
explicit startup entry points.
"""

import logging
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from ai_docs_agent.agent import get_retrieval_score_threshold
from ai_docs_agent.config import AppSettings, get_settings
from ai_docs_agent.integrated_agent import (
    IntegratedConversationAgentService,
    build_integrated_service,
)
from ai_docs_agent.models import IntegratedAgentResult
from ai_docs_agent.observability import (
    hash_session_id,
    log_exception_safely,
    request_logging_context,
)

_TELEGRAM_MAX_MESSAGE_LENGTH = 4000
_STARTUP_SUMMARY_BOT_DATA_KEY = "telegram_startup_summary"

_START_MESSAGE = (
    "Привет! Я отвечаю на вопросы по индексированной технической документации и могу "
    "узнать актуальные данные пакета на PyPI (например, последнюю версию).\n\n"
    "Команда «Запомни: ...» явно сохраняет ваше личное предпочтение — например: "
    "«Запомни: в примерах я предпочитаю httpx.» Обычные сообщения никогда не "
    "сохраняются как предпочтения.\n\n"
    "Я учитываю до 10 последних сообщений текущего диалога для связности.\n"
    "Команда /reset очищает контекст текущего диалога, но не удаляет явно "
    "сохранённые предпочтения."
)

_RESET_CONFIRMATION = (
    "Контекст текущего диалога очищен. Явно сохранённые предпочтения "
    "(через «Запомни: ...») не удалены."
)

_ANSWER_FAILURE_MESSAGE = "Не удалось подготовить ответ. Попробуйте повторить запрос позже."

_MEMORY_SOURCE_LABEL = "Источник: ваше сохранённое предпочтение (персональная память)"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TelegramStartupSummary:
    """Safe, non-secret runtime settings worth logging at bot startup."""

    pinecone_index_name: str
    pinecone_namespace: str
    embedding_model: str
    retrieval_top_k: int
    score_threshold: float


def split_telegram_message(
    text: str, *, max_length: int = _TELEGRAM_MAX_MESSAGE_LENGTH
) -> list[str]:
    """Split `text` into parts of at most `max_length` characters.

    Prefers splitting at a newline boundary within the current window, falling
    back to a hard cut at `max_length` otherwise. Order is preserved, no part
    is ever empty, and joining the returned parts reproduces `text` exactly
    (no content is discarded).
    """
    if max_length <= 0:
        raise ValueError("max_length must be a positive integer.")
    if not text:
        raise ValueError("text must not be empty.")

    if len(text) <= max_length:
        return [text]

    parts: list[str] = []
    remaining = text
    while len(remaining) > max_length:
        newline_index = remaining.rfind("\n", 0, max_length)
        cut = newline_index + 1 if newline_index > 0 else max_length
        parts.append(remaining[:cut])
        remaining = remaining[cut:]
    if remaining:
        parts.append(remaining)
    return parts


def format_integrated_answer(result: IntegratedAgentResult) -> str:
    """Render an IntegratedAgentResult as the deterministic Telegram response text."""
    if result.remember_command_detected:
        # Memory confirmations are self-contained; no source block applies.
        return result.answer

    lines = [result.answer, ""]
    if result.sources:
        lines.append("Источники:")
        for rank, source in enumerate(result.sources, start=1):
            lines.append(f"{rank}. {source.title} — {source.url}")
    elif "user_memory_recall" in result.tools_used and result.outcome == "success":
        # A recalled preference is user-provided memory, never documentation
        # or PyPI data, so it gets its own clearly labeled source type.
        lines.append(_MEMORY_SOURCE_LABEL)
    else:
        lines.append("Источники: не найдены")
    return "\n".join(lines)


class TelegramBotService:
    """Telegram-specific handlers wrapping IntegratedConversationAgentService.

    Holds no RAG, agent, or memory logic of its own; every message is handled
    through the injected integrated service, keyed by str(chat_id) as
    session_id.
    """

    def __init__(self, integrated_service: IntegratedConversationAgentService) -> None:
        self._integrated_service = integrated_service

    async def handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._reply(update, _START_MESSAGE)

    async def handle_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        session_id = self._session_id(update)
        self._integrated_service.reset(session_id)
        await self._reply(update, _RESET_CONFIRMATION)

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if message is None or not message.text:
            return
        question = message.text.strip()
        if not question:
            return

        session_id = self._session_id(update)

        try:
            chat = update.effective_chat
            if chat is not None:
                await context.bot.send_chat_action(chat_id=chat.id, action=ChatAction.TYPING)
        except Exception:
            pass  # the typing indicator is best-effort only

        try:
            with request_logging_context(session_id=session_id):
                result = self._integrated_service.handle_message(session_id, question)
                response_text = format_integrated_answer(result)
        except Exception as exc:
            # Expected domain failures arrive as IntegratedAgentError; anything
            # else is mapped to the same safe user-facing message so the bot
            # never leaks internals or crashes the handler.
            log_exception_safely(
                logger,
                (
                    "Telegram request failed "
                    f"session_hash={self._session_hash(session_id)} "
                    f"question_length={len(question)}"
                ),
                exc=exc,
            )
            response_text = _ANSWER_FAILURE_MESSAGE

        for part in split_telegram_message(response_text):
            await self._reply(update, part)

    @staticmethod
    def _session_id(update: Update) -> str:
        chat = update.effective_chat
        return str(chat.id if chat is not None else update.effective_message.chat_id)

    @staticmethod
    def _session_hash(session_id: str) -> str:
        return hash_session_id(session_id)

    @staticmethod
    async def _reply(update: Update, text: str) -> None:
        message = update.effective_message
        if message is None:
            return
        try:
            await message.reply_text(text)
        except Exception:
            pass  # Telegram send/runtime failure at the outer boundary: nothing more to do.


def build_application(
    settings: AppSettings | None = None,
    *,
    chat_model: BaseChatModel | None = None,
) -> Application:
    """Construct the Telegram Application with all handlers registered.

    Loads settings and builds the integrated agent/memory service graph only
    when this function is called explicitly; importing this module never does.
    `chat_model` is injectable so tests can drive the same factory graph with a
    scripted fake tool-calling model.
    """
    resolved_settings = settings or get_settings()
    integrated_service = build_integrated_service(resolved_settings, chat_model=chat_model)
    bot_service = TelegramBotService(integrated_service)

    application = (
        ApplicationBuilder()
        .token(resolved_settings.telegram_bot_token.get_secret_value())
        .build()
    )
    application.bot_data[_STARTUP_SUMMARY_BOT_DATA_KEY] = build_startup_summary(
        resolved_settings
    )
    application.add_handler(CommandHandler("start", bot_service.handle_start))
    application.add_handler(CommandHandler("reset", bot_service.handle_reset))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, bot_service.handle_text)
    )
    return application


def run_bot(settings: AppSettings | None = None) -> None:
    """Build the Telegram application and start long polling. Blocks until stopped."""
    application = build_application(settings)
    application.run_polling()


def build_startup_summary(settings: AppSettings) -> TelegramStartupSummary:
    """Build the safe startup summary logged by the live bot entry point."""
    return TelegramStartupSummary(
        pinecone_index_name=settings.pinecone_index_name,
        pinecone_namespace=settings.pinecone_documents_namespace,
        embedding_model=settings.openai_embedding_model,
        retrieval_top_k=settings.retrieval_top_k,
        score_threshold=get_retrieval_score_threshold(),
    )
