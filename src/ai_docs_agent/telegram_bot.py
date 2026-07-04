"""Minimal Telegram interface: chat_id -> session_id -> ConversationAnswerService.

Pipeline: Telegram text message -> str(chat_id) as session_id ->
ConversationAnswerService.answer() -> deterministic answer + source-list text ->
Telegram response, split into <=4000-character parts on newline boundaries
where practical. /start sends a short introduction; /reset clears only the
current chat's history via ConversationAnswerService.reset(). Only question
and answer text ever pass through the existing memory layer -- no Telegram
message objects, usernames, or other metadata are stored, and Pinecone
namespaces/top_k/model names are never exposed to chat users. No URL
indexing, tools, admin controls, or persistence are implemented here.

Importing this module builds nothing and makes no network calls: settings,
the answer/memory service stack, and the Telegram Application are all
constructed lazily by build_application()/run_bot(), which are the only
explicit startup entry points.
"""

import logging
from dataclasses import dataclass

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

from ai_docs_agent.agent import (
    AnswerServiceError,
    DocumentationAnswerService,
    get_retrieval_score_threshold,
)
from ai_docs_agent.config import AppSettings, get_settings
from ai_docs_agent.memory import ConversationAnswerService, InMemoryConversationMemory
from ai_docs_agent.models import GroundedAnswerResult
from ai_docs_agent.observability import (
    hash_session_id,
    log_exception_safely,
    request_logging_context,
)

_TELEGRAM_MAX_MESSAGE_LENGTH = 4000
_STARTUP_SUMMARY_BOT_DATA_KEY = "telegram_startup_summary"

_START_MESSAGE = (
    "Привет! Я отвечаю на вопросы по индексированной технической документации.\n\n"
    "Я запоминаю до 10 последних сообщений в текущем диалоге — ваши вопросы и мои ответы.\n\n"
    "Команда /reset очищает историю текущего диалога.\n"
    "После перезапуска бота история не сохраняется."
)

_RESET_CONFIRMATION = "История текущего диалога очищена."

_ANSWER_FAILURE_MESSAGE = "Не удалось подготовить ответ. Попробуйте повторить запрос позже."

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


def format_answer(result: GroundedAnswerResult) -> str:
    """Render a GroundedAnswerResult as the deterministic Telegram response text."""
    lines = [result.answer, ""]
    if not result.sources:
        lines.append("Источники: не найдены")
    else:
        lines.append("Источники:")
        for rank, source in enumerate(result.sources, start=1):
            lines.append(f"{rank}. {source.title} — {source.url}")
    return "\n".join(lines)


class TelegramBotService:
    """Telegram-specific handlers wrapping ConversationAnswerService.

    Holds no RAG or memory logic of its own; every question is answered
    through the injected ConversationAnswerService, keyed by
    str(chat_id) as session_id.
    """

    def __init__(self, conversation_service: ConversationAnswerService) -> None:
        self._conversation_service = conversation_service

    async def handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._reply(update, _START_MESSAGE)

    async def handle_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        session_id = self._session_id(update)
        self._conversation_service.reset(session_id)
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
                result = self._conversation_service.answer(session_id, question)
                response_text = format_answer(result)
        except AnswerServiceError as exc:
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


def build_application(settings: AppSettings | None = None) -> Application:
    """Construct the Telegram Application with all handlers registered.

    Loads settings and builds the RAG/memory service stack only when this
    function is called explicitly; importing this module never does.
    """
    resolved_settings = settings or get_settings()
    answer_service = DocumentationAnswerService(resolved_settings)
    memory = InMemoryConversationMemory()
    conversation_service = ConversationAnswerService(answer_service, memory=memory)
    bot_service = TelegramBotService(conversation_service)

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
