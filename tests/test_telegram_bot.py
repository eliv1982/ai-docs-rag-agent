"""Unit tests for the Telegram bot boundary.

Uses fake Telegram update/context objects and a fake ConversationAnswerService;
no real Telegram, OpenAI, or Pinecone calls, and no polling is ever started.
"""

import asyncio
import logging
from typing import Any

import pytest
from pydantic import ValidationError
from telegram.ext import CommandHandler, MessageHandler

from ai_docs_agent.agent import (
    AnswerGenerationError,
    AnswerRetrievalError,
    DocumentationAnswerService,
)
from ai_docs_agent.config import AppSettings
from ai_docs_agent.memory import ConversationAnswerService, InMemoryConversationMemory
from ai_docs_agent.models import (
    AnswerSource,
    ConversationMessage,
    GroundedAnswerResult,
    RetrievalResult,
    RetrievedChunk,
)
from ai_docs_agent.observability import hash_session_id
from ai_docs_agent.telegram_bot import (
    _STARTUP_SUMMARY_BOT_DATA_KEY,
    TelegramBotService,
    TelegramStartupSummary,
    build_application,
    build_startup_summary,
    format_answer,
    split_telegram_message,
)

_REQUIRED: dict[str, Any] = {
    "openai_api_key": "sk-test-openai",
    "pinecone_api_key": "pc-test-key",
    "openai_chat_model": "gpt-4o-mini",
    "telegram_bot_token": "test-telegram-token",
    "user_memory_hash_secret": "unit-test-user-memory-secret",
}


def make_settings(**overrides: Any) -> AppSettings:
    return AppSettings(_env_file=None, **{**_REQUIRED, **overrides})


def make_source(**overrides: Any) -> AnswerSource:
    defaults: dict[str, Any] = {
        "title": "Example Page",
        "url": "https://docs.example.com/page",
        "document_id": "doc-abc123",
        "chunk_index": 0,
        "chunk_count": 1,
    }
    return AnswerSource(**{**defaults, **overrides})


def make_result(**overrides: Any) -> GroundedAnswerResult:
    sources = overrides.pop("sources", (make_source(),))
    defaults: dict[str, Any] = {
        "question": "how do I configure the client?",
        "answer": "Set the API key via the documented environment variable.",
        "sources": sources,
        "retrieved_chunk_count": 1,
    }
    return GroundedAnswerResult(**{**defaults, **overrides})


def make_chunk(**overrides: Any) -> RetrievedChunk:
    defaults: dict[str, Any] = {
        "chunk_id": "doc-abc123-chunk-0000",
        "score": 0.9,
        "document_id": "doc-abc123",
        "source_url": "https://docs.example.com/page",
        "final_url": "https://docs.example.com/page",
        "title": "Example Page",
        "content_hash": "hash-value",
        "chunk_index": 0,
        "chunk_count": 1,
        "text": "The client is configured via the OPENAI_API_KEY environment variable.",
    }
    return RetrievedChunk(**{**defaults, **overrides})


def make_retrieval_result(**overrides: Any) -> RetrievalResult:
    matches = overrides.pop("matches", (make_chunk(),))
    defaults: dict[str, Any] = {
        "query": "how do I configure the client?",
        "namespace": "documentation",
        "top_k": 5,
        "matches": matches,
    }
    return RetrievalResult(**{**defaults, **overrides})


def run(coro: Any) -> Any:
    return asyncio.run(coro)


# --- fake Telegram objects ------------------------------------------------------


class FakeMessage:
    def __init__(self, text: str | None, *, chat_id: int = 123) -> None:
        self.text = text
        self.chat_id = chat_id
        self.replies: list[str] = []

    async def reply_text(self, text: str) -> None:
        self.replies.append(text)


class FakeChat:
    def __init__(self, chat_id: int) -> None:
        self.id = chat_id


class FakeUpdate:
    def __init__(self, *, message: FakeMessage | None, chat_id: int = 123) -> None:
        self.effective_message = message
        self.effective_chat = FakeChat(chat_id) if message is not None else None


class FakeBot:
    def __init__(self) -> None:
        self.typing_calls: list[int] = []

    async def send_chat_action(self, *, chat_id: int, action: str) -> None:
        self.typing_calls.append(chat_id)


class FakeContext:
    def __init__(self) -> None:
        self.bot = FakeBot()


class FakeConversationService:
    """Fake ConversationAnswerService recording the exact arguments it receives."""

    def __init__(
        self,
        *,
        result: GroundedAnswerResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self._result = result
        self._error = error
        self.answer_calls: list[tuple[str, str]] = []
        self.reset_calls: list[str] = []

    def answer(
        self,
        session_id: str,
        question: str,
        *,
        top_k: int | None = None,
        namespace: str | None = None,
    ) -> GroundedAnswerResult:
        self.answer_calls.append((session_id, question))
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result

    def reset(self, session_id: str) -> None:
        self.reset_calls.append(session_id)


class FactoryFakeRetrievalService:
    """Fake RetrievalService installed under the real Telegram factory."""

    instances: list["FactoryFakeRetrievalService"] = []

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self.calls: list[tuple[str, int | None, str | None]] = []
        FactoryFakeRetrievalService.instances.append(self)

    def search(
        self, query: str, *, top_k: int | None = None, namespace: str | None = None
    ) -> RetrievalResult:
        self.calls.append((query, top_k, namespace))
        resolved_query = query.strip()
        if resolved_query == _ALIAS_SET_QUESTION:
            return make_retrieval_result(
                query=resolved_query,
                matches=(_SPLITTER_DOC_CHUNK,),
            )
        if resolved_query == _ALIAS_FOLLOW_UP_QUESTION:
            return make_retrieval_result(
                query=resolved_query,
                matches=(
                    make_chunk(
                        chunk_id="low-score-1",
                        score=0.2007,
                        text="LEAK_DIRECT_LOW_SCORE_1",
                    ),
                    make_chunk(
                        chunk_id="low-score-2",
                        score=0.1597,
                        text="LEAK_DIRECT_LOW_SCORE_2",
                    ),
                ),
            )
        if "RecursiveCharacterTextSplitter" in resolved_query:
            return make_retrieval_result(
                query=resolved_query,
                matches=(_SPLITTER_DOC_CHUNK,),
            )
        return make_retrieval_result(query=resolved_query, matches=())


class FactoryFakeChatClient:
    """Fake OpenAIChatClient installed under the real Telegram factory."""

    instances: list["FactoryFakeChatClient"] = []

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self.calls: list[dict[str, str]] = []
        FactoryFakeChatClient.instances.append(self)

    def complete(self, *, model: str, system_prompt: str, user_prompt: str) -> str:
        self.calls.append(
            {"model": model, "system_prompt": system_prompt, "user_prompt": user_prompt}
        )
        if "standalone semantic-search query" in system_prompt:
            return _ALIAS_FOLLOW_UP_QUESTION
        if user_prompt.endswith(
            f"Current user message (respond to this message only):\n{_ALIAS_SET_QUESTION}"
        ):
            return _ALIAS_ACKNOWLEDGEMENT
        if user_prompt.endswith(
            f"Current user message (respond to this message only):\n{_ALIAS_FOLLOW_UP_QUESTION}"
        ):
            return _SPLITTER_GROUNDED_ANSWER
        if user_prompt.endswith(
            "Current user message (respond to this message only):\n"
            "Для чего нужен RecursiveCharacterTextSplitter?"
        ):
            return _SPLITTER_GROUNDED_ANSWER
        return "Fallback answer."


_ALIAS_SET_QUESTION = "В этом диалоге называй RecursiveCharacterTextSplitter Резаком."
_ALIAS_FOLLOW_UP_QUESTION = "Для чего нужен Резак?"
_ALIAS_ACKNOWLEDGEMENT = (
    "Хорошо, в рамках текущего диалога буду называть "
    "RecursiveCharacterTextSplitter Резаком."
)
_SPLITTER_GROUNDED_ANSWER = (
    "RecursiveCharacterTextSplitter нужен, чтобы рекурсивно разбивать текст на части."
)
_SPLITTER_DOC_CHUNK = make_chunk(
    score=0.62,
    title="Recursive Splitter Docs",
    final_url="https://docs.example.com/recursive-splitter",
    source_url="https://docs.example.com/recursive-splitter",
    document_id="doc-splitter",
    text="LEAK_SPLITTER_DOC_BODY RecursiveCharacterTextSplitter splits long text recursively.",
)


def build_live_factory_harness(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, TelegramBotService, FactoryFakeRetrievalService, FactoryFakeChatClient]:
    import ai_docs_agent.agent as agent_module

    FactoryFakeRetrievalService.instances = []
    FactoryFakeChatClient.instances = []
    monkeypatch.setattr(agent_module, "RetrievalService", FactoryFakeRetrievalService)
    monkeypatch.setattr(agent_module, "OpenAIChatClient", FactoryFakeChatClient)

    application = build_application(make_settings())
    handlers = application.handlers[0]
    message_handler = next(h for h in handlers if isinstance(h, MessageHandler))
    bot_service = message_handler.callback.__self__
    assert isinstance(bot_service, TelegramBotService)
    retrieval = FactoryFakeRetrievalService.instances[0]
    chat = FactoryFakeChatClient.instances[0]
    return application, bot_service, retrieval, chat


# --- /start ------------------------------------------------------------------------


def test_handle_start_sends_introduction() -> None:
    conversation = FakeConversationService(result=make_result())
    service = TelegramBotService(conversation)
    message = FakeMessage("/start")
    update = FakeUpdate(message=message, chat_id=1)

    run(service.handle_start(update, FakeContext()))

    assert len(message.replies) == 1
    assert "документац" in message.replies[0].lower()
    assert "/reset" in message.replies[0]
    assert "10" in message.replies[0]


# --- /reset --------------------------------------------------------------------------


def test_handle_reset_calls_reset_with_chat_id_session() -> None:
    conversation = FakeConversationService(result=make_result())
    service = TelegramBotService(conversation)
    update = FakeUpdate(message=FakeMessage("/reset"), chat_id=42)

    run(service.handle_reset(update, FakeContext()))

    assert conversation.reset_calls == ["42"]


def test_handle_reset_sends_confirmation() -> None:
    conversation = FakeConversationService(result=make_result())
    service = TelegramBotService(conversation)
    message = FakeMessage("/reset")
    update = FakeUpdate(message=message, chat_id=42)

    run(service.handle_reset(update, FakeContext()))

    assert message.replies == ["История текущего диалога очищена."]


def test_handle_reset_of_empty_session_succeeds() -> None:
    conversation = FakeConversationService(result=make_result())
    service = TelegramBotService(conversation)
    update = FakeUpdate(message=FakeMessage("/reset"), chat_id=999)

    run(service.handle_reset(update, FakeContext()))  # must not raise

    assert conversation.reset_calls == ["999"]


# --- ordinary text questions ---------------------------------------------------------


def test_handle_text_uses_correct_session_id() -> None:
    conversation = FakeConversationService(result=make_result())
    service = TelegramBotService(conversation)
    update = FakeUpdate(message=FakeMessage("What is LangChain?"), chat_id=777)

    run(service.handle_text(update, FakeContext()))

    assert conversation.answer_calls == [("777", "What is LangChain?")]


def test_handle_text_different_chats_remain_distinct() -> None:
    conversation = FakeConversationService(result=make_result())
    service = TelegramBotService(conversation)

    run(service.handle_text(FakeUpdate(message=FakeMessage("q1"), chat_id=1), FakeContext()))
    run(service.handle_text(FakeUpdate(message=FakeMessage("q2"), chat_id=2), FakeContext()))

    assert conversation.answer_calls == [("1", "q1"), ("2", "q2")]


def test_handle_text_strips_question() -> None:
    conversation = FakeConversationService(result=make_result())
    service = TelegramBotService(conversation)
    update = FakeUpdate(message=FakeMessage("   how do I configure it?   "), chat_id=1)

    run(service.handle_text(update, FakeContext()))

    assert conversation.answer_calls == [("1", "how do I configure it?")]


def test_handle_text_successful_answer_formatting() -> None:
    result = make_result(answer="The answer text.", sources=(make_source(),))
    conversation = FakeConversationService(result=result)
    service = TelegramBotService(conversation)
    message = FakeMessage("query")
    update = FakeUpdate(message=message, chat_id=1)

    run(service.handle_text(update, FakeContext()))

    assert len(message.replies) == 1
    reply = message.replies[0]
    assert "The answer text." in reply
    assert "Источники:" in reply
    assert "1. Example Page — https://docs.example.com/page" in reply


def test_handle_text_deterministic_source_order() -> None:
    sources = (
        make_source(title="Page A", url="https://docs.example.com/a"),
        make_source(title="Page B", url="https://docs.example.com/b"),
    )
    result = make_result(sources=sources, retrieved_chunk_count=2)
    conversation = FakeConversationService(result=result)
    service = TelegramBotService(conversation)
    message = FakeMessage("query")
    update = FakeUpdate(message=message, chat_id=1)

    run(service.handle_text(update, FakeContext()))

    reply = message.replies[0]
    assert reply.index("1. Page A") < reply.index("2. Page B")


def test_handle_text_empty_source_formatting() -> None:
    result = make_result(
        answer="В базе знаний не найдено достаточно информации для ответа на этот вопрос.",
        sources=(),
        retrieved_chunk_count=0,
    )
    conversation = FakeConversationService(result=result)
    service = TelegramBotService(conversation)
    message = FakeMessage("query")
    update = FakeUpdate(message=message, chat_id=1)

    run(service.handle_text(update, FakeContext()))

    assert "Источники: не найдены" in message.replies[0]


def test_handle_text_sends_typing_indicator() -> None:
    conversation = FakeConversationService(result=make_result())
    service = TelegramBotService(conversation)
    update = FakeUpdate(message=FakeMessage("query"), chat_id=55)
    context = FakeContext()

    run(service.handle_text(update, context))

    assert context.bot.typing_calls == [55]


def test_handle_text_retrieval_error_produces_safe_message() -> None:
    conversation = FakeConversationService(error=AnswerRetrievalError("boom"))
    service = TelegramBotService(conversation)
    message = FakeMessage("query")
    update = FakeUpdate(message=message, chat_id=1)

    run(service.handle_text(update, FakeContext()))

    assert message.replies == ["Не удалось подготовить ответ. Попробуйте повторить запрос позже."]


def test_handle_text_generation_error_produces_safe_message() -> None:
    conversation = FakeConversationService(error=AnswerGenerationError("boom"))
    service = TelegramBotService(conversation)
    message = FakeMessage("query")
    update = FakeUpdate(message=message, chat_id=1)

    run(service.handle_text(update, FakeContext()))

    assert message.replies == ["Не удалось подготовить ответ. Попробуйте повторить запрос позже."]


def test_handle_text_logs_service_exception_safely(
    caplog: pytest.LogCaptureFixture,
) -> None:
    conversation = FakeConversationService(
        error=AnswerRetrievalError("sk-super-secret-leaked pc-super-secret-leaked")
    )
    service = TelegramBotService(conversation)
    message = FakeMessage("How do I configure the client?", chat_id=424242)
    update = FakeUpdate(message=message, chat_id=424242)

    with caplog.at_level(logging.ERROR):
        run(service.handle_text(update, FakeContext()))

    assert message.replies == ["Не удалось подготовить ответ. Попробуйте повторить запрос позже."]
    assert "Traceback" in caplog.text
    assert "AnswerRetrievalError" in caplog.text
    assert "424242" not in caplog.text
    assert "How do I configure the client?" not in caplog.text
    assert "sk-super-secret-leaked" not in caplog.text
    assert "pc-super-secret-leaked" not in caplog.text
    assert hash_session_id("424242") in caplog.text


def test_handle_text_error_output_does_not_leak_secret_shaped_text() -> None:
    conversation = FakeConversationService(
        error=AnswerRetrievalError("sk-super-secret-leaked pc-super-secret-leaked")
    )
    service = TelegramBotService(conversation)
    message = FakeMessage("query")
    update = FakeUpdate(message=message, chat_id=1)

    run(service.handle_text(update, FakeContext()))

    reply = message.replies[0]
    assert "sk-super-secret-leaked" not in reply
    assert "pc-super-secret-leaked" not in reply
    assert "Traceback" not in reply


def test_handle_text_blank_text_is_ignored_safely() -> None:
    conversation = FakeConversationService(result=make_result())
    service = TelegramBotService(conversation)
    message = FakeMessage("   ")
    update = FakeUpdate(message=message, chat_id=1)

    run(service.handle_text(update, FakeContext()))

    assert conversation.answer_calls == []
    assert message.replies == []


def test_handle_text_non_text_message_is_ignored_safely() -> None:
    conversation = FakeConversationService(result=make_result())
    service = TelegramBotService(conversation)
    message = FakeMessage(None)  # e.g. a photo/sticker with no text
    update = FakeUpdate(message=message, chat_id=1)

    run(service.handle_text(update, FakeContext()))

    assert conversation.answer_calls == []
    assert message.replies == []


def test_handle_text_missing_message_is_ignored_safely() -> None:
    conversation = FakeConversationService(result=make_result())
    service = TelegramBotService(conversation)
    update = FakeUpdate(message=None)

    run(service.handle_text(update, FakeContext()))  # must not raise

    assert conversation.answer_calls == []


# --- format_answer ---------------------------------------------------------------


def test_format_answer_includes_sources_in_order() -> None:
    sources = (
        make_source(title="Page A", url="https://docs.example.com/a"),
        make_source(title="Page B", url="https://docs.example.com/b"),
    )
    result = make_result(answer="Answer text.", sources=sources, retrieved_chunk_count=2)

    text = format_answer(result)

    assert text.startswith("Answer text.")
    assert text.index("1. Page A") < text.index("2. Page B")


def test_format_answer_empty_sources() -> None:
    result = make_result(answer="Answer text.", sources=(), retrieved_chunk_count=0)

    text = format_answer(result)

    assert "Источники: не найдены" in text


# --- message splitting -----------------------------------------------------------


def test_split_short_text_remains_one_part() -> None:
    assert split_telegram_message("hello world") == ["hello world"]


def test_split_at_exact_limit_remains_one_part() -> None:
    text = "a" * 4000

    parts = split_telegram_message(text)

    assert parts == [text]


def test_split_over_limit_text_is_split() -> None:
    text = "a" * 4001

    parts = split_telegram_message(text)

    assert len(parts) == 2
    assert all(len(part) <= 4000 for part in parts)


def test_split_prefers_newline_boundary() -> None:
    text = ("x" * 10 + "\n") * 500  # well over 4000 chars, many newline opportunities

    parts = split_telegram_message(text, max_length=100)

    assert all(part.endswith("\n") or part == parts[-1] for part in parts[:-1])


def test_split_preserves_all_content() -> None:
    text = "line-one\n" * 1000

    parts = split_telegram_message(text, max_length=250)

    assert "".join(parts) == text


def test_split_preserves_order() -> None:
    text = "".join(f"segment-{i}\n" for i in range(500))

    parts = split_telegram_message(text, max_length=200)

    reconstructed = "".join(parts)
    assert reconstructed == text
    assert reconstructed.index("segment-0") < reconstructed.index("segment-499")


def test_split_never_returns_empty_parts() -> None:
    text = "y" * 9000

    parts = split_telegram_message(text, max_length=4000)

    assert all(len(part) > 0 for part in parts)


def test_split_very_long_unbroken_text_without_newlines() -> None:
    text = "z" * 12000

    parts = split_telegram_message(text, max_length=4000)

    assert len(parts) == 3
    assert "".join(parts) == text


def test_split_handles_unicode_text() -> None:
    text = ("документация русский текст " * 5 + "\n") * 200

    parts = split_telegram_message(text, max_length=500)

    assert "".join(parts) == text
    assert all(len(part) > 0 for part in parts)


def test_split_rejects_empty_text() -> None:
    with pytest.raises(ValueError):
        split_telegram_message("")


def test_split_rejects_non_positive_max_length() -> None:
    with pytest.raises(ValueError):
        split_telegram_message("hello", max_length=0)


# --- startup / factory -----------------------------------------------------------


def test_build_application_registers_all_handlers() -> None:
    settings = make_settings()

    application = build_application(settings)

    handlers = application.handlers[0]
    assert len(handlers) == 3
    command_handlers = [h for h in handlers if isinstance(h, CommandHandler)]
    message_handlers = [h for h in handlers if isinstance(h, MessageHandler)]
    assert len(message_handlers) == 1
    registered_commands = {command for h in command_handlers for command in h.commands}
    assert registered_commands == {"start", "reset"}


def test_build_application_records_safe_startup_summary() -> None:
    settings = make_settings(
        pinecone_index_name="docs-index",
        pinecone_documents_namespace="documentation-live-check",
        openai_embedding_model="text-embedding-3-small",
        retrieval_top_k=7,
    )

    application = build_application(settings)

    assert application.bot_data[_STARTUP_SUMMARY_BOT_DATA_KEY] == TelegramStartupSummary(
        pinecone_index_name="docs-index",
        pinecone_namespace="documentation-live-check",
        embedding_model="text-embedding-3-small",
        retrieval_top_k=7,
        score_threshold=build_startup_summary(settings).score_threshold,
    )


def test_build_application_uses_injected_settings_for_answer_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ai_docs_agent.telegram_bot as telegram_bot_module

    settings = make_settings(pinecone_documents_namespace="docs-live")
    captured: dict[str, Any] = {}

    class RecordingAnswerService:
        def __init__(self, received_settings: AppSettings) -> None:
            captured["settings"] = received_settings

    monkeypatch.setattr(
        telegram_bot_module,
        "DocumentationAnswerService",
        RecordingAnswerService,
    )

    build_application(settings)

    assert captured["settings"] is settings


def test_build_application_accepts_injected_settings_without_network() -> None:
    settings = make_settings(telegram_bot_token="another-fake-token")

    application = build_application(settings)  # must not raise or touch the network

    assert application is not None


def test_build_application_real_factory_alias_follow_up_uses_memory_and_contextual_retry(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    application, bot_service, retrieval, chat = build_live_factory_harness(monkeypatch)
    message_handler = next(
        h for h in application.handlers[0] if isinstance(h, MessageHandler)
    )

    assert isinstance(bot_service._conversation_service, ConversationAnswerService)
    assert isinstance(bot_service._conversation_service._memory, InMemoryConversationMemory)
    assert isinstance(
        bot_service._conversation_service._answer_service,
        DocumentationAnswerService,
    )

    alias_message = FakeMessage(_ALIAS_SET_QUESTION, chat_id=101)
    run(message_handler.callback(FakeUpdate(message=alias_message, chat_id=101), FakeContext()))

    history = bot_service._conversation_service._memory.get_history("101")
    assert isinstance(history, tuple)
    assert len(history) == 2
    assert all(isinstance(message, ConversationMessage) for message in history)
    assert [message.role for message in history] == ["user", "assistant"]

    follow_up_message = FakeMessage(_ALIAS_FOLLOW_UP_QUESTION, chat_id=101)
    with caplog.at_level(logging.INFO):
        run(
            message_handler.callback(
                FakeUpdate(message=follow_up_message, chat_id=101),
                FakeContext(),
            )
        )

    assert retrieval.calls[0] == (_ALIAS_SET_QUESTION, None, None)
    assert retrieval.calls[1] == (_ALIAS_FOLLOW_UP_QUESTION, None, None)
    assert len(retrieval.calls) == 3
    assert "RecursiveCharacterTextSplitter" in retrieval.calls[2][0]
    assert len(chat.calls) == 3
    assert _SPLITTER_GROUNDED_ANSWER in follow_up_message.replies[0]
    assert "https://docs.example.com/recursive-splitter" in follow_up_message.replies[0]
    assert "retrieval_pass=contextual" in caplog.text
    assert "history_message_count=2" in caplog.text
    assert "history_turn_count=1" in caplog.text
    assert "contextual_retry_eligible=true" in caplog.text
    assert "contextual_retry_reason=eligible" in caplog.text
    assert _ALIAS_SET_QUESTION not in caplog.text
    assert _ALIAS_FOLLOW_UP_QUESTION not in caplog.text
    assert "LEAK_SPLITTER_DOC_BODY" not in caplog.text
    assert "LEAK_DIRECT_LOW_SCORE_1" not in caplog.text
    assert "101" not in caplog.text


def test_build_application_real_factory_session_isolation_and_reset_block_alias_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application, bot_service, retrieval, _chat = build_live_factory_harness(monkeypatch)
    message_handler = next(
        h for h in application.handlers[0] if isinstance(h, MessageHandler)
    )
    reset_handler = next(
        h
        for h in application.handlers[0]
        if isinstance(h, CommandHandler) and "reset" in h.commands
    )

    run(
        message_handler.callback(
            FakeUpdate(message=FakeMessage(_ALIAS_SET_QUESTION, chat_id=1), chat_id=1),
            FakeContext(),
        )
    )

    other_chat_message = FakeMessage(_ALIAS_FOLLOW_UP_QUESTION, chat_id=2)
    run(
        message_handler.callback(
            FakeUpdate(message=other_chat_message, chat_id=2),
            FakeContext(),
        )
    )

    assert other_chat_message.replies
    assert "не найдено" in other_chat_message.replies[0]
    assert retrieval.calls == [
        (_ALIAS_SET_QUESTION, None, None),
        (_ALIAS_FOLLOW_UP_QUESTION, None, None),
    ]

    run(
        reset_handler.callback(
            FakeUpdate(message=FakeMessage("/reset", chat_id=1), chat_id=1),
            FakeContext(),
        )
    )
    assert bot_service._conversation_service._memory.get_history("1") == ()

    reset_follow_up_message = FakeMessage(_ALIAS_FOLLOW_UP_QUESTION, chat_id=1)
    run(
        message_handler.callback(
            FakeUpdate(message=reset_follow_up_message, chat_id=1),
            FakeContext(),
        )
    )

    assert "не найдено" in reset_follow_up_message.replies[0]
    assert retrieval.calls == [
        (_ALIAS_SET_QUESTION, None, None),
        (_ALIAS_FOLLOW_UP_QUESTION, None, None),
        (_ALIAS_FOLLOW_UP_QUESTION, None, None),
    ]


def test_build_application_real_factory_direct_full_name_success_uses_one_retrieval(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    application, _bot_service, retrieval, chat = build_live_factory_harness(monkeypatch)
    message_handler = next(
        h for h in application.handlers[0] if isinstance(h, MessageHandler)
    )

    full_name_message = FakeMessage(
        "Для чего нужен RecursiveCharacterTextSplitter?",
        chat_id=7,
    )
    with caplog.at_level(logging.INFO):
        run(
            message_handler.callback(
                FakeUpdate(message=full_name_message, chat_id=7),
                FakeContext(),
            )
        )

    assert retrieval.calls == [("Для чего нужен RecursiveCharacterTextSplitter?", None, None)]
    assert len(chat.calls) == 1
    assert _SPLITTER_GROUNDED_ANSWER in full_name_message.replies[0]
    assert "contextual_retry_eligible=false" in caplog.text
    assert "contextual_retry_reason=direct_context_found" in caplog.text


def test_build_application_fails_concisely_on_missing_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    from ai_docs_agent.config import get_settings

    for name in (
        "OPENAI_API_KEY",
        "PINECONE_API_KEY",
        "OPENAI_CHAT_MODEL",
        "TELEGRAM_BOT_TOKEN",
        "USER_MEMORY_HASH_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    monkeypatch.setenv("PINECONE_API_KEY", "pc-test-key")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("USER_MEMORY_HASH_SECRET", "env-user-memory-secret")
    # TELEGRAM_BOT_TOKEN intentionally left unset to exercise the failure path.
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()

    try:
        with pytest.raises(ValidationError) as exc_info:
            build_application()
    finally:
        get_settings.cache_clear()

    assert "TELEGRAM_BOT_TOKEN" in str(exc_info.value)
    assert "test-telegram-token" not in str(exc_info.value)


def test_import_creates_no_polling_or_global_state() -> None:
    import ai_docs_agent.telegram_bot as telegram_bot_module

    # Re-importing must not construct an Application, bot, or memory store at
    # module scope; only functions/classes are defined.
    assert not hasattr(telegram_bot_module, "application")
    assert not hasattr(telegram_bot_module, "bot")
