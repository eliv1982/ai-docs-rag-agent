"""Telegram runtime contract: non-blocking handlers and private-chat-only dispatch.

Two properties are exercised here, both fully offline (no Telegram, OpenAI,
Pinecone or PyPI call, and no polling):

* The synchronous integrated service must run off the asyncio event loop, with
  its request-scoped context (logging session hash, memory identity) still bound.
* Handlers must be reachable only by new private-chat messages. Real PTB
  ``Update`` objects are matched against the handlers registered by
  ``build_application`` using the same ``check_update`` rule PTB's
  ``Application.process_update`` applies, so group/supergroup/channel updates and
  every edited update provably never reach the agent, memory or reset.
"""

import asyncio
import threading
import time
from collections.abc import Callable
from functools import lru_cache
from typing import Any

import pytest
import telegram
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from telegram import Update

import ai_docs_agent.telegram_bot as telegram_bot_module
from ai_docs_agent.config import AppSettings
from ai_docs_agent.integrated_agent import IntegratedConversationAgentService
from ai_docs_agent.langchain_agent import LangChainToolCallingAgent
from ai_docs_agent.memory import InMemoryConversationMemory
from ai_docs_agent.models import (
    IntegratedAgentResult,
    UserMemoryRecallResult,
)
from ai_docs_agent.observability import current_request_session_hash, hash_session_id
from ai_docs_agent.telegram_bot import TelegramBotService, build_application

_REQUIRED: dict[str, Any] = {
    "openai_api_key": "sk-test-openai",
    "pinecone_api_key": "pc-test-key",
    "openai_chat_model": "gpt-4o-mini",
    "telegram_bot_token": "test-telegram-token",
    "user_memory_hash_secret": "unit-test-user-memory-secret",
}

_PRIVATE_CHAT_ID = 4242
_GROUP_CHAT_ID = -1001234567890
_REMEMBER_TEXT = "Запомни: в примерах я предпочитаю httpx."


def make_settings() -> AppSettings:
    return AppSettings(_env_file=None, **_REQUIRED)


def make_result(answer: str = "Ответ по документации.") -> IntegratedAgentResult:
    return IntegratedAgentResult(
        question="question",
        answer=answer,
        sources=(),
        tools_used=("documentation_search",),
        tool_call_count=1,
        used_no_tool=False,
        outcome="success",
        failure_category=None,
    )


# --- fakes -----------------------------------------------------------------------------


class FakeMessage:
    def __init__(self, text: str, *, chat_id: int) -> None:
        self.text = text
        self.chat_id = chat_id
        self.replies: list[str] = []

    async def reply_text(self, text: str) -> None:
        self.replies.append(text)


class FakeChat:
    def __init__(self, chat_id: int) -> None:
        self.id = chat_id


class FakeUpdate:
    def __init__(self, text: str, *, chat_id: int = _PRIVATE_CHAT_ID) -> None:
        self.effective_message = FakeMessage(text, chat_id=chat_id)
        self.effective_chat = FakeChat(chat_id)


class FakeBot:
    async def send_chat_action(self, *, chat_id: int, action: str) -> None:
        return None


class FakeContext:
    def __init__(self) -> None:
        self.bot = FakeBot()


class SpyIntegratedService:
    """Records every call; optionally blocks like the real synchronous pipeline."""

    def __init__(self, *, delay_seconds: float = 0.0) -> None:
        self._delay_seconds = delay_seconds
        self.handle_calls: list[tuple[str, str]] = []
        self.reset_calls: list[str] = []
        self.handler_thread_ids: list[int] = []
        self.observed_session_hashes: list[str | None] = []

    def handle_message(self, session_id: str, text: str) -> IntegratedAgentResult:
        self.handle_calls.append((session_id, text))
        self.handler_thread_ids.append(threading.get_ident())
        self.observed_session_hashes.append(current_request_session_hash())
        time.sleep(self._delay_seconds)  # blocking, like real network I/O
        return make_result()

    def reset(self, session_id: str) -> None:
        self.reset_calls.append(session_id)


# --- event-loop correctness -----------------------------------------------------------


def test_blocking_service_does_not_stall_the_event_loop() -> None:
    blocking_seconds = 0.8
    service = SpyIntegratedService(delay_seconds=blocking_seconds)
    bot_service = TelegramBotService(service)  # type: ignore[arg-type]
    update = FakeUpdate("Что такое embeddings?")

    async def scenario() -> tuple[int, list[float], float]:
        loop_thread_id = threading.get_ident()
        gaps: list[float] = []
        stop = asyncio.Event()

        async def ticker() -> None:
            last = time.monotonic()
            while not stop.is_set():
                await asyncio.sleep(0.01)
                now = time.monotonic()
                gaps.append(now - last)
                last = now

        ticker_task = asyncio.create_task(ticker())
        await asyncio.sleep(0.05)
        started = time.monotonic()
        await bot_service.handle_text(update, FakeContext())  # type: ignore[arg-type]
        elapsed = time.monotonic() - started
        stop.set()
        await ticker_task
        return loop_thread_id, gaps, elapsed

    loop_thread_id, gaps, elapsed = asyncio.run(scenario())

    # The blocking service really did block for its full duration...
    assert elapsed >= blocking_seconds * 0.9
    # ...on a worker thread, not on the event-loop thread (deterministic check)...
    assert len(service.handler_thread_ids) == 1
    assert service.handler_thread_ids[0] != loop_thread_id
    # ...so the loop kept ticking: run inline it would have gone silent for ~0.8s.
    assert len(gaps) >= 10
    assert max(gaps) < blocking_seconds / 2
    assert update.effective_message.replies == ["Ответ по документации.\n\nИсточники: не найдены"]


def test_request_logging_context_is_bound_inside_the_worker_thread() -> None:
    service = SpyIntegratedService()
    bot_service = TelegramBotService(service)  # type: ignore[arg-type]

    async def scenario() -> str | None:
        await bot_service.handle_text(
            FakeUpdate("вопрос", chat_id=_PRIVATE_CHAT_ID),
            FakeContext(),  # type: ignore[arg-type]
        )
        return current_request_session_hash()

    leaked_into_loop = asyncio.run(scenario())

    assert service.observed_session_hashes == [hash_session_id(str(_PRIVATE_CHAT_ID))]
    assert leaked_into_loop is None  # the loop's own context is never polluted


class _ScriptedRecallModel(BaseChatModel):
    """Always asks for the request-scoped user_memory_recall tool."""

    @property
    def _llm_type(self) -> str:
        return "scripted-recall"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_ScriptedRecallModel":
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "user_memory_recall",
                    "args": {"query": "какая библиотека"},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


class _RecordingMemoryService:
    def __init__(self) -> None:
        self.recall_identifiers: list[str] = []
        self.recall_threads: list[int] = []

    def recall(self, user_identifier: str, query: str) -> UserMemoryRecallResult:
        self.recall_identifiers.append(user_identifier)
        self.recall_threads.append(threading.get_ident())
        return UserMemoryRecallResult(
            matches=(),
            found=False,
            threshold=0.35,
            top_k=5,
            raw_candidate_count=0,
            identity_digest="abc123def456",
        )

    def remember(self, user_identifier: str, statement: str) -> Any:
        raise AssertionError("not used by this test")


def test_memory_identity_survives_the_thread_hop_through_the_real_agent() -> None:
    memory_service = _RecordingMemoryService()
    agent = LangChainToolCallingAgent(
        make_settings(),
        documentation_service=object(),  # type: ignore[arg-type]
        pypi_service=object(),  # type: ignore[arg-type]
        user_memory_service=memory_service,  # type: ignore[arg-type]
        chat_model=_ScriptedRecallModel(),
    )
    integrated = IntegratedConversationAgentService(
        agent=agent,
        user_memory_service=memory_service,  # type: ignore[arg-type]
        memory=InMemoryConversationMemory(),
    )
    bot_service = TelegramBotService(integrated)
    update = FakeUpdate("Какую библиотеку я предпочитаю?", chat_id=_PRIVATE_CHAT_ID)

    async def scenario() -> int:
        await bot_service.handle_text(update, FakeContext())  # type: ignore[arg-type]
        return threading.get_ident()

    loop_thread_id = asyncio.run(scenario())

    # The trusted identity bound by the service is the private chat id, and the
    # recall tool still saw it although the whole pipeline ran off the loop.
    assert memory_service.recall_identifiers == [str(_PRIVATE_CHAT_ID)]
    assert memory_service.recall_threads[0] != loop_thread_id
    assert update.effective_message.replies  # an answer was still delivered


# --- private-chat-only dispatch via real PTB updates ----------------------------------


class _OfflineBot(telegram.Bot):
    """A real PTB Bot that knows its own username without calling get_me().

    CommandHandler needs the bot username to resolve '/cmd@botname'; nothing here
    ever performs a network request.
    """

    @property
    def bot(self) -> telegram.User:
        return telegram.User(id=1, first_name="Bot", is_bot=True, username="test_bot")


@lru_cache(maxsize=1)
def _offline_bot() -> _OfflineBot:
    return _OfflineBot("123456:TEST-offline-token")


def _command_entity(command: str) -> list[dict[str, Any]]:
    return [{"type": "bot_command", "offset": 0, "length": len(command)}]


def make_update(
    kind: str,
    chat_type: str,
    text: str,
    *,
    command: bool = False,
) -> Update:
    """Build a real PTB Update of `kind` (message/edited_message/channel_post/...)."""
    chat_id = _PRIVATE_CHAT_ID if chat_type == "private" else _GROUP_CHAT_ID
    payload: dict[str, Any] = {
        "message_id": 10,
        "date": 1_700_000_000,
        "chat": {"id": chat_id, "type": chat_type},
        "text": text,
    }
    if kind.endswith("message"):
        payload["from"] = {"id": _PRIVATE_CHAT_ID, "is_bot": False, "first_name": "Test"}
    if command:
        payload["entities"] = _command_entity(text.split()[0])
    return Update.de_json({"update_id": 1, kind: payload}, _offline_bot())


@pytest.fixture
def dispatch_app(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, SpyIntegratedService, list[tuple[int, str]]]:
    """The real application factory wired to a spy service; replies are recorded."""
    spy = SpyIntegratedService()
    monkeypatch.setattr(
        telegram_bot_module,
        "build_integrated_service",
        lambda settings, chat_model=None: spy,
    )
    replies: list[tuple[int, str]] = []

    async def fake_reply_text(self: telegram.Message, text: str, *args: Any, **kwargs: Any) -> None:
        replies.append((self.chat_id, text))

    monkeypatch.setattr(telegram.Message, "reply_text", fake_reply_text)
    application = build_application(make_settings())
    return application, spy, replies


def dispatch(application: Any, update: Update) -> Callable[..., Any] | None:
    """Return the callback PTB would run for `update`, mirroring process_update."""
    for group in sorted(application.handlers):
        for handler in application.handlers[group]:
            check = handler.check_update(update)
            if check is not None and check is not False:
                return handler.callback
    return None


async def feed(application: Any, update: Update) -> bool:
    callback = dispatch(application, update)
    if callback is None:
        return False
    await callback(update, FakeContext())
    return True


_ACCEPTED = [
    pytest.param(("message", "private", "Как настроить клиент?", False), id="private-text"),
    pytest.param(("message", "private", _REMEMBER_TEXT, False), id="private-memory-write"),
    pytest.param(("message", "private", "/reset", True), id="private-reset"),
    pytest.param(("message", "private", "/start", True), id="private-start"),
]

_IGNORED = [
    pytest.param(("message", "group", "Как настроить клиент?", False), id="group-text"),
    pytest.param(("message", "supergroup", "Как настроить клиент?", False), id="supergroup-text"),
    pytest.param(("channel_post", "channel", "Как настроить клиент?", False), id="channel-post"),
    pytest.param(
        ("edited_channel_post", "channel", "Как настроить клиент?", False),
        id="edited-channel-post",
    ),
    pytest.param(
        ("edited_message", "private", "Как настроить клиент?", False), id="edited-private"
    ),
    pytest.param(("edited_message", "group", "Как настроить клиент?", False), id="edited-group"),
    pytest.param(("message", "group", _REMEMBER_TEXT, False), id="group-memory-write"),
    pytest.param(("message", "supergroup", _REMEMBER_TEXT, False), id="supergroup-memory-write"),
    pytest.param(("edited_message", "private", _REMEMBER_TEXT, False), id="edited-memory-write"),
    pytest.param(("message", "group", "/reset", True), id="group-reset"),
    pytest.param(("message", "supergroup", "/reset", True), id="supergroup-reset"),
    pytest.param(("edited_message", "private", "/reset", True), id="edited-private-reset"),
    pytest.param(("edited_message", "group", "/reset", True), id="edited-group-reset"),
    pytest.param(("message", "group", "/start", True), id="group-start"),
]


@pytest.mark.parametrize("case", _ACCEPTED)
def test_new_private_messages_are_dispatched(
    dispatch_app: tuple[Any, SpyIntegratedService, list[tuple[int, str]]],
    case: tuple[str, str, str, bool],
) -> None:
    application, _spy, _replies = dispatch_app
    kind, chat_type, text, is_command = case

    callback = dispatch(application, make_update(kind, chat_type, text, command=is_command))

    assert callback is not None


def test_private_text_memory_write_and_reset_reach_the_service(
    dispatch_app: tuple[Any, SpyIntegratedService, list[tuple[int, str]]],
) -> None:
    application, spy, replies = dispatch_app

    async def scenario() -> list[bool]:
        return [
            await feed(application, make_update("message", "private", "Как настроить клиент?")),
            await feed(application, make_update("message", "private", _REMEMBER_TEXT)),
            await feed(
                application, make_update("message", "private", "/reset", command=True)
            ),
        ]

    handled = asyncio.run(scenario())

    assert handled == [True, True, True]
    chat = str(_PRIVATE_CHAT_ID)
    assert spy.handle_calls == [(chat, "Как настроить клиент?"), (chat, _REMEMBER_TEXT)]
    assert spy.reset_calls == [chat]
    assert len(replies) == 3  # two answers plus the reset confirmation
    assert all(chat_id == _PRIVATE_CHAT_ID for chat_id, _text in replies)


@pytest.mark.parametrize("case", _IGNORED)
def test_group_channel_and_edited_updates_are_not_dispatched(
    dispatch_app: tuple[Any, SpyIntegratedService, list[tuple[int, str]]],
    case: tuple[str, str, str, bool],
) -> None:
    application, spy, replies = dispatch_app
    kind, chat_type, text, is_command = case

    handled = asyncio.run(
        feed(application, make_update(kind, chat_type, text, command=is_command))
    )

    # No handler matches, so PTB would run nothing: the agent, memory writes,
    # short-term history and reset are all untouched and nothing is sent.
    assert handled is False
    assert spy.handle_calls == []
    assert spy.reset_calls == []
    assert replies == []


def test_ignored_updates_leave_a_real_session_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the real service graph, an ignored update cannot add to short-term history."""
    memory = InMemoryConversationMemory()
    memory_service = _RecordingMemoryService()
    agent = LangChainToolCallingAgent(
        make_settings(),
        documentation_service=object(),  # type: ignore[arg-type]
        pypi_service=object(),  # type: ignore[arg-type]
        user_memory_service=memory_service,  # type: ignore[arg-type]
        chat_model=_ScriptedRecallModel(),
    )
    integrated = IntegratedConversationAgentService(
        agent=agent,
        user_memory_service=memory_service,  # type: ignore[arg-type]
        memory=memory,
    )
    monkeypatch.setattr(
        telegram_bot_module,
        "build_integrated_service",
        lambda settings, chat_model=None: integrated,
    )
    application = build_application(make_settings())
    group_update = make_update("message", "group", "Какую библиотеку я предпочитаю?")

    handled = asyncio.run(feed(application, group_update))

    assert handled is False
    assert memory.get_history(str(_GROUP_CHAT_ID)) == ()
    assert memory.get_history(str(_PRIVATE_CHAT_ID)) == ()
    assert memory_service.recall_identifiers == []

