"""Integration-style tests for the final integrated agent flow (Stage 4I).

All tests use scripted fake tool-calling chat models and fake services only.
No real OpenAI, Pinecone, PyPI, Telegram, or DNS calls occur here.
"""

import logging
from collections.abc import Callable, Sequence
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from ai_docs_agent.config import AppSettings
from ai_docs_agent.integrated_agent import (
    IntegratedAgentError,
    IntegratedConversationAgentService,
)
from ai_docs_agent.langchain_agent import LangChainToolCallingAgent, build_langchain_tools
from ai_docs_agent.memory import InMemoryConversationMemory
from ai_docs_agent.models import (
    AnswerSource,
    ConversationMessage,
    GroundedAnswerResult,
    IntegratedAgentResult,
    PyPIPackageInfo,
    UserMemoryMatch,
    UserMemoryRecallResult,
    UserMemoryRecallToolInput,
    UserMemoryWriteResult,
)
from ai_docs_agent.observability import hash_session_id
from ai_docs_agent.user_memory import (
    InvalidMemoryStatementError,
    MemoryStorageError,
    statement_content_hash,
)

_REQUIRED: dict[str, Any] = {
    "openai_api_key": "sk-test-openai",
    "pinecone_api_key": "pc-test-key",
    "openai_chat_model": "gpt-4o-mini",
    "telegram_bot_token": "test-telegram-token",
    "user_memory_hash_secret": "unit-test-user-memory-secret",
}

_SESSION = "555777"
_OTHER_SESSION = "888999"
_REMEMBER_COMMAND = "Запомни: в примерах я предпочитаю httpx."
_PREFERENCE_STATEMENT = "в примерах я предпочитаю httpx."
_PREFERENCE_QUESTION = "Какую HTTP-библиотеку я предпочитаю?"
_DOCS_QUESTION = "Что такое embeddings в OpenAI API?"
_PYPI_QUESTION = "Какая последняя версия пакета httpx на PyPI?"
_BORSCHT_QUESTION = "Как сварить борщ?"
_ALIAS_SET_MESSAGE = "В этом диалоге называй RecursiveCharacterTextSplitter Резаком."
_ALIAS_FOLLOW_UP = "Для чего нужен Резак?"
_RESOLVED_ALIAS_QUESTION = "Для чего нужен RecursiveCharacterTextSplitter?"
_NO_CONTEXT_ANSWER = (
    "В базе знаний не найдено достаточно информации для ответа на этот вопрос."
)


def make_settings(**overrides: Any) -> AppSettings:
    return AppSettings(_env_file=None, **{**_REQUIRED, **overrides})


def make_source(**overrides: Any) -> AnswerSource:
    defaults: dict[str, Any] = {
        "title": "OpenAI Embeddings Guide",
        "url": "https://docs.example.com/embeddings",
        "document_id": "doc-embeddings",
        "chunk_index": 0,
        "chunk_count": 1,
    }
    return AnswerSource(**{**defaults, **overrides})


def make_grounded_result(**overrides: Any) -> GroundedAnswerResult:
    sources = overrides.pop("sources", (make_source(),))
    defaults: dict[str, Any] = {
        "question": _DOCS_QUESTION,
        "answer": "Embeddings are vector representations of text.",
        "sources": sources,
        "retrieved_chunk_count": len(sources),
    }
    return GroundedAnswerResult(**{**defaults, **overrides})


def make_no_context_result(question: str) -> GroundedAnswerResult:
    return GroundedAnswerResult(
        question=question,
        answer=_NO_CONTEXT_ANSWER,
        sources=(),
        retrieved_chunk_count=0,
    )


def make_pypi_info(**overrides: Any) -> PyPIPackageInfo:
    defaults: dict[str, Any] = {
        "package_name": "httpx",
        "latest_version": "9.9.9",
        "summary": "HTTP client for Python.",
        "requires_python": ">=3.8",
        "pypi_url": "https://pypi.org/project/httpx/",
        "project_url": "https://www.python-httpx.org/",
    }
    return PyPIPackageInfo(**{**defaults, **overrides})


def make_tool_call_message(name: str, args: dict[str, Any]) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": "call_1", "type": "tool_call"}],
    )


def _last_human_text(messages: Sequence[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    raise AssertionError("no HumanMessage found in agent input")


# --- fakes -------------------------------------------------------------------------


class ScriptedToolCallingModel(BaseChatModel):
    """Deterministic fake tool-calling chat model driven by a script callable."""

    script: Callable[[list[BaseMessage]], AIMessage] | None = None
    bound_tool_names: list[list[str]] = Field(default_factory=list)
    model_calls: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-tool-calling"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedToolCallingModel":
        self.bound_tool_names.append([tool.name for tool in tools])
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.model_calls.append(list(messages))
        assert self.script is not None
        message = self.script(list(messages))
        return ChatResult(generations=[ChatGeneration(message=message)])


def routing_script(messages: list[BaseMessage]) -> AIMessage:
    """Deterministic stand-in for the model's autonomous tool selection."""
    text = _last_human_text(messages)
    lowered = text.lower()
    if "версия" in lowered and ("пакет" in lowered or "pypi" in lowered):
        return make_tool_call_message("pypi_lookup", {"package_name": "httpx"})
    if "предпочитаю" in lowered or "предпочтени" in lowered:
        return make_tool_call_message("user_memory_recall", {"query": text})
    if "борщ" in lowered:
        return AIMessage(content="Борщ варят из свёклы, капусты и картофеля.")
    resolved = text
    if "Резак" in text:
        for message in messages:
            if (
                isinstance(message, HumanMessage)
                and "RecursiveCharacterTextSplitter" in str(message.content)
                and "Резак" in str(message.content)
            ):
                resolved = _RESOLVED_ALIAS_QUESTION
                break
    return make_tool_call_message("documentation_search", {"question": resolved})


class FakeDocumentationService:
    """Fake DocumentationAnswerService recording questions and supplied history."""

    def __init__(
        self,
        *,
        responder: Callable[[str], GroundedAnswerResult] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._responder = responder
        self._error = error
        self.calls: list[dict[str, Any]] = []

    def answer(
        self,
        question: str,
        *,
        history: Sequence[ConversationMessage] = (),
    ) -> GroundedAnswerResult:
        self.calls.append({"question": question, "history": tuple(history)})
        if self._error is not None:
            raise self._error
        if self._responder is not None:
            return self._responder(question)
        return make_grounded_result(question=question)


def documentation_responder(question: str) -> GroundedAnswerResult:
    if "RecursiveCharacterTextSplitter" in question:
        return make_grounded_result(
            question=question,
            answer="RecursiveCharacterTextSplitter рекурсивно разбивает текст на чанки.",
            sources=(
                make_source(
                    title="Recursive Splitter Docs",
                    url="https://docs.example.com/recursive-splitter",
                    document_id="doc-splitter",
                ),
            ),
        )
    if "embeddings" in question.lower():
        return make_grounded_result(question=question)
    return make_no_context_result(question)


class FakePyPIService:
    def __init__(
        self,
        *,
        result: PyPIPackageInfo | None = None,
        error: Exception | None = None,
    ) -> None:
        self._result = result or make_pypi_info()
        self._error = error
        self.calls: list[str] = []

    def lookup(self, package_name: str) -> PyPIPackageInfo:
        self.calls.append(package_name)
        if self._error is not None:
            raise self._error
        return self._result


class FakeUserMemoryService:
    """Fake UserMemoryService with a per-identifier in-memory store."""

    def __init__(
        self,
        *,
        remember_error: Exception | None = None,
        recall_error: Exception | None = None,
    ) -> None:
        self._remember_error = remember_error
        self._recall_error = recall_error
        self.remember_calls: list[tuple[str, str]] = []
        self.recall_calls: list[tuple[str, str]] = []
        self.store: dict[str, dict[str, str]] = {}

    def remember(self, user_identifier: str, statement: str) -> UserMemoryWriteResult:
        self.remember_calls.append((user_identifier, statement))
        if self._remember_error is not None:
            raise self._remember_error
        stripped = statement.strip()
        memory_id = "memory-" + statement_content_hash(stripped)[:32]
        user_store = self.store.setdefault(user_identifier, {})
        status = "duplicate" if memory_id in user_store else "created"
        user_store[memory_id] = stripped
        return UserMemoryWriteResult(
            status=status, memory_id=memory_id, identity_digest="abc123def456"
        )

    def recall(self, user_identifier: str, query: str) -> UserMemoryRecallResult:
        self.recall_calls.append((user_identifier, query))
        if self._recall_error is not None:
            raise self._recall_error
        stored = self.store.get(user_identifier, {})
        matches = tuple(
            UserMemoryMatch(
                memory_id=memory_id, text=text, score=0.91, memory_kind="user_memory"
            )
            for memory_id, text in list(stored.items())[:1]
        )
        return UserMemoryRecallResult(
            matches=matches,
            found=bool(matches),
            threshold=0.35,
            top_k=5,
            raw_candidate_count=len(matches),
            identity_digest="abc123def456",
        )


def make_integrated_service(
    *,
    script: Callable[[list[BaseMessage]], AIMessage] = routing_script,
    documentation_service: FakeDocumentationService | None = None,
    pypi_service: FakePyPIService | None = None,
    user_memory_service: FakeUserMemoryService | None = None,
) -> tuple[
    IntegratedConversationAgentService,
    ScriptedToolCallingModel,
    FakeDocumentationService,
    FakePyPIService,
    FakeUserMemoryService,
]:
    model = ScriptedToolCallingModel(script=script)
    documentation_service = documentation_service or FakeDocumentationService(
        responder=documentation_responder
    )
    pypi_service = pypi_service or FakePyPIService()
    user_memory_service = user_memory_service or FakeUserMemoryService()
    agent = LangChainToolCallingAgent(
        make_settings(),
        documentation_service=documentation_service,  # type: ignore[arg-type]
        pypi_service=pypi_service,  # type: ignore[arg-type]
        user_memory_service=user_memory_service,  # type: ignore[arg-type]
        chat_model=model,
    )
    service = IntegratedConversationAgentService(
        agent=agent,
        user_memory_service=user_memory_service,  # type: ignore[arg-type]
        memory=InMemoryConversationMemory(),
    )
    return service, model, documentation_service, pypi_service, user_memory_service


# --- tool wiring -------------------------------------------------------------------


def test_agent_binds_all_three_tools() -> None:
    service, model, _docs, _pypi, _memory = make_integrated_service()

    service.handle_message(_SESSION, _DOCS_QUESTION)

    assert model.bound_tool_names
    assert model.bound_tool_names[0] == [
        "documentation_search",
        "pypi_lookup",
        "user_memory_recall",
    ]


def test_memory_recall_tool_schema_contains_only_the_semantic_query() -> None:
    tools = build_langchain_tools(
        documentation_service=FakeDocumentationService(),  # type: ignore[arg-type]
        pypi_service=FakePyPIService(),  # type: ignore[arg-type]
        user_memory_service=FakeUserMemoryService(),  # type: ignore[arg-type]
    )
    memory_tool = next(tool for tool in tools if tool.name == "user_memory_recall")

    assert memory_tool.args_schema is UserMemoryRecallToolInput
    assert set(UserMemoryRecallToolInput.model_fields) == {"query"}


# --- documentation flow --------------------------------------------------------------


def test_documentation_question_selects_documentation_search() -> None:
    service, _model, docs, pypi, memory = make_integrated_service()

    result = service.handle_message(_SESSION, _DOCS_QUESTION)

    assert isinstance(result, IntegratedAgentResult)
    assert result.tools_used == ("documentation_search",)
    assert result.tool_call_count == 1
    assert result.outcome == "success"
    assert docs.calls[0]["question"] == _DOCS_QUESTION
    assert pypi.calls == []
    assert memory.recall_calls == []


def test_documentation_answer_preserves_real_document_sources() -> None:
    service, _model, _docs, _pypi, _memory = make_integrated_service()

    result = service.handle_message(_SESSION, _DOCS_QUESTION)

    assert result.sources == (make_source(),)
    assert result.sources[0].url == "https://docs.example.com/embeddings"


# --- PyPI flow ------------------------------------------------------------------------


def test_pypi_version_question_selects_pypi_lookup() -> None:
    service, _model, docs, pypi, _memory = make_integrated_service()

    result = service.handle_message(_SESSION, _PYPI_QUESTION)

    assert result.tools_used == ("pypi_lookup",)
    assert result.tool_call_count == 1
    assert pypi.calls == ["httpx"]
    assert docs.calls == []


def test_pypi_version_comes_from_tool_result_not_model_memory() -> None:
    service, _model, _docs, _pypi, _memory = make_integrated_service(
        pypi_service=FakePyPIService(result=make_pypi_info(latest_version="7.7.7"))
    )

    result = service.handle_message(_SESSION, _PYPI_QUESTION)

    assert "7.7.7" in result.answer


def test_pypi_url_is_preserved_in_sources() -> None:
    service, _model, _docs, _pypi, _memory = make_integrated_service()

    result = service.handle_message(_SESSION, _PYPI_QUESTION)

    assert result.sources[0].url == "https://pypi.org/project/httpx/"


# --- explicit remember ------------------------------------------------------------------


def test_explicit_russian_remember_writes_once_and_skips_agent() -> None:
    service, model, _docs, _pypi, memory = make_integrated_service()

    result = service.handle_message(_SESSION, _REMEMBER_COMMAND)

    assert result.remember_command_detected is True
    assert result.memory_written is True
    assert result.memory_write_status == "created"
    assert result.tools_used == ()
    assert result.tool_call_count == 0
    assert memory.remember_calls == [(_SESSION, _PREFERENCE_STATEMENT)]
    assert model.model_calls == []  # the agent was never invoked
    assert "сохранено" in result.answer.lower()


def test_duplicate_remember_returns_safe_duplicate_confirmation() -> None:
    service, _model, _docs, _pypi, memory = make_integrated_service()

    service.handle_message(_SESSION, _REMEMBER_COMMAND)
    result = service.handle_message(_SESSION, _REMEMBER_COMMAND)

    assert result.memory_write_status == "duplicate"
    assert result.memory_written is False
    assert len(memory.remember_calls) == 2
    assert len(memory.store[_SESSION]) == 1
    assert "уже" in result.answer.lower()


def test_english_remember_command_works() -> None:
    service, model, _docs, _pypi, memory = make_integrated_service()

    result = service.handle_message(_SESSION, "Remember: I prefer httpx in examples.")

    assert result.remember_command_detected is True
    assert result.memory_write_status == "created"
    assert memory.remember_calls == [(_SESSION, "I prefer httpx in examples.")]
    assert model.model_calls == []


def test_remember_confirmation_exposes_no_record_id_namespace_or_digest() -> None:
    service, _model, _docs, _pypi, _memory = make_integrated_service()

    result = service.handle_message(_SESSION, _REMEMBER_COMMAND)

    assert "memory-" not in result.answer
    assert "user-memory" not in result.answer
    assert "abc123def456" not in result.answer


@pytest.mark.parametrize(
    "message",
    [
        "Я предпочитаю httpx.",
        "Мне нравится requests.",
        "Расскажи про httpx.",
    ],
)
def test_ordinary_preference_statement_does_not_write_memory(message: str) -> None:
    service, _model, _docs, _pypi, memory = make_integrated_service()

    result = service.handle_message(_SESSION, message)

    assert result.remember_command_detected is False
    assert result.memory_write_status is None
    assert memory.remember_calls == []


def test_remember_prefix_without_statement_fails_safely() -> None:
    service, model, _docs, _pypi, _memory = make_integrated_service(
        user_memory_service=FakeUserMemoryService(
            remember_error=InvalidMemoryStatementError("blank")
        )
    )

    result = service.handle_message(_SESSION, "Запомни:   ")

    assert result.remember_command_detected is True
    assert result.outcome == "safe_fallback"
    assert result.failure_category == "invalid_memory_statement"
    assert model.model_calls == []


def test_remember_storage_failure_is_safe_and_does_not_corrupt_history() -> None:
    memory_service = FakeUserMemoryService(remember_error=MemoryStorageError("boom"))
    service, _model, _docs, _pypi, _memory = make_integrated_service(
        user_memory_service=memory_service
    )

    result = service.handle_message(_SESSION, _REMEMBER_COMMAND)

    assert result.outcome == "safe_fallback"
    assert result.failure_category == "memory_write_failure"
    assert "boom" not in result.answer
    assert service._memory.get_history(_SESSION) == ()


# --- memory recall through the agent ----------------------------------------------------


def test_preference_question_selects_user_memory_recall() -> None:
    service, _model, docs, pypi, memory = make_integrated_service()
    service.handle_message(_SESSION, _REMEMBER_COMMAND)

    result = service.handle_message(_SESSION, _PREFERENCE_QUESTION)

    assert result.tools_used == ("user_memory_recall",)
    assert result.tool_call_count == 1
    assert result.outcome == "success"
    assert "httpx" in result.answer
    assert docs.calls == []
    assert pypi.calls == []
    assert len(memory.recall_calls) == 1


def test_memory_recall_tool_receives_no_model_supplied_user_identifier() -> None:
    service, _model, _docs, _pypi, memory = make_integrated_service()
    service.handle_message(_SESSION, _REMEMBER_COMMAND)

    service.handle_message(_SESSION, _PREFERENCE_QUESTION)

    # The model's tool call carried only the semantic query; the trusted
    # session identifier was bound by application code outside the schema.
    identifier, query = memory.recall_calls[0]
    assert identifier == _SESSION
    assert query == _PREFERENCE_QUESTION
    assert _SESSION not in query


def test_trusted_session_identifier_is_bound_outside_tool_schema() -> None:
    service, model, _docs, _pypi, memory = make_integrated_service()
    service.handle_message(_SESSION, _REMEMBER_COMMAND)

    service.handle_message(_SESSION, _PREFERENCE_QUESTION)

    # No message shown to the model contains the raw session identifier.
    for call in model.model_calls:
        for message in call:
            assert _SESSION not in str(message.content)
    assert memory.recall_calls[0][0] == _SESSION


def test_different_sessions_remain_isolated() -> None:
    service, _model, _docs, _pypi, _memory = make_integrated_service()
    service.handle_message(_SESSION, _REMEMBER_COMMAND)

    other_result = service.handle_message(_OTHER_SESSION, _PREFERENCE_QUESTION)

    assert other_result.tools_used == ("user_memory_recall",)
    assert other_result.outcome == "safe_fallback"
    assert other_result.failure_category == "memory_no_match"
    assert "httpx" not in other_result.answer


def test_user_memory_is_not_mislabeled_as_documentation_or_pypi() -> None:
    service, _model, _docs, _pypi, _memory = make_integrated_service()
    service.handle_message(_SESSION, _REMEMBER_COMMAND)

    result = service.handle_message(_SESSION, _PREFERENCE_QUESTION)

    assert result.sources == ()  # no fabricated documentation/PyPI source
    assert "сохранённое предпочтение" in result.answer.lower()


def test_memory_recall_failure_is_mapped_safely() -> None:
    service, _model, _docs, _pypi, _memory = make_integrated_service(
        user_memory_service=FakeUserMemoryService(
            recall_error=MemoryStorageError("pinecone down")
        )
    )

    result = service.handle_message(_SESSION, _PREFERENCE_QUESTION)

    assert result.outcome == "safe_fallback"
    assert result.failure_category == "memory_recall_failure"
    assert "pinecone down" not in result.answer


# --- short-term memory, aliases, /reset ---------------------------------------------------


def test_alias_follow_up_resolves_through_short_term_history() -> None:
    service, _model, docs, _pypi, _memory = make_integrated_service()

    service.handle_message(_SESSION, _ALIAS_SET_MESSAGE)
    result = service.handle_message(_SESSION, _ALIAS_FOLLOW_UP)

    assert result.tools_used == ("documentation_search",)
    assert result.outcome == "success"
    assert "RecursiveCharacterTextSplitter" in result.answer
    assert docs.calls[-1]["question"] == _RESOLVED_ALIAS_QUESTION
    # The documentation tool received the short-term history via the trusted
    # request-scoped adapter.
    assert any(
        _ALIAS_SET_MESSAGE == message.content
        for message in docs.calls[-1]["history"]
    )


def test_reset_clears_short_term_alias_context() -> None:
    service, _model, docs, _pypi, _memory = make_integrated_service()
    service.handle_message(_SESSION, _ALIAS_SET_MESSAGE)

    service.reset(_SESSION)
    result = service.handle_message(_SESSION, _ALIAS_FOLLOW_UP)

    assert docs.calls[-1]["question"] == _ALIAS_FOLLOW_UP  # alias no longer resolves
    assert docs.calls[-1]["history"] == ()
    assert result.outcome == "safe_fallback"
    assert result.failure_category == "no_context"


def test_reset_preserves_persistent_memory() -> None:
    service, _model, _docs, _pypi, memory = make_integrated_service()
    service.handle_message(_SESSION, _REMEMBER_COMMAND)

    service.reset(_SESSION)
    result = service.handle_message(_SESSION, _PREFERENCE_QUESTION)

    assert service._memory.get_history(_SESSION) != ()  # only the new exchange
    assert memory.store[_SESSION]  # persistent record untouched by reset
    assert result.outcome == "success"
    assert "httpx" in result.answer


def test_documentation_history_is_not_exposed_as_a_source() -> None:
    service, _model, _docs, _pypi, _memory = make_integrated_service()

    service.handle_message(_SESSION, _ALIAS_SET_MESSAGE)
    result = service.handle_message(_SESSION, _ALIAS_FOLLOW_UP)

    assert all(
        source.url == "https://docs.example.com/recursive-splitter"
        for source in result.sources
    )


def test_successful_exchanges_are_added_to_short_term_history() -> None:
    service, _model, _docs, _pypi, _memory = make_integrated_service()

    service.handle_message(_SESSION, _DOCS_QUESTION)

    history = service._memory.get_history(_SESSION)
    assert [message.role for message in history] == ["user", "assistant"]
    assert history[0].content == _DOCS_QUESTION


def test_agent_failure_does_not_corrupt_history() -> None:
    def exploding_script(messages: list[BaseMessage]) -> AIMessage:
        raise RuntimeError("model exploded")

    service, _model, _docs, _pypi, _memory = make_integrated_service(
        script=exploding_script
    )

    with pytest.raises(IntegratedAgentError):
        service.handle_message(_SESSION, _DOCS_QUESTION)

    assert service._memory.get_history(_SESSION) == ()


# --- out-of-scope policy ---------------------------------------------------------------


def test_out_of_scope_question_returns_safe_fallback_without_sources() -> None:
    service, _model, _docs, _pypi, _memory = make_integrated_service()

    result = service.handle_message(_SESSION, _BORSCHT_QUESTION)

    assert result.outcome == "safe_fallback"
    assert result.failure_category == "out_of_scope_no_tool"
    assert result.sources == ()
    assert result.answer == _NO_CONTEXT_ANSWER
    assert "свёкл" not in result.answer  # the no-tool model answer never leaks


def test_no_tool_model_answer_cannot_bypass_out_of_scope_policy() -> None:
    def direct_answer_script(messages: list[BaseMessage]) -> AIMessage:
        return AIMessage(content="Pretrained knowledge answer that must not be shown.")

    service, _model, _docs, _pypi, _memory = make_integrated_service(
        script=direct_answer_script
    )

    result = service.handle_message(_SESSION, "Любой вопрос не по теме")

    assert result.used_no_tool is True
    assert result.outcome == "safe_fallback"
    assert "Pretrained knowledge" not in result.answer


# --- bounded execution ------------------------------------------------------------------


def test_clear_single_tool_request_invokes_only_one_tool() -> None:
    service, model, docs, pypi, memory = make_integrated_service()

    service.handle_message(_SESSION, _PYPI_QUESTION)

    assert pypi.calls == ["httpx"]
    assert docs.calls == []
    assert memory.recall_calls == []
    assert len(model.model_calls) == 1


def test_agent_execution_remains_bounded() -> None:
    service, model, _docs, _pypi, _memory = make_integrated_service()

    result = service.handle_message(_SESSION, _DOCS_QUESTION)

    # One model step selects the tool; the recursion limit stops the graph
    # after the single tool step, and the answer comes from the tool trace.
    assert len(model.model_calls) == 1
    assert result.tool_call_count == 1


# --- observability -----------------------------------------------------------------------


def test_logs_are_privacy_safe_and_show_tool_selection(
    caplog: pytest.LogCaptureFixture,
) -> None:
    service, _model, _docs, _pypi, _memory = make_integrated_service()

    with caplog.at_level(logging.INFO):
        service.handle_message(_SESSION, _REMEMBER_COMMAND)
        service.handle_message(_SESSION, _PREFERENCE_QUESTION)
        service.handle_message(_SESSION, _DOCS_QUESTION)

    log_text = caplog.text
    assert "remember_command=true" in log_text
    assert "memory_write_status=created" in log_text
    assert "user_memory_recall" in log_text
    assert "documentation_search" in log_text
    assert f"session_hash={hash_session_id(_SESSION)}" in log_text
    # Private content and identifiers never appear.
    assert _PREFERENCE_STATEMENT not in log_text
    assert "httpx" not in log_text
    assert _PREFERENCE_QUESTION not in log_text
    assert _DOCS_QUESTION not in log_text
    assert _SESSION not in log_text  # raw chat/session ID never appears
    assert "unit-test-user-memory-secret" not in log_text
    assert "vector=" not in log_text


# --- import safety -----------------------------------------------------------------------


def test_importing_integrated_module_performs_no_external_work() -> None:
    import importlib.util
    from pathlib import Path

    module_path = (
        Path(__file__).resolve().parents[1] / "src" / "ai_docs_agent" / "integrated_agent.py"
    )
    spec = importlib.util.spec_from_file_location("integrated_agent_import_check", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert not hasattr(module, "service")
    assert not hasattr(module, "agent")
    assert not hasattr(module, "settings")
