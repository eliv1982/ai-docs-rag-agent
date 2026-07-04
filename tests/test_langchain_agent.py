"""Unit tests for the real LangChain tool-calling agent layer.

All tests use fake chat models and fake services only. No real OpenAI, Pinecone, PyPI,
Telegram, or DNS calls occur here.
"""

import importlib
import json
import logging
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from pydantic import Field

from ai_docs_agent.agent import AnswerGenerationError
from ai_docs_agent.config import AppSettings
from ai_docs_agent.langchain_agent import (
    LangChainToolCallingAgent,
    build_langchain_tools,
)
from ai_docs_agent.models import (
    AnswerSource,
    GroundedAnswerResult,
    LangChainAgentResult,
    PyPIPackageInfo,
)
from ai_docs_agent.observability import hash_session_id
from ai_docs_agent.pypi import (
    InvalidPackageNameError,
    MalformedPyPIResponseError,
    PackageNotFoundError,
    PyPINetworkError,
    PyPITimeoutError,
    PyPIUpstreamHTTPError,
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
        "document_id": "doc-example",
        "chunk_index": 0,
        "chunk_count": 1,
    }
    return AnswerSource(**{**defaults, **overrides})


def make_grounded_result(**overrides: Any) -> GroundedAnswerResult:
    sources = overrides.pop("sources", (make_source(),))
    defaults: dict[str, Any] = {
        "question": "Что такое embeddings в OpenAI API?",
        "answer": "Embeddings are vector representations of text.",
        "sources": sources,
        "retrieved_chunk_count": len(sources),
    }
    return GroundedAnswerResult(**{**defaults, **overrides})


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


def make_tool_call_message(
    name: str,
    args: dict[str, Any],
    *,
    call_id: str = "call_1",
) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


class FakeDocumentationService:
    """Fake DocumentationAnswerService for tool-calling tests."""

    def __init__(
        self,
        *,
        result: GroundedAnswerResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self._result = result
        self._error = error
        self.calls: list[str] = []

    def answer(self, question: str) -> GroundedAnswerResult:
        self.calls.append(question)
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


class FakePyPIService:
    """Fake PyPILookupService for tool-calling tests."""

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


class ToolFriendlyFakeModel(FakeMessagesListChatModel):
    """Fake chat model that records LangChain's tool binding calls."""

    bind_tools_calls: list[dict[str, Any]] = Field(default_factory=list)

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ToolFriendlyFakeModel":
        self.bind_tools_calls.append(
            {
                "tool_names": [tool.name for tool in tools],
                "kwargs": kwargs,
            }
        )
        return self


def make_agent(
    *,
    model: ToolFriendlyFakeModel,
    documentation_service: FakeDocumentationService | None = None,
    pypi_service: FakePyPIService | None = None,
) -> tuple[LangChainToolCallingAgent, FakeDocumentationService, FakePyPIService]:
    documentation_service = documentation_service or FakeDocumentationService(
        result=make_grounded_result()
    )
    pypi_service = pypi_service or FakePyPIService(result=make_pypi_info())
    agent = LangChainToolCallingAgent(
        make_settings(),
        documentation_service=documentation_service,
        pypi_service=pypi_service,
        chat_model=model,
    )
    return agent, documentation_service, pypi_service


def test_pypi_version_question_calls_only_pypi_tool_and_uses_tool_result() -> None:
    model = ToolFriendlyFakeModel(
        responses=[
            make_tool_call_message("pypi_lookup", {"package_name": "httpx"}),
            AIMessage(content="This second response should never be needed."),
        ]
    )
    agent, documentation_service, pypi_service = make_agent(model=model)

    result = agent.answer("Какая последняя версия пакета httpx на PyPI?")

    assert result.tools_used == ("pypi_lookup",)
    assert result.tool_call_count == 1
    assert result.used_no_tool is False
    assert result.outcome == "success"
    assert "9.9.9" in result.answer
    assert pypi_service.calls == ["httpx"]
    assert documentation_service.calls == []
    assert result.sources[0].url == "https://pypi.org/project/httpx/"
    assert model.bind_tools_calls
    assert model.bind_tools_calls[0]["tool_names"] == [
        "documentation_search",
        "pypi_lookup",
    ]


def test_agent_blocks_silent_current_version_answer_without_tool_call() -> None:
    model = ToolFriendlyFakeModel(responses=[AIMessage(content="Latest version is 0.0.0.")])
    agent, documentation_service, pypi_service = make_agent(model=model)

    result = agent.answer("Какая последняя версия пакета httpx на PyPI?")

    assert result.used_no_tool is True
    assert result.tools_used == ()
    assert result.outcome == "safe_fallback"
    assert result.failure_category == "missing_required_tool_call"
    assert "0.0.0" not in result.answer
    assert pypi_service.calls == []
    assert documentation_service.calls == []


@pytest.mark.parametrize(
    ("question", "expected_question"),
    [
        ("Что такое embeddings в OpenAI API?", "Что такое embeddings в OpenAI API?"),
        ("Как работает semantic search в Pinecone?", "Как работает semantic search в Pinecone?"),
        (
            "Для чего нужен RecursiveCharacterTextSplitter?",
            "Для чего нужен RecursiveCharacterTextSplitter?",
        ),
    ],
)
def test_documentation_questions_call_only_documentation_tool(
    question: str,
    expected_question: str,
) -> None:
    source = make_source(title="Docs Page", url="https://docs.example.com/embeddings")
    grounded = make_grounded_result(
        question=expected_question,
        answer="Grounded documentation answer.",
        sources=(source,),
        retrieved_chunk_count=1,
    )
    model = ToolFriendlyFakeModel(
        responses=[
            make_tool_call_message("documentation_search", {"question": expected_question}),
            AIMessage(content="This second response should never be needed."),
        ]
    )
    agent, documentation_service, pypi_service = make_agent(
        model=model,
        documentation_service=FakeDocumentationService(result=grounded),
    )

    result = agent.answer(question)

    assert result.tools_used == ("documentation_search",)
    assert result.tool_call_count == 1
    assert result.answer == "Grounded documentation answer."
    assert result.sources == (source,)
    assert documentation_service.calls == [expected_question]
    assert pypi_service.calls == []


def test_unknown_package_still_calls_pypi_and_returns_safe_not_found_result() -> None:
    model = ToolFriendlyFakeModel(
        responses=[
            make_tool_call_message("pypi_lookup", {"package_name": "definitely-not-real-package"}),
        ]
    )
    agent, documentation_service, pypi_service = make_agent(
        model=model,
        pypi_service=FakePyPIService(error=PackageNotFoundError("not found")),
    )

    result = agent.answer("Какая последняя версия несуществующего пакета ...?")

    assert result.tools_used == ("pypi_lookup",)
    assert result.outcome == "safe_fallback"
    assert result.failure_category == "package_not_found"
    assert "не найден" in result.answer.lower()
    assert documentation_service.calls == []
    assert pypi_service.calls == ["definitely-not-real-package"]


def test_invalid_package_name_failure_is_safe() -> None:
    model = ToolFriendlyFakeModel(
        responses=[
            make_tool_call_message("pypi_lookup", {"package_name": "requests/httpx"}),
        ]
    )
    agent, _documentation_service, pypi_service = make_agent(
        model=model,
        pypi_service=FakePyPIService(error=InvalidPackageNameError("unsafe name")),
    )

    result = agent.answer("Какая последняя версия пакета requests/httpx на PyPI?")

    assert result.failure_category == "invalid_package_name"
    assert "requests/httpx" not in result.answer
    assert pypi_service.calls == ["requests/httpx"]


@pytest.mark.parametrize(
    ("error", "expected_category", "expected_fragment"),
    [
        (PyPITimeoutError("timed out"), "timeout", "PyPI"),
        (PyPINetworkError("dns failure"), "network_error", "PyPI"),
        (MalformedPyPIResponseError("bad payload"), "malformed_response", "PyPI"),
        (PyPIUpstreamHTTPError("500"), "upstream_http_error", "PyPI"),
    ],
)
def test_pypi_failures_are_mapped_safely(
    error: Exception,
    expected_category: str,
    expected_fragment: str,
) -> None:
    model = ToolFriendlyFakeModel(
        responses=[make_tool_call_message("pypi_lookup", {"package_name": "httpx"})]
    )
    agent, _documentation_service, pypi_service = make_agent(
        model=model,
        pypi_service=FakePyPIService(error=error),
    )

    result = agent.answer("Какая последняя версия пакета httpx на PyPI?")

    assert result.outcome == "safe_fallback"
    assert result.failure_category == expected_category
    assert expected_fragment in result.answer
    assert pypi_service.calls == ["httpx"]


def test_documentation_no_context_behavior_is_safe() -> None:
    model = ToolFriendlyFakeModel(
        responses=[
            make_tool_call_message("documentation_search", {"question": "Что такое embeddings?"}),
        ]
    )
    no_context_result = make_grounded_result(
        answer="В базе знаний не найдено достаточно информации для ответа на этот вопрос.",
        sources=(),
        retrieved_chunk_count=0,
    )
    agent, documentation_service, _pypi_service = make_agent(
        model=model,
        documentation_service=FakeDocumentationService(result=no_context_result),
    )

    result = agent.answer("Что такое embeddings?")

    assert result.outcome == "safe_fallback"
    assert result.failure_category == "no_context"
    assert result.sources == ()
    assert documentation_service.calls == ["Что такое embeddings?"]


def test_documentation_generation_failure_does_not_leak_exception_details() -> None:
    model = ToolFriendlyFakeModel(
        responses=[
            make_tool_call_message("documentation_search", {"question": "Что такое embeddings?"}),
        ]
    )
    agent, _documentation_service, _pypi_service = make_agent(
        model=model,
        documentation_service=FakeDocumentationService(
            error=AnswerGenerationError("LEAK_CHUNK_BODY sk-live-secret vector=[1,2,3]")
        ),
    )

    result = agent.answer("Что такое embeddings?")

    assert result.outcome == "safe_fallback"
    assert result.failure_category == "generation_failure"
    assert "LEAK_CHUNK_BODY" not in result.answer
    assert "sk-live-secret" not in result.answer
    assert "vector=" not in result.answer


def test_tool_outputs_do_not_expose_chunk_bodies_or_vectors() -> None:
    tools = build_langchain_tools(
        documentation_service=FakeDocumentationService(
            result=make_grounded_result(
                answer="Grounded answer only.",
                sources=(make_source(title="Docs", url="https://docs.example.com/page"),),
                retrieved_chunk_count=1,
            )
        ),
        pypi_service=FakePyPIService(result=make_pypi_info()),
    )
    documentation_tool = next(tool for tool in tools if tool.name == "documentation_search")

    payload = json.loads(documentation_tool.invoke({"question": "Что такое embeddings?"}))

    assert set(payload) == {"status", "answer", "sources", "context_found"}
    assert payload["answer"] == "Grounded answer only."
    assert "vector=" not in json.dumps(payload)
    assert "chunk body" not in json.dumps(payload).lower()


def test_agent_execution_is_bounded_to_one_tool_call() -> None:
    grounded = make_grounded_result(answer="Bounded answer.", retrieved_chunk_count=1)
    model = ToolFriendlyFakeModel(
        responses=[
            make_tool_call_message("documentation_search", {"question": "Q1"}),
            make_tool_call_message("documentation_search", {"question": "Q2"}, call_id="call_2"),
        ]
    )
    agent, documentation_service, _pypi_service = make_agent(
        model=model,
        documentation_service=FakeDocumentationService(result=grounded),
    )

    result = agent.answer("Что такое embeddings в OpenAI API?")

    assert result.tool_call_count == 1
    assert documentation_service.calls == ["Q1"]
    assert result.answer == "Bounded answer."


def test_logs_are_privacy_safe_and_include_tool_diagnostics(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = ToolFriendlyFakeModel(
        responses=[make_tool_call_message("pypi_lookup", {"package_name": "httpx"})]
    )
    agent, _documentation_service, _pypi_service = make_agent(model=model)

    with caplog.at_level(logging.INFO):
        result = agent.answer(
            "LEAK_QUESTION Какая последняя версия пакета httpx на PyPI?",
            session_id="chat-123456",
        )

    assert result.tools_used == ("pypi_lookup",)
    assert "session_hash=" + hash_session_id("chat-123456") in caplog.text
    assert "question_length=" in caplog.text
    assert "tool_call_count=1" in caplog.text
    assert "pypi_lookup" in caplog.text
    assert "LEAK_QUESTION" not in caplog.text
    assert "sk-" not in caplog.text
    assert "vector=" not in caplog.text
    assert "chunk body" not in caplog.text.lower()


def test_importing_module_does_not_construct_services_or_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import langchain_openai

    import ai_docs_agent.agent as agent_module
    import ai_docs_agent.langchain_agent as langchain_agent_module
    import ai_docs_agent.pypi as pypi_module

    def fail(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("constructor should not run at import time")

    monkeypatch.setattr(agent_module.DocumentationAnswerService, "__init__", fail)
    monkeypatch.setattr(pypi_module.PyPILookupService, "__init__", fail)
    monkeypatch.setattr(langchain_openai.ChatOpenAI, "__init__", fail)

    reloaded = importlib.reload(langchain_agent_module)

    assert hasattr(reloaded, "LangChainToolCallingAgent")


def test_agent_result_model_supports_safe_fallback_without_tool_use() -> None:
    result = LangChainAgentResult(
        question="Какая последняя версия пакета httpx на PyPI?",
        answer="Не удалось подтвердить актуальные данные пакета без обращения к PyPI.",
        sources=(),
        tools_used=(),
        tool_call_count=0,
        used_no_tool=True,
        outcome="safe_fallback",
        failure_category="missing_required_tool_call",
    )

    assert result.used_no_tool is True
