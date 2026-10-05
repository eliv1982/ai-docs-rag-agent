"""Unit tests for the real LangChain tool-calling agent layer.

All tests use fake chat models and fake services only. No real OpenAI, Pinecone, PyPI,
Telegram, or DNS calls occur here.
"""

import importlib
import json
import logging
from typing import Any

import httpx
import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from pydantic import Field

from ai_docs_agent.agent import AnswerGenerationError
from ai_docs_agent.config import AppSettings
from ai_docs_agent.langchain_agent import (
    LangChainAgentExecutionError,
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
        self.history_calls: list[tuple[Any, ...]] = []

    def answer(
        self, question: str, *, history: tuple[Any, ...] = ()
    ) -> GroundedAnswerResult:
        self.calls.append(question)
        self.history_calls.append(tuple(history))
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


class CountingFakeModel(ToolFriendlyFakeModel):
    """Counts real model steps (FakeMessagesListChatModel.i cycles, so it cannot)."""

    generate_calls: int = 0

    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        self.generate_calls += 1
        return super()._generate(*args, **kwargs)


class _GraphSpy:
    """Delegates to the real compiled create_agent graph, recording invoke() traffic."""

    def __init__(self, graph: Any) -> None:
        self._graph = graph
        self.configs: list[Any] = []
        self.errors: list[BaseException] = []

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        self.configs.append(config)
        try:
            return self._graph.invoke(input, config=config, **kwargs)
        except BaseException as exc:
            self.errors.append(exc)
            raise


def _bounded_agent(
    model: ToolFriendlyFakeModel,
    documentation_service: FakeDocumentationService,
) -> tuple[LangChainToolCallingAgent, _GraphSpy]:
    spies: list[_GraphSpy] = []

    def spying_create_agent(**kwargs: Any) -> _GraphSpy:
        spies.append(_GraphSpy(create_agent(**kwargs)))
        return spies[0]

    agent = LangChainToolCallingAgent(
        make_settings(),
        documentation_service=documentation_service,
        pypi_service=FakePyPIService(result=make_pypi_info()),
        chat_model=model,
        agent_factory=spying_create_agent,
    )
    return agent, spies[0]


def test_agent_execution_is_bounded_to_one_tool_call() -> None:
    """Contract: recursion_limit=2 means one model step plus one tool step, then stop.

    The graph is the real create_agent graph. A model that would happily keep
    calling tools (and has a third scripted step ready) gets exactly one model
    step and one tool execution; the resulting GraphRecursionError is raised by
    LangGraph and absorbed into the normal tool-trace result.
    """
    grounded = make_grounded_result(answer="Bounded answer.", retrieved_chunk_count=1)
    model = CountingFakeModel(
        responses=[
            make_tool_call_message("documentation_search", {"question": "Q1"}),
            make_tool_call_message("documentation_search", {"question": "Q2"}, call_id="call_2"),
            AIMessage(content="A third model step must never run."),
        ]
    )
    documentation_service = FakeDocumentationService(result=grounded)
    agent, spy = _bounded_agent(model, documentation_service)

    result = agent.answer("Что такое embeddings в OpenAI API?")

    assert model.generate_calls == 1  # no second model step after the tool step
    assert documentation_service.calls == ["Q1"]  # Q2 was never executed
    assert result.tool_call_count == 1
    assert result.outcome == "success"
    assert result.answer == "Bounded answer."
    assert "third model step" not in result.answer
    assert spy.configs == [{"recursion_limit": 2}]
    assert [type(error) for error in spy.errors] == [GraphRecursionError]


@pytest.mark.parametrize(
    "tool_call",
    [
        pytest.param(("no_such_tool", {"question": "Q1"}), id="unknown-tool"),
        pytest.param(("documentation_search", {"wrong_argument": "Q1"}), id="invalid-arguments"),
    ],
)
def test_recursion_limit_without_a_tool_result_gives_the_iteration_limit_fallback(
    tool_call: tuple[str, dict[str, Any]],
) -> None:
    """A model step the tools node cannot satisfy must not buy a second model step.

    The tools node answers with an error message instead of running a tool, so
    the graph would loop back to the model; the recursion limit stops it first.
    With no tool trace the agent returns its fixed iteration-limit fallback.
    """
    model = CountingFakeModel(
        responses=[
            make_tool_call_message(*tool_call),
            make_tool_call_message("documentation_search", {"question": "Q2"}, call_id="call_2"),
        ]
    )
    documentation_service = FakeDocumentationService(result=make_grounded_result())
    agent, spy = _bounded_agent(model, documentation_service)

    result = agent.answer("Что такое embeddings в OpenAI API?")

    assert model.generate_calls == 1
    assert documentation_service.calls == []
    assert [type(error) for error in spy.errors] == [GraphRecursionError]
    assert result.outcome == "safe_fallback"
    assert result.failure_category == "agent_iteration_limit"
    assert result.answer == "Не удалось безопасно подготовить ответ для этого запроса."
    assert result.used_no_tool is True
    assert result.tool_call_count == 0
    assert result.tools_used == ()
    assert result.sources == ()


def test_only_graph_recursion_is_absorbed_other_graph_failures_are_wrapped() -> None:
    """The recursion handler is narrow: any other graph error is not turned into a result."""

    class BrokenGraph:
        def invoke(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("graph exploded")

    agent = LangChainToolCallingAgent(
        make_settings(),
        documentation_service=FakeDocumentationService(result=make_grounded_result()),
        pypi_service=FakePyPIService(result=make_pypi_info()),
        chat_model=ToolFriendlyFakeModel(responses=[AIMessage(content="unused")]),
        agent_factory=lambda **kwargs: BrokenGraph(),
    )

    with pytest.raises(LangChainAgentExecutionError):
        agent.answer("Что такое embeddings в OpenAI API?")


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


# --- one-tool-call contract (real create_agent graph) ----------------------------------

_MULTI_CALL_CATEGORY = "multiple_tool_calls"


def make_multi_tool_call_message(*calls: tuple[str, dict[str, Any]]) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": args, "id": f"call_{index}", "type": "tool_call"}
            for index, (name, args) in enumerate(calls, start=1)
        ],
    )


_DOCS_CALL = ("documentation_search", {"question": "Что такое embeddings?"})
_PYPI_CALL = ("pypi_lookup", {"package_name": "httpx"})
_PARALLEL_CALL_ORDERS = [
    pytest.param([_DOCS_CALL, _PYPI_CALL], id="docs-then-pypi"),
    pytest.param([_PYPI_CALL, _DOCS_CALL], id="pypi-then-docs"),
    pytest.param([_DOCS_CALL, _DOCS_CALL], id="docs-twice"),
]


def _agent_with_parallel_attempt(
    calls: list[tuple[str, dict[str, Any]]],
    *,
    agent_factory: Any = create_agent,
) -> tuple[LangChainToolCallingAgent, ToolFriendlyFakeModel, Any, Any]:
    model = ToolFriendlyFakeModel(
        responses=[
            make_multi_tool_call_message(*calls),
            AIMessage(content="This second response should never be needed."),
        ]
    )
    documentation_service = FakeDocumentationService(result=make_grounded_result())
    pypi_service = FakePyPIService(result=make_pypi_info())
    agent = LangChainToolCallingAgent(
        make_settings(),
        documentation_service=documentation_service,
        pypi_service=pypi_service,
        chat_model=model,
        agent_factory=agent_factory,
    )
    return agent, model, documentation_service, pypi_service


def test_agent_asks_the_provider_not_to_emit_parallel_tool_calls() -> None:
    model = ToolFriendlyFakeModel(
        responses=[make_tool_call_message("pypi_lookup", {"package_name": "httpx"})]
    )
    agent, _documentation_service, _pypi_service = make_agent(model=model)

    result = agent.answer("Какая последняя версия пакета httpx на PyPI?")

    assert result.outcome == "success"
    assert model.bind_tools_calls
    assert model.bind_tools_calls[0]["kwargs"]["parallel_tool_calls"] is False


@pytest.mark.parametrize("calls", _PARALLEL_CALL_ORDERS)
def test_parallel_tool_calls_are_rejected_before_any_tool_executes(
    calls: list[tuple[str, dict[str, Any]]],
) -> None:
    agent, _model, documentation_service, pypi_service = _agent_with_parallel_attempt(calls)

    result = agent.answer("Что такое embeddings и какая последняя версия httpx на PyPI?")

    # Nothing ran, so no side effect can have happened, and the answer is the
    # fixed safe fallback rather than whichever call happened to be first/last.
    assert documentation_service.calls == []
    assert pypi_service.calls == []
    assert result.outcome == "safe_fallback"
    assert result.failure_category == _MULTI_CALL_CATEGORY
    assert result.tool_call_count == 0
    assert result.tools_used == ()
    assert result.used_no_tool is True
    assert result.sources == ()
    assert "9.9.9" not in result.answer
    assert "Embeddings" not in result.answer


def test_rejected_parallel_tool_calls_give_the_same_result_in_any_order() -> None:
    results = []
    for calls in ([_DOCS_CALL, _PYPI_CALL], [_PYPI_CALL, _DOCS_CALL]):
        agent, *_services = _agent_with_parallel_attempt(calls)
        results.append(agent.answer("Что такое embeddings и какая версия httpx на PyPI?"))

    assert results[0] == results[1]


def _create_agent_without_guard(**kwargs: Any) -> Any:
    kwargs.pop("middleware")
    return create_agent(**kwargs)


def test_defensive_fallback_is_order_independent_even_if_parallel_calls_execute() -> None:
    """Second line of defence: if an unguarded graph ran several tools anyway."""
    results = []
    for calls in ([_DOCS_CALL, _PYPI_CALL], [_PYPI_CALL, _DOCS_CALL]):
        agent, _model, documentation_service, pypi_service = _agent_with_parallel_attempt(
            calls, agent_factory=_create_agent_without_guard
        )
        result = agent.answer("Что такое embeddings и какая версия httpx на PyPI?")
        results.append(result)

        # Both tools genuinely ran in this unguarded graph...
        assert documentation_service.calls == [_DOCS_CALL[1]["question"]]
        assert pypi_service.calls == ["httpx"]
        # ...yet the answer is neither tool's output.
        assert result.outcome == "safe_fallback"
        assert result.failure_category == _MULTI_CALL_CATEGORY
        assert result.tool_call_count == 2
        assert result.tools_used == ("documentation_search", "pypi_lookup")
        assert "9.9.9" not in result.answer
        assert "Embeddings" not in result.answer
        assert result.sources == ()

    assert results[0] == results[1]


def _openai_tool_call(index: int, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"call_{index}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _real_chat_openai_over_mock_transport(
    tool_calls: list[dict[str, Any]], payloads: list[dict[str, Any]]
) -> ChatOpenAI:
    """A real ChatOpenAI whose HTTP layer is a local mock: no network, real payloads."""

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {"role": "assistant", "content": None, "tool_calls": tool_calls},
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    return ChatOpenAI(
        model="gpt-4o-mini",
        api_key="sk-test-openai",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_real_chat_openai_request_disables_parallel_tool_calls() -> None:
    payloads: list[dict[str, Any]] = []
    chat_model = _real_chat_openai_over_mock_transport(
        [_openai_tool_call(1, "pypi_lookup", {"package_name": "httpx"})], payloads
    )
    pypi_service = FakePyPIService(result=make_pypi_info())
    agent = LangChainToolCallingAgent(
        make_settings(),
        documentation_service=FakeDocumentationService(result=make_grounded_result()),
        pypi_service=pypi_service,
        chat_model=chat_model,
    )

    result = agent.answer("Какая последняя версия пакета httpx на PyPI?")

    assert result.outcome == "success"
    assert pypi_service.calls == ["httpx"]
    assert payloads[0]["parallel_tool_calls"] is False
    assert {tool["function"]["name"] for tool in payloads[0]["tools"]} == {
        "documentation_search",
        "pypi_lookup",
    }


def test_real_chat_openai_parallel_response_is_rejected_without_running_tools() -> None:
    """Even a provider that ignores parallel_tool_calls=false cannot get two tools run."""
    payloads: list[dict[str, Any]] = []
    chat_model = _real_chat_openai_over_mock_transport(
        [
            _openai_tool_call(1, "documentation_search", {"question": "Что такое embeddings?"}),
            _openai_tool_call(2, "pypi_lookup", {"package_name": "httpx"}),
        ],
        payloads,
    )
    documentation_service = FakeDocumentationService(result=make_grounded_result())
    pypi_service = FakePyPIService(result=make_pypi_info())
    agent = LangChainToolCallingAgent(
        make_settings(),
        documentation_service=documentation_service,
        pypi_service=pypi_service,
        chat_model=chat_model,
    )

    result = agent.answer("Что такое embeddings и какая версия httpx на PyPI?")

    assert len(payloads) == 1  # the graph stopped after the rejected model step
    assert documentation_service.calls == []
    assert pypi_service.calls == []
    assert result.outcome == "safe_fallback"
    assert result.failure_category == _MULTI_CALL_CATEGORY
    assert result.tool_call_count == 0
