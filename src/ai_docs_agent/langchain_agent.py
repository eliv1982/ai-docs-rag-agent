"""Real LangChain tool-calling orchestration over documentation search and PyPI lookup.

The model's responsibility in this stage is autonomous tool selection. The final
user-facing answer is rendered from the authoritative tool output, not from an
additional model synthesis pass, so current package metadata and documentation
sources stay grounded in the actual tool results.
"""

import logging
import re
import time
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool, StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError

from ai_docs_agent.agent import (
    AnswerGenerationError,
    AnswerRetrievalError,
    AnswerServiceError,
    DocumentationAnswerService,
)
from ai_docs_agent.config import AppSettings, get_settings
from ai_docs_agent.models import (
    AgentToolName,
    AnswerSource,
    DocumentationSearchToolInput,
    DocumentationToolResult,
    LangChainAgentResult,
    PyPILookupToolInput,
    PyPIToolResult,
)
from ai_docs_agent.observability import current_request_session_hash, request_logging_context
from ai_docs_agent.pypi import (
    InvalidPackageNameError,
    MalformedPyPIResponseError,
    PackageNotFoundError,
    PyPILookupService,
    PyPINetworkError,
    PyPITimeoutError,
    PyPIUpstreamHTTPError,
)
from ai_docs_agent.tools import answer_documentation_question, lookup_pypi_package

_AGENT_NAME = "ai_docs_tool_calling_agent"
_AGENT_RECURSION_LIMIT = 2
_DOCUMENTATION_SEARCH_TOOL_NAME = "documentation_search"
_PYPI_LOOKUP_TOOL_NAME = "pypi_lookup"

_NO_TOOL_ANSWER = "Не удалось безопасно подготовить ответ для этого запроса."
_NO_TOOL_CURRENT_PACKAGE_METADATA_ANSWER = (
    "Не удалось подтвердить актуальные данные пакета без обращения к PyPI."
)
_DOCUMENTATION_FAILURE_ANSWER = (
    "Не удалось получить подтвержденный ответ из базы документации. "
    "Попробуйте повторить запрос позже."
)
_INVALID_PACKAGE_NAME_ANSWER = "Не удалось выполнить запрос к PyPI: некорректное имя пакета."
_PACKAGE_NOT_FOUND_ANSWER = "Пакет не найден на PyPI."
_PYPI_TIMEOUT_ANSWER = "PyPI не ответил вовремя. Попробуйте повторить запрос позже."
_PYPI_NETWORK_ERROR_ANSWER = "Не удалось связаться с PyPI. Попробуйте повторить запрос позже."
_PYPI_MALFORMED_RESPONSE_ANSWER = "PyPI вернул некорректные данные о пакете."
_PYPI_UPSTREAM_HTTP_ERROR_ANSWER = "PyPI вернул ошибку при обработке запроса."

_CURRENT_PACKAGE_METADATA_PATTERNS = (
    re.compile(r"\b(latest|current)\b.*\b(version|release)\b", re.IGNORECASE),
    re.compile(r"\b(version|release)\b.*\b(latest|current)\b", re.IGNORECASE),
    re.compile(r"\bпоследн\w+\b.*\bверс\w+\b", re.IGNORECASE),
    re.compile(r"\bверс\w+\b.*\bпоследн\w+\b", re.IGNORECASE),
    re.compile(r"\bактуальн\w+\b.*\bверс\w+\b", re.IGNORECASE),
)
_PACKAGE_METADATA_HINT_PATTERN = re.compile(r"\b(pypi|package)\b|пакет", re.IGNORECASE)

_SYSTEM_PROMPT = (
    "You are a tool-calling assistant for one repository.\n"
    "- You have exactly two tools: documentation_search and pypi_lookup.\n"
    "- Use pypi_lookup for current or latest PyPI package metadata such as version, "
    "package existence, summary, requires_python, or project links.\n"
    "- Use documentation_search for technical questions about the indexed documentation "
    "such as OpenAI, Pinecone, LangChain, RecursiveCharacterTextSplitter, and other "
    "documentation topics already indexed in the vector store.\n"
    "- Tool outputs are authoritative. Do not invent package versions, package existence, "
    "documentation facts, sources, or URLs.\n"
    "- Conversation text is not documentary evidence.\n"
    "- Use at most one tool call for a single request in this stage. Choose the single "
    "best tool and stop.\n"
    "- If a tool reports a failure or no context, treat that report as authoritative and "
    "do not overwrite it with guesses.\n"
    "- If no tool is needed, answer briefly and safely."
)

_DOCUMENTATION_SEARCH_TOOL_DESCRIPTION = (
    "Search the indexed technical documentation and return a grounded answer with sources. "
    "Use this for documentation questions about OpenAI, Pinecone, LangChain, "
    "RecursiveCharacterTextSplitter, embeddings, semantic search, and other indexed docs. "
    "Do not use this for current or latest PyPI package metadata."
)
_PYPI_LOOKUP_TOOL_DESCRIPTION = (
    "Look up current PyPI package metadata from the real PyPI JSON API. "
    "Use this for latest/current version questions, package existence checks, summaries, "
    "requires_python, PyPI URLs, and project URLs. Always use this for current or latest "
    "PyPI package metadata instead of answering from memory."
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ToolTrace:
    """One safe tool invocation record collected during a single agent run."""

    name: AgentToolName
    result: DocumentationToolResult | PyPIToolResult


_CURRENT_TOOL_TRACE: ContextVar[list[_ToolTrace] | None] = ContextVar(
    "ai_docs_agent_current_tool_trace",
    default=None,
)


class LangChainAgentExecutionError(Exception):
    """Raised when the LangChain agent cannot complete a request safely."""


def build_chat_model(settings: AppSettings) -> BaseChatModel:
    """Build the real chat model used by the LangChain agent."""
    kwargs: dict[str, Any] = {
        "model": settings.openai_chat_model,
        "api_key": settings.openai_api_key.get_secret_value(),
        "temperature": 0,
    }
    if settings.openai_base_url is not None:
        kwargs["base_url"] = settings.openai_base_url
    return ChatOpenAI(**kwargs)


def build_langchain_tools(
    *,
    documentation_service: DocumentationAnswerService,
    pypi_service: PyPILookupService,
) -> list[BaseTool]:
    """Build the two LangChain tools used by the agent."""
    return [
        _build_documentation_search_tool(documentation_service),
        _build_pypi_lookup_tool(pypi_service),
    ]


class LangChainToolCallingAgent:
    """Autonomous LangChain tool-calling agent over docs search and PyPI lookup."""

    def __init__(
        self,
        settings: AppSettings | None = None,
        *,
        documentation_service: DocumentationAnswerService | None = None,
        pypi_service: PyPILookupService | None = None,
        chat_model: BaseChatModel | None = None,
        agent_factory: Callable[..., Any] = create_agent,
    ) -> None:
        self._settings = settings or get_settings()
        self._documentation_service = documentation_service or DocumentationAnswerService(
            self._settings
        )
        self._pypi_service = pypi_service or PyPILookupService(self._settings)
        self._chat_model = chat_model or build_chat_model(self._settings)
        self._tools = build_langchain_tools(
            documentation_service=self._documentation_service,
            pypi_service=self._pypi_service,
        )
        self._graph = agent_factory(
            model=self._chat_model,
            tools=self._tools,
            system_prompt=_SYSTEM_PROMPT,
            name=_AGENT_NAME,
        )

    def answer(self, question: str, *, session_id: str | None = None) -> LangChainAgentResult:
        """Answer one natural-language request through real LangChain tool calling."""
        stripped_question = question.strip()
        if not stripped_question:
            raise ValueError("question must not be blank.")

        started_at = time.monotonic()
        trace: list[_ToolTrace] = []
        trace_token: Token[list[_ToolTrace] | None] = _CURRENT_TOOL_TRACE.set(trace)
        logging_context = (
            request_logging_context(session_id=session_id)
            if session_id is not None
            else nullcontext()
        )

        with logging_context:
            try:
                try:
                    raw_result = self._graph.invoke(
                        {"messages": [{"role": "user", "content": stripped_question}]},
                        config={"recursion_limit": _AGENT_RECURSION_LIMIT},
                    )
                except GraphRecursionError:
                    result = self._build_result_from_trace_or_limit(stripped_question, trace)
                    self._log_agent_outcome(
                        question=stripped_question,
                        trace=trace,
                        outcome=result.outcome,
                        failure_category=result.failure_category,
                        elapsed_seconds=time.monotonic() - started_at,
                    )
                    return result
            except Exception as exc:
                self._log_unhandled_failure(
                    question=stripped_question,
                    trace=trace,
                    elapsed_seconds=time.monotonic() - started_at,
                    exc=exc,
                )
                raise LangChainAgentExecutionError(
                    "Failed to execute the LangChain tool-calling agent."
                ) from exc
            finally:
                _CURRENT_TOOL_TRACE.reset(trace_token)

        result = self._build_result(
            question=stripped_question,
            raw_messages=raw_result["messages"],
            trace=trace,
        )
        self._log_agent_outcome(
            question=stripped_question,
            trace=trace,
            outcome=result.outcome,
            failure_category=result.failure_category,
            elapsed_seconds=time.monotonic() - started_at,
        )
        return result

    @staticmethod
    def _build_result_from_trace_or_limit(
        question: str,
        trace: Sequence[_ToolTrace],
    ) -> LangChainAgentResult:
        if trace:
            return LangChainToolCallingAgent._build_result_from_trace(question, trace)
        return LangChainAgentResult(
            question=question,
            answer=_NO_TOOL_ANSWER,
            sources=(),
            tools_used=(),
            tool_call_count=0,
            used_no_tool=True,
            outcome="safe_fallback",
            failure_category="agent_iteration_limit",
        )

    @staticmethod
    def _build_result(
        *,
        question: str,
        raw_messages: Sequence[BaseMessage],
        trace: Sequence[_ToolTrace],
    ) -> LangChainAgentResult:
        if trace:
            return LangChainToolCallingAgent._build_result_from_trace(question, trace)

        direct_answer = LangChainToolCallingAgent._extract_final_ai_message_text(raw_messages)
        if _requires_current_package_metadata_tool(question):
            return LangChainAgentResult(
                question=question,
                answer=_NO_TOOL_CURRENT_PACKAGE_METADATA_ANSWER,
                sources=(),
                tools_used=(),
                tool_call_count=0,
                used_no_tool=True,
                outcome="safe_fallback",
                failure_category="missing_required_tool_call",
            )

        if direct_answer is None:
            return LangChainAgentResult(
                question=question,
                answer=_NO_TOOL_ANSWER,
                sources=(),
                tools_used=(),
                tool_call_count=0,
                used_no_tool=True,
                outcome="safe_fallback",
                failure_category="no_tool_output",
            )

        return LangChainAgentResult(
            question=question,
            answer=direct_answer,
            sources=(),
            tools_used=(),
            tool_call_count=0,
            used_no_tool=True,
            outcome="success",
            failure_category=None,
        )

    @staticmethod
    def _build_result_from_trace(
        question: str,
        trace: Sequence[_ToolTrace],
    ) -> LangChainAgentResult:
        authoritative_trace = trace[-1]
        tools_used = _ordered_unique_tool_names(trace)

        if authoritative_trace.name == _DOCUMENTATION_SEARCH_TOOL_NAME:
            tool_result = authoritative_trace.result
            assert isinstance(tool_result, DocumentationToolResult)
            return LangChainAgentResult(
                question=question,
                answer=tool_result.answer,
                sources=tool_result.sources,
                tools_used=tools_used,
                tool_call_count=len(trace),
                used_no_tool=False,
                outcome="success" if tool_result.status == "success" else "safe_fallback",
                failure_category=None if tool_result.status == "success" else tool_result.status,
            )

        tool_result = authoritative_trace.result
        assert isinstance(tool_result, PyPIToolResult)
        return LangChainAgentResult(
            question=question,
            answer=_render_pypi_answer(tool_result),
            sources=_build_pypi_sources(tool_result),
            tools_used=tools_used,
            tool_call_count=len(trace),
            used_no_tool=False,
            outcome="success" if tool_result.status == "success" else "safe_fallback",
            failure_category=None if tool_result.status == "success" else tool_result.status,
        )

    @staticmethod
    def _extract_final_ai_message_text(raw_messages: Sequence[BaseMessage]) -> str | None:
        for message in reversed(raw_messages):
            if not isinstance(message, AIMessage):
                continue
            content = message.content
            if isinstance(content, str):
                stripped = content.strip()
                if stripped:
                    return stripped
        return None

    @staticmethod
    def _log_agent_outcome(
        *,
        question: str,
        trace: Sequence[_ToolTrace],
        outcome: str,
        failure_category: str | None,
        elapsed_seconds: float,
    ) -> None:
        session_hash = current_request_session_hash() or "-"
        logger.info(
            "Agent request session_hash=%s question_length=%d tool_names=%s tool_call_count=%d "
            "outcome=%s failure_category=%s elapsed_ms=%d",
            session_hash,
            len(question),
            list(_ordered_unique_tool_names(trace)),
            len(trace),
            outcome,
            failure_category or "-",
            round(elapsed_seconds * 1000),
        )

    @staticmethod
    def _log_unhandled_failure(
        *,
        question: str,
        trace: Sequence[_ToolTrace],
        elapsed_seconds: float,
        exc: BaseException,
    ) -> None:
        session_hash = current_request_session_hash() or "-"
        logger.exception(
            "Agent request failed session_hash=%s question_length=%d tool_names=%s "
            "tool_call_count=%d elapsed_ms=%d",
            session_hash,
            len(question),
            list(_ordered_unique_tool_names(trace)),
            len(trace),
            round(elapsed_seconds * 1000),
            exc_info=(type(exc), type(exc)(), exc.__traceback__),
        )


def _build_documentation_search_tool(
    documentation_service: DocumentationAnswerService,
) -> BaseTool:
    def documentation_search(question: str) -> str:
        result = _run_documentation_search_tool(
            question=question,
            documentation_service=documentation_service,
        )
        _record_tool_trace(_DOCUMENTATION_SEARCH_TOOL_NAME, result)
        return result.model_dump_json()

    return StructuredTool.from_function(
        func=documentation_search,
        name=_DOCUMENTATION_SEARCH_TOOL_NAME,
        description=_DOCUMENTATION_SEARCH_TOOL_DESCRIPTION,
        args_schema=DocumentationSearchToolInput,
    )


def _build_pypi_lookup_tool(pypi_service: PyPILookupService) -> BaseTool:
    def pypi_lookup(package_name: str) -> str:
        result = _run_pypi_lookup_tool(package_name=package_name, pypi_service=pypi_service)
        _record_tool_trace(_PYPI_LOOKUP_TOOL_NAME, result)
        return result.model_dump_json()

    return StructuredTool.from_function(
        func=pypi_lookup,
        name=_PYPI_LOOKUP_TOOL_NAME,
        description=_PYPI_LOOKUP_TOOL_DESCRIPTION,
        args_schema=PyPILookupToolInput,
    )


def _run_documentation_search_tool(
    *,
    question: str,
    documentation_service: DocumentationAnswerService,
) -> DocumentationToolResult:
    try:
        grounded_result = answer_documentation_question(question, service=documentation_service)
    except AnswerRetrievalError:
        return DocumentationToolResult(
            status="retrieval_failure",
            answer=_DOCUMENTATION_FAILURE_ANSWER,
            sources=(),
            context_found=False,
        )
    except AnswerGenerationError:
        return DocumentationToolResult(
            status="generation_failure",
            answer=_DOCUMENTATION_FAILURE_ANSWER,
            sources=(),
            context_found=False,
        )
    except AnswerServiceError:
        return DocumentationToolResult(
            status="generation_failure",
            answer=_DOCUMENTATION_FAILURE_ANSWER,
            sources=(),
            context_found=False,
        )

    if grounded_result.retrieved_chunk_count == 0:
        return DocumentationToolResult(
            status="no_context",
            answer=grounded_result.answer,
            sources=(),
            context_found=False,
        )

    return DocumentationToolResult(
        status="success",
        answer=grounded_result.answer,
        sources=grounded_result.sources,
        context_found=True,
    )


def _run_pypi_lookup_tool(
    *,
    package_name: str,
    pypi_service: PyPILookupService,
) -> PyPIToolResult:
    try:
        package_info = lookup_pypi_package(package_name, service=pypi_service)
    except InvalidPackageNameError:
        return PyPIToolResult(status="invalid_package_name")
    except PackageNotFoundError:
        return PyPIToolResult(status="package_not_found")
    except PyPITimeoutError:
        return PyPIToolResult(status="timeout")
    except PyPINetworkError:
        return PyPIToolResult(status="network_error")
    except MalformedPyPIResponseError:
        return PyPIToolResult(status="malformed_response")
    except PyPIUpstreamHTTPError:
        return PyPIToolResult(status="upstream_http_error")

    return PyPIToolResult(
        status="success",
        package_name=package_info.package_name,
        latest_version=package_info.latest_version,
        summary=package_info.summary,
        requires_python=package_info.requires_python,
        pypi_url=package_info.pypi_url,
        project_url=package_info.project_url,
    )


def _record_tool_trace(
    name: AgentToolName,
    result: DocumentationToolResult | PyPIToolResult,
) -> None:
    trace = _CURRENT_TOOL_TRACE.get()
    if trace is None:
        return
    trace.append(_ToolTrace(name=name, result=result))


def _ordered_unique_tool_names(trace: Sequence[_ToolTrace]) -> tuple[AgentToolName, ...]:
    names: list[AgentToolName] = []
    seen: set[AgentToolName] = set()
    for item in trace:
        if item.name in seen:
            continue
        seen.add(item.name)
        names.append(item.name)
    return tuple(names)


def _build_pypi_sources(result: PyPIToolResult) -> tuple[AnswerSource, ...]:
    if result.status != "success":
        return ()

    assert result.package_name is not None
    assert result.pypi_url is not None

    sources = [
        AnswerSource(
            title=f"PyPI package page: {result.package_name}",
            url=result.pypi_url,
            document_id=f"pypi:{result.package_name}",
            chunk_index=0,
            chunk_count=1,
        )
    ]

    if result.project_url is not None:
        sources.append(
            AnswerSource(
                title=f"Project URL for {result.package_name}",
                url=result.project_url,
                document_id=f"project:{result.package_name}",
                chunk_index=0,
                chunk_count=1,
            )
        )

    return tuple(sources)


def _render_pypi_answer(result: PyPIToolResult) -> str:
    if result.status == "invalid_package_name":
        return _INVALID_PACKAGE_NAME_ANSWER
    if result.status == "package_not_found":
        return _PACKAGE_NOT_FOUND_ANSWER
    if result.status == "timeout":
        return _PYPI_TIMEOUT_ANSWER
    if result.status == "network_error":
        return _PYPI_NETWORK_ERROR_ANSWER
    if result.status == "malformed_response":
        return _PYPI_MALFORMED_RESPONSE_ANSWER
    if result.status == "upstream_http_error":
        return _PYPI_UPSTREAM_HTTP_ERROR_ANSWER

    assert result.package_name is not None
    assert result.latest_version is not None

    lines = [f"Последняя версия пакета {result.package_name} на PyPI: {result.latest_version}."]
    if result.summary is not None:
        lines.append(f"Summary: {result.summary}")
    if result.requires_python is not None:
        lines.append(f"Requires Python: {result.requires_python}")
    if result.project_url is not None:
        lines.append(f"Project URL: {result.project_url}")
    return "\n".join(lines)


def _requires_current_package_metadata_tool(question: str) -> bool:
    if _PACKAGE_METADATA_HINT_PATTERN.search(question) is None:
        return False
    return any(pattern.search(question) for pattern in _CURRENT_PACKAGE_METADATA_PATTERNS)
