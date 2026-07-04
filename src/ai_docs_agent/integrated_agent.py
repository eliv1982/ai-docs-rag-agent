"""Final integrated application flow: Telegram -> agent + short/long-term memory.

One service combines the already-implemented components without redesigning them:

    IntegratedConversationAgentService
    -> deterministic explicit-remember pre-routing (parse_remember_command):
       only "Запомни: ..." / "Remember: ..." triggers UserMemoryService.remember;
       the model never decides whether a message is persisted
    -> everything else: LangChainToolCallingAgent with autonomous tool selection
       between documentation_search, pypi_lookup and the request-scoped
       user_memory_recall tool
    -> short-term InMemoryConversationMemory per session_id supplies recent turns
       for reference resolution and the documentation tool's contextual retrieval

Out-of-scope policy: a no-tool model answer never reaches the user — if the
agent answered without any tool, the integrated service replaces it with the
safe no-context fallback, so arbitrary pretrained-knowledge answers cannot
bypass the grounded flow.

reset() clears only the session's short-term history; persistent Pinecone user
memory is never deleted here. Ordinary conversation is never written into
vector memory. Importing this module performs no network calls; the factory
functions build the service graph lazily.
"""

import logging
import time

from langchain_core.language_models.chat_models import BaseChatModel

from ai_docs_agent.agent import DocumentationAnswerService
from ai_docs_agent.config import AppSettings, get_settings
from ai_docs_agent.langchain_agent import (
    LangChainAgentExecutionError,
    LangChainToolCallingAgent,
    build_chat_model,
)
from ai_docs_agent.memory import InMemoryConversationMemory
from ai_docs_agent.models import IntegratedAgentResult, LangChainAgentResult
from ai_docs_agent.observability import current_request_session_hash, request_logging_context
from ai_docs_agent.pypi import PyPILookupService
from ai_docs_agent.user_memory import (
    InvalidMemoryStatementError,
    UserMemoryError,
    UserMemoryService,
    parse_remember_command,
)

_OUT_OF_SCOPE_FALLBACK_ANSWER = (
    "В базе знаний не найдено достаточно информации для ответа на этот вопрос."
)
_OUT_OF_SCOPE_FAILURE_CATEGORY = "out_of_scope_no_tool"

_MEMORY_CREATED_CONFIRMATION = (
    "Предпочтение сохранено. Я буду учитывать его, когда вы спросите о нём."
)
_MEMORY_DUPLICATE_CONFIRMATION = "Это предпочтение уже было сохранено ранее."
_MEMORY_INVALID_STATEMENT_MESSAGE = (
    "Не удалось сохранить: после \"Запомни:\" нужен непустой текст предпочтения "
    "разумной длины."
)
_MEMORY_WRITE_FAILURE_MESSAGE = (
    "Не удалось сохранить предпочтение. Попробуйте повторить запрос позже."
)

logger = logging.getLogger(__name__)


class IntegratedAgentError(Exception):
    """Raised when the integrated flow cannot complete a request safely."""


class IntegratedConversationAgentService:
    """One Telegram-facing flow over the agent, short-term and persistent memory."""

    def __init__(
        self,
        *,
        agent: LangChainToolCallingAgent,
        user_memory_service: UserMemoryService,
        memory: InMemoryConversationMemory | None = None,
    ) -> None:
        self._agent = agent
        self._user_memory_service = user_memory_service
        self._memory = memory or InMemoryConversationMemory()

    def handle_message(self, session_id: str, text: str) -> IntegratedAgentResult:
        """Handle one user message: explicit remember command or agent flow."""
        stripped_text = text.strip()
        if not stripped_text:
            raise ValueError("text must not be blank.")

        started_at = time.monotonic()
        with request_logging_context(session_id=session_id):
            statement = parse_remember_command(stripped_text)
            if statement is not None:
                result = self._handle_remember(session_id, stripped_text, statement)
            else:
                result = self._handle_agent_question(session_id, stripped_text)

            self._log_request(
                result=result,
                question_length=len(stripped_text),
                elapsed_seconds=time.monotonic() - started_at,
            )
            return result

    def reset(self, session_id: str) -> None:
        """Clear only this session's short-term history; persistent memory stays."""
        self._memory.clear(session_id)

    # --- explicit remember path (never delegated to the model) ---------------------

    def _handle_remember(
        self, session_id: str, original_text: str, statement: str
    ) -> IntegratedAgentResult:
        try:
            write_result = self._user_memory_service.remember(session_id, statement)
        except InvalidMemoryStatementError:
            return self._remember_failure_result(
                original_text, _MEMORY_INVALID_STATEMENT_MESSAGE, "invalid_memory_statement"
            )
        except UserMemoryError:
            return self._remember_failure_result(
                original_text, _MEMORY_WRITE_FAILURE_MESSAGE, "memory_write_failure"
            )

        answer = (
            _MEMORY_CREATED_CONFIRMATION
            if write_result.status == "created"
            else _MEMORY_DUPLICATE_CONFIRMATION
        )
        # The confirmation joins short-term history so the dialogue stays coherent;
        # no record ID, namespace, or identity digest ever reaches the user.
        self._memory.add_exchange(
            session_id, user_message=original_text, assistant_message=answer
        )
        return IntegratedAgentResult(
            question=original_text,
            answer=answer,
            sources=(),
            tools_used=(),
            tool_call_count=0,
            used_no_tool=True,
            outcome="success",
            failure_category=None,
            remember_command_detected=True,
            memory_written=write_result.status == "created",
            memory_write_status=write_result.status,
        )

    @staticmethod
    def _remember_failure_result(
        original_text: str, answer: str, failure_category: str
    ) -> IntegratedAgentResult:
        return IntegratedAgentResult(
            question=original_text,
            answer=answer,
            sources=(),
            tools_used=(),
            tool_call_count=0,
            used_no_tool=True,
            outcome="safe_fallback",
            failure_category=failure_category,
            remember_command_detected=True,
            memory_written=False,
            memory_write_status=None,
        )

    # --- ordinary questions: autonomous LangChain tool selection --------------------

    def _handle_agent_question(self, session_id: str, question: str) -> IntegratedAgentResult:
        history = self._memory.get_history(session_id)

        try:
            agent_result = self._agent.answer(
                question,
                session_id=session_id,
                history=history,
                memory_user_identifier=session_id,
            )
        except LangChainAgentExecutionError as exc:
            raise IntegratedAgentError(
                "Failed to execute the integrated agent flow."
            ) from exc

        result = self._enforce_out_of_scope_policy(agent_result)
        # Only a successfully handled exchange joins short-term history; a raised
        # failure above leaves the session history untouched.
        self._memory.add_exchange(
            session_id, user_message=agent_result.question, assistant_message=result.answer
        )
        return result

    @staticmethod
    def _enforce_out_of_scope_policy(
        agent_result: LangChainAgentResult,
    ) -> IntegratedAgentResult:
        if agent_result.used_no_tool and agent_result.outcome == "success":
            # A direct pretrained-knowledge answer must never bypass the grounded
            # flow: replace it with the safe source-less fallback.
            return IntegratedAgentResult(
                question=agent_result.question,
                answer=_OUT_OF_SCOPE_FALLBACK_ANSWER,
                sources=(),
                tools_used=(),
                tool_call_count=0,
                used_no_tool=True,
                outcome="safe_fallback",
                failure_category=_OUT_OF_SCOPE_FAILURE_CATEGORY,
            )

        return IntegratedAgentResult(
            question=agent_result.question,
            answer=agent_result.answer,
            sources=agent_result.sources,
            tools_used=agent_result.tools_used,
            tool_call_count=agent_result.tool_call_count,
            used_no_tool=agent_result.used_no_tool,
            outcome=agent_result.outcome,
            failure_category=agent_result.failure_category,
        )

    # --- observability ---------------------------------------------------------------

    @staticmethod
    def _log_request(
        *,
        result: IntegratedAgentResult,
        question_length: int,
        elapsed_seconds: float,
    ) -> None:
        session_hash = current_request_session_hash() or "-"
        logger.info(
            "Integrated request session_hash=%s question_length=%d remember_command=%s "
            "tools=%s tool_call_count=%d outcome=%s memory_write_status=%s source_count=%d "
            "failure_category=%s elapsed_ms=%d",
            session_hash,
            question_length,
            str(result.remember_command_detected).lower(),
            list(result.tools_used),
            result.tool_call_count,
            result.outcome,
            result.memory_write_status or "-",
            len(result.sources),
            result.failure_category or "-",
            round(elapsed_seconds * 1000),
        )


def build_integrated_service(
    settings: AppSettings | None = None,
    *,
    chat_model: BaseChatModel | None = None,
    memory: InMemoryConversationMemory | None = None,
) -> IntegratedConversationAgentService:
    """Build the full integrated service graph used by the live Telegram bot.

    Constructs the real DocumentationAnswerService, PyPILookupService,
    UserMemoryService and LangChainToolCallingAgent; `chat_model` and `memory`
    are injectable for tests. Nothing here performs a network call at build
    time.
    """
    resolved_settings = settings or get_settings()
    documentation_service = DocumentationAnswerService(resolved_settings)
    pypi_service = PyPILookupService(resolved_settings)
    user_memory_service = UserMemoryService(resolved_settings)
    agent = LangChainToolCallingAgent(
        resolved_settings,
        documentation_service=documentation_service,
        pypi_service=pypi_service,
        user_memory_service=user_memory_service,
        chat_model=chat_model or build_chat_model(resolved_settings),
    )
    return IntegratedConversationAgentService(
        agent=agent,
        user_memory_service=user_memory_service,
        memory=memory or InMemoryConversationMemory(),
    )
