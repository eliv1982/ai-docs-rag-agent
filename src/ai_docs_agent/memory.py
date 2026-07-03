"""Short-term, process-local conversation memory (per session_id).

Stores only the last `max_messages` user/assistant ConversationMessage turns per
session, oldest first discarded. This is an intentional homework-MVP
limitation: memory lives entirely in a plain in-process dict and is lost on
process restart. Retrieved chunks, prompts, configuration, and exceptions are
never stored -- only the normalized question text and the returned answer
text. Persistent storage (e.g. SQLite) is deferred to a future version.
"""

from ai_docs_agent.agent import DocumentationAnswerService
from ai_docs_agent.models import ConversationMessage, GroundedAnswerResult

_DEFAULT_MAX_MESSAGES = 10


class ConversationMemoryError(Exception):
    """Raised for invalid conversation-memory input, such as a blank session_id."""


class InMemoryConversationMemory:
    """Bounded, per-session, process-local store of ConversationMessage turns."""

    def __init__(self, *, max_messages: int = _DEFAULT_MAX_MESSAGES) -> None:
        if isinstance(max_messages, bool) or not isinstance(max_messages, int):
            raise ConversationMemoryError("max_messages must be a positive integer.")
        if max_messages <= 0:
            raise ConversationMemoryError("max_messages must be a positive integer.")
        self._max_messages = max_messages
        self._sessions: dict[str, list[ConversationMessage]] = {}

    def get_history(self, session_id: str) -> tuple[ConversationMessage, ...]:
        """Return the stored history for `session_id`, oldest first; () if unknown."""
        key = self._normalize_session_id(session_id)
        return tuple(self._sessions.get(key, ()))

    def add_message(self, session_id: str, message: ConversationMessage) -> None:
        """Append one message, trimming the oldest messages beyond `max_messages`."""
        key = self._normalize_session_id(session_id)
        if not isinstance(message, ConversationMessage):
            raise ConversationMemoryError("message must be a ConversationMessage.")
        history = self._sessions.setdefault(key, [])
        history.append(message)
        if len(history) > self._max_messages:
            del history[: len(history) - self._max_messages]

    def add_exchange(self, session_id: str, *, user_message: str, assistant_message: str) -> None:
        """Append a user message followed by an assistant message as one exchange."""
        key = self._normalize_session_id(session_id)
        self.add_message(key, ConversationMessage(role="user", content=user_message))
        self.add_message(key, ConversationMessage(role="assistant", content=assistant_message))

    def clear(self, session_id: str) -> None:
        """Discard all history for `session_id`. Harmless if the session is unknown."""
        key = self._normalize_session_id(session_id)
        self._sessions.pop(key, None)

    @staticmethod
    def _normalize_session_id(session_id: str) -> str:
        if not isinstance(session_id, str):
            raise ConversationMemoryError("session_id must be a string.")
        stripped = session_id.strip()
        if not stripped:
            raise ConversationMemoryError("session_id must not be blank.")
        return stripped


class ConversationAnswerService:
    """Adds short-term conversation memory around DocumentationAnswerService.

    Reads the session's stored history, forwards it to
    DocumentationAnswerService.answer(), and -- only once that call succeeds --
    appends the normalized question and returned answer to the session. If
    retrieval or generation fails, the exception propagates and memory is left
    unchanged.
    """

    def __init__(
        self,
        answer_service: DocumentationAnswerService,
        *,
        memory: InMemoryConversationMemory | None = None,
    ) -> None:
        self._answer_service = answer_service
        self._memory = memory or InMemoryConversationMemory()

    def answer(
        self,
        session_id: str,
        question: str,
        *,
        top_k: int | None = None,
        namespace: str | None = None,
    ) -> GroundedAnswerResult:
        history = self._memory.get_history(session_id)

        result = self._answer_service.answer(
            question, history=history, top_k=top_k, namespace=namespace
        )

        self._memory.add_exchange(
            session_id, user_message=result.question, assistant_message=result.answer
        )
        return result

    def reset(self, session_id: str) -> None:
        """Clear only the given session's stored history."""
        self._memory.clear(session_id)
