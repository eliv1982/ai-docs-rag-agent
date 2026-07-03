"""Unit tests for InMemoryConversationMemory and ConversationAnswerService.

Uses a fake DocumentationAnswerService; no real network, OpenAI, or Pinecone
calls, and no file/database access (memory is a plain in-process dict).
"""

from typing import Any

import pytest

from ai_docs_agent.memory import (
    ConversationAnswerService,
    ConversationMemoryError,
    InMemoryConversationMemory,
)
from ai_docs_agent.models import AnswerSource, ConversationMessage, GroundedAnswerResult


def make_message(role: str = "user", content: str = "hello") -> ConversationMessage:
    return ConversationMessage(role=role, content=content)  # type: ignore[arg-type]


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


# --- InMemoryConversationMemory ----------------------------------------------------


def test_unknown_session_returns_empty_tuple() -> None:
    memory = InMemoryConversationMemory()

    assert memory.get_history("unknown-session") == ()


def test_add_one_message_then_get_history() -> None:
    memory = InMemoryConversationMemory()

    memory.add_message("session-a", make_message("user", "hello"))

    assert memory.get_history("session-a") == (make_message("user", "hello"),)


def test_chronological_order_is_preserved() -> None:
    memory = InMemoryConversationMemory()

    memory.add_message("session-a", make_message("user", "first"))
    memory.add_message("session-a", make_message("assistant", "second"))
    memory.add_message("session-a", make_message("user", "third"))

    history = memory.get_history("session-a")

    assert [m.content for m in history] == ["first", "second", "third"]


def test_add_exchange_appends_user_then_assistant() -> None:
    memory = InMemoryConversationMemory()

    memory.add_exchange("session-a", user_message="What is X?", assistant_message="X is Y.")

    history = memory.get_history("session-a")

    assert len(history) == 2
    assert history[0].role == "user"
    assert history[0].content == "What is X?"
    assert history[1].role == "assistant"
    assert history[1].content == "X is Y."


def test_exact_maximum_of_ten_messages_is_kept() -> None:
    memory = InMemoryConversationMemory(max_messages=10)

    for i in range(10):
        memory.add_message("session-a", make_message("user", f"turn-{i:02d}"))

    assert len(memory.get_history("session-a")) == 10


def test_oldest_message_is_trimmed_beyond_maximum() -> None:
    memory = InMemoryConversationMemory(max_messages=10)

    for i in range(12):
        memory.add_message("session-a", make_message("user", f"turn-{i:02d}"))

    history = memory.get_history("session-a")

    assert len(history) == 10
    assert history[0].content == "turn-02"
    assert history[-1].content == "turn-11"


def test_default_max_messages_is_exactly_ten() -> None:
    memory = InMemoryConversationMemory()

    for i in range(11):
        memory.add_message("session-a", make_message("user", f"turn-{i:02d}"))

    assert len(memory.get_history("session-a")) == 10


def test_sessions_are_fully_isolated() -> None:
    memory = InMemoryConversationMemory()

    memory.add_message("session-a", make_message("user", "for a"))
    memory.add_message("session-b", make_message("user", "for b"))

    assert memory.get_history("session-a") == (make_message("user", "for a"),)
    assert memory.get_history("session-b") == (make_message("user", "for b"),)


def test_clear_removes_only_the_selected_session() -> None:
    memory = InMemoryConversationMemory()
    memory.add_message("session-a", make_message("user", "for a"))
    memory.add_message("session-b", make_message("user", "for b"))

    memory.clear("session-a")

    assert memory.get_history("session-a") == ()
    assert memory.get_history("session-b") == (make_message("user", "for b"),)


def test_clear_unknown_session_is_harmless() -> None:
    memory = InMemoryConversationMemory()

    memory.clear("never-seen-session")  # must not raise


def test_blank_session_id_is_rejected_by_get_history() -> None:
    memory = InMemoryConversationMemory()

    with pytest.raises(ConversationMemoryError):
        memory.get_history("   ")


def test_blank_session_id_is_rejected_by_add_message() -> None:
    memory = InMemoryConversationMemory()

    with pytest.raises(ConversationMemoryError):
        memory.add_message("   ", make_message())


def test_blank_session_id_is_rejected_by_clear() -> None:
    memory = InMemoryConversationMemory()

    with pytest.raises(ConversationMemoryError):
        memory.clear("   ")


def test_zero_max_messages_is_rejected() -> None:
    with pytest.raises(ConversationMemoryError):
        InMemoryConversationMemory(max_messages=0)


def test_negative_max_messages_is_rejected() -> None:
    with pytest.raises(ConversationMemoryError):
        InMemoryConversationMemory(max_messages=-1)


def test_non_integer_max_messages_is_rejected() -> None:
    with pytest.raises(ConversationMemoryError):
        InMemoryConversationMemory(max_messages="10")  # type: ignore[arg-type]


def test_bool_max_messages_is_rejected() -> None:
    with pytest.raises(ConversationMemoryError):
        InMemoryConversationMemory(max_messages=True)


def test_returned_history_is_a_snapshot_not_a_live_view() -> None:
    memory = InMemoryConversationMemory()
    memory.add_message("session-a", make_message("user", "first"))

    snapshot = memory.get_history("session-a")
    memory.add_message("session-a", make_message("assistant", "second"))

    assert snapshot == (make_message("user", "first"),)
    assert len(memory.get_history("session-a")) == 2


# --- ConversationAnswerService -------------------------------------------------------


class FakeAnswerService:
    """Fake DocumentationAnswerService recording the exact arguments it was called with."""

    def __init__(
        self,
        *,
        result: GroundedAnswerResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self._result = result
        self._error = error
        self.calls: list[dict[str, Any]] = []

    def answer(
        self,
        question: str,
        *,
        history: tuple[ConversationMessage, ...] = (),
        top_k: int | None = None,
        namespace: str | None = None,
    ) -> GroundedAnswerResult:
        self.calls.append(
            {"question": question, "history": history, "top_k": top_k, "namespace": namespace}
        )
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


def test_first_answer_receives_empty_history() -> None:
    fake_answer_service = FakeAnswerService(result=make_result())
    service = ConversationAnswerService(fake_answer_service)

    service.answer("session-a", "how do I configure the client?")

    assert fake_answer_service.calls[0]["history"] == ()


def test_subsequent_answer_receives_the_previous_exchange() -> None:
    fake_answer_service = FakeAnswerService(
        result=make_result(question="first question", answer="first answer")
    )
    service = ConversationAnswerService(fake_answer_service)

    service.answer("session-a", "first question")
    fake_answer_service._result = make_result(question="second question", answer="second answer")
    service.answer("session-a", "second question")

    second_call_history = fake_answer_service.calls[1]["history"]
    assert [m.content for m in second_call_history] == ["first question", "first answer"]


def test_successful_exchange_is_stored() -> None:
    fake_answer_service = FakeAnswerService(
        result=make_result(question="normalized question", answer="the answer")
    )
    memory = InMemoryConversationMemory()
    service = ConversationAnswerService(fake_answer_service, memory=memory)

    service.answer("session-a", "  normalized question  ")

    history = memory.get_history("session-a")
    assert [m.content for m in history] == ["normalized question", "the answer"]
    assert history[0].role == "user"
    assert history[1].role == "assistant"


def test_answer_failure_does_not_change_memory() -> None:
    fake_answer_service = FakeAnswerService(error=RuntimeError("retrieval boom"))
    memory = InMemoryConversationMemory()
    service = ConversationAnswerService(fake_answer_service, memory=memory)

    with pytest.raises(RuntimeError):
        service.answer("session-a", "query")

    assert memory.get_history("session-a") == ()


def test_no_context_fallback_result_is_stored() -> None:
    fallback_result = make_result(
        answer="В базе знаний не найдено достаточно информации для ответа на этот вопрос.",
        sources=(),
        retrieved_chunk_count=0,
    )
    fake_answer_service = FakeAnswerService(result=fallback_result)
    memory = InMemoryConversationMemory()
    service = ConversationAnswerService(fake_answer_service, memory=memory)

    service.answer("session-a", "query")

    history = memory.get_history("session-a")
    assert len(history) == 2
    assert "не найдено" in history[1].content


def test_reset_removes_only_the_chosen_session() -> None:
    fake_answer_service = FakeAnswerService(result=make_result())
    memory = InMemoryConversationMemory()
    service = ConversationAnswerService(fake_answer_service, memory=memory)
    service.answer("session-a", "query")
    service.answer("session-b", "query")

    service.reset("session-a")

    assert memory.get_history("session-a") == ()
    assert len(memory.get_history("session-b")) == 2


def test_top_k_and_namespace_are_forwarded_unchanged() -> None:
    fake_answer_service = FakeAnswerService(result=make_result())
    service = ConversationAnswerService(fake_answer_service)

    service.answer("session-a", "query", top_k=3, namespace="custom-ns")

    call = fake_answer_service.calls[0]
    assert call["top_k"] == 3
    assert call["namespace"] == "custom-ns"


def test_sources_and_retrieved_chunks_are_not_stored_in_memory() -> None:
    fake_answer_service = FakeAnswerService(result=make_result())
    memory = InMemoryConversationMemory()
    service = ConversationAnswerService(fake_answer_service, memory=memory)

    service.answer("session-a", "query")

    for message in memory.get_history("session-a"):
        assert "docs.example.com" not in message.content
