"""Unit tests for InMemoryConversationMemory and ConversationAnswerService.

Uses a fake DocumentationAnswerService; no real network, OpenAI, or Pinecone
calls, and no file/database access (memory is a plain in-process dict).
"""

from typing import Any

import pytest

from ai_docs_agent.agent import DocumentationAnswerService
from ai_docs_agent.config import AppSettings
from ai_docs_agent.memory import (
    ConversationAnswerService,
    ConversationMemoryError,
    InMemoryConversationMemory,
)
from ai_docs_agent.models import (
    AnswerSource,
    ConversationMessage,
    GroundedAnswerResult,
    RetrievalResult,
    RetrievedChunk,
)


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


def make_settings(**overrides: Any) -> AppSettings:
    required: dict[str, Any] = {
        "openai_api_key": "sk-test-openai",
        "pinecone_api_key": "pc-test-key",
        "openai_chat_model": "gpt-4o-mini",
        "telegram_bot_token": "test-telegram-token",
    }
    return AppSettings(_env_file=None, **{**required, **overrides})


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
        "text": "RecursiveCharacterTextSplitter recursively splits text.",
    }
    return RetrievedChunk(**{**defaults, **overrides})


def make_retrieval_result(**overrides: Any) -> RetrievalResult:
    matches = overrides.pop("matches", ())
    defaults: dict[str, Any] = {
        "query": "query",
        "namespace": "documentation",
        "top_k": 5,
        "matches": matches,
    }
    return RetrievalResult(**{**defaults, **overrides})


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


class FakeRetrievalService:
    def __init__(self, *, results: list[RetrievalResult]) -> None:
        self._results = list(results)
        self.calls: list[tuple[str, int | None, str | None]] = []

    def search(
        self, query: str, *, top_k: int | None = None, namespace: str | None = None
    ) -> RetrievalResult:
        self.calls.append((query, top_k, namespace))
        assert self._results, "FakeRetrievalService.search called more times than expected."
        return self._results.pop(0)


class FakeChatClient:
    def __init__(self, *, answers: list[str]) -> None:
        self._answers = list(answers)
        self.calls: list[dict[str, str]] = []

    def complete(self, *, model: str, system_prompt: str, user_prompt: str) -> str:
        self.calls.append(
            {"model": model, "system_prompt": system_prompt, "user_prompt": user_prompt}
        )
        assert self._answers, "FakeChatClient.complete called more times than expected."
        return self._answers.pop(0)


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


def test_alias_exchange_is_available_to_the_next_turn() -> None:
    alias_question = "В этом диалоге будем называть RecursiveCharacterTextSplitter «Резак»."
    alias_ack = (
        "Хорошо, в рамках текущего диалога буду называть "
        "RecursiveCharacterTextSplitter «Резак»."
    )
    fake_answer_service = FakeAnswerService(
        result=make_result(question=alias_question, answer=alias_ack)
    )
    memory = InMemoryConversationMemory()
    service = ConversationAnswerService(fake_answer_service, memory=memory)

    service.answer("session-a", alias_question)
    fake_answer_service._result = make_result(
        question="Что делает Резак?", answer="Резак делит текст рекурсивно."
    )
    service.answer("session-a", "Что делает Резак?")

    second_call_history = fake_answer_service.calls[1]["history"]
    assert any("Резак" in message.content for message in second_call_history)


def test_alias_exchange_is_stored_only_in_the_correct_session() -> None:
    alias_question = "В этом диалоге называй RecursiveCharacterTextSplitter Резаком."
    alias_ack = "Хорошо, в рамках текущего диалога буду называть его Резаком."
    fake_answer_service = FakeAnswerService(
        result=make_result(question=alias_question, answer=alias_ack)
    )
    memory = InMemoryConversationMemory()
    service = ConversationAnswerService(fake_answer_service, memory=memory)

    service.answer("session-a", alias_question)

    assert [message.content for message in memory.get_history("session-a")] == [
        alias_question,
        alias_ack,
    ]
    assert memory.get_history("session-b") == ()


def test_reset_removes_the_alias_history() -> None:
    alias_question = "В этом диалоге будем называть RecursiveCharacterTextSplitter «Резак»."
    alias_ack = "Хорошо, буду называть его «Резак»."
    fake_answer_service = FakeAnswerService(
        result=make_result(question=alias_question, answer=alias_ack)
    )
    memory = InMemoryConversationMemory()
    service = ConversationAnswerService(fake_answer_service, memory=memory)
    service.answer("session-a", alias_question)

    service.reset("session-a")

    assert memory.get_history("session-a") == ()


def test_reset_clears_alias_context_so_follow_up_returns_normal_no_context_fallback() -> None:
    alias_question = "В этом диалоге называй RecursiveCharacterTextSplitter Резаком."
    settings = make_settings()
    retrieval = FakeRetrievalService(
        results=[
            make_retrieval_result(query=alias_question, matches=(make_chunk(score=0.6),)),
            make_retrieval_result(query="Для чего нужен Резак?", matches=()),
        ]
    )
    chat = FakeChatClient(
        answers=[
            "Хорошо, в рамках текущего диалога буду называть его Резаком.",
            "Рекурсивный сплиттер разбивает текст.",
        ]
    )
    answer_service = DocumentationAnswerService(
        settings, retrieval_service=retrieval, chat_client=chat
    )
    memory = InMemoryConversationMemory()
    service = ConversationAnswerService(answer_service, memory=memory)

    service.answer("session-a", alias_question)
    service.reset("session-a")
    result = service.answer("session-a", "Для чего нужен Резак?")

    assert result.retrieved_chunk_count == 0
    assert result.sources == ()
    assert "не найдено" in result.answer
    assert retrieval.calls == [
        (alias_question, None, None),
        ("Для чего нужен Резак?", None, None),
    ]


def test_reset_removes_only_the_chosen_session() -> None:
    fake_answer_service = FakeAnswerService(result=make_result())
    memory = InMemoryConversationMemory()
    service = ConversationAnswerService(fake_answer_service, memory=memory)
    service.answer("session-a", "query")
    service.answer("session-b", "query")

    service.reset("session-a")

    assert memory.get_history("session-a") == ()
    assert len(memory.get_history("session-b")) == 2


def test_alias_context_does_not_leak_between_sessions() -> None:
    alias_question = "В этом диалоге называй RecursiveCharacterTextSplitter Резаком."
    settings = make_settings()
    splitter_chunk = make_chunk(
        score=0.62,
        title="Recursive Splitter Docs",
        final_url="https://docs.example.com/recursive-splitter",
        source_url="https://docs.example.com/recursive-splitter",
        document_id="doc-splitter",
    )
    retrieval = FakeRetrievalService(
        results=[
            make_retrieval_result(query=alias_question, matches=(make_chunk(score=0.6),)),
            make_retrieval_result(query="Для чего нужен Резак?", matches=()),
            make_retrieval_result(
                query="Для чего нужен RecursiveCharacterTextSplitter?",
                matches=(splitter_chunk,),
            ),
            make_retrieval_result(query="Для чего нужен Резак?", matches=()),
        ]
    )
    chat = FakeChatClient(
        answers=[
            "Хорошо, в рамках текущего диалога буду называть его Резаком.",
            "Для чего нужен RecursiveCharacterTextSplitter?",
            "Он нужен, чтобы рекурсивно разбивать текст.",
        ]
    )
    answer_service = DocumentationAnswerService(
        settings, retrieval_service=retrieval, chat_client=chat
    )
    memory = InMemoryConversationMemory()
    service = ConversationAnswerService(answer_service, memory=memory)

    service.answer("session-a", alias_question)
    session_a_result = service.answer("session-a", "Для чего нужен Резак?")
    session_b_result = service.answer("session-b", "Для чего нужен Резак?")

    assert session_a_result.retrieved_chunk_count == 1
    assert session_b_result.retrieved_chunk_count == 0
    assert "не найдено" in session_b_result.answer


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
