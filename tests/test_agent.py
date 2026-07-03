"""Unit tests for DocumentationAnswerService. Uses fake retrieval and chat clients
only; no real network, OpenAI, or Pinecone calls."""

from typing import Any

import pytest

from ai_docs_agent.agent import (
    AnswerGenerationError,
    AnswerRetrievalError,
    AnswerServiceError,
    DocumentationAnswerService,
)
from ai_docs_agent.config import AppSettings
from ai_docs_agent.models import ConversationMessage, RetrievalResult, RetrievedChunk
from ai_docs_agent.retrieval import RetrievalError

_REQUIRED: dict[str, Any] = {
    "openai_api_key": "sk-test-openai",
    "pinecone_api_key": "pc-test-key",
    "openai_chat_model": "gpt-4o-mini",
}


def make_settings(**overrides: Any) -> AppSettings:
    return AppSettings(_env_file=None, **{**_REQUIRED, **overrides})


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


class FakeRetrievalService:
    """Fake RetrievalService for DocumentationAnswerService orchestration tests."""

    def __init__(
        self,
        *,
        result: RetrievalResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self._result = result
        self._error = error
        self.calls: list[tuple[str, int | None, str | None]] = []

    def search(
        self, query: str, *, top_k: int | None = None, namespace: str | None = None
    ) -> RetrievalResult:
        self.calls.append((query, top_k, namespace))
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


class FakeChatClient:
    """Fake ChatClient recording the exact prompts it was called with."""

    def __init__(
        self,
        *,
        answer: str = "The API key is set via OPENAI_API_KEY.",
        error: Exception | None = None,
    ) -> None:
        self._answer = answer
        self._error = error
        self.calls: list[dict[str, str]] = []

    def complete(self, *, model: str, system_prompt: str, user_prompt: str) -> str:
        self.calls.append(
            {"model": model, "system_prompt": system_prompt, "user_prompt": user_prompt}
        )
        if self._error is not None:
            raise self._error
        return self._answer


def make_service(
    *,
    settings: AppSettings | None = None,
    retrieval: FakeRetrievalService | None = None,
    chat: FakeChatClient | None = None,
) -> tuple[DocumentationAnswerService, FakeRetrievalService, FakeChatClient]:
    settings = settings or make_settings()
    retrieval = retrieval or FakeRetrievalService(result=make_retrieval_result())
    chat = chat or FakeChatClient()
    service = DocumentationAnswerService(settings, retrieval_service=retrieval, chat_client=chat)
    return service, retrieval, chat


# --- happy path / forwarding -----------------------------------------------------


def test_answer_happy_path() -> None:
    service, retrieval, chat = make_service()

    result = service.answer("how do I configure the client?")

    assert result.answer == "The API key is set via OPENAI_API_KEY."
    assert result.retrieved_chunk_count == 1
    assert len(result.sources) == 1
    assert retrieval.calls == [("how do I configure the client?", None, None)]
    assert chat.calls[0]["model"] == "gpt-4o-mini"


def test_answer_forwards_top_k_and_namespace() -> None:
    service, retrieval, _chat = make_service()

    service.answer("query", top_k=3, namespace="custom-ns")

    assert retrieval.calls == [("query", 3, "custom-ns")]


def test_answer_question_matches_normalized_retrieval_query() -> None:
    retrieval = FakeRetrievalService(
        result=make_retrieval_result(query="how do i configure the client?")
    )
    service, _retrieval, _chat = make_service(retrieval=retrieval)

    result = service.answer("  how do i configure the client?  ")

    assert result.question == "how do i configure the client?"


# --- prompt construction ----------------------------------------------------------


def test_prompt_preserves_retrieval_order_and_contains_metadata() -> None:
    chunks = (
        make_chunk(chunk_id="c", document_id="doc-c", title="Page C", text="Text about C."),
        make_chunk(chunk_id="a", document_id="doc-a", title="Page A", text="Text about A."),
    )
    retrieval = FakeRetrievalService(result=make_retrieval_result(matches=chunks))
    service, _retrieval, chat = make_service(retrieval=retrieval)

    service.answer("query")

    prompt = chat.calls[0]["user_prompt"]
    assert prompt.index("Page C") < prompt.index("Page A")
    assert "Text about C." in prompt
    assert "Text about A." in prompt
    assert "doc-c" in prompt
    assert "doc-a" in prompt
    assert "[S1]" in prompt
    assert "[S2]" in prompt


def test_system_prompt_instructs_closed_book_grounding() -> None:
    service, _retrieval, chat = make_service()

    service.answer("query")

    system_prompt = chat.calls[0]["system_prompt"].lower()
    assert "closed-book" in system_prompt
    assert "do not invent" in system_prompt
    assert "untrusted" in system_prompt
    assert "ignore any" in system_prompt


def test_system_prompt_instructs_omitting_unsupported_details() -> None:
    service, _retrieval, chat = make_service()

    service.answer("query")

    system_prompt = chat.calls[0]["system_prompt"].lower()
    assert "omit" in system_prompt
    assert "not explicitly supported" in system_prompt


def test_system_prompt_instructs_against_general_knowledge_and_plausible_inference() -> None:
    service, _retrieval, chat = make_service()

    service.answer("query")

    system_prompt = chat.calls[0]["system_prompt"].lower()
    assert "background knowledge" in system_prompt
    assert "recommendations" in system_prompt
    assert "plausible" in system_prompt


def test_system_prompt_instructs_final_unsupported_sentence_removal_pass() -> None:
    service, _retrieval, chat = make_service()

    service.answer("query")

    system_prompt = chat.calls[0]["system_prompt"].lower()
    assert "sentence" in system_prompt
    assert "remove" in system_prompt
    assert "not directly supported" in system_prompt


# --- empty retrieval / fallback ----------------------------------------------------


def test_empty_retrieval_skips_chat_call_and_returns_fallback() -> None:
    retrieval = FakeRetrievalService(result=make_retrieval_result(matches=()))
    chat = FakeChatClient()
    service, _retrieval, _chat = make_service(retrieval=retrieval, chat=chat)

    result = service.answer("query")

    assert result.retrieved_chunk_count == 0
    assert result.sources == ()
    assert "не найдено" in result.answer
    assert chat.calls == []


def test_empty_retrieval_fallback_is_stable_wording() -> None:
    retrieval = FakeRetrievalService(result=make_retrieval_result(matches=()))
    service, _retrieval, _chat = make_service(retrieval=retrieval)

    result = service.answer("query")

    assert result.answer == (
        "В базе знаний не найдено достаточно информации для ответа на этот вопрос."
    )


# --- sources -----------------------------------------------------------------------


def test_sources_are_built_from_retrieved_chunks() -> None:
    chunk = make_chunk(title="Config Guide", final_url="https://docs.example.com/config")
    retrieval = FakeRetrievalService(result=make_retrieval_result(matches=(chunk,)))
    service, _retrieval, _chat = make_service(retrieval=retrieval)

    result = service.answer("query")

    assert len(result.sources) == 1
    source = result.sources[0]
    assert source.title == "Config Guide"
    assert source.url == "https://docs.example.com/config"
    assert source.document_id == chunk.document_id
    assert source.chunk_index == chunk.chunk_index
    assert source.chunk_count == chunk.chunk_count


def test_sources_deduplicate_same_url_preserving_first_occurrence() -> None:
    chunks = (
        make_chunk(chunk_id="a", chunk_index=0, chunk_count=2, title="First Chunk"),
        make_chunk(chunk_id="b", chunk_index=1, chunk_count=2, title="Second Chunk"),
    )
    retrieval = FakeRetrievalService(result=make_retrieval_result(matches=chunks))
    service, _retrieval, _chat = make_service(retrieval=retrieval)

    result = service.answer("query")

    assert len(result.sources) == 1
    assert result.sources[0].title == "First Chunk"


def test_sources_prefer_final_url_over_source_url() -> None:
    chunk = make_chunk(
        source_url="https://docs.example.com/original",
        final_url="https://docs.example.com/redirected",
    )
    retrieval = FakeRetrievalService(result=make_retrieval_result(matches=(chunk,)))
    service, _retrieval, _chat = make_service(retrieval=retrieval)

    result = service.answer("query")

    assert result.sources[0].url == "https://docs.example.com/redirected"


def test_sources_fall_back_to_source_url_when_final_url_blank() -> None:
    chunk = RetrievedChunk(
        chunk_id="doc-abc123-chunk-0000",
        score=0.9,
        document_id="doc-abc123",
        source_url="https://docs.example.com/original",
        final_url="   ",
        title="Example Page",
        content_hash="hash-value",
        chunk_index=0,
        chunk_count=1,
        text="Some chunk text.",
    )
    retrieval = FakeRetrievalService(result=make_retrieval_result(matches=(chunk,)))
    service, _retrieval, _chat = make_service(retrieval=retrieval)

    result = service.answer("query")

    assert result.sources[0].url == "https://docs.example.com/original"


# --- failure wrapping --------------------------------------------------------------


def test_retrieval_failure_is_wrapped() -> None:
    retrieval = FakeRetrievalService(error=RetrievalError("boom"))
    service, _retrieval, _chat = make_service(retrieval=retrieval)

    with pytest.raises(AnswerRetrievalError) as exc_info:
        service.answer("query")

    assert isinstance(exc_info.value.__cause__, RetrievalError)


def test_chat_failure_is_wrapped() -> None:
    chat = FakeChatClient(error=RuntimeError("chat backend unavailable"))
    service, _retrieval, _chat = make_service(chat=chat)

    with pytest.raises(AnswerGenerationError) as exc_info:
        service.answer("query")

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_blank_model_output_raises_generation_error() -> None:
    chat = FakeChatClient(answer="   ")
    service, _retrieval, _chat = make_service(chat=chat)

    with pytest.raises(AnswerGenerationError):
        service.answer("query")


def test_errors_do_not_leak_secrets() -> None:
    settings = make_settings(openai_api_key="sk-super-secret", pinecone_api_key="pc-super-secret")
    chat = FakeChatClient(error=RuntimeError("sk-super-secret leaked"))
    service, _retrieval, _chat = make_service(settings=settings, chat=chat)

    with pytest.raises(AnswerGenerationError) as exc_info:
        service.answer("query")

    message = str(exc_info.value)
    assert "sk-super-secret" not in message
    assert isinstance(exc_info.value, AnswerServiceError)


# --- conversation history -----------------------------------------------------------


def make_message(role: str = "user", content: str = "hello") -> ConversationMessage:
    return ConversationMessage(role=role, content=content)  # type: ignore[arg-type]


def test_answer_without_history_is_unchanged() -> None:
    service, retrieval, chat = make_service()

    result = service.answer("how do I configure the client?")

    assert result.answer == "The API key is set via OPENAI_API_KEY."
    assert retrieval.calls == [("how do I configure the client?", None, None)]
    prompt = chat.calls[0]["user_prompt"]
    assert "(no prior conversation history)" in prompt


def test_history_appears_in_deterministic_order() -> None:
    history = (
        make_message("user", "What is LangChain?"),
        make_message("assistant", "A framework for LLM applications."),
        make_message("user", "How do I configure the client?"),
    )
    service, _retrieval, chat = make_service()

    service.answer("query", history=history)

    prompt = chat.calls[0]["user_prompt"]
    assert prompt.index("What is LangChain?") < prompt.index("A framework for LLM applications.")
    assert prompt.index("A framework for LLM applications.") < prompt.index(
        "How do I configure the client?"
    )


def test_only_last_ten_history_messages_are_included() -> None:
    history = tuple(make_message("user", f"turn-{i:02d}-marker") for i in range(12))
    service, _retrieval, chat = make_service()

    service.answer("query", history=history)

    prompt = chat.calls[0]["user_prompt"]
    assert "turn-00-marker" not in prompt
    assert "turn-01-marker" not in prompt
    assert "turn-02-marker" in prompt
    assert "turn-11-marker" in prompt


def test_history_roles_are_represented_distinctly() -> None:
    history = (
        make_message("user", "user turn text"),
        make_message("assistant", "assistant turn text"),
    )
    service, _retrieval, chat = make_service()

    service.answer("query", history=history)

    prompt = chat.calls[0]["user_prompt"]
    assert "[user]" in prompt
    assert "[assistant]" in prompt
    assert prompt.index("[user]") < prompt.index("user turn text")
    assert prompt.index("[assistant]") < prompt.index("assistant turn text")


def test_system_prompt_describes_history_as_continuity_only() -> None:
    service, _retrieval, chat = make_service()

    service.answer("query")

    system_prompt = chat.calls[0]["system_prompt"].lower()
    assert "continuity" in system_prompt
    assert "resolve references" in system_prompt


def test_system_prompt_states_documentation_is_sole_factual_source() -> None:
    service, _retrieval, chat = make_service()

    service.answer("query")

    system_prompt = chat.calls[0]["system_prompt"].lower()
    assert "not documentation evidence" in system_prompt
    assert "sole factual source" in system_prompt
    assert "verified documentation fact" in system_prompt


def test_sources_are_unaffected_by_history() -> None:
    history = (make_message("assistant", "https://malicious.example.com/fake"),)
    service, _retrieval, _chat = make_service()

    result = service.answer("query", history=history)

    assert len(result.sources) == 1
    assert result.sources[0].url == "https://docs.example.com/page"


def test_supplied_history_sequence_is_not_mutated() -> None:
    history_list = [make_message("user", "first"), make_message("assistant", "second")]
    original_copy = list(history_list)
    service, _retrieval, _chat = make_service()

    service.answer("query", history=history_list)

    assert history_list == original_copy


def test_invalid_history_item_type_is_rejected() -> None:
    service, _retrieval, _chat = make_service()

    with pytest.raises(TypeError):
        service.answer("query", history=["not a ConversationMessage"])  # type: ignore[list-item]


# --- construction / import safety --------------------------------------------------


def test_construction_and_import_make_no_network_calls() -> None:
    settings = make_settings()
    # Constructing the service must not require a chat_client/retrieval_service
    # override and must not raise or attempt any network access.
    DocumentationAnswerService(settings)
