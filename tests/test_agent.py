"""Unit tests for DocumentationAnswerService. Uses fake retrieval and chat clients
only; no real network, OpenAI, or Pinecone calls."""

import logging
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
from ai_docs_agent.observability import hash_session_id, request_logging_context
from ai_docs_agent.retrieval import RetrievalError

_REQUIRED: dict[str, Any] = {
    "openai_api_key": "sk-test-openai",
    "pinecone_api_key": "pc-test-key",
    "openai_chat_model": "gpt-4o-mini",
    "telegram_bot_token": "test-telegram-token",
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
        results: list[RetrievalResult] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._result = result
        self._results = list(results) if results is not None else None
        self._error = error
        self.calls: list[tuple[str, int | None, str | None]] = []

    def search(
        self, query: str, *, top_k: int | None = None, namespace: str | None = None
    ) -> RetrievalResult:
        self.calls.append((query, top_k, namespace))
        if self._error is not None:
            raise self._error
        if self._results is not None:
            assert self._results, "FakeRetrievalService.search called more times than expected."
            return self._results.pop(0)
        assert self._result is not None
        return self._result


class FakeChatClient:
    """Fake ChatClient recording the exact prompts it was called with."""

    def __init__(
        self,
        *,
        answer: str = "The API key is set via OPENAI_API_KEY.",
        answers: list[str] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._answer = answer
        self._answers = list(answers) if answers is not None else None
        self._error = error
        self.calls: list[dict[str, str]] = []

    def complete(self, *, model: str, system_prompt: str, user_prompt: str) -> str:
        self.calls.append(
            {"model": model, "system_prompt": system_prompt, "user_prompt": user_prompt}
        )
        if self._error is not None:
            raise self._error
        if self._answers is not None:
            assert self._answers, "FakeChatClient.complete called more times than expected."
            return self._answers.pop(0)
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


def test_answer_logs_safe_request_diagnostics_for_grounded_answer(
    caplog: pytest.LogCaptureFixture,
) -> None:
    chunk = make_chunk(
        score=0.91,
        text="LEAK_CHUNK_BODY sk-live-secret vector=[0.1,0.2,0.3]",
    )
    question = "What are text splitters used for in LangChain?"
    retrieval = FakeRetrievalService(
        result=make_retrieval_result(query=question, matches=(chunk,))
    )
    service, _retrieval, _chat = make_service(retrieval=retrieval)

    with request_logging_context(session_id="chat-123456"):
        with caplog.at_level(logging.INFO):
            service.answer(question)

    assert "session_hash=" + hash_session_id("chat-123456") in caplog.text
    assert "question_length=46" in caplog.text
    assert "raw_candidate_count=1" in caplog.text
    assert "accepted_candidate_count=1" in caplog.text
    assert "top_candidate_scores=[0.91]" in caplog.text
    assert "outcome=grounded" in caplog.text
    assert "score_threshold=0.25" in caplog.text
    assert "chat-123456" not in caplog.text
    assert question not in caplog.text
    assert "LEAK_CHUNK_BODY" not in caplog.text
    assert "sk-live-secret" not in caplog.text
    assert "vector=[0.1,0.2,0.3]" not in caplog.text


# --- minimum relevance score gate ---------------------------------------------------


def test_low_relevance_matches_trigger_no_context_fallback_and_skip_chat_call() -> None:
    chunk = make_chunk(score=0.0556)
    retrieval = FakeRetrievalService(result=make_retrieval_result(matches=(chunk,)))
    chat = FakeChatClient()
    service, _retrieval, _chat = make_service(retrieval=retrieval, chat=chat)

    result = service.answer("Как приготовить борщ?")

    assert result.retrieved_chunk_count == 0
    assert result.sources == ()
    assert "не найдено" in result.answer
    assert chat.calls == []


def test_no_context_fallback_logs_candidate_and_threshold_info_safely(
    caplog: pytest.LogCaptureFixture,
) -> None:
    chunk = make_chunk(
        score=0.0556,
        text="LEAK_CHUNK_BODY should-not-appear",
    )
    question = "How do I configure the client?"
    retrieval = FakeRetrievalService(
        result=make_retrieval_result(query=question, matches=(chunk,))
    )
    service, _retrieval, _chat = make_service(retrieval=retrieval)

    with request_logging_context(session_id="987654321"):
        with caplog.at_level(logging.INFO):
            result = service.answer(question)

    assert result.retrieved_chunk_count == 0
    assert "session_hash=" + hash_session_id("987654321") in caplog.text
    assert "question_length=30" in caplog.text
    assert "raw_candidate_count=1" in caplog.text
    assert "accepted_candidate_count=0" in caplog.text
    assert "top_candidate_scores=[0.0556]" in caplog.text
    assert "score_threshold=0.25" in caplog.text
    assert "outcome=no_context" in caplog.text
    assert "987654321" not in caplog.text
    assert question not in caplog.text
    assert "LEAK_CHUNK_BODY" not in caplog.text


def test_direct_retrieval_first_pass_uses_only_latest_question_even_with_alias_history() -> None:
    history = (
        make_message(
            "user", "В этом диалоге называй RecursiveCharacterTextSplitter Резаком."
        ),
        make_message(
            "assistant",
            "Хорошо, в рамках текущего диалога буду называть "
            "RecursiveCharacterTextSplitter Резаком.",
        ),
    )
    retrieval = FakeRetrievalService(
        results=[
            make_retrieval_result(query="Для чего нужен Резак?", matches=()),
            make_retrieval_result(
                query="Для чего нужен RecursiveCharacterTextSplitter?",
                matches=(make_chunk(score=0.58),),
            ),
        ]
    )
    chat = FakeChatClient(
        answers=[
            "Для чего нужен RecursiveCharacterTextSplitter?",
            "Он нужен, чтобы рекурсивно разбивать текст.",
        ]
    )
    service, _retrieval, _chat = make_service(retrieval=retrieval, chat=chat)

    service.answer("Для чего нужен Резак?", history=history)

    assert retrieval.calls[0] == ("Для чего нужен Резак?", None, None)
    assert "RecursiveCharacterTextSplitter" not in retrieval.calls[0][0]


def test_relevant_documentation_match_clears_the_relevance_gate() -> None:
    chunk = make_chunk(score=0.6474)
    retrieval = FakeRetrievalService(result=make_retrieval_result(matches=(chunk,)))
    service, _retrieval, chat = make_service(retrieval=retrieval)

    result = service.answer(
        "Как RecursiveCharacterTextSplitter определяет, где разделять текст?"
    )

    assert result.retrieved_chunk_count == 1
    assert chat.calls


def test_relevance_gate_filters_only_the_low_score_chunks() -> None:
    relevant = make_chunk(
        chunk_id="rel", document_id="doc-rel", score=0.6, title="Relevant Page"
    )
    irrelevant = make_chunk(
        chunk_id="irr", document_id="doc-irr", score=0.05, title="Irrelevant Page"
    )
    retrieval = FakeRetrievalService(result=make_retrieval_result(matches=(relevant, irrelevant)))
    service, _retrieval, chat = make_service(retrieval=retrieval)

    result = service.answer("query")

    assert result.retrieved_chunk_count == 1
    assert result.sources[0].title == "Relevant Page"
    prompt = chat.calls[0]["user_prompt"]
    assert "Relevant Page" in prompt
    assert "Irrelevant Page" not in prompt


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


def test_system_prompt_instructs_answering_current_message_only() -> None:
    service, _retrieval, chat = make_service()

    service.answer("query")

    system_prompt = chat.calls[0]["system_prompt"].lower()
    assert "current user message" in system_prompt
    assert "the only message you must" in system_prompt
    assert "do not repeat, continue, or restate a previous answer" in system_prompt


def test_system_prompt_instructs_acknowledging_conversational_messages() -> None:
    service, _retrieval, chat = make_service()

    service.answer("query")

    system_prompt = chat.calls[0]["system_prompt"].lower()
    assert "conversational instruction" in system_prompt
    assert "naming/alias request" in system_prompt
    assert "acknowledge it briefly" in system_prompt
    assert "instead of summarizing the documentation context" in system_prompt


def test_current_message_is_placed_after_history_and_context() -> None:
    history = (make_message("user", "HISTORY_MARKER"),)
    chunk = make_chunk(text="Some text about CONTEXT_MARKER here.")
    retrieval = FakeRetrievalService(
        result=make_retrieval_result(query="CURRENT_MESSAGE_MARKER", matches=(chunk,))
    )
    service, _retrieval, chat = make_service(retrieval=retrieval)

    service.answer("anything", history=history)

    prompt = chat.calls[0]["user_prompt"]
    assert prompt.index("HISTORY_MARKER") < prompt.index("CONTEXT_MARKER")
    assert prompt.index("CONTEXT_MARKER") < prompt.index("CURRENT_MESSAGE_MARKER")


def test_contextual_retry_runs_after_direct_no_context_result_when_history_exists() -> None:
    history = (
        make_message(
            "user", "В этом диалоге называй RecursiveCharacterTextSplitter Резаком."
        ),
        make_message(
            "assistant",
            "Хорошо, в рамках текущего диалога буду называть "
            "RecursiveCharacterTextSplitter Резаком.",
        ),
    )
    splitter_chunk = make_chunk(
        score=0.61,
        title="Splitting recursively - Text splitter integration guide - Docs by LangChain",
        final_url="https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter",
        source_url="https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter",
        document_id="langchain-splitter-doc",
        text="RecursiveCharacterTextSplitter recursively splits text until chunks fit.",
    )
    retrieval = FakeRetrievalService(
        results=[
            make_retrieval_result(query="Для чего нужен Резак?", matches=()),
            make_retrieval_result(
                query="Для чего нужен RecursiveCharacterTextSplitter?",
                matches=(splitter_chunk,),
            ),
        ]
    )
    chat = FakeChatClient(
        answers=[
            "Для чего нужен RecursiveCharacterTextSplitter?",
            "Он нужен, чтобы рекурсивно разбивать текст на части подходящего размера.",
        ]
    )
    service, _retrieval, _chat = make_service(retrieval=retrieval, chat=chat)

    result = service.answer("Для чего нужен Резак?", history=history)

    assert retrieval.calls == [
        ("Для чего нужен Резак?", None, None),
        ("Для чего нужен RecursiveCharacterTextSplitter?", None, None),
    ]
    assert len(chat.calls) == 2
    assert result.question == "Для чего нужен Резак?"
    assert result.retrieved_chunk_count == 1
    assert result.sources[0].url == (
        "https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter"
    )
    assert result.sources[0].title.startswith("Splitting recursively")


def test_contextual_retry_preserves_grounding_and_never_uses_history_as_source() -> None:
    history = (
        make_message(
            "user", "В этом диалоге называй RecursiveCharacterTextSplitter Резаком."
        ),
        make_message("assistant", "Хорошо, буду называть его Резаком."),
    )
    splitter_chunk = make_chunk(
        score=0.58,
        title="Recursive Splitter Docs",
        final_url="https://docs.example.com/recursive-splitter",
        source_url="https://docs.example.com/recursive-splitter",
        document_id="doc-splitter",
        text="RecursiveCharacterTextSplitter splits long text recursively by separators.",
    )
    retrieval = FakeRetrievalService(
        results=[
            make_retrieval_result(query="Для чего нужен Резак?", matches=()),
            make_retrieval_result(
                query="Для чего нужен RecursiveCharacterTextSplitter?",
                matches=(splitter_chunk,),
            ),
        ]
    )
    chat = FakeChatClient(
        answers=[
            "Для чего нужен RecursiveCharacterTextSplitter?",
            "Он нужен, чтобы разбивать длинный текст на части.",
        ]
    )
    service, _retrieval, _chat = make_service(retrieval=retrieval, chat=chat)

    result = service.answer("Для чего нужен Резак?", history=history)

    assert result.answer == "Он нужен, чтобы разбивать длинный текст на части."
    assert len(result.sources) == 1
    assert result.sources[0].title == "Recursive Splitter Docs"
    assert "Резак" not in result.sources[0].title
    assert "Резак" not in result.sources[0].url


def test_contextual_retry_uses_history_augmented_query_when_rewrite_is_unchanged() -> None:
    history = (
        make_message(
            "user", "В этом диалоге называй RecursiveCharacterTextSplitter Резаком."
        ),
        make_message(
            "assistant",
            "Хорошо, в рамках текущего диалога буду называть "
            "RecursiveCharacterTextSplitter Резаком.",
        ),
    )
    retrieval = FakeRetrievalService(
        results=[
            make_retrieval_result(query="Для чего нужен Резак?", matches=()),
            make_retrieval_result(
                query=(
                    "Current documentation question:\n"
                    "Для чего нужен Резак?\n\n"
                    "Recent conversation for reference resolution:\n"
                    "[user] В этом диалоге называй RecursiveCharacterTextSplitter Резаком.\n"
                    "[assistant] Хорошо, в рамках текущего диалога буду называть "
                    "RecursiveCharacterTextSplitter Резаком."
                ),
                matches=(make_chunk(score=0.61),),
            ),
        ]
    )
    chat = FakeChatClient(
        answers=[
            "Для чего нужен Резак?",
            "Он нужен, чтобы рекурсивно разбивать текст на части.",
        ]
    )
    service, _retrieval, _chat = make_service(retrieval=retrieval, chat=chat)

    result = service.answer("Для чего нужен Резак?", history=history)

    assert retrieval.calls[0] == ("Для чего нужен Резак?", None, None)
    assert len(retrieval.calls) == 2
    assert "RecursiveCharacterTextSplitter" in retrieval.calls[1][0]
    assert "[user]" in retrieval.calls[1][0]
    assert result.retrieved_chunk_count == 1
    assert result.question == "Для чего нужен Резак?"


def test_direct_retrieval_success_skips_contextual_retry_even_with_history() -> None:
    history = (make_message("user", "Назовем библиотеку коротко."),)
    retrieval = FakeRetrievalService(
        result=make_retrieval_result(
            query="Что делает RecursiveCharacterTextSplitter?",
            matches=(make_chunk(score=0.64),),
        )
    )
    chat = FakeChatClient(answer="Он рекурсивно разбивает текст.")
    service, _retrieval, _chat = make_service(retrieval=retrieval, chat=chat)

    result = service.answer("Что делает RecursiveCharacterTextSplitter?", history=history)

    assert retrieval.calls == [("Что делает RecursiveCharacterTextSplitter?", None, None)]
    assert len(chat.calls) == 1
    assert result.retrieved_chunk_count == 1


def test_contextual_retry_logs_only_safe_metadata(
    caplog: pytest.LogCaptureFixture,
) -> None:
    history = (
        make_message(
            "user", "LEAK_ALIAS В этом диалоге называй RecursiveCharacterTextSplitter Резаком."
        ),
        make_message("assistant", "LEAK_ALIAS_ACK Хорошо, буду называть его Резаком."),
    )
    splitter_chunk = make_chunk(
        score=0.58,
        title="Recursive Splitter Docs",
        final_url="https://docs.example.com/recursive-splitter",
        source_url="https://docs.example.com/recursive-splitter",
        document_id="doc-splitter",
        text="LEAK_CHUNK_BODY RecursiveCharacterTextSplitter splits long text.",
    )
    retrieval = FakeRetrievalService(
        results=[
            make_retrieval_result(query="Для чего нужен Резак?", matches=()),
            make_retrieval_result(
                query="Для чего нужен RecursiveCharacterTextSplitter?",
                matches=(splitter_chunk,),
            ),
        ]
    )
    chat = FakeChatClient(
        answers=[
            "Для чего нужен RecursiveCharacterTextSplitter?",
            "Он нужен, чтобы разбивать длинный текст на части.",
        ]
    )
    service, _retrieval, _chat = make_service(retrieval=retrieval, chat=chat)

    with request_logging_context(session_id="chat-424242"):
        with caplog.at_level(logging.INFO):
            service.answer("Для чего нужен Резак?", history=history)

    assert "session_hash=" + hash_session_id("chat-424242") in caplog.text
    assert "retrieval_pass=direct" in caplog.text
    assert "retrieval_pass=contextual" in caplog.text
    assert "query_length=" in caplog.text
    assert "raw_candidate_count=0" in caplog.text
    assert "accepted_candidate_count=1" in caplog.text
    assert "LEAK_ALIAS" not in caplog.text
    assert "LEAK_ALIAS_ACK" not in caplog.text
    assert "Для чего нужен Резак?" not in caplog.text
    assert "chat-424242" not in caplog.text
    assert "LEAK_CHUNK_BODY" not in caplog.text


def test_conversational_message_can_be_acknowledged_without_documentation_summary() -> None:
    chunk = make_chunk(score=0.55)
    retrieval = FakeRetrievalService(result=make_retrieval_result(matches=(chunk,)))
    acknowledgement = (
        "Хорошо, в рамках текущего диалога буду называть "
        "RecursiveCharacterTextSplitter «Резак»."
    )
    chat = FakeChatClient(answer=acknowledgement)
    service, _retrieval, _chat = make_service(retrieval=retrieval, chat=chat)

    result = service.answer(
        "В этом диалоге будем называть RecursiveCharacterTextSplitter «Резак»."
    )

    assert result.answer == acknowledgement


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
