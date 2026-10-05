"""Every production OpenAI client gets the one configurable finite timeout.

The real client classes are constructed (construction never touches the network)
and their effective timeout is inspected. No network access.
"""

from typing import Any

import pytest
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI
from pydantic import ValidationError

from ai_docs_agent.agent import OpenAIChatClient
from ai_docs_agent.config import AppSettings
from ai_docs_agent.langchain_agent import build_chat_model
from ai_docs_agent.pinecone_store import PineconeStore

_REQUIRED: dict[str, Any] = {
    "openai_api_key": "sk-test-openai",
    "pinecone_api_key": "pc-test-key",
    "openai_chat_model": "gpt-4o-mini",
    "telegram_bot_token": "test-telegram-token",
    "user_memory_hash_secret": "unit-test-user-memory-secret",
}


def make_settings(**overrides: Any) -> AppSettings:
    return AppSettings(_env_file=None, **{**_REQUIRED, **overrides})


def test_langchain_chat_model_uses_the_configured_timeout() -> None:
    model = build_chat_model(make_settings(openai_timeout_seconds=7.5))

    assert isinstance(model, ChatOpenAI)
    assert model.request_timeout == 7.5
    assert model.root_client.timeout == 7.5


def test_langchain_embeddings_use_the_configured_timeout() -> None:
    embeddings = PineconeStore(make_settings(openai_timeout_seconds=7.5))._embedding_client

    assert isinstance(embeddings, OpenAIEmbeddings)
    assert embeddings.request_timeout == 7.5
    assert embeddings.client._client.timeout == 7.5


def test_direct_grounded_answer_client_uses_the_configured_timeout() -> None:
    client = OpenAIChatClient(make_settings(openai_timeout_seconds=7.5))._openai_client

    assert isinstance(client, OpenAI)
    assert client.timeout == 7.5


def test_all_three_clients_default_to_one_finite_timeout() -> None:
    settings = make_settings()

    timeouts = {
        build_chat_model(settings).request_timeout,
        PineconeStore(settings)._embedding_client.request_timeout,
        OpenAIChatClient(settings)._openai_client.timeout,
    }

    assert timeouts == {settings.openai_timeout_seconds}
    assert 0 < settings.openai_timeout_seconds < float("inf")


def test_timeout_is_configurable_through_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "21")

    settings = make_settings()

    assert settings.openai_timeout_seconds == 21
    assert build_chat_model(settings).request_timeout == 21


@pytest.mark.parametrize("value", [0, -1])
def test_non_positive_timeout_is_rejected(value: float) -> None:
    with pytest.raises(ValidationError):
        make_settings(openai_timeout_seconds=value)
