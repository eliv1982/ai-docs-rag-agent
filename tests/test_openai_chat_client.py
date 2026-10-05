"""OpenAIChatClient request construction, inspected on the wire.

The real OpenAI SDK client is driven through an httpx.MockTransport, so the JSON
body that would be sent to the API is captured and asserted. No network access.
"""

import json
import logging
from typing import Any

import httpx
import pytest
from openai import OpenAI

from ai_docs_agent.agent import OpenAIChatClient
from ai_docs_agent.config import AppSettings

_REQUIRED: dict[str, Any] = {
    "openai_api_key": "sk-test-openai",
    "pinecone_api_key": "pc-test-key",
    "openai_chat_model": "gpt-4o-mini",
    "telegram_bot_token": "test-telegram-token",
    "user_memory_hash_secret": "unit-test-user-memory-secret",
}


def make_settings(**overrides: Any) -> AppSettings:
    return AppSettings(_env_file=None, **{**_REQUIRED, **overrides})


def make_wired_client(
    settings: AppSettings, *, finish_reason: str = "stop"
) -> tuple[OpenAIChatClient, list[dict[str, Any]]]:
    """An OpenAIChatClient whose SDK client records every request body it would send."""
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": settings.openai_chat_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Grounded answer."},
                        "finish_reason": finish_reason,
                    }
                ],
            },
        )

    client = OpenAIChatClient(settings)
    client._client = OpenAI(
        api_key="sk-test-openai",
        timeout=settings.openai_timeout_seconds,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return client, bodies


def test_decoding_controls_reach_the_outgoing_request() -> None:
    client, bodies = make_wired_client(make_settings())

    answer = client.complete(
        model="gpt-4o-mini",
        system_prompt="SYSTEM",
        user_prompt="USER",
        temperature=0.0,
        max_output_tokens=1024,
    )

    assert answer == "Grounded answer."
    (body,) = bodies
    assert body["model"] == "gpt-4o-mini"
    assert body["messages"] == [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "USER"},
    ]
    assert body["temperature"] == 0
    assert body["max_completion_tokens"] == 1024
    # The deprecated parameter must not be sent alongside the supported one.
    assert "max_tokens" not in body


def test_request_without_controls_is_unchanged() -> None:
    client, bodies = make_wired_client(make_settings())

    client.complete(model="gpt-4o-mini", system_prompt="SYSTEM", user_prompt="USER")

    (body,) = bodies
    assert set(body) == {"model", "messages"}


def test_timeout_and_retry_behavior_are_preserved() -> None:
    settings = make_settings(openai_timeout_seconds=7.5)

    sdk_client = OpenAIChatClient(settings)._openai_client

    assert sdk_client.timeout == 7.5
    assert sdk_client.max_retries == OpenAI(api_key="sk-test-openai").max_retries


def test_truncation_at_the_output_limit_is_logged_without_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client, _bodies = make_wired_client(make_settings(), finish_reason="length")

    with caplog.at_level(logging.WARNING):
        answer = client.complete(
            model="gpt-4o-mini",
            system_prompt="SYSTEM",
            user_prompt="LEAK_USER_PROMPT",
            temperature=0.0,
            max_output_tokens=1024,
        )

    assert answer == "Grounded answer."
    assert "output token limit" in caplog.text
    assert "max_output_tokens=1024" in caplog.text
    assert "LEAK_USER_PROMPT" not in caplog.text
    assert "Grounded answer." not in caplog.text


def test_normal_completion_logs_no_truncation_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client, _bodies = make_wired_client(make_settings())

    with caplog.at_level(logging.WARNING):
        client.complete(
            model="gpt-4o-mini", system_prompt="S", user_prompt="U", max_output_tokens=1024
        )

    assert "output token limit" not in caplog.text
