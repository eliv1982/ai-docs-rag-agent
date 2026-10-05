"""Keyed (HMAC) session identifiers used in logs. No network access."""

import hashlib
import hmac

import pytest

from ai_docs_agent.config import AppSettings
from ai_docs_agent.observability import (
    configure_session_hash_secret,
    current_request_session_hash,
    hash_session_id,
    request_logging_context,
)
from ai_docs_agent.user_memory import UserMemoryService

_SECRET = "unit-test-observability-secret"
_CHAT_ID = "424242424"


def expected_hash(secret: str, session_id: str) -> str:
    return hmac.new(
        secret.encode(), b"observability-session:" + session_id.encode(), hashlib.sha256
    ).hexdigest()[:12]


def test_same_id_and_secret_give_a_stable_identifier() -> None:
    configure_session_hash_secret(_SECRET)

    assert hash_session_id(_CHAT_ID) == hash_session_id(_CHAT_ID)
    assert hash_session_id(_CHAT_ID) == expected_hash(_SECRET, _CHAT_ID)


def test_identifier_is_short_lowercase_hex() -> None:
    configure_session_hash_secret(_SECRET)

    identifier = hash_session_id(_CHAT_ID)

    assert len(identifier) == 12
    assert identifier == identifier.lower()
    int(identifier, 16)  # raises ValueError if not hex


def test_different_ids_give_different_identifiers() -> None:
    configure_session_hash_secret(_SECRET)

    identifiers = {hash_session_id(str(chat_id)) for chat_id in range(1000, 1500)}

    assert len(identifiers) == 500


def test_different_secrets_give_different_identifiers() -> None:
    configure_session_hash_secret(_SECRET)
    first = hash_session_id(_CHAT_ID)
    configure_session_hash_secret("a-different-secret")
    second = hash_session_id(_CHAT_ID)

    assert first != second
    assert second == expected_hash("a-different-secret", _CHAT_ID)


def test_raw_numeric_id_is_absent_from_the_identifier() -> None:
    configure_session_hash_secret(_SECRET)

    assert _CHAT_ID not in hash_session_id(_CHAT_ID)
    with request_logging_context(session_id=_CHAT_ID):
        bound = current_request_session_hash()
    assert bound is not None
    assert _CHAT_ID not in bound


def test_plain_sha256_of_the_id_does_not_reproduce_the_identifier() -> None:
    configure_session_hash_secret(_SECRET)
    identifier = hash_session_id(_CHAT_ID)

    plain = hashlib.sha256(_CHAT_ID.encode()).hexdigest()

    assert identifier != plain[:12]
    # An observer who can only compute unsalted hashes of numeric IDs cannot find it.
    enumerated = {
        hashlib.sha256(str(candidate).encode()).hexdigest()[:12]
        for candidate in range(420_000_000, 420_000_000 + 5000)
    } | {plain[:12]}
    assert identifier not in enumerated


def test_identifier_does_not_expose_the_secret() -> None:
    configure_session_hash_secret(_SECRET)

    assert _SECRET not in hash_session_id(_CHAT_ID)


def test_identifier_is_domain_separated_from_the_long_term_memory_identity() -> None:
    configure_session_hash_secret(_SECRET)
    settings = AppSettings(
        _env_file=None,
        openai_api_key="sk-test-openai",
        pinecone_api_key="pc-test-key",
        openai_chat_model="gpt-4o-mini",
        telegram_bot_token="test-telegram-token",
        user_memory_hash_secret=_SECRET,
    )
    # Identity derivation is purely local; constructing the service makes no network call.
    memory_identity = UserMemoryService(settings)._derive_identity(_CHAT_ID)

    assert hash_session_id(_CHAT_ID) != memory_identity.safe_digest
    assert hash_session_id(_CHAT_ID) != hmac.new(
        _SECRET.encode(), _CHAT_ID.encode(), hashlib.sha256
    ).hexdigest()[:12]


def test_unconfigured_process_still_hashes_with_a_random_key_not_the_plain_hash() -> None:
    # No configure call: the per-process random fallback key applies.
    first = hash_session_id(_CHAT_ID)

    assert first == hash_session_id(_CHAT_ID)
    assert first != hashlib.sha256(_CHAT_ID.encode()).hexdigest()[:12]
    assert _CHAT_ID not in first


@pytest.mark.parametrize("blank", ["", "   ", "\n"])
def test_blank_secret_is_rejected(blank: str) -> None:
    with pytest.raises(ValueError):
        configure_session_hash_secret(blank)


def test_request_logging_context_binds_the_keyed_identifier() -> None:
    configure_session_hash_secret(_SECRET)

    assert current_request_session_hash() is None
    with request_logging_context(session_id=_CHAT_ID):
        assert current_request_session_hash() == expected_hash(_SECRET, _CHAT_ID)
    assert current_request_session_hash() is None
