"""Privacy-safe helpers for request-scoped operational logging."""

import hashlib
import hmac
import logging
import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token

_REQUEST_SESSION_HASH: ContextVar[str | None] = ContextVar(
    "ai_docs_agent_request_session_hash",
    default=None,
)

# Domain separation: long-term memory derives its identity as
# HMAC(secret, raw_identifier) with no prefix, so the observability hash is keyed
# over a distinct, prefixed input domain and can never equal a memory digest.
_SESSION_HASH_DOMAIN = b"observability-session:"
_SESSION_HASH_CHARS = 12

# Until the runtime supplies its secret, a random per-process key is used: hashes
# stay stable within the process but are never reproducible from the numeric ID.
_session_hash_key: bytes = secrets.token_bytes(32)


def configure_session_hash_secret(secret: str) -> None:
    """Key session hashing with a runtime secret so hashes are stable across restarts."""
    global _session_hash_key
    if not secret.strip():
        raise ValueError("The session hash secret must not be empty.")
    _session_hash_key = secret.encode("utf-8")


def hash_session_id(session_id: str) -> str:
    """Return a short, stable, keyed identifier for a session; the raw ID is never exposed.

    HMAC-SHA256 over a domain-prefixed ID: without the key, a short digest cannot
    be used to enumerate predictable numeric chat IDs.
    """
    digest = hmac.new(
        _session_hash_key, _SESSION_HASH_DOMAIN + session_id.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return digest[:_SESSION_HASH_CHARS]


def current_request_session_hash() -> str | None:
    """Return the current request-scoped session hash, if one is set."""
    return _REQUEST_SESSION_HASH.get()


@contextmanager
def request_logging_context(*, session_id: str | None = None) -> Iterator[None]:
    """Attach a privacy-safe session hash to logs emitted during the current request."""
    token: Token[str | None] | None = None
    if session_id is not None:
        token = _REQUEST_SESSION_HASH.set(hash_session_id(session_id))
    try:
        yield
    finally:
        if token is not None:
            _REQUEST_SESSION_HASH.reset(token)


def log_exception_safely(
    logger: logging.Logger,
    message: str,
    *,
    exc: BaseException,
) -> None:
    """Log a traceback without re-emitting an exception's potentially sensitive message."""
    try:
        sanitized_exception = type(exc)()
    except Exception:
        sanitized_exception = Exception()

    logger.exception(
        message,
        exc_info=(type(exc), sanitized_exception, exc.__traceback__),
    )
