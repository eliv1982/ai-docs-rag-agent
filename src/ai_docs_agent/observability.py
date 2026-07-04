"""Privacy-safe helpers for request-scoped operational logging."""

import hashlib
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token

_REQUEST_SESSION_HASH: ContextVar[str | None] = ContextVar(
    "ai_docs_agent_request_session_hash",
    default=None,
)


def hash_session_id(session_id: str) -> str:
    """Return a short, stable, privacy-safe hash for a session identifier."""
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:12]


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
