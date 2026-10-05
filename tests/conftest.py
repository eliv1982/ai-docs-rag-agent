"""Shared pytest fixtures."""

from collections.abc import Iterator

import pytest

from ai_docs_agent import observability


@pytest.fixture(autouse=True)
def _restore_session_hash_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Undo any configure_session_hash_secret() call so tests never leak the keyed state."""
    monkeypatch.setattr(observability, "_session_hash_key", observability._session_hash_key)
    yield
