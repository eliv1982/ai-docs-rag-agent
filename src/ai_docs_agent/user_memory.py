"""Controlled long-term user memory in Pinecone (explicit-write only).

Memory is written only through an explicit ``remember(user_identifier, statement)``
call (or the deliberate chat command recognized by ``parse_remember_command``);
ordinary conversation, questions and documentation answers are never stored.

Identity: the raw external user identifier (e.g. a Telegram chat ID) is never
stored or logged. A stable pseudonymous identity is derived as
``HMAC-SHA256(USER_MEMORY_HASH_SECRET, raw_identifier)``; each user gets a
dedicated Pinecone namespace ``<prefix>-<first 32 hex chars of the digest>``.
Rotating USER_MEMORY_HASH_SECRET changes every derived namespace, which makes
previously written memory unreachable through the new identity (the old records
stay in their old namespaces but are never queried again).

Deduplication: the content hash is SHA-256 of the *normalized* statement
(Unicode NFC, casefold, whitespace runs collapsed to single spaces, stripped).
The record ID is derived from that hash, so repeating the same normalized
statement addresses the same record and returns a ``duplicate`` status instead
of creating a second record. The stored human-readable text is only stripped of
leading/trailing whitespace, never rewritten.

This module reuses the existing PineconeStore (embeddings, index readiness,
response parsing) and never touches the documentation namespace.
"""

import hashlib
import hmac
import logging
import re
import time
import unicodedata
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, NamedTuple

from ai_docs_agent.config import AppSettings
from ai_docs_agent.models import (
    PineconeQueryMatch,
    UserMemoryMatch,
    UserMemoryRecallResult,
    UserMemoryRecord,
    UserMemoryWriteResult,
)
from ai_docs_agent.pinecone_store import (
    PineconeEmbeddingError,
    PineconeQueryError,
    PineconeStore,
    PineconeStoreError,
)

logger = logging.getLogger(__name__)

_MEMORY_KIND = "user_memory"
_SCHEMA_VERSION = 1
_MEMORY_ID_PREFIX = "memory-"
_MEMORY_ID_HASH_CHARS = 32
_NAMESPACE_DIGEST_CHARS = 32
_SAFE_DIGEST_CHARS = 12
_MAX_USER_IDENTIFIER_LENGTH = 256
_MEMORY_KIND_FILTER: dict[str, Any] = {"kind": {"$eq": _MEMORY_KIND}}

# Deliberate, deterministic command forms only; ordinary sentences never match.
_REMEMBER_COMMAND_PATTERN = re.compile(
    r"^\s*(?:запомни|remember)\s*:\s*(?P<statement>.*)$",
    re.IGNORECASE | re.DOTALL,
)

_ALLOWED_CONTROL_CHARS = frozenset({"\t", "\n", "\r"})


class UserMemoryError(Exception):
    """Base class for domain errors raised by the long-term user memory service."""


class InvalidUserIdentifierError(UserMemoryError):
    """Raised when the raw user identifier is empty, oversized, or malformed."""


class InvalidMemoryStatementError(UserMemoryError):
    """Raised when a memory statement or recall query fails validation."""


class MemoryIdentityConfigurationError(UserMemoryError):
    """Raised when the pseudonymous-identity configuration (hash secret) is unusable."""


class MemoryEmbeddingError(UserMemoryError):
    """Raised when creating an embedding for a statement or recall query fails."""


class MemoryStorageError(UserMemoryError):
    """Raised when writing a memory record to Pinecone fails or cannot be confirmed."""


class MemoryRecallError(UserMemoryError):
    """Raised when querying Pinecone for memory records fails."""


class MalformedMemoryRecordError(UserMemoryError):
    """Raised when a Pinecone match does not carry valid user-memory metadata."""


class _UserIdentity(NamedTuple):
    namespace: str
    safe_digest: str


def parse_remember_command(text: str) -> str | None:
    """Extract the statement from an explicit remember command; None otherwise.

    Only the deliberate forms "Запомни: <statement>" / "Remember: <statement>"
    (case-insensitive, colon required) count as a command. Ordinary messages
    such as "Я использую httpx." return None and must never trigger storage.
    An empty statement after the prefix is returned as "" so the write path
    can reject it explicitly.
    """
    match = _REMEMBER_COMMAND_PATTERN.match(text)
    if match is None:
        return None
    return match.group("statement").strip()


def normalize_statement(text: str) -> str:
    """Return the documented normalized form used for hashing and deduplication.

    Normalization: Unicode NFC, casefold, whitespace runs collapsed to single
    spaces, stripped. Used only for the content hash; the stored text is never
    rewritten with this form.
    """
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def statement_content_hash(text: str) -> str:
    """Return the SHA-256 hex digest of the normalized statement."""
    return hashlib.sha256(normalize_statement(text).encode("utf-8")).hexdigest()


class UserMemoryService:
    """Explicit-write long-term user memory in per-user Pinecone namespaces."""

    def __init__(
        self,
        settings: AppSettings,
        *,
        pinecone_store: PineconeStore | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self._settings = settings
        self._pinecone_store = pinecone_store or PineconeStore(settings)
        self._clock = clock
        self._wall_clock = wall_clock

    def remember(self, user_identifier: str, statement: str) -> UserMemoryWriteResult:
        """Explicitly store one memory statement for the pseudonymous user."""
        started_at = self._clock()
        identity = self._derive_identity(user_identifier)
        record = self._build_record(statement)

        if self._record_exists(record.memory_id, identity):
            elapsed = self._clock() - started_at
            self._log_write(identity, record, status="duplicate", elapsed=elapsed)
            return UserMemoryWriteResult(
                status="duplicate",
                memory_id=record.memory_id,
                identity_digest=identity.safe_digest,
            )

        embedding = self._embed_statement(record.text, identity)
        self._upsert_record(record, embedding, identity)

        elapsed = self._clock() - started_at
        self._log_write(identity, record, status="created", elapsed=elapsed)
        return UserMemoryWriteResult(
            status="created",
            memory_id=record.memory_id,
            identity_digest=identity.safe_digest,
        )

    def recall(self, user_identifier: str, query: str) -> UserMemoryRecallResult:
        """Semantically search only the pseudonymous user's own memory namespace."""
        started_at = self._clock()
        identity = self._derive_identity(user_identifier)
        resolved_query = self._validate_text(query, description="Recall query")
        top_k = self._settings.user_memory_top_k
        threshold = self._settings.user_memory_score_threshold

        try:
            vector = self._pinecone_store.embed_query(resolved_query)
        except PineconeStoreError as exc:
            self._log_failure(identity, "recall", "embedding_failure")
            raise MemoryEmbeddingError(
                "Failed to create an embedding for the recall query."
            ) from exc

        try:
            raw_matches = self._pinecone_store.query_similar(
                vector,
                namespace=identity.namespace,
                top_k=top_k,
                metadata_filter=dict(_MEMORY_KIND_FILTER),
            )
        except PineconeQueryError as exc:
            self._log_failure(identity, "recall", "retrieval_failure")
            raise MemoryRecallError("Failed to query the user memory namespace.") from exc
        except PineconeStoreError as exc:
            self._log_failure(identity, "recall", "retrieval_failure")
            raise MemoryRecallError("Failed to query the user memory namespace.") from exc

        decoded = [self._decode_match(match) for match in raw_matches]
        accepted = sorted(
            (match for match in decoded if match.score >= threshold),
            key=lambda match: match.score,
            reverse=True,
        )

        elapsed = self._clock() - started_at
        top_scores = ", ".join(f"{match.score:.4f}" for match in accepted[:3]) or "none"
        logger.info(
            "User memory recall identity=%s query_length=%d raw_count=%d accepted_count=%d "
            "threshold=%.2f top_k=%d top_scores=[%s] elapsed_seconds=%.3f",
            identity.safe_digest,
            len(resolved_query),
            len(raw_matches),
            len(accepted),
            threshold,
            top_k,
            top_scores,
            elapsed,
        )

        return UserMemoryRecallResult(
            matches=tuple(accepted),
            found=bool(accepted),
            threshold=threshold,
            top_k=top_k,
            raw_candidate_count=len(raw_matches),
            identity_digest=identity.safe_digest,
        )

    # --- identity -----------------------------------------------------------------

    def _derive_identity(self, user_identifier: str) -> _UserIdentity:
        if not isinstance(user_identifier, str):
            raise InvalidUserIdentifierError("User identifier must be a string.")
        stripped = user_identifier.strip()
        if not stripped:
            raise InvalidUserIdentifierError("User identifier must not be blank.")
        if len(stripped) > _MAX_USER_IDENTIFIER_LENGTH:
            raise InvalidUserIdentifierError(
                f"User identifier must not exceed {_MAX_USER_IDENTIFIER_LENGTH} characters."
            )
        if any(ch for ch in stripped if unicodedata.category(ch) == "Cc"):
            raise InvalidUserIdentifierError(
                "User identifier must not contain control characters."
            )

        secret = self._settings.user_memory_hash_secret.get_secret_value()
        if not secret.strip():
            raise MemoryIdentityConfigurationError(
                "USER_MEMORY_HASH_SECRET is empty; pseudonymous identity cannot be derived."
            )

        digest = hmac.new(
            secret.encode("utf-8"), stripped.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        namespace = (
            f"{self._settings.user_memory_namespace_prefix}-"
            f"{digest[:_NAMESPACE_DIGEST_CHARS]}"
        )
        return _UserIdentity(namespace=namespace, safe_digest=digest[:_SAFE_DIGEST_CHARS])

    # --- write path ---------------------------------------------------------------

    def _build_record(self, statement: str) -> UserMemoryRecord:
        text = self._validate_text(statement, description="Memory statement")
        content_hash = statement_content_hash(text)
        memory_id = f"{_MEMORY_ID_PREFIX}{content_hash[:_MEMORY_ID_HASH_CHARS]}"
        created_at = datetime.fromtimestamp(self._wall_clock(), tz=UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        return UserMemoryRecord(
            memory_id=memory_id,
            text=text,
            content_hash=content_hash,
            memory_kind=_MEMORY_KIND,
            schema_version=_SCHEMA_VERSION,
            created_at=created_at,
        )

    def _validate_text(self, text: str, *, description: str) -> str:
        if not isinstance(text, str):
            raise InvalidMemoryStatementError(f"{description} must be a string.")
        stripped = text.strip()
        if not stripped:
            raise InvalidMemoryStatementError(f"{description} must not be blank.")
        max_length = self._settings.user_memory_max_statement_length
        if len(stripped) > max_length:
            raise InvalidMemoryStatementError(
                f"{description} must not exceed {max_length} characters."
            )
        for ch in stripped:
            if unicodedata.category(ch) == "Cc" and ch not in _ALLOWED_CONTROL_CHARS:
                raise InvalidMemoryStatementError(
                    f"{description} must not contain control characters."
                )
        return stripped

    def _record_exists(self, memory_id: str, identity: _UserIdentity) -> bool:
        try:
            existing = self._pinecone_store.fetch_existing_ids(
                [memory_id], namespace=identity.namespace
            )
        except PineconeStoreError as exc:
            self._log_failure(identity, "remember", "storage_failure")
            raise MemoryStorageError(
                "Failed to check the user memory namespace for an existing record."
            ) from exc
        return memory_id in existing

    def _embed_statement(self, text: str, identity: _UserIdentity) -> list[float]:
        try:
            embeddings = self._pinecone_store.embed_documents([text])
        except PineconeEmbeddingError as exc:
            self._log_failure(identity, "remember", "embedding_failure")
            raise MemoryEmbeddingError(
                "Failed to create an embedding for the memory statement."
            ) from exc
        except PineconeStoreError as exc:
            self._log_failure(identity, "remember", "embedding_failure")
            raise MemoryEmbeddingError(
                "Failed to create an embedding for the memory statement."
            ) from exc
        if len(embeddings) != 1:
            self._log_failure(identity, "remember", "embedding_failure")
            raise MemoryEmbeddingError(
                f"Expected exactly one embedding, received {len(embeddings)}."
            )
        return embeddings[0]

    def _upsert_record(
        self, record: UserMemoryRecord, embedding: list[float], identity: _UserIdentity
    ) -> None:
        vector: dict[str, object] = {
            "id": record.memory_id,
            "values": embedding,
            "metadata": {
                "kind": record.memory_kind,
                "text": record.text,
                "content_hash": record.content_hash,
                "schema_version": record.schema_version,
                "created_at": record.created_at,
            },
        }
        try:
            upserted = self._pinecone_store.upsert_vectors(
                [vector], namespace=identity.namespace
            )
        except PineconeStoreError as exc:
            self._log_failure(identity, "remember", "storage_failure")
            raise MemoryStorageError("Failed to store the memory record in Pinecone.") from exc
        if upserted != 1:
            self._log_failure(identity, "remember", "storage_failure")
            raise MemoryStorageError(
                f"Pinecone confirmed {upserted} upserted record(s); expected exactly 1."
            )

    # --- recall decoding ----------------------------------------------------------

    def _decode_match(self, match: PineconeQueryMatch) -> UserMemoryMatch:
        metadata = match.metadata
        kind = metadata.get("kind")
        if kind != _MEMORY_KIND:
            raise MalformedMemoryRecordError(
                f"Memory match '{match.id}' has unexpected kind '{kind}'."
            )
        text = metadata.get("text")
        if not isinstance(text, str) or not text.strip():
            raise MalformedMemoryRecordError(
                f"Memory match '{match.id}' metadata 'text' must be a non-blank string."
            )
        content_hash = metadata.get("content_hash")
        if not isinstance(content_hash, str) or not content_hash.strip():
            raise MalformedMemoryRecordError(
                f"Memory match '{match.id}' metadata 'content_hash' must be a "
                "non-blank string."
            )
        return UserMemoryMatch(
            memory_id=match.id,
            text=text,
            score=match.score,
            memory_kind=kind,
        )

    # --- observability ------------------------------------------------------------

    @staticmethod
    def _log_write(
        identity: _UserIdentity,
        record: UserMemoryRecord,
        *,
        status: str,
        elapsed: float,
    ) -> None:
        logger.info(
            "User memory remember identity=%s status=%s statement_length=%d "
            "elapsed_seconds=%.3f",
            identity.safe_digest,
            status,
            len(record.text),
            elapsed,
        )

    @staticmethod
    def _log_failure(identity: _UserIdentity, operation: str, category: str) -> None:
        logger.warning(
            "User memory %s failed identity=%s category=%s",
            operation,
            identity.safe_digest,
            category,
        )
