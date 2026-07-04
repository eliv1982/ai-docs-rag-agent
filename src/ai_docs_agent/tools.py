"""Thin tool-facing adapters over the project's read-only service layer."""

from collections.abc import Sequence

from ai_docs_agent.agent import DocumentationAnswerService
from ai_docs_agent.config import AppSettings, get_settings
from ai_docs_agent.models import ConversationMessage, GroundedAnswerResult, PyPIPackageInfo
from ai_docs_agent.pypi import PyPILookupService


def answer_documentation_question(
    question: str,
    *,
    service: DocumentationAnswerService | None = None,
    settings: AppSettings | None = None,
    history: Sequence[ConversationMessage] = (),
) -> GroundedAnswerResult:
    """Answer one documentation question through the existing grounded RAG path.

    `history`, when supplied by a trusted request-scoped adapter, only feeds the
    existing contextual-retrieval behavior; it is never documentary evidence.
    """
    resolved_service = service or DocumentationAnswerService(settings or get_settings())
    return resolved_service.answer(question, history=history)


def lookup_pypi_package(
    package_name: str,
    *,
    service: PyPILookupService | None = None,
    settings: AppSettings | None = None,
) -> PyPIPackageInfo:
    """Look up one package through the real PyPI JSON API via a typed service."""
    resolved_service = service or PyPILookupService(settings or get_settings())
    return resolved_service.lookup(package_name)
