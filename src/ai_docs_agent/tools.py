"""Thin tool-facing adapters over external read-only API integrations.

LangChain agent routing is not implemented yet; this module currently provides only
small typed adapter functions that can be wrapped later.
"""

from ai_docs_agent.config import AppSettings, get_settings
from ai_docs_agent.models import PyPIPackageInfo
from ai_docs_agent.pypi import PyPILookupService


def lookup_pypi_package(
    package_name: str,
    *,
    service: PyPILookupService | None = None,
    settings: AppSettings | None = None,
) -> PyPIPackageInfo:
    """Look up one package through the real PyPI JSON API via a typed service."""
    resolved_service = service or PyPILookupService(settings or get_settings())
    return resolved_service.lookup(package_name)
