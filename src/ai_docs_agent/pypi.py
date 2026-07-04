"""Typed read-only PyPI package metadata lookup via the JSON API.

Pipeline: validate package name -> GET /pypi/{package_name}/json with an explicit
timeout -> decode JSON -> validate required fields -> PyPIPackageInfo. No retry
logic, LangChain integration, or Telegram-specific behavior is implemented here.
"""

import re
from typing import Any

import httpx

from ai_docs_agent.config import AppSettings
from ai_docs_agent.models import PyPIPackageInfo

_PACKAGE_NAME_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,199})?$")
_PREFERRED_PROJECT_URL_LABELS = (
    "Homepage",
    "Home",
    "Source",
    "Source Code",
    "Repository",
    "Documentation",
)


class PyPILookupError(Exception):
    """Base class for domain errors raised during a PyPI package lookup."""


class InvalidPackageNameError(PyPILookupError):
    """Raised when the requested package name is blank, unsafe, or malformed."""


class PackageNotFoundError(PyPILookupError):
    """Raised when PyPI returns HTTP 404 for the requested package name."""


class PyPITimeoutError(PyPILookupError):
    """Raised when the request to the PyPI JSON API times out."""


class PyPINetworkError(PyPILookupError):
    """Raised when the PyPI JSON API request fails before a response arrives."""


class MalformedPyPIResponseError(PyPILookupError):
    """Raised when PyPI returns non-JSON or an invalid JSON payload shape."""


class PyPIUpstreamHTTPError(PyPILookupError):
    """Raised when PyPI returns an unexpected non-404 HTTP status."""


def _normalize_and_validate_package_name(package_name: str) -> str:
    if not isinstance(package_name, str):
        raise InvalidPackageNameError("Package name must be a string.")

    stripped = package_name.strip()
    if not stripped:
        raise InvalidPackageNameError("Package name must not be blank.")
    if len(stripped) > 200:
        raise InvalidPackageNameError("Package name is too long.")
    if any(character in stripped for character in ("/", "\\", "?", "#", "&", "=")):
        raise InvalidPackageNameError("Package name contains invalid characters.")
    if "://" in stripped or stripped.lower().startswith(("http:", "https:")):
        raise InvalidPackageNameError("Package name must not be a URL.")
    if not _PACKAGE_NAME_RE.fullmatch(stripped):
        raise InvalidPackageNameError("Package name contains invalid characters.")
    return stripped


def _require_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MalformedPyPIResponseError(f"PyPI response field '{field_name}' must be an object.")
    return value


def _require_non_blank_string(
    payload: dict[str, Any], *, field_name: str, response_context: str
) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise MalformedPyPIResponseError(
            f"PyPI response {response_context} field '{field_name}' must be a non-blank string."
        )
    return value.strip()


def _get_optional_string(
    payload: dict[str, Any], *, field_name: str, response_context: str
) -> str | None:
    if field_name not in payload or payload[field_name] is None:
        return None
    value = payload[field_name]
    if not isinstance(value, str):
        raise MalformedPyPIResponseError(
            f"PyPI response {response_context} field '{field_name}' must be a string or null."
        )
    stripped = value.strip()
    return stripped or None


def _resolve_pypi_url(info: dict[str, Any]) -> str:
    for field_name in ("package_url", "project_url"):
        value = info.get(field_name)
        if value is None:
            continue
        if not isinstance(value, str):
            raise MalformedPyPIResponseError(
                f"PyPI response info field '{field_name}' must be a string."
            )
        stripped = value.strip()
        if stripped:
            return stripped
    raise MalformedPyPIResponseError(
        "PyPI response info must include a non-blank 'package_url' or 'project_url'."
    )


def _resolve_project_url(info: dict[str, Any], *, pypi_url: str) -> str | None:
    project_urls = info.get("project_urls")
    if project_urls is not None:
        project_url_mapping = _require_mapping(project_urls, field_name="project_urls")
        for label in _PREFERRED_PROJECT_URL_LABELS:
            candidate = project_url_mapping.get(label)
            if candidate is None:
                continue
            if not isinstance(candidate, str):
                raise MalformedPyPIResponseError(
                    "PyPI response info.project_urls values must be strings."
                )
            stripped = candidate.strip()
            if stripped and stripped != pypi_url:
                return stripped
        for candidate in project_url_mapping.values():
            if not isinstance(candidate, str):
                raise MalformedPyPIResponseError(
                    "PyPI response info.project_urls values must be strings."
                )
            stripped = candidate.strip()
            if stripped and stripped != pypi_url:
                return stripped

    for field_name in ("home_page", "project_url"):
        candidate = _get_optional_string(
            info, field_name=field_name, response_context="info"
        )
        if candidate is not None and candidate != pypi_url:
            return candidate
    return None


class PyPILookupService:
    """Fetches typed package metadata from the PyPI JSON API."""

    def __init__(self, settings: AppSettings, *, http_client: httpx.Client | None = None) -> None:
        self._settings = settings
        self._injected_client = http_client

    def lookup(self, package_name: str) -> PyPIPackageInfo:
        resolved_package_name = _normalize_and_validate_package_name(package_name)
        request_url = self._build_request_url(resolved_package_name)

        client = self._injected_client if self._injected_client is not None else httpx.Client(
            timeout=self._settings.pypi_timeout_seconds,
            follow_redirects=False,
            trust_env=False,
        )
        owns_client = self._injected_client is None
        try:
            try:
                response = client.get(
                    request_url,
                    headers={"Accept": "application/json"},
                )
            except httpx.TimeoutException as exc:
                raise PyPITimeoutError(
                    "Timed out while requesting package metadata from PyPI."
                ) from exc
            except httpx.RequestError as exc:
                raise PyPINetworkError(
                    "Failed to reach PyPI for package metadata."
                ) from exc
        finally:
            if owns_client:
                client.close()

        if response.status_code == 404:
            raise PackageNotFoundError(
                f"Package '{resolved_package_name}' was not found on PyPI."
            )
        if response.status_code != 200:
            raise PyPIUpstreamHTTPError(
                f"PyPI returned unexpected HTTP status {response.status_code} for package "
                f"'{resolved_package_name}'."
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise MalformedPyPIResponseError(
                "PyPI returned malformed JSON package metadata."
            ) from exc

        root = _require_mapping(payload, field_name="root")
        info = _require_mapping(root.get("info"), field_name="info")
        canonical_name = _require_non_blank_string(
            info,
            field_name="name",
            response_context="info",
        )
        latest_version = _require_non_blank_string(
            info,
            field_name="version",
            response_context="info",
        )
        pypi_url = _resolve_pypi_url(info)

        return PyPIPackageInfo(
            package_name=canonical_name,
            latest_version=latest_version,
            summary=_get_optional_string(info, field_name="summary", response_context="info"),
            requires_python=_get_optional_string(
                info,
                field_name="requires_python",
                response_context="info",
            ),
            pypi_url=pypi_url,
            project_url=_resolve_project_url(info, pypi_url=pypi_url),
        )

    def _build_request_url(self, package_name: str) -> str:
        return f"{self._settings.pypi_base_url}/pypi/{package_name}/json"
