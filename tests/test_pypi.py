"""Unit tests for the PyPI JSON API lookup service and thin tool adapter.

No real network access occurs: HTTP is faked via httpx.MockTransport or injected fake
service objects only.
"""

from typing import Any

import httpx
import pytest

from ai_docs_agent.config import AppSettings
from ai_docs_agent.models import PyPIPackageInfo
from ai_docs_agent.pypi import (
    InvalidPackageNameError,
    MalformedPyPIResponseError,
    PackageNotFoundError,
    PyPILookupService,
    PyPINetworkError,
    PyPITimeoutError,
    PyPIUpstreamHTTPError,
)
from ai_docs_agent.tools import lookup_pypi_package

_REQUIRED: dict[str, Any] = {
    "openai_api_key": "sk-test-openai",
    "pinecone_api_key": "pc-test-key",
    "openai_chat_model": "gpt-4o-mini",
    "telegram_bot_token": "test-telegram-token",
    "user_memory_hash_secret": "unit-test-user-memory-secret",
}


def make_settings(**overrides: Any) -> AppSettings:
    return AppSettings(_env_file=None, **{**_REQUIRED, **overrides})


def make_client(handler: Any) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def forbidden_transport_handler(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"unexpected HTTP request to {request.url!r}")


def make_payload(**info_overrides: Any) -> dict[str, Any]:
    info: dict[str, Any] = {
        "name": "httpx",
        "version": "9.9.9",
        "summary": "HTTP client for Python.",
        "requires_python": ">=3.8",
        "package_url": "https://pypi.org/project/httpx/",
        "project_url": "https://pypi.org/project/httpx/",
        "project_urls": {
            "Homepage": "https://www.python-httpx.org/",
            "Source": "https://github.com/encode/httpx",
        },
        "home_page": "https://www.python-httpx.org/",
    }
    info.update(info_overrides)
    return {
        "info": info,
        "last_serial": 1,
        "releases": {},
        "urls": [],
        "vulnerabilities": [],
    }


def test_lookup_success_returns_typed_package_info_for_httpx() -> None:
    service = PyPILookupService(
        make_settings(),
        http_client=make_client(lambda request: httpx.Response(200, json=make_payload())),
    )

    result = service.lookup("HTTPX")

    assert result == PyPIPackageInfo(
        package_name="httpx",
        latest_version="9.9.9",
        summary="HTTP client for Python.",
        requires_python=">=3.8",
        pypi_url="https://pypi.org/project/httpx/",
        project_url="https://www.python-httpx.org/",
    )


@pytest.mark.parametrize("summary_mode", ["missing", "null"])
def test_lookup_summary_absent_or_null_maps_to_none(summary_mode: str) -> None:
    payload = make_payload()
    if summary_mode == "missing":
        payload["info"].pop("summary")
    else:
        payload["info"]["summary"] = None
    service = PyPILookupService(
        make_settings(),
        http_client=make_client(lambda request: httpx.Response(200, json=payload)),
    )

    result = service.lookup("httpx")

    assert result.summary is None


@pytest.mark.parametrize("requires_python_mode", ["missing", "null"])
def test_lookup_requires_python_absent_or_null_maps_to_none(requires_python_mode: str) -> None:
    payload = make_payload()
    if requires_python_mode == "missing":
        payload["info"].pop("requires_python")
    else:
        payload["info"]["requires_python"] = None
    service = PyPILookupService(
        make_settings(),
        http_client=make_client(lambda request: httpx.Response(200, json=payload)),
    )

    result = service.lookup("httpx")

    assert result.requires_python is None


def test_lookup_project_url_falls_back_to_home_page_when_project_urls_missing() -> None:
    payload = make_payload(project_urls=None, home_page="https://example.com/project-home")
    service = PyPILookupService(
        make_settings(),
        http_client=make_client(lambda request: httpx.Response(200, json=payload)),
    )

    result = service.lookup("httpx")

    assert result.project_url == "https://example.com/project-home"


def test_lookup_project_url_is_none_when_only_pypi_page_is_available() -> None:
    payload = make_payload(project_urls=None, home_page=None, project_url="https://pypi.org/project/httpx/")
    service = PyPILookupService(
        make_settings(),
        http_client=make_client(lambda request: httpx.Response(200, json=payload)),
    )

    result = service.lookup("httpx")

    assert result.project_url is None


@pytest.mark.parametrize("package_name", ["typing-extensions", "zope.interface"])
def test_lookup_accepts_legitimate_hyphen_and_dot_package_names(package_name: str) -> None:
    requested_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        return httpx.Response(
            200,
            json=make_payload(
                name=package_name,
                package_url=f"https://pypi.org/project/{package_name}/",
                project_url=f"https://pypi.org/project/{package_name}/",
            ),
        )

    service = PyPILookupService(make_settings(), http_client=make_client(handler))

    result = service.lookup(package_name)

    assert result.package_name == package_name
    assert requested_urls == [f"https://pypi.org/pypi/{package_name}/json"]


@pytest.mark.parametrize(
    "package_name",
    [
        "",
        "   ",
        "requests/httpx",
        r"requests\httpx",
        "https://pypi.org/project/httpx/",
        "httpx?format=json",
        "httpx#frag",
    ],
)
def test_lookup_rejects_invalid_or_unsafe_package_names_before_http(package_name: str) -> None:
    service = PyPILookupService(
        make_settings(),
        http_client=make_client(forbidden_transport_handler),
    )

    with pytest.raises(InvalidPackageNameError):
        service.lookup(package_name)


def test_lookup_404_maps_to_package_not_found() -> None:
    service = PyPILookupService(
        make_settings(),
        http_client=make_client(lambda request: httpx.Response(404)),
    )

    with pytest.raises(PackageNotFoundError):
        service.lookup("httpx")


def test_lookup_timeout_maps_to_domain_timeout_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    service = PyPILookupService(make_settings(), http_client=make_client(handler))

    with pytest.raises(PyPITimeoutError):
        service.lookup("httpx")


def test_lookup_network_error_maps_to_domain_network_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("dns failure", request=request)

    service = PyPILookupService(make_settings(), http_client=make_client(handler))

    with pytest.raises(PyPINetworkError):
        service.lookup("httpx")


def test_lookup_malformed_json_maps_to_malformed_response_error() -> None:
    service = PyPILookupService(
        make_settings(),
        http_client=make_client(
            lambda request: httpx.Response(
                200,
                content=b"{ definitely not json",
                headers={"content-type": "application/json"},
            )
        ),
    )

    with pytest.raises(MalformedPyPIResponseError):
        service.lookup("httpx")


def test_lookup_missing_info_maps_to_malformed_response_error() -> None:
    service = PyPILookupService(
        make_settings(),
        http_client=make_client(lambda request: httpx.Response(200, json={"urls": []})),
    )

    with pytest.raises(MalformedPyPIResponseError):
        service.lookup("httpx")


@pytest.mark.parametrize("bad_name", [None, "", "   ", 123])
def test_lookup_missing_or_invalid_canonical_name_maps_to_malformed_response_error(
    bad_name: object,
) -> None:
    payload = make_payload(name=bad_name)
    service = PyPILookupService(
        make_settings(),
        http_client=make_client(lambda request: httpx.Response(200, json=payload)),
    )

    with pytest.raises(MalformedPyPIResponseError):
        service.lookup("httpx")


@pytest.mark.parametrize("bad_version", [None, "", "   ", 123])
def test_lookup_missing_or_invalid_version_maps_to_malformed_response_error(
    bad_version: object,
) -> None:
    payload = make_payload(version=bad_version)
    service = PyPILookupService(
        make_settings(),
        http_client=make_client(lambda request: httpx.Response(200, json=payload)),
    )

    with pytest.raises(MalformedPyPIResponseError):
        service.lookup("httpx")


@pytest.mark.parametrize("status_code", [429, 500])
def test_lookup_unexpected_http_status_maps_to_upstream_http_error(status_code: int) -> None:
    service = PyPILookupService(
        make_settings(),
        http_client=make_client(lambda request: httpx.Response(status_code)),
    )

    with pytest.raises(PyPIUpstreamHTTPError):
        service.lookup("httpx")


def test_lookup_constructs_expected_request_url_and_timeout_when_owning_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ai_docs_agent.pypi as pypi_module

    created_clients: list[Any] = []

    class RecordingClient:
        def __init__(self, *, timeout: float, follow_redirects: bool, trust_env: bool) -> None:
            self.timeout = timeout
            self.follow_redirects = follow_redirects
            self.trust_env = trust_env
            self.closed = False
            self.calls: list[tuple[str, dict[str, str]]] = []
            created_clients.append(self)

        def get(self, url: str, headers: dict[str, str]) -> httpx.Response:
            self.calls.append((url, headers))
            return httpx.Response(200, json=make_payload())

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(pypi_module.httpx, "Client", RecordingClient)
    service = PyPILookupService(
        make_settings(pypi_timeout_seconds=12.5, pypi_base_url="https://pypi.org")
    )

    result = service.lookup("httpx")

    assert result.package_name == "httpx"
    assert len(created_clients) == 1
    client = created_clients[0]
    assert client.timeout == 12.5
    assert client.follow_redirects is False
    assert client.trust_env is False
    assert client.calls == [
        ("https://pypi.org/pypi/httpx/json", {"Accept": "application/json"})
    ]
    assert client.closed is True


def test_lookup_with_injected_client_does_not_construct_default_http_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ai_docs_agent.pypi as pypi_module

    injected_client = make_client(lambda request: httpx.Response(200, json=make_payload()))
    monkeypatch.setattr(
        pypi_module.httpx,
        "Client",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("unexpected real client creation")),
    )
    service = PyPILookupService(
        make_settings(),
        http_client=injected_client,
    )

    result = service.lookup("httpx")

    assert result.package_name == "httpx"


def test_lookup_pypi_package_tool_forwards_to_injected_service() -> None:
    result = PyPIPackageInfo(
        package_name="httpx",
        latest_version="9.9.9",
        summary="HTTP client for Python.",
        requires_python=">=3.8",
        pypi_url="https://pypi.org/project/httpx/",
        project_url="https://www.python-httpx.org/",
    )

    class FakeService:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def lookup(self, package_name: str) -> PyPIPackageInfo:
            self.calls.append(package_name)
            return result

    fake_service = FakeService()

    tool_result = lookup_pypi_package("httpx", service=fake_service)  # type: ignore[arg-type]

    assert tool_result == result
    assert fake_service.calls == ["httpx"]


def test_lookup_pypi_package_tool_builds_default_service_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ai_docs_agent.tools as tools_module

    captured: dict[str, Any] = {}
    expected_result = PyPIPackageInfo(
        package_name="httpx",
        latest_version="9.9.9",
        summary=None,
        requires_python=None,
        pypi_url="https://pypi.org/project/httpx/",
        project_url=None,
    )

    class RecordingService:
        def __init__(self, settings: AppSettings) -> None:
            captured["settings"] = settings

        def lookup(self, package_name: str) -> PyPIPackageInfo:
            captured["package_name"] = package_name
            return expected_result

    settings = make_settings()
    monkeypatch.setattr(tools_module, "PyPILookupService", RecordingService)

    result = tools_module.lookup_pypi_package("httpx", settings=settings)

    assert result == expected_result
    assert captured == {"settings": settings, "package_name": "httpx"}
