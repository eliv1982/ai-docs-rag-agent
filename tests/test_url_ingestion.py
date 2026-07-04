"""Unit tests for UrlIngestionService. No real network access or DNS resolution:
HTTP is faked via httpx.MockTransport and DNS via an injected resolver callable.
"""

from collections.abc import Callable
from typing import Any

import httpx
import pytest
from bs4 import BeautifulSoup

from ai_docs_agent.config import AppSettings
from ai_docs_agent.models import UrlProcessingResult
from ai_docs_agent.url_ingestion import (
    ContentExtractionError,
    InvalidUrlError,
    PageTooLargeError,
    TooManyRedirectsError,
    UnsafeUrlError,
    UnsupportedContentTypeError,
    UrlFetchError,
    UrlIngestionService,
    _extract_charset,
    _extract_pre_blocks,
    _make_unique_placeholder_token,
    _normalize_pre_text,
    _normalize_text,
    _trim_chunk_text,
)

_REQUIRED: dict[str, Any] = {
    "openai_api_key": "sk-test-openai",
    "pinecone_api_key": "pc-test-key",
    "openai_chat_model": "gpt-4o-mini",
    "telegram_bot_token": "test-telegram-token",
}


def make_settings(**overrides: Any) -> AppSettings:
    return AppSettings(_env_file=None, **{**_REQUIRED, **overrides})


def make_resolver(mapping: dict[str, list[str]]) -> Callable[[str], list[str]]:
    def resolver(hostname: str) -> list[str]:
        if hostname not in mapping:
            raise AssertionError(f"unexpected DNS lookup for {hostname!r}")
        return mapping[hostname]

    return resolver


def forbidden_transport_handler(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"unexpected HTTP request to {request.url!r}")


def make_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def html_page(*, title: str = "Doc Title", paragraph_words: int = 60) -> bytes:
    body = " ".join(["word"] * paragraph_words)
    return (
        f"<html><head><title>{title}</title></head>"
        f"<body><main><p>{body}</p></main></body></html>"
    ).encode()


def html_response(
    *, content_type: str = "text/html; charset=utf-8", **kwargs: Any
) -> httpx.Response:
    return httpx.Response(
        200, headers={"content-type": content_type}, content=html_page(**kwargs)
    )


# --- URL validation ------------------------------------------------------


def test_process_url_rejects_empty_url() -> None:
    service = UrlIngestionService(
        make_settings(),
        http_client=make_client(forbidden_transport_handler),
        host_resolver=make_resolver({}),
    )
    with pytest.raises(InvalidUrlError):
        service.process_url("")


def test_process_url_rejects_relative_url() -> None:
    service = UrlIngestionService(
        make_settings(),
        http_client=make_client(forbidden_transport_handler),
        host_resolver=make_resolver({}),
    )
    with pytest.raises(InvalidUrlError):
        service.process_url("/relative/path")


def test_process_url_rejects_unsupported_scheme() -> None:
    service = UrlIngestionService(
        make_settings(),
        http_client=make_client(forbidden_transport_handler),
        host_resolver=make_resolver({}),
    )
    with pytest.raises(InvalidUrlError):
        service.process_url("ftp://docs.example.com/page")


def test_process_url_rejects_credentials_in_url() -> None:
    service = UrlIngestionService(
        make_settings(),
        http_client=make_client(forbidden_transport_handler),
        host_resolver=make_resolver({}),
    )
    with pytest.raises(InvalidUrlError):
        service.process_url("https://user:pass@docs.example.com/page")


def test_process_url_rejects_localhost() -> None:
    service = UrlIngestionService(
        make_settings(),
        http_client=make_client(forbidden_transport_handler),
        host_resolver=make_resolver({}),
    )
    with pytest.raises(UnsafeUrlError):
        service.process_url("http://localhost/page")


def test_process_url_rejects_dot_localhost_suffix() -> None:
    service = UrlIngestionService(
        make_settings(),
        http_client=make_client(forbidden_transport_handler),
        host_resolver=make_resolver({}),
    )
    with pytest.raises(UnsafeUrlError):
        service.process_url("http://myapp.localhost/page")


def test_process_url_rejects_localhost_with_trailing_dot() -> None:
    service = UrlIngestionService(
        make_settings(),
        http_client=make_client(forbidden_transport_handler),
        host_resolver=make_resolver({}),
    )
    with pytest.raises(UnsafeUrlError):
        service.process_url("http://localhost./page")


def test_process_url_rejects_mixed_case_localhost_with_trailing_dot() -> None:
    service = UrlIngestionService(
        make_settings(),
        http_client=make_client(forbidden_transport_handler),
        host_resolver=make_resolver({}),
    )
    with pytest.raises(UnsafeUrlError):
        service.process_url("http://LOCALHOST./page")


def test_process_url_rejects_dot_localhost_suffix_with_trailing_dot() -> None:
    service = UrlIngestionService(
        make_settings(),
        http_client=make_client(forbidden_transport_handler),
        host_resolver=make_resolver({}),
    )
    with pytest.raises(UnsafeUrlError):
        service.process_url("http://app.localhost./page")


def test_process_url_rejects_loopback_ipv4_literal_with_trailing_dot() -> None:
    service = UrlIngestionService(
        make_settings(),
        http_client=make_client(forbidden_transport_handler),
        host_resolver=make_resolver({}),
    )
    with pytest.raises(UnsafeUrlError):
        service.process_url("http://127.0.0.1./page")


def test_process_url_accepts_public_fqdn_with_trailing_dot_via_fake_resolver() -> None:
    settings = make_settings(url_min_text_chars=10)
    service = UrlIngestionService(
        settings,
        http_client=make_client(lambda request: html_response()),
        host_resolver=make_resolver({"docs.example.com": ["8.8.8.8"]}),
    )

    result = service.process_url("http://docs.example.com./page")

    assert result.chunk_count >= 1


def test_process_url_rejects_private_ipv4_literal() -> None:
    service = UrlIngestionService(
        make_settings(),
        http_client=make_client(forbidden_transport_handler),
        host_resolver=make_resolver({"10.0.0.5": ["10.0.0.5"]}),
    )
    with pytest.raises(UnsafeUrlError):
        service.process_url("http://10.0.0.5/page")


def test_process_url_rejects_loopback_ipv4_literal() -> None:
    service = UrlIngestionService(
        make_settings(),
        http_client=make_client(forbidden_transport_handler),
        host_resolver=make_resolver({"127.0.0.1": ["127.0.0.1"]}),
    )
    with pytest.raises(UnsafeUrlError):
        service.process_url("http://127.0.0.1/page")


def test_process_url_rejects_private_ipv6_literal() -> None:
    service = UrlIngestionService(
        make_settings(),
        http_client=make_client(forbidden_transport_handler),
        host_resolver=make_resolver({"fd00::1": ["fd00::1"]}),
    )
    with pytest.raises(UnsafeUrlError):
        service.process_url("http://[fd00::1]/page")


def test_process_url_rejects_dns_returning_private_ip() -> None:
    service = UrlIngestionService(
        make_settings(),
        http_client=make_client(forbidden_transport_handler),
        host_resolver=make_resolver({"internal.example.com": ["10.1.2.3"]}),
    )
    with pytest.raises(UnsafeUrlError):
        service.process_url("http://internal.example.com/page")


def test_process_url_rejects_dns_returning_mixed_public_and_private_ip() -> None:
    service = UrlIngestionService(
        make_settings(),
        http_client=make_client(forbidden_transport_handler),
        host_resolver=make_resolver({"mixed.example.com": ["8.8.8.8", "10.1.2.3"]}),
    )
    with pytest.raises(UnsafeUrlError):
        service.process_url("http://mixed.example.com/page")


def test_process_url_accepts_valid_public_url() -> None:
    settings = make_settings(url_min_text_chars=10)
    service = UrlIngestionService(
        settings,
        http_client=make_client(lambda request: html_response()),
        host_resolver=make_resolver({"docs.example.com": ["8.8.8.8"]}),
    )

    result = service.process_url("http://docs.example.com/page")

    assert result.source_url == "http://docs.example.com/page"
    assert result.final_url == "http://docs.example.com/page"


# --- Redirects -------------------------------------------------------------


def test_redirect_relative_location_resolves_and_succeeds() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "/final"})
        return html_response()

    settings = make_settings(url_min_text_chars=10)
    service = UrlIngestionService(
        settings,
        http_client=make_client(handler),
        host_resolver=make_resolver({"docs.example.com": ["8.8.8.8"]}),
    )

    result = service.process_url("http://docs.example.com/start")

    assert result.final_url == "http://docs.example.com/final"


def test_redirect_to_public_url_succeeds() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "old.example.com":
            return httpx.Response(302, headers={"location": "http://new.example.com/page"})
        return html_response()

    settings = make_settings(url_min_text_chars=10)
    service = UrlIngestionService(
        settings,
        http_client=make_client(handler),
        host_resolver=make_resolver(
            {"old.example.com": ["8.8.8.8"], "new.example.com": ["8.8.8.8"]}
        ),
    )

    result = service.process_url("http://old.example.com/page")

    assert result.final_url == "http://new.example.com/page"


def test_redirect_to_private_target_is_blocked() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://internal.example.com/page"})

    settings = make_settings(url_min_text_chars=10)
    service = UrlIngestionService(
        settings,
        http_client=make_client(handler),
        host_resolver=make_resolver(
            {"docs.example.com": ["8.8.8.8"], "internal.example.com": ["10.0.0.1"]}
        ),
    )

    with pytest.raises(UnsafeUrlError):
        service.process_url("http://docs.example.com/page")


def test_redirect_without_location_raises_fetch_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302)

    settings = make_settings()
    service = UrlIngestionService(
        settings,
        http_client=make_client(handler),
        host_resolver=make_resolver({"docs.example.com": ["8.8.8.8"]}),
    )

    with pytest.raises(UrlFetchError):
        service.process_url("http://docs.example.com/page")


def test_redirect_limit_exceeded_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "/next"})

    settings = make_settings(url_max_redirects=2)
    service = UrlIngestionService(
        settings,
        http_client=make_client(handler),
        host_resolver=make_resolver({"docs.example.com": ["8.8.8.8"]}),
    )

    with pytest.raises(TooManyRedirectsError):
        service.process_url("http://docs.example.com/start")


# --- Fetch -------------------------------------------------------------------


def test_fetch_non_2xx_status_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, headers={"content-type": "text/html"}, content=b"nope")

    service = UrlIngestionService(
        make_settings(),
        http_client=make_client(handler),
        host_resolver=make_resolver({"docs.example.com": ["8.8.8.8"]}),
    )

    with pytest.raises(UrlFetchError):
        service.process_url("http://docs.example.com/page")


def test_fetch_unsupported_content_type_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/json"}, content=b"{}")

    service = UrlIngestionService(
        make_settings(),
        http_client=make_client(handler),
        host_resolver=make_resolver({"docs.example.com": ["8.8.8.8"]}),
    )

    with pytest.raises(UnsupportedContentTypeError):
        service.process_url("http://docs.example.com/page")


def test_fetch_content_length_header_exceeds_limit_raises_before_reading_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html", "content-length": "999999"},
            content=b"<html></html>",
        )

    settings = make_settings(url_max_response_bytes=100)
    service = UrlIngestionService(
        settings,
        http_client=make_client(handler),
        host_resolver=make_resolver({"docs.example.com": ["8.8.8.8"]}),
    )

    with pytest.raises(PageTooLargeError):
        service.process_url("http://docs.example.com/page")


def test_fetch_streaming_body_exceeds_limit_without_content_length() -> None:
    consumed_chunks = {"count": 0}

    def body_gen():
        for _ in range(3):
            consumed_chunks["count"] += 1
            yield b"x" * 300
        raise AssertionError("streaming must stop once the byte limit is exceeded")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html"}, content=body_gen())

    settings = make_settings(url_max_response_bytes=500)
    service = UrlIngestionService(
        settings,
        http_client=make_client(handler),
        host_resolver=make_resolver({"docs.example.com": ["8.8.8.8"]}),
    )

    with pytest.raises(PageTooLargeError):
        service.process_url("http://docs.example.com/page")

    assert consumed_chunks["count"] <= 2


def test_fetch_transport_error_is_wrapped_with_chaining() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    service = UrlIngestionService(
        make_settings(),
        http_client=make_client(handler),
        host_resolver=make_resolver({"docs.example.com": ["8.8.8.8"]}),
    )

    with pytest.raises(UrlFetchError) as exc_info:
        service.process_url("http://docs.example.com/page")

    assert isinstance(exc_info.value.__cause__, httpx.ConnectError)


def test_fetch_invalid_charset_raises_content_extraction_error_with_chaining() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=definitely-not-a-codec"},
            content=b"<html><body>SECRET-MARKER " + b"word " * 60 + b"</body></html>",
        )

    service = UrlIngestionService(
        make_settings(url_min_text_chars=10),
        http_client=make_client(handler),
        host_resolver=make_resolver({"docs.example.com": ["8.8.8.8"]}),
    )

    with pytest.raises(ContentExtractionError) as exc_info:
        service.process_url("http://docs.example.com/page")

    assert isinstance(exc_info.value.__cause__, LookupError)
    assert "SECRET-MARKER" not in str(exc_info.value)


# --- Content-Length edge cases --------------------------------------------


def test_content_length_negative_raises_url_fetch_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html", "content-length": "-5"},
            content=b"<html></html>",
        )

    service = UrlIngestionService(
        make_settings(),
        http_client=make_client(handler),
        host_resolver=make_resolver({"docs.example.com": ["8.8.8.8"]}),
    )

    with pytest.raises(UrlFetchError):
        service.process_url("http://docs.example.com/page")


def test_content_length_non_numeric_raises_url_fetch_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html", "content-length": "not-a-number"},
            content=b"<html></html>",
        )

    service = UrlIngestionService(
        make_settings(),
        http_client=make_client(handler),
        host_resolver=make_resolver({"docs.example.com": ["8.8.8.8"]}),
    )

    with pytest.raises(UrlFetchError):
        service.process_url("http://docs.example.com/page")


def test_content_length_comma_separated_conflicting_values_raises_url_fetch_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html", "content-length": "10, 20"},
            content=b"<html></html>",
        )

    service = UrlIngestionService(
        make_settings(),
        http_client=make_client(handler),
        host_resolver=make_resolver({"docs.example.com": ["8.8.8.8"]}),
    )

    with pytest.raises(UrlFetchError):
        service.process_url("http://docs.example.com/page")


def test_content_length_equal_repeated_values_are_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html", "content-length": "10, 10"},
            content=b"<html></html>",
        )

    service = UrlIngestionService(
        make_settings(),
        http_client=make_client(handler),
        host_resolver=make_resolver({"docs.example.com": ["8.8.8.8"]}),
    )

    with pytest.raises(UrlFetchError):
        service.process_url("http://docs.example.com/page")


def test_content_length_repeated_identical_raw_headers_are_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers=[
                ("content-type", "text/html"),
                ("content-length", "10"),
                ("content-length", "10"),
            ],
            content=b"<html></html>",
        )

    service = UrlIngestionService(
        make_settings(),
        http_client=make_client(handler),
        host_resolver=make_resolver({"docs.example.com": ["8.8.8.8"]}),
    )

    with pytest.raises(UrlFetchError):
        service.process_url("http://docs.example.com/page")


def test_content_length_with_surrounding_whitespace_is_accepted() -> None:
    body = html_page(paragraph_words=5)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html", "content-length": f" {len(body)} "},
            content=body,
        )

    settings = make_settings(url_max_response_bytes=len(body), url_min_text_chars=1)
    service = UrlIngestionService(
        settings,
        http_client=make_client(handler),
        host_resolver=make_resolver({"docs.example.com": ["8.8.8.8"]}),
    )

    result = service.process_url("http://docs.example.com/page")

    assert result.chunk_count >= 1


def test_content_length_with_leading_plus_sign_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html", "content-length": "+10"},
            content=b"<html></html>",
        )

    service = UrlIngestionService(
        make_settings(),
        http_client=make_client(handler),
        host_resolver=make_resolver({"docs.example.com": ["8.8.8.8"]}),
    )

    with pytest.raises(UrlFetchError):
        service.process_url("http://docs.example.com/page")


def test_content_length_with_unicode_decimal_digits_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "content-type": "text/html",
                # Arabic-indic digits ("10"); passed pre-encoded since httpx's
                # header normalization only accepts ASCII str header values.
                "content-length": "١٠".encode(),
            },
            content=b"<html></html>",
        )

    service = UrlIngestionService(
        make_settings(),
        http_client=make_client(handler),
        host_resolver=make_resolver({"docs.example.com": ["8.8.8.8"]}),
    )

    with pytest.raises(UrlFetchError):
        service.process_url("http://docs.example.com/page")


def test_malformed_content_length_prevents_body_stream_from_being_read() -> None:
    read_flag = {"read": False}

    def body_gen():
        read_flag["read"] = True
        yield b"x" * 10

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html", "content-length": "10, 20"},
            content=body_gen(),
        )

    service = UrlIngestionService(
        make_settings(),
        http_client=make_client(handler),
        host_resolver=make_resolver({"docs.example.com": ["8.8.8.8"]}),
    )

    with pytest.raises(UrlFetchError):
        service.process_url("http://docs.example.com/page")

    assert read_flag["read"] is False


def test_content_length_exactly_at_limit_is_accepted() -> None:
    body = html_page(paragraph_words=5)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html", "content-length": str(len(body))},
            content=body,
        )

    settings = make_settings(url_max_response_bytes=len(body), url_min_text_chars=1)
    service = UrlIngestionService(
        settings,
        http_client=make_client(handler),
        host_resolver=make_resolver({"docs.example.com": ["8.8.8.8"]}),
    )

    result = service.process_url("http://docs.example.com/page")

    assert result.chunk_count >= 1


def test_content_length_one_byte_over_limit_raises_page_too_large() -> None:
    body = html_page(paragraph_words=5)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html", "content-length": str(len(body))},
            content=body,
        )

    settings = make_settings(url_max_response_bytes=len(body) - 1)
    service = UrlIngestionService(
        settings,
        http_client=make_client(handler),
        host_resolver=make_resolver({"docs.example.com": ["8.8.8.8"]}),
    )

    with pytest.raises(PageTooLargeError):
        service.process_url("http://docs.example.com/page")


def test_valid_content_length_over_limit_prevents_body_stream_from_being_read() -> None:
    read_flag = {"read": False}

    def body_gen():
        read_flag["read"] = True
        yield b"x" * 10

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html", "content-length": "999999"},
            content=body_gen(),
        )

    settings = make_settings(url_max_response_bytes=100)
    service = UrlIngestionService(
        settings,
        http_client=make_client(handler),
        host_resolver=make_resolver({"docs.example.com": ["8.8.8.8"]}),
    )

    with pytest.raises(PageTooLargeError):
        service.process_url("http://docs.example.com/page")

    assert read_flag["read"] is False


# --- Response and client lifecycle ----------------------------------------


def test_response_stream_closed_after_successful_fetch() -> None:
    captured: dict[str, httpx.Response] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        response = html_response()
        captured["response"] = response
        return response

    service = UrlIngestionService(
        make_settings(url_min_text_chars=10),
        http_client=make_client(handler),
        host_resolver=make_resolver({"docs.example.com": ["8.8.8.8"]}),
    )

    service.process_url("http://docs.example.com/page")

    assert captured["response"].is_closed


def test_response_stream_closed_after_page_too_large_error() -> None:
    captured: dict[str, httpx.Response] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        response = httpx.Response(
            200,
            headers={"content-type": "text/html", "content-length": "999999"},
            content=b"<html></html>",
        )
        captured["response"] = response
        return response

    settings = make_settings(url_max_response_bytes=100)
    service = UrlIngestionService(
        settings,
        http_client=make_client(handler),
        host_resolver=make_resolver({"docs.example.com": ["8.8.8.8"]}),
    )

    with pytest.raises(PageTooLargeError):
        service.process_url("http://docs.example.com/page")

    assert captured["response"].is_closed


def test_response_stream_closed_after_invalid_charset_error() -> None:
    captured: dict[str, httpx.Response] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        response = httpx.Response(
            200,
            headers={"content-type": "text/html; charset=bogus-codec"},
            content=html_page(),
        )
        captured["response"] = response
        return response

    service = UrlIngestionService(
        make_settings(url_min_text_chars=10),
        http_client=make_client(handler),
        host_resolver=make_resolver({"docs.example.com": ["8.8.8.8"]}),
    )

    with pytest.raises(ContentExtractionError):
        service.process_url("http://docs.example.com/page")

    assert captured["response"].is_closed


def test_response_stream_closed_after_unsupported_content_type_error() -> None:
    captured: dict[str, httpx.Response] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        response = httpx.Response(200, headers={"content-type": "application/json"}, content=b"{}")
        captured["response"] = response
        return response

    service = UrlIngestionService(
        make_settings(),
        http_client=make_client(handler),
        host_resolver=make_resolver({"docs.example.com": ["8.8.8.8"]}),
    )

    with pytest.raises(UnsupportedContentTypeError):
        service.process_url("http://docs.example.com/page")

    assert captured["response"].is_closed


def test_response_stream_closed_after_malformed_content_length_error() -> None:
    captured: dict[str, httpx.Response] = {}
    read_flag = {"read": False}

    def body_gen():
        read_flag["read"] = True
        yield b"x" * 10

    def handler(request: httpx.Request) -> httpx.Response:
        response = httpx.Response(
            200,
            headers={"content-type": "text/html", "content-length": "10, 20"},
            content=body_gen(),
        )
        captured["response"] = response
        return response

    service = UrlIngestionService(
        make_settings(),
        http_client=make_client(handler),
        host_resolver=make_resolver({"docs.example.com": ["8.8.8.8"]}),
    )

    with pytest.raises(UrlFetchError):
        service.process_url("http://docs.example.com/page")

    assert captured["response"].is_closed
    assert read_flag["read"] is False


def test_redirect_response_is_closed_before_next_request() -> None:
    captured: dict[str, httpx.Response] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            response = httpx.Response(302, headers={"location": "/final"})
            captured["redirect_response"] = response
            return response
        response = html_response()
        captured["final_response"] = response
        return response

    settings = make_settings(url_min_text_chars=10)
    service = UrlIngestionService(
        settings,
        http_client=make_client(handler),
        host_resolver=make_resolver({"docs.example.com": ["8.8.8.8"]}),
    )

    service.process_url("http://docs.example.com/start")

    assert captured["redirect_response"].is_closed
    assert captured["final_response"].is_closed


def test_redirect_does_not_close_injected_client() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "/final"})
        return html_response()

    client = httpx.Client(transport=httpx.MockTransport(handler))
    settings = make_settings(url_min_text_chars=10)
    service = UrlIngestionService(
        settings,
        http_client=client,
        host_resolver=make_resolver({"docs.example.com": ["8.8.8.8"]}),
    )

    service.process_url("http://docs.example.com/start")

    assert client.is_closed is False
    client.close()


def test_injected_client_is_not_closed_by_service() -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda request: html_response()))
    service = UrlIngestionService(
        make_settings(url_min_text_chars=10),
        http_client=client,
        host_resolver=make_resolver({"docs.example.com": ["8.8.8.8"]}),
    )

    service.process_url("http://docs.example.com/page")

    assert client.is_closed is False
    client.close()


def test_internally_created_client_is_closed_after_process_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = html_response()

    class _FakeStreamContext:
        def __enter__(self) -> httpx.Response:
            return response

        def __exit__(self, *exc_info: object) -> bool:
            response.close()
            return False

    class _FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            self.closed = False

        def stream(
            self,
            method: str,
            url: str,
            *,
            headers: dict[str, str] | None = None,
            follow_redirects: bool = False,
        ) -> _FakeStreamContext:
            return _FakeStreamContext()

        def close(self) -> None:
            self.closed = True

    fake_clients: list[_FakeClient] = []

    def fake_client_factory(**kwargs: Any) -> _FakeClient:
        client = _FakeClient(**kwargs)
        fake_clients.append(client)
        return client

    monkeypatch.setattr(httpx, "Client", fake_client_factory)

    service = UrlIngestionService(
        make_settings(url_min_text_chars=10),
        host_resolver=make_resolver({"docs.example.com": ["8.8.8.8"]}),
    )

    service.process_url("http://docs.example.com/page")

    assert len(fake_clients) == 1
    assert fake_clients[0].closed is True


# --- Extraction ----------------------------------------------------------


def make_extraction_service(**settings_overrides: Any) -> UrlIngestionService:
    settings = make_settings(**settings_overrides)
    return UrlIngestionService(
        settings,
        http_client=make_client(forbidden_transport_handler),
        host_resolver=make_resolver({}),
    )


def test_extract_page_strips_script_nav_and_footer() -> None:
    html = (
        "<html><head><title>T</title></head><body>"
        "<nav>NAV</nav><header>HEADER</header>"
        "<main><p>" + " ".join(["kept"] * 60) + "</p></main>"
        "<footer>FOOTER</footer><script>evil()</script>"
        "</body></html>"
    )
    service = make_extraction_service(url_min_text_chars=10)

    page = service._extract_page(source_url="http://x/", final_url="http://x/", html=html)

    assert "NAV" not in page.text
    assert "HEADER" not in page.text
    assert "FOOTER" not in page.text
    assert "evil" not in page.text
    assert "kept" in page.text


def test_extract_page_prefers_main_over_body() -> None:
    html = (
        "<html><head><title>T</title></head><body>"
        "<p>" + " ".join(["outside"] * 60) + "</p>"
        "<main><p>" + " ".join(["inside"] * 60) + "</p></main>"
        "</body></html>"
    )
    service = make_extraction_service(url_min_text_chars=10)

    page = service._extract_page(source_url="http://x/", final_url="http://x/", html=html)

    assert "inside" in page.text
    assert "outside" not in page.text


def test_extract_page_uses_article_when_no_main() -> None:
    html = (
        "<html><head><title>T</title></head><body>"
        "<article><p>" + " ".join(["article-text"] * 60) + "</p></article>"
        "</body></html>"
    )
    service = make_extraction_service(url_min_text_chars=10)

    page = service._extract_page(source_url="http://x/", final_url="http://x/", html=html)

    assert "article-text" in page.text


def test_extract_page_title_falls_back_to_h1() -> None:
    html = "<html><body><h1>Fallback Heading</h1><p>" + " ".join(["p"] * 60) + "</p></body></html>"
    service = make_extraction_service(url_min_text_chars=10)

    page = service._extract_page(source_url="http://x/", final_url="http://x/", html=html)

    assert page.title == "Fallback Heading"


def test_extract_page_title_falls_back_to_hostname() -> None:
    html = "<html><body><p>" + " ".join(["p"] * 60) + "</p></body></html>"
    service = make_extraction_service(url_min_text_chars=10)

    page = service._extract_page(
        source_url="http://docs.example.com/x",
        final_url="http://docs.example.com/x",
        html=html,
    )

    assert page.title == "docs.example.com"


def test_extract_page_preserves_code_and_pre_content() -> None:
    html = (
        "<html><head><title>T</title></head><body><main>"
        "<p>" + " ".join(["intro"] * 60) + "</p>"
        "<pre><code>def f():\n    return 1</code></pre>"
        "</main></body></html>"
    )
    service = make_extraction_service(url_min_text_chars=10)

    page = service._extract_page(source_url="http://x/", final_url="http://x/", html=html)

    assert "def f():" in page.text
    assert "return 1" in page.text


def test_extract_page_preserves_leading_indentation_in_python_pre_code() -> None:
    html = (
        "<html><head><title>T</title></head><body><main>"
        "<p>" + " ".join(["intro"] * 60) + "</p>"
        "<pre><code>def f():\n    return 1</code></pre>"
        "</main></body></html>"
    )
    service = make_extraction_service(url_min_text_chars=10)

    page = service._extract_page(source_url="http://x/", final_url="http://x/", html=html)

    assert "def f():\n    return 1" in page.text


def test_extract_page_preserves_nested_indentation_in_yaml_like_pre_block() -> None:
    yaml_like = "top:\n  child: 1\n  nested:\n    deep: true"
    html = (
        "<html><head><title>T</title></head><body><main>"
        "<p>" + " ".join(["intro"] * 60) + "</p>"
        f"<pre><code>{yaml_like}</code></pre>"
        "</main></body></html>"
    )
    service = make_extraction_service(url_min_text_chars=10)

    page = service._extract_page(source_url="http://x/", final_url="http://x/", html=html)

    assert yaml_like in page.text


def test_extract_page_pre_code_content_is_not_duplicated() -> None:
    html = (
        "<html><head><title>T</title></head><body><main>"
        "<p>" + " ".join(["intro"] * 60) + "</p>"
        "<pre><code>unique-marker-xyz</code></pre>"
        "</main></body></html>"
    )
    service = make_extraction_service(url_min_text_chars=10)

    page = service._extract_page(source_url="http://x/", final_url="http://x/", html=html)

    assert page.text.count("unique-marker-xyz") == 1


def test_extract_page_still_normalizes_prose_outside_pre_blocks() -> None:
    html = (
        "<html><head><title>T</title></head><body><main>"
        "<p>lots   of    spaces   here " + " ".join(["word"] * 55) + "</p>"
        "<pre><code>keep    spacing</code></pre>"
        "</main></body></html>"
    )
    service = make_extraction_service(url_min_text_chars=10)

    page = service._extract_page(source_url="http://x/", final_url="http://x/", html=html)

    assert "lots of spaces here" in page.text
    assert "keep    spacing" in page.text


def test_normalize_pre_text_preserves_indentation_and_trims_outer_blank_lines() -> None:
    raw = "\n\n  line one\n    line two  \n\n"

    normalized = _normalize_pre_text(raw)

    assert normalized == "  line one\n    line two"


def test_trim_chunk_text_preserves_leading_indentation() -> None:
    assert _trim_chunk_text("\n  first\n    second\n  third\n") == "  first\n    second\n  third"


def test_trim_chunk_text_whitespace_only_becomes_empty() -> None:
    assert _trim_chunk_text("   \n\t\n   ") == ""


# --- Placeholder collision resistance -------------------------------------


def test_extract_pre_blocks_returns_token_to_text_mapping() -> None:
    soup = BeautifulSoup("<div><pre><code>hello</code></pre></div>", "html.parser")

    placeholders = _extract_pre_blocks(soup)

    assert list(placeholders.values()) == ["hello"]
    assert soup.get_text() in placeholders


def test_make_unique_placeholder_token_avoids_visible_text_collision() -> None:
    base_token = _make_unique_placeholder_token("print(1)", 0, "", set())
    colliding_visible_text = f"before {base_token} after"

    token = _make_unique_placeholder_token("print(1)", 0, colliding_visible_text, set())

    assert token != base_token
    assert token not in colliding_visible_text


def test_make_unique_placeholder_token_avoids_already_chosen_token_collision() -> None:
    base_token = _make_unique_placeholder_token("print(1)", 0, "", set())

    token = _make_unique_placeholder_token("print(1)", 0, "", {base_token})

    assert token != base_token


def test_extract_page_preserves_literal_text_matching_old_placeholder_pattern() -> None:
    html = (
        "<html><head><title>T</title></head><body><main>"
        "<p>" + " ".join(["intro"] * 60) + " \x00PRE_BLOCK_0\x00 tail text</p>"
        "<pre><code>print(1)</code></pre>"
        "</main></body></html>"
    )
    service = make_extraction_service(url_min_text_chars=10)

    page = service._extract_page(source_url="http://x/", final_url="http://x/", html=html)

    assert "\x00PRE_BLOCK_0\x00" in page.text
    assert "print(1)" in page.text


def test_extract_pre_blocks_falls_back_to_next_candidate_on_collision() -> None:
    # Visible page text happens to contain the exact string the placeholder
    # algorithm would naturally pick first for this block.
    base_token = _make_unique_placeholder_token("print(1)", 0, "", set())
    html = (
        "<html><head><title>T</title></head><body><main>"
        f"<p>{' '.join(['intro'] * 60)} {base_token} tail text</p>"
        "<pre><code>print(1)</code></pre>"
        "</main></body></html>"
    )
    service = make_extraction_service(url_min_text_chars=10)

    page = service._extract_page(source_url="http://x/", final_url="http://x/", html=html)

    # The literal visible text is preserved verbatim (it is real page content)...
    assert page.text.count(base_token) == 1
    # ...and the <pre> block still restores correctly exactly once, meaning the
    # algorithm fell back to a different, non-colliding token internally.
    assert page.text.count("print(1)") == 1
    remaining_text = page.text.replace(base_token, "")
    assert "\x00" not in remaining_text


def test_extract_pre_blocks_preserves_order_across_multiple_blocks() -> None:
    html = (
        "<html><head><title>T</title></head><body><main>"
        "<p>" + " ".join(["intro"] * 60) + "</p>"
        "<pre><code>first-block</code></pre>"
        "<p>" + " ".join(["middle"] * 60) + "</p>"
        "<pre><code>second-block</code></pre>"
        "</main></body></html>"
    )
    service = make_extraction_service(url_min_text_chars=10)

    page = service._extract_page(source_url="http://x/", final_url="http://x/", html=html)

    assert page.text.index("first-block") < page.text.index("second-block")


def test_extract_pre_blocks_does_not_mix_identical_blocks() -> None:
    html = (
        "<html><head><title>T</title></head><body><main>"
        "<p>" + " ".join(["intro"] * 60) + "</p>"
        "<pre><code>same-content</code></pre>"
        "<p>" + " ".join(["middle"] * 60) + "</p>"
        "<pre><code>same-content</code></pre>"
        "</main></body></html>"
    )
    service = make_extraction_service(url_min_text_chars=10)

    page = service._extract_page(source_url="http://x/", final_url="http://x/", html=html)

    assert page.text.count("same-content") == 2


def test_extract_pre_blocks_is_deterministic_for_identical_html() -> None:
    html = (
        "<html><head><title>T</title></head><body><main>"
        "<p>" + " ".join(["intro"] * 60) + "</p>"
        "<pre><code>def f():\n    return 1</code></pre>"
        "</main></body></html>"
    )
    service = make_extraction_service(url_min_text_chars=10)

    first = service._extract_page(source_url="http://x/", final_url="http://x/", html=html)
    second = service._extract_page(source_url="http://x/", final_url="http://x/", html=html)

    assert first.text == second.text


def test_extract_pre_blocks_leaves_no_placeholder_tokens_in_final_text() -> None:
    html = (
        "<html><head><title>T</title></head><body><main>"
        "<p>" + " ".join(["intro"] * 60) + "</p>"
        "<pre><code>first-block</code></pre>"
        "<p>" + " ".join(["middle"] * 60) + "</p>"
        "<pre><code>second-block</code></pre>"
        "</main></body></html>"
    )
    service = make_extraction_service(url_min_text_chars=10)

    page = service._extract_page(source_url="http://x/", final_url="http://x/", html=html)

    assert "\x00" not in page.text


def test_extract_page_rejects_too_short_text() -> None:
    html = "<html><body><main><p>short</p></main></body></html>"
    service = make_extraction_service(url_min_text_chars=200)

    with pytest.raises(ContentExtractionError):
        service._extract_page(source_url="http://x/", final_url="http://x/", html=html)


def test_normalize_text_collapses_whitespace_and_blank_lines() -> None:
    raw = "line one   with   spaces\n\n\n\nline\ttwo\n   \n"

    normalized = _normalize_text(raw)

    assert normalized == "line one with spaces\n\nline two"
    assert "\n\n\n" not in normalized


def test_extract_charset_reads_content_type_parameter() -> None:
    assert _extract_charset("text/html; charset=iso-8859-1") == "iso-8859-1"
    assert _extract_charset("text/html") == "utf-8"
    assert _extract_charset(None) == "utf-8"


def test_content_hash_is_stable_for_identical_html() -> None:
    service = make_extraction_service(url_min_text_chars=10)
    html = html_page().decode("utf-8")

    first = service._extract_page(source_url="http://x/", final_url="http://x/", html=html)
    second = service._extract_page(source_url="http://x/", final_url="http://x/", html=html)

    assert first.content_hash == second.content_hash


def test_content_hash_differs_for_different_html() -> None:
    service = make_extraction_service(url_min_text_chars=10)
    html_a = html_page(title="A").decode("utf-8")
    html_b = html_page(title="B", paragraph_words=61).decode("utf-8")

    page_a = service._extract_page(source_url="http://x/", final_url="http://x/", html=html_a)
    page_b = service._extract_page(source_url="http://x/", final_url="http://x/", html=html_b)

    assert page_a.content_hash != page_b.content_hash


# --- Chunking ----------------------------------------------------------------


def _make_chunking_service(*, paragraph_words: int, **settings_overrides: Any) -> tuple[
    UrlIngestionService, bytes
]:
    settings = make_settings(
        url_min_text_chars=10, chunk_size=50, chunk_overlap=10, **settings_overrides
    )
    body = html_page(paragraph_words=paragraph_words)
    service = UrlIngestionService(
        settings,
        http_client=make_client(
            lambda request: httpx.Response(
                200, headers={"content-type": "text/html"}, content=body
            )
        ),
        host_resolver=make_resolver({"docs.example.com": ["8.8.8.8"]}),
    )
    return service, body


def test_process_url_produces_multiple_sequential_chunks() -> None:
    service, _ = _make_chunking_service(paragraph_words=200)

    result = service.process_url("http://docs.example.com/page")

    assert result.chunk_count > 1
    assert result.chunk_count == len(result.chunks)
    assert [chunk.chunk_index for chunk in result.chunks] == list(range(result.chunk_count))
    assert all(chunk.chunk_count == result.chunk_count for chunk in result.chunks)


def test_process_url_ids_are_deterministic_for_same_url_and_content() -> None:
    service_a, _ = _make_chunking_service(paragraph_words=200)
    service_b, _ = _make_chunking_service(paragraph_words=200)

    result_a = service_a.process_url("http://docs.example.com/page")
    result_b = service_b.process_url("http://docs.example.com/page")

    assert result_a.document_id == result_b.document_id
    assert [c.id for c in result_a.chunks] == [c.id for c in result_b.chunks]


def test_process_url_changed_content_produces_new_document_id() -> None:
    service_a, _ = _make_chunking_service(paragraph_words=200)
    service_b, _ = _make_chunking_service(paragraph_words=201)

    result_a = service_a.process_url("http://docs.example.com/page")
    result_b = service_b.process_url("http://docs.example.com/page")

    assert result_a.document_id != result_b.document_id
    assert result_a.chunks[0].id != result_b.chunks[0].id


def test_process_url_chunk_metadata_is_flat_with_exact_fields() -> None:
    service, _ = _make_chunking_service(paragraph_words=200)

    result: UrlProcessingResult = service.process_url("http://docs.example.com/page")
    metadata = result.chunks[0].to_pinecone_metadata()

    assert set(metadata.keys()) == {
        "kind",
        "text",
        "document_id",
        "source_url",
        "final_url",
        "title",
        "content_hash",
        "chunk_index",
        "chunk_count",
    }
    for value in metadata.values():
        assert isinstance(value, str | int)
        assert not isinstance(value, dict)


# --- Chunk-level indentation preservation (end-to-end) ----------------------


class _FixedSplitter:
    """A TextSplitter stand-in that returns pre-determined chunks regardless of input.

    Used to deterministically simulate a splitter chunk boundary landing right
    at a <pre> block's indented continuation line, without depending on the
    exact (and somewhat opaque) behavior of RecursiveCharacterTextSplitter.
    """

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    def split_text(self, text: str) -> list[str]:
        return list(self._chunks)


def _make_fixed_chunk_service(
    chunks: list[str], html: bytes | str, **settings_overrides: Any
) -> UrlIngestionService:
    settings = make_settings(url_min_text_chars=1, **settings_overrides)
    body = html.encode() if isinstance(html, str) else html
    return UrlIngestionService(
        settings,
        http_client=make_client(
            lambda request: httpx.Response(
                200, headers={"content-type": "text/html"}, content=body
            )
        ),
        host_resolver=make_resolver({"docs.example.com": ["8.8.8.8"]}),
        text_splitter=_FixedSplitter(chunks),
    )


def test_process_url_chunk_preserves_leading_indentation_at_chunk_boundary() -> None:
    html = (
        "<html><head><title>T</title></head><body><main>"
        "<p>" + " ".join(["intro"] * 60) + "</p>"
        "<pre><code>def f():\n    return 1</code></pre>"
        "</main></body></html>"
    )
    service = _make_fixed_chunk_service(["intro text\ndef f():", "    return 1"], html)

    result = service.process_url("http://docs.example.com/page")

    assert result.chunks[1].text == "    return 1"


def test_process_url_chunk_preserves_nested_yaml_indentation() -> None:
    service = _make_fixed_chunk_service(
        ["  nested:\n    deep: true"], html_page(paragraph_words=60)
    )

    result = service.process_url("http://docs.example.com/page")

    assert result.chunks[0].text == "  nested:\n    deep: true"


def test_process_url_whitespace_only_chunk_is_filtered_out() -> None:
    service = _make_fixed_chunk_service(
        ["real chunk text", "   \n\t  \n  "], html_page(paragraph_words=60)
    )

    result = service.process_url("http://docs.example.com/page")

    assert result.chunk_count == 1
    assert result.chunks[0].text == "real chunk text"


def test_process_url_pre_content_chunking_is_deterministic_across_runs() -> None:
    html = (
        "<html><head><title>T</title></head><body><main>"
        "<p>" + " ".join(["intro"] * 80) + "</p>"
        "<pre><code>" + "\n".join(f"    line {i}" for i in range(20)) + "</code></pre>"
        "</main></body></html>"
    )

    def make_service() -> UrlIngestionService:
        settings = make_settings(url_min_text_chars=10, chunk_size=50, chunk_overlap=10)
        return UrlIngestionService(
            settings,
            http_client=make_client(
                lambda request: httpx.Response(
                    200, headers={"content-type": "text/html"}, content=html.encode()
                )
            ),
            host_resolver=make_resolver({"docs.example.com": ["8.8.8.8"]}),
        )

    result_a = make_service().process_url("http://docs.example.com/page")
    result_b = make_service().process_url("http://docs.example.com/page")

    assert result_a.document_id == result_b.document_id
    assert [c.id for c in result_a.chunks] == [c.id for c in result_b.chunks]
    assert [c.text for c in result_a.chunks] == [c.text for c in result_b.chunks]
