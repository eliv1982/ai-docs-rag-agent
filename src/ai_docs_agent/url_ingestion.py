"""Fetching and cleaning HTML content from documentation URLs.

Pipeline: URL validation (scheme/host/SSRF guard) -> bounded-redirect HTTP fetch ->
bounded response size -> HTML extraction -> normalized text -> deterministic
document/chunk IDs -> RecursiveCharacterTextSplitter -> deterministic chunks.
"""

import hashlib
import ipaddress
import re
import socket
from collections.abc import Callable, Sequence
from typing import Protocol
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ai_docs_agent.config import AppSettings
from ai_docs_agent.models import DocumentChunk, FetchedPage, UrlProcessingResult

_ALLOWED_SCHEMES = frozenset({"http", "https"})
_ALLOWED_CONTENT_TYPES = frozenset({"text/html", "application/xhtml+xml"})
_REDIRECT_STATUS_CODES = frozenset({301, 302, 303, 307, 308})
_STRIP_TAG_NAMES = (
    "script",
    "style",
    "noscript",
    "svg",
    "nav",
    "footer",
    "header",
    "aside",
    "form",
    "template",
)
_CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
_WHITESPACE_RUN = re.compile(r"[ \t]+")
_PRE_PLACEHOLDER_PREFIX = "\x00\x01PRE-BLOCK:"
_PRE_PLACEHOLDER_SUFFIX = ":\x01\x00"
_CONTENT_LENGTH_RE = re.compile(r"^[0-9]+$")

HostResolver = Callable[[str], Sequence[str]]


class TextSplitter(Protocol):
    """Structural interface for the text splitter used by UrlIngestionService."""

    def split_text(self, text: str) -> list[str]: ...


class UrlIngestionError(Exception):
    """Base class for all domain errors raised while ingesting a URL."""


class InvalidUrlError(UrlIngestionError):
    """Raised when a URL is malformed, uses an unsupported scheme, or lacks a hostname."""


class UnsafeUrlError(UrlIngestionError):
    """Raised when a URL's hostname (or a resolved address) is disallowed for fetching."""


class UrlFetchError(UrlIngestionError):
    """Raised when the HTTP request for a URL fails or returns an unexpected status."""


class UnsupportedContentTypeError(UrlIngestionError):
    """Raised when a fetched response's Content-Type is not supported for ingestion."""


class PageTooLargeError(UrlIngestionError):
    """Raised when a fetched response exceeds the configured maximum size."""


class ContentExtractionError(UrlIngestionError):
    """Raised when a fetched page cannot be turned into enough usable text or chunks."""


class TooManyRedirectsError(UrlIngestionError):
    """Raised when a URL's redirect chain exceeds the configured maximum length."""


def _default_resolve_host(hostname: str) -> list[str]:
    try:
        infos = socket.getaddrinfo(hostname, None)
    except OSError as exc:
        raise UnsafeUrlError(f"Could not resolve hostname '{hostname}'.") from exc
    return sorted({info[4][0] for info in infos})


def _normalize_and_validate_url(url: str) -> str:
    if not url or not url.strip():
        raise InvalidUrlError("URL must not be empty.")

    parsed = urlsplit(url.strip())
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise InvalidUrlError(f"Unsupported or missing URL scheme in '{url}'.")
    if not parsed.hostname:
        raise InvalidUrlError(f"URL '{url}' must include a hostname.")
    if parsed.username is not None or parsed.password is not None:
        raise InvalidUrlError("URL must not include embedded credentials.")

    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))


def _normalize_hostname_for_checks(hostname: str) -> str:
    """Normalize a hostname for security checks and DNS resolution.

    Lowercases and strips trailing dots so that DNS-equivalent forms like
    'LOCALHOST.' or 'app.localhost.' cannot bypass the literal-hostname guard,
    which previously ran against the raw, unnormalized hostname.
    """
    normalized = hostname.strip().lower().rstrip(".")
    if not normalized:
        raise UnsafeUrlError("URL hostname must not be empty.")
    return normalized


def _validate_hostname_literal(hostname: str) -> None:
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise UnsafeUrlError(f"Hostname '{hostname}' is not allowed.")


def _try_parse_ip_literal(
    hostname: str,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(hostname)
    except ValueError:
        return None


def _validate_resolved_addresses(addresses: Sequence[str], hostname: str) -> None:
    if not addresses:
        raise UnsafeUrlError(f"Hostname '{hostname}' did not resolve to any address.")

    for raw_address in addresses:
        address = ipaddress.ip_address(raw_address)
        if (
            address.is_loopback
            or address.is_private
            or address.is_link_local
            or address.is_multicast
            or address.is_reserved
            or address.is_unspecified
        ):
            raise UnsafeUrlError(
                f"Hostname '{hostname}' resolves to a disallowed address '{raw_address}'."
            )


def _parse_content_length(header_value: str | None) -> int | None:
    """Strictly parse a single Content-Length header value.

    Returns None when the header is absent, so the caller falls back to the
    streaming byte-count guard. Raises UrlFetchError for anything that is not
    exactly one ASCII decimal non-negative integer (surrounding whitespace is
    tolerated): negative values, signs, decimals, non-ASCII digits, and any
    comma-separated/multiple/repeated value (including repeated identical
    headers, which httpx joins into a single comma-separated string) are all
    treated as malformed.
    """
    if header_value is None:
        return None

    if "," in header_value:
        raise UrlFetchError(f"Malformed Content-Length header '{header_value}'.")

    candidate = header_value.strip()
    if not _CONTENT_LENGTH_RE.fullmatch(candidate):
        raise UrlFetchError(f"Malformed Content-Length header '{header_value}'.")

    return int(candidate)


def _make_document_id(final_url: str, content_hash: str) -> str:
    seed = f"{final_url}\n{content_hash}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return f"doc-{digest[:24]}"


def _extract_charset(content_type_header: str | None) -> str:
    if not content_type_header:
        return "utf-8"
    for part in content_type_header.split(";")[1:]:
        key, _, value = part.strip().partition("=")
        if key.strip().lower() == "charset":
            charset = value.strip().strip('"').strip("'")
            return charset or "utf-8"
    return "utf-8"


def _decode_body(body: bytes, charset: str) -> str:
    try:
        return body.decode(charset, errors="replace")
    except LookupError as exc:
        raise ContentExtractionError(
            f"Response declared an unsupported or unknown charset '{charset}'."
        ) from exc


def _normalize_text(raw_text: str) -> str:
    normalized_lines: list[str] = []
    previous_blank = False
    for line in raw_text.splitlines():
        stripped = _WHITESPACE_RUN.sub(" ", line.strip())
        if stripped:
            normalized_lines.append(stripped)
            previous_blank = False
        elif not previous_blank:
            normalized_lines.append("")
            previous_blank = True
    return "\n".join(normalized_lines).strip()


def _normalize_pre_text(raw_text: str) -> str:
    """Preserve a <pre> block's internal formatting.

    Only CRLF/CR normalization, per-line trailing-whitespace removal, and
    leading/trailing blank lines are trimmed -- significant leading indentation
    and internal blank lines are left untouched.
    """
    unified_newlines = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in unified_newlines.split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _trim_chunk_text(raw_text: str) -> str:
    """Trim a splitter-produced chunk without disturbing significant indentation.

    A plain .strip() would eat a <pre> block's leading indentation whenever the
    splitter happens to place that line at the start of a chunk. Reuses the
    same outer-blank-line/trailing-whitespace trim used for <pre> blocks, so a
    chunk of only whitespace normalizes to "" (and is filtered out as empty).
    """
    return _normalize_pre_text(raw_text)


def _make_unique_placeholder_token(
    block_text: str, index: int, visible_text: str, chosen_tokens: set[str]
) -> str:
    """Build a placeholder token guaranteed absent from `visible_text` and `chosen_tokens`.

    The base candidate is derived from a SHA-256 digest of the block's index
    and content, so two identical <pre> blocks at different positions never
    collide. If that (extremely unlikely) candidate still collides with the
    page's own visible text or an already-chosen token, a deterministic
    counter suffix is appended until a safe token is found -- no randomness is
    used, so extraction stays fully deterministic.
    """
    seed = f"{index}\n{block_text}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    counter = 0
    while True:
        suffix = "" if counter == 0 else f"-{counter}"
        candidate = f"{_PRE_PLACEHOLDER_PREFIX}{digest}{suffix}{_PRE_PLACEHOLDER_SUFFIX}"
        if candidate not in visible_text and candidate not in chosen_tokens:
            return candidate
        counter += 1


def _extract_pre_blocks(content_root: BeautifulSoup | Tag) -> dict[str, str]:
    """Replace each <pre> element with a collision-free placeholder token.

    Returns a token -> preserved-text mapping (in document order) used to
    restore the exact formatting after the later prose normalization pass,
    which cannot alter the tokens since they contain no whitespace runs.
    """
    visible_text = content_root.get_text(separator="\n")
    chosen_tokens: set[str] = set()
    placeholders: dict[str, str] = {}
    for index, pre in enumerate(content_root.find_all("pre")):
        code = pre.find("code")
        source = code if code is not None else pre
        block_text = _normalize_pre_text(source.get_text())
        token = _make_unique_placeholder_token(block_text, index, visible_text, chosen_tokens)
        chosen_tokens.add(token)
        placeholders[token] = block_text
        pre.replace_with(NavigableString(token))
    return placeholders


def _restore_pre_blocks(text: str, placeholders: dict[str, str]) -> str:
    for token, block_text in placeholders.items():
        text = text.replace(token, block_text)
    return text


class UrlIngestionService:
    """Validates, fetches, extracts, and chunks a single documentation URL."""

    def __init__(
        self,
        settings: AppSettings,
        *,
        http_client: httpx.Client | None = None,
        host_resolver: HostResolver | None = None,
        text_splitter: TextSplitter | None = None,
    ) -> None:
        self._settings = settings
        self._injected_client = http_client
        self._resolve_host = host_resolver or _default_resolve_host
        self._splitter: TextSplitter = text_splitter or RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            separators=_CHUNK_SEPARATORS,
        )

    def process_url(self, url: str) -> UrlProcessingResult:
        """Fetch, extract, and chunk a single URL into a deterministic result."""
        source_url = self._validate_url_is_safe(url)

        client = self._injected_client if self._injected_client is not None else httpx.Client(
            timeout=self._settings.url_fetch_timeout_seconds,
            follow_redirects=False,
            trust_env=False,
        )
        owns_client = self._injected_client is None
        try:
            final_url, charset, body = self._fetch_with_redirects(client, source_url)
        finally:
            if owns_client:
                client.close()

        html = _decode_body(body, charset)
        page = self._extract_page(source_url=source_url, final_url=final_url, html=html)
        document_id = _make_document_id(page.final_url, page.content_hash)
        chunks = self._build_chunks(page, document_id)

        return UrlProcessingResult(
            source_url=page.source_url,
            final_url=page.final_url,
            title=page.title,
            document_id=document_id,
            content_hash=page.content_hash,
            text_char_count=len(page.text),
            chunk_count=len(chunks),
            chunks=chunks,
        )

    def _validate_url_is_safe(self, url: str) -> str:
        normalized = _normalize_and_validate_url(url)
        hostname = urlsplit(normalized).hostname
        assert hostname is not None  # guaranteed by _normalize_and_validate_url
        check_hostname = _normalize_hostname_for_checks(hostname)
        _validate_hostname_literal(check_hostname)

        ip_literal = _try_parse_ip_literal(check_hostname)
        if ip_literal is not None:
            _validate_resolved_addresses([str(ip_literal)], check_hostname)
        else:
            addresses = self._resolve_host(check_hostname)
            _validate_resolved_addresses(addresses, check_hostname)
        return normalized

    def _fetch_with_redirects(
        self, client: httpx.Client, start_url: str
    ) -> tuple[str, str, bytes]:
        current_url = start_url
        headers = {
            "User-Agent": self._settings.url_user_agent,
            "Accept": "text/html, application/xhtml+xml",
        }
        max_attempts = self._settings.url_max_redirects + 1

        for _attempt in range(max_attempts):
            try:
                with client.stream(
                    "GET", current_url, headers=headers, follow_redirects=False
                ) as response:
                    if response.status_code in _REDIRECT_STATUS_CODES:
                        location = response.headers.get("location")
                        if not location:
                            raise UrlFetchError(
                                f"Redirect response from '{current_url}' had no Location header."
                            )
                        current_url = self._validate_url_is_safe(
                            urljoin(current_url, location)
                        )
                        continue

                    if not 200 <= response.status_code < 300:
                        raise UrlFetchError(
                            f"Unexpected HTTP status {response.status_code} from "
                            f"'{current_url}'."
                        )

                    content_type_header = response.headers.get("content-type")
                    self._validate_content_type(content_type_header)
                    self._validate_content_length(response.headers.get("content-length"))
                    body = self._read_bounded_body(response)
                    charset = _extract_charset(content_type_header)
                    return current_url, charset, body
            except httpx.HTTPError as exc:
                raise UrlFetchError(f"Failed to fetch '{current_url}'.") from exc

        raise TooManyRedirectsError(
            f"Exceeded maximum of {self._settings.url_max_redirects} redirect(s) "
            f"starting at '{start_url}'."
        )

    def _validate_content_type(self, header_value: str | None) -> None:
        if header_value is None:
            raise UnsupportedContentTypeError("Response did not include a Content-Type header.")
        media_type = header_value.split(";", 1)[0].strip().lower()
        if media_type not in _ALLOWED_CONTENT_TYPES:
            raise UnsupportedContentTypeError(f"Unsupported Content-Type '{media_type}'.")

    def _validate_content_length(self, header_value: str | None) -> None:
        declared_bytes = _parse_content_length(header_value)
        if declared_bytes is None:
            return
        limit = self._settings.url_max_response_bytes
        if declared_bytes > limit:
            raise PageTooLargeError(
                f"Declared response size {declared_bytes} bytes exceeds the "
                f"{limit}-byte limit."
            )

    def _read_bounded_body(self, response: httpx.Response) -> bytes:
        limit = self._settings.url_max_response_bytes
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_bytes():
            total += len(chunk)
            if total > limit:
                raise PageTooLargeError(
                    f"Response body exceeded the {limit}-byte limit while streaming."
                )
            chunks.append(chunk)
        return b"".join(chunks)

    def _extract_page(self, *, source_url: str, final_url: str, html: str) -> FetchedPage:
        soup = BeautifulSoup(html, "html.parser")
        for tag_name in _STRIP_TAG_NAMES:
            for element in soup.find_all(tag_name):
                element.decompose()

        title = self._extract_title(soup, final_url)
        content_root = soup.find("main") or soup.find("article") or soup.find("body") or soup
        pre_blocks = _extract_pre_blocks(content_root)
        raw_text = content_root.get_text(separator="\n")
        cleaned_text = _restore_pre_blocks(_normalize_text(raw_text), pre_blocks)

        if len(cleaned_text) < self._settings.url_min_text_chars:
            raise ContentExtractionError(
                f"Extracted text ({len(cleaned_text)} chars) is shorter than the "
                f"required {self._settings.url_min_text_chars} characters."
            )

        content_hash = hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest()
        return FetchedPage(
            source_url=source_url,
            final_url=final_url,
            title=title,
            text=cleaned_text,
            content_hash=content_hash,
        )

    def _extract_title(self, soup: BeautifulSoup, final_url: str) -> str:
        title_tag = soup.find("title")
        if title_tag is not None:
            text = title_tag.get_text(strip=True)
            if text:
                return text

        h1_tag = soup.find("h1")
        if h1_tag is not None:
            text = h1_tag.get_text(strip=True)
            if text:
                return text

        hostname = urlsplit(final_url).hostname
        return hostname or final_url

    def _build_chunks(
        self, page: FetchedPage, document_id: str
    ) -> tuple[DocumentChunk, ...]:
        raw_chunks = [_trim_chunk_text(chunk) for chunk in self._splitter.split_text(page.text)]
        non_empty_chunks = [chunk for chunk in raw_chunks if chunk]
        if not non_empty_chunks:
            raise ContentExtractionError("Splitting the extracted text produced no chunks.")

        chunk_count = len(non_empty_chunks)
        return tuple(
            DocumentChunk(
                id=f"{document_id}-chunk-{index:04d}",
                document_id=document_id,
                source_url=page.source_url,
                final_url=page.final_url,
                title=page.title,
                text=text,
                chunk_index=index,
                chunk_count=chunk_count,
                content_hash=page.content_hash,
            )
            for index, text in enumerate(non_empty_chunks)
        )
