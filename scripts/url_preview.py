"""Preview the URL ingestion pipeline for a single documentation URL.

Fetches, extracts, and chunks the given URL via UrlIngestionService.process_url.
Does not call OpenAI or Pinecone, and does not write anything to a vector store.
"""

import argparse
import sys

from ai_docs_agent.config import get_settings
from ai_docs_agent.models import UrlProcessingResult
from ai_docs_agent.url_ingestion import UrlIngestionError, UrlIngestionService

_PREVIEW_CHAR_LIMIT = 500


def format_preview_report(result: UrlProcessingResult) -> list[str]:
    """Render a human-readable report line list for a UrlProcessingResult."""
    first_chunk = result.chunks[0]
    preview_text = first_chunk.text[:_PREVIEW_CHAR_LIMIT]
    return [
        "URL processing OK",
        f"  source URL:     {result.source_url}",
        f"  final URL:      {result.final_url}",
        f"  title:          {result.title}",
        f"  document id:    {result.document_id}",
        f"  content hash:   {result.content_hash}",
        f"  text chars:     {result.text_char_count}",
        f"  chunk count:    {result.chunk_count}",
        f"  first chunk id: {first_chunk.id}",
        "  first chunk preview:",
        f"    {preview_text}",
    ]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch a documentation URL and preview its extracted chunks."
    )
    parser.add_argument("url", help="The documentation page URL to fetch and chunk.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, service: UrlIngestionService | None = None) -> int:
    args = _parse_args(argv)
    if service is None:
        service = UrlIngestionService(get_settings())

    try:
        result = service.process_url(args.url)
    except UrlIngestionError as exc:
        print(f"URL processing FAILED: {exc}")
        return 1
    except Exception:
        print("URL preview FAILED: unexpected internal error")
        return 1

    for line in format_preview_report(result):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
