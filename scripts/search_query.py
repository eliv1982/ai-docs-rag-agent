"""Search the indexed documentation in Pinecone for chunks similar to a text query.

Performs real network calls against OpenAI (query embedding) and Pinecone (query)
using credentials from the environment (or a local .env file). Read-only: never
upserts, deletes, re-indexes, or otherwise modifies index/namespace data. Not part
of the automated test suite.
"""

import argparse
import sys

from ai_docs_agent.config import get_settings
from ai_docs_agent.models import RetrievalResult
from ai_docs_agent.retrieval import RetrievalError, RetrievalService

_TEXT_PREVIEW_CHAR_LIMIT = 500


def format_search_report(result: RetrievalResult) -> list[str]:
    """Render a human-readable report line list for a RetrievalResult."""
    lines = [
        "Retrieval search OK",
        f"  query:     {result.query}",
        f"  namespace: {result.namespace}",
        f"  top_k:     {result.top_k}",
        f"  matches:   {len(result.matches)}",
    ]

    if not result.matches:
        lines.append("  (no matches found)")
        return lines

    for rank, chunk in enumerate(result.matches, start=1):
        preview_text = chunk.text[:_TEXT_PREVIEW_CHAR_LIMIT]
        lines.extend(
            [
                "",
                f"  #{rank}",
                f"    score:            {chunk.score:.4f}",
                f"    title:            {chunk.title}",
                f"    source URL:       {chunk.source_url}",
                f"    final URL:        {chunk.final_url}",
                f"    document id:      {chunk.document_id}",
                f"    chunk index/count: {chunk.chunk_index}/{chunk.chunk_count}",
                "    text:",
                f"      {preview_text}",
            ]
        )
    return lines


def _configure_stream_errors(stream: object) -> None:
    """Make `stream` replace unencodable characters instead of raising.

    Retrieved chunk text and argparse error messages are arbitrary/user-supplied
    and may contain characters a narrow console codepage can't represent.
    `reconfigure` may be absent, non-callable, or itself raise; none of that
    should prevent the CLI from running.
    """
    reconfigure = getattr(stream, "reconfigure", None)
    if not callable(reconfigure):
        return

    try:
        reconfigure(errors="replace")
    except Exception:
        return


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search the indexed documentation in Pinecone for chunks similar to a query."
    )
    parser.add_argument("query", help="The text query to search for.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Maximum number of results to return (defaults to RETRIEVAL_TOP_K).",
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help="Pinecone namespace to search (defaults to PINECONE_DOCUMENTS_NAMESPACE).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, service: RetrievalService | None = None) -> int:
    _configure_stream_errors(sys.stdout)
    _configure_stream_errors(sys.stderr)

    args = _parse_args(argv)

    if service is None:
        service = RetrievalService(get_settings())

    try:
        result = service.search(args.query, top_k=args.top_k, namespace=args.namespace)
    except RetrievalError as exc:
        print(f"Retrieval search FAILED: {exc}")
        return 1
    except Exception:
        print("Retrieval search FAILED: unexpected internal error")
        return 1

    for line in format_search_report(result):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
