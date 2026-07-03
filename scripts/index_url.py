"""Index a single documentation URL into Pinecone: fetch -> embed -> upsert -> verify -> cleanup.

Performs real network calls against the target URL, OpenAI, and Pinecone using
credentials from the environment (or a local .env file). Not part of the
automated test suite.

Exit codes: 0 = indexing, verification, and any requested cleanup all
succeeded; 1 = domain/execution error; 2 = indexing and verification
succeeded but cleanup of stale page versions failed.
"""

import argparse
import sys

from ai_docs_agent.config import get_settings
from ai_docs_agent.indexing import DocumentIndexingError, DocumentIndexingService
from ai_docs_agent.models import DocumentIndexingResult
from ai_docs_agent.url_ingestion import UrlIngestionError


def format_index_report(result: DocumentIndexingResult) -> tuple[list[str], int]:
    """Render a human-readable report line list and exit code for an indexing result."""
    cleanup_failed = (
        result.old_versions_cleanup_requested and result.old_versions_cleanup_succeeded is False
    )
    if cleanup_failed:
        lines = ["URL indexing FAILED: cleanup of old page versions did not complete"]
        exit_code = 2
    else:
        lines = ["URL indexing OK"]
        exit_code = 0

    if result.old_versions_cleanup_succeeded is None:
        cleanup_status = "not requested"
    elif result.old_versions_cleanup_succeeded:
        cleanup_status = "ok"
    else:
        cleanup_status = "FAILED"

    lines.extend(
        [
            f"  source URL:      {result.source_url}",
            f"  final URL:       {result.final_url}",
            f"  document id:     {result.document_id}",
            f"  content hash:    {result.content_hash}",
            f"  namespace:       {result.namespace}",
            f"  chunk count:     {result.chunk_count}",
            f"  embedded count:  {result.embedded_count}",
            f"  upserted count:  {result.upserted_count}",
            f"  verified count:  {result.verified_count}",
            f"  cleanup:         {cleanup_status}",
            f"  elapsed:         {result.elapsed_seconds:.2f}s",
        ]
    )
    return lines, exit_code


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch, embed, and index a documentation URL into Pinecone."
    )
    parser.add_argument("url", help="The documentation page URL to fetch, embed, and index.")
    parser.add_argument(
        "--namespace",
        default=None,
        help="Pinecone namespace to write to (defaults to PINECONE_DOCUMENTS_NAMESPACE).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, service: DocumentIndexingService | None = None) -> int:
    args = _parse_args(argv)
    if service is None:
        service = DocumentIndexingService(get_settings())

    try:
        result = service.index_url(args.url, namespace=args.namespace)
    except (DocumentIndexingError, UrlIngestionError) as exc:
        print(f"URL indexing FAILED: {exc}")
        return 1
    except Exception:
        print("URL indexing FAILED: unexpected internal error")
        return 1

    lines, exit_code = format_index_report(result)
    for line in lines:
        print(line)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
