"""Ask a grounded question against the indexed documentation (retrieval + LLM answer).

Performs real network calls against OpenAI (query embedding + chat completion) and
Pinecone (query) using credentials from the environment (or a local .env file).
Read-only with respect to Pinecone: never upserts, deletes, re-indexes, or
otherwise modifies index/namespace data. Not part of the automated test suite.
"""

import argparse
import sys

from ai_docs_agent.agent import AnswerServiceError, DocumentationAnswerService
from ai_docs_agent.config import get_settings
from ai_docs_agent.models import GroundedAnswerResult


def format_answer_report(result: GroundedAnswerResult) -> list[str]:
    """Render a human-readable report line list for a GroundedAnswerResult."""
    lines = ["Answer:", result.answer, ""]

    if not result.sources:
        lines.append("Sources: none")
        return lines

    lines.append("Sources:")
    for rank, source in enumerate(result.sources, start=1):
        lines.append(f"{rank}. {source.title} — {source.url}")
    return lines


def _configure_stream_errors(stream: object) -> None:
    """Make `stream` replace unencodable characters instead of raising.

    Retrieved chunk text, model answers, and argparse error messages are
    arbitrary/user-supplied and may contain characters a narrow console
    codepage can't represent. `reconfigure` may be absent, non-callable, or
    itself raise; none of that should prevent the CLI from running.
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
        description="Ask a grounded question against the indexed documentation."
    )
    parser.add_argument("question", help="The question to answer from indexed documentation.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Maximum number of retrieved chunks to use (defaults to RETRIEVAL_TOP_K).",
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help="Pinecone namespace to search (defaults to PINECONE_DOCUMENTS_NAMESPACE).",
    )
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None, *, service: DocumentationAnswerService | None = None
) -> int:
    _configure_stream_errors(sys.stdout)
    _configure_stream_errors(sys.stderr)

    args = _parse_args(argv)

    if service is None:
        service = DocumentationAnswerService(get_settings())

    try:
        result = service.answer(args.question, top_k=args.top_k, namespace=args.namespace)
    except AnswerServiceError as exc:
        print(f"Answer FAILED: {exc}")
        return 1
    except Exception:
        print("Answer FAILED: unexpected internal error")
        return 1

    for line in format_answer_report(result):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
