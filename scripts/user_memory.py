"""Explicitly store or recall one long-term user memory statement (live Pinecone/OpenAI).

Usage:
    python scripts/user_memory.py remember <user-identifier> "<statement>"
    python scripts/user_memory.py recall <user-identifier> "<query>"

Uses the real OpenAI embedding and Pinecone services and writes/reads only the
derived pseudonymous per-user namespace. Never prints the raw namespace, the
hash secret, or any API keys. Not part of the automated test suite; tests
inject a fake service only.
"""

import argparse
import sys

from ai_docs_agent.config import get_settings
from ai_docs_agent.models import UserMemoryRecallResult, UserMemoryWriteResult
from ai_docs_agent.user_memory import UserMemoryError, UserMemoryService


def format_write_report(result: UserMemoryWriteResult) -> list[str]:
    """Render a human-readable report line list for one remember operation."""
    return [
        f"User memory remember: {result.status}",
        f"Memory ID: {result.memory_id}",
        f"Identity digest: {result.identity_digest}",
    ]


def format_recall_report(result: UserMemoryRecallResult) -> list[str]:
    """Render a human-readable report line list for one recall operation."""
    lines = [
        (
            f"User memory recall: {len(result.matches)} of {result.raw_candidate_count} "
            f"candidate(s) accepted (threshold {result.threshold}, top_k {result.top_k})"
        ),
        f"Identity digest: {result.identity_digest}",
    ]
    if not result.found:
        lines.append("No stored memories matched the query.")
        return lines
    for position, match in enumerate(result.matches, start=1):
        lines.append(f"{position}. [{match.score:.4f}] {match.text}")
    return lines


def _configure_stream_errors(stream: object) -> None:
    """Make `stream` replace unencodable characters instead of raising."""
    reconfigure = getattr(stream, "reconfigure", None)
    if not callable(reconfigure):
        return

    try:
        reconfigure(errors="replace")
    except Exception:
        return


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explicitly store or recall one long-term user memory statement."
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)

    remember_parser = subparsers.add_parser(
        "remember", help="Explicitly store one memory statement."
    )
    remember_parser.add_argument("user_identifier", help="The external user identifier.")
    remember_parser.add_argument("statement", help="The memory statement to store.")

    recall_parser = subparsers.add_parser(
        "recall", help="Semantically search the user's own stored memories."
    )
    recall_parser.add_argument("user_identifier", help="The external user identifier.")
    recall_parser.add_argument("query", help="The recall query text.")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, service: UserMemoryService | None = None) -> int:
    _configure_stream_errors(sys.stdout)
    _configure_stream_errors(sys.stderr)

    args = _parse_args(argv)

    try:
        if service is None:
            service = UserMemoryService(get_settings())

        if args.operation == "remember":
            lines = format_write_report(service.remember(args.user_identifier, args.statement))
        else:
            lines = format_recall_report(service.recall(args.user_identifier, args.query))
    except UserMemoryError as exc:
        print(f"User memory FAILED: {exc}")
        return 1
    except Exception:
        print("User memory FAILED: unexpected internal error")
        return 1

    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
