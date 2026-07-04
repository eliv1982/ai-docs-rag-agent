"""Ask the real LangChain tool-calling agent one question.

Performs read-only calls only: OpenAI for tool selection and the existing documentation
answer path, Pinecone for documentation retrieval, and PyPI for package metadata when the
agent selects that tool.
"""

import argparse
import sys

from ai_docs_agent.config import get_settings
from ai_docs_agent.langchain_agent import LangChainAgentExecutionError, LangChainToolCallingAgent
from ai_docs_agent.models import LangChainAgentResult


def format_agent_report(result: LangChainAgentResult) -> list[str]:
    """Render a human-readable report line list for one agent answer."""
    lines = ["Answer:", result.answer, ""]

    if not result.sources:
        lines.append("Sources: none")
    else:
        lines.append("Sources:")
        for rank, source in enumerate(result.sources, start=1):
            lines.append(f"{rank}. {source.title} — {source.url}")

    tools_used = ", ".join(result.tools_used) if result.tools_used else "no tool"
    lines.extend(["", f"Tools used: {tools_used}"])
    if result.outcome == "safe_fallback":
        lines.append(f"Outcome: {result.outcome} ({result.failure_category})")
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
    parser = argparse.ArgumentParser(description="Ask the LangChain tool-calling agent.")
    parser.add_argument("question", help="The user question to answer.")
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
    *,
    service: LangChainToolCallingAgent | None = None,
) -> int:
    _configure_stream_errors(sys.stdout)
    _configure_stream_errors(sys.stderr)

    args = _parse_args(argv)

    try:
        resolved_service = service or LangChainToolCallingAgent(get_settings())
        result = resolved_service.answer(args.question)
    except LangChainAgentExecutionError:
        print("Agent FAILED: unexpected orchestration error")
        return 1
    except Exception:
        print("Agent FAILED: unexpected startup error")
        return 1

    for line in format_agent_report(result):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
