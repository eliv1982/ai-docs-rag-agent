"""Look up one package via the real read-only PyPI JSON API.

Performs a single GET request to https://pypi.org/pypi/{package_name}/json (or the
configured PYPI_BASE_URL equivalent) and prints a concise deterministic summary. Not
part of the automated test suite; tests inject fake services only.
"""

import argparse
import sys

from ai_docs_agent.config import get_settings
from ai_docs_agent.models import PyPIPackageInfo
from ai_docs_agent.pypi import PyPILookupError, PyPILookupService

_NOT_SPECIFIED = "not specified"


def format_pypi_report(result: PyPIPackageInfo) -> list[str]:
    """Render a human-readable report line list for one PyPI package lookup."""
    return [
        f"Package: {result.package_name}",
        f"Latest version: {result.latest_version}",
        f"Summary: {result.summary or _NOT_SPECIFIED}",
        f"Requires Python: {result.requires_python or _NOT_SPECIFIED}",
        f"PyPI URL: {result.pypi_url}",
        f"Project URL: {result.project_url or _NOT_SPECIFIED}",
    ]


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
    parser = argparse.ArgumentParser(description="Look up one package via the PyPI JSON API.")
    parser.add_argument("package_name", help="The PyPI package name to look up.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, service: PyPILookupService | None = None) -> int:
    _configure_stream_errors(sys.stdout)
    _configure_stream_errors(sys.stderr)

    args = _parse_args(argv)

    if service is None:
        service = PyPILookupService(get_settings())

    try:
        result = service.lookup(args.package_name)
    except PyPILookupError as exc:
        print(f"PyPI lookup FAILED: {exc}")
        return 1
    except Exception:
        print("PyPI lookup FAILED: unexpected internal error")
        return 1

    for line in format_pypi_report(result):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
