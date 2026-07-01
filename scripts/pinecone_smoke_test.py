"""Live smoke test: OpenAI embedding -> Pinecone upsert -> query -> cleanup.

Performs real network calls against OpenAI and Pinecone using credentials from
the environment (or a local .env file). Not part of the automated test suite.

Exit codes: 0 = full success (including cleanup); 1 = execution/domain error;
2 = the smoke-test pipeline succeeded but cleanup of the test vector failed.
"""

import sys

from ai_docs_agent.config import get_settings
from ai_docs_agent.models import PineconeSmokeTestResult
from ai_docs_agent.pinecone_store import PineconeSmokeTestError, PineconeStore, PineconeStoreError


def format_smoke_test_report(result: PineconeSmokeTestResult) -> tuple[list[str], int]:
    """Render a human-readable report line list and exit code for a smoke-test result."""
    if result.cleanup_succeeded:
        lines = ["Pinecone smoke test OK"]
        exit_code = 0
    else:
        lines = ["Pinecone smoke test FAILED: cleanup did not complete"]
        exit_code = 2

    lines.extend(
        [
            f"  index name:      {result.index_name}",
            f"  namespace:       {result.namespace}",
            f"  embedding model: {result.embedding_model}",
            f"  dimension:       {result.dimension}",
            f"  matched id:      {result.matched_id}",
            f"  score:           {result.score:.4f}",
            f"  cleanup:         {'ok' if result.cleanup_succeeded else 'FAILED'}",
            f"  elapsed:         {result.elapsed_seconds:.2f}s",
        ]
    )
    return lines, exit_code


def main() -> int:
    settings = get_settings()
    store = PineconeStore(settings)

    try:
        result = store.smoke_test()
    except (PineconeStoreError, PineconeSmokeTestError) as exc:
        print(f"Pinecone smoke test FAILED: {exc}")
        return 1

    lines, exit_code = format_smoke_test_report(result)
    for line in lines:
        print(line)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
