"""Smoke test: verify the ai_docs_agent package is importable and set up correctly."""

import ai_docs_agent

PROJECT_NAME = "ai-docs-rag-agent"


def main() -> None:
    print(f"{PROJECT_NAME} {ai_docs_agent.__version__}")


if __name__ == "__main__":
    main()
