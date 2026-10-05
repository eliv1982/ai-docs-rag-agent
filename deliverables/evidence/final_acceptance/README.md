# Final acceptance evidence

Screenshots from one manual, live run of the integrated Telegram bot
(`python scripts/run_telegram_bot.py`) against a real Pinecone index, captured on
2026-07-04 at commit `db7d147` ("Integrate Telegram agent and persistent memory"). The
conversations are in Russian, matching the bot's user-facing strings.

These screenshots are historical acceptance evidence from 2026-07-04, not current-state
verification. They predate the October 2026 hardening commits. Current behavior is verified by
the automated test suite and CI.

A tenth screenshot of console logs was removed from this set: it showed session hashes from the
earlier unkeyed scheme, a derived memory-identity digest, and index and namespace names, and
publishing them serves no purpose. Log content and its absence of question text, memory text and
raw chat IDs are covered by the automated tests instead.

| File | Prompt | What it shows |
| --- | --- | --- |
| `01_start.png` | `/start` | Introduction: documentation answers, PyPI lookup, the explicit `Запомни:` command, and `/reset` semantics |
| `02_documentation_agent.png` | "Что такое embeddings в OpenAI API?" | Documentation answer with an OpenAI docs source URL (`documentation_search`) |
| `03_pypi_agent.png` | "Какая последняя версия пакета httpx на PyPI?" | Live PyPI metadata with PyPI and project source URLs (`pypi_lookup`) |
| `04_memory_remember.png` | "Запомни: в примерах я предпочитаю httpx." | Confirmation only; no record ID, namespace or digest is shown |
| `05_memory_recall.png` | "Какую HTTP-библиотеку я предпочитаю?" | Recalled preference with the personal-memory label (`user_memory_recall`) |
| `06_memory_survives_reset.png` | `/reset`, then the same question | The dialogue context is cleared; the saved preference is still recalled |
| `07_short_term_alias.png` | "В этом диалоге называй RecursiveCharacterTextSplitter Резаком. Для чего он нужен?", then "Какие параметры есть у Резака?" | Short-term context resolves the alias; answers cite the LangChain splitter guide |
| `08_reset_clears_alias.png` | `/reset`, then "Какие параметры есть у Резака?" | The alias is gone; the fixed no-context fallback is returned without sources |
| `09_out_of_scope.png` | "Как сварить борщ?" | Fixed fallback without sources; no answer from general knowledge |
