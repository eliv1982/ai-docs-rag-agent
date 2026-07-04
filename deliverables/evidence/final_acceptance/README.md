# Final acceptance evidence (Stage 4I — integrated Telegram agent)

Screenshots are created manually in Telegram after the live acceptance run of
`python scripts/run_telegram_bot.py`. Nothing here is fabricated: this manifest
only lists the expected filenames and what each screenshot must show.

Checklist / expected files:

- [x] `01_start.png` — `/start` reply advertising documentation answers, PyPI
  lookup, the explicit `Запомни: ...` command, and the accurate `/reset`
  semantics (dialogue context cleared, saved preferences kept).
- [x] `02_documentation_agent.png` — "Что такое embeddings в OpenAI API?" →
  grounded answer with a real OpenAI documentation source
  (documentation_search).
- [x] `03_pypi_agent.png` — "Какая последняя версия пакета httpx на PyPI?" →
  current live version with the PyPI source URL (pypi_lookup).
- [x] `04_memory_remember.png` — "Запомни: в примерах я предпочитаю httpx." →
  created/duplicate confirmation; no record ID, namespace, or digest shown.
- [x] `05_memory_recall.png` — "Какую HTTP-библиотеку я предпочитаю?" →
  answer contains httpx with the personal-memory source label
  (user_memory_recall).
- [x] `06_memory_survives_reset.png` — `/reset` followed by the same preference
  question; httpx is still recalled from persistent memory.
- [x] `07_short_term_alias.png` — alias set ("...называй
  RecursiveCharacterTextSplitter Резаком.") followed by "Для чего нужен
  Резак?" → grounded documentation answer.
- [x] `08_reset_clears_alias.png` — `/reset` followed by "Для чего нужен
  Резак?" → alias no longer resolves (safe no-context fallback).
- [x] `09_out_of_scope.png` — "Как сварить борщ?" → safe fallback without
  sources.
- [x] `10_safe_agent_logs.png` — console logs proving tool selection and
  memory outcomes (session hash, tool names, created/duplicate, counts) with
  no question text, memory text, raw chat ID, user-memory namespace, or secrets.
