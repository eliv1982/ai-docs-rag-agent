# AI Docs RAG Agent

Telegram RAG-агент для работы с технической документацией.

**Статус: typed configuration + Pinecone index management/smoke-test + URL ingestion/chunking
+ URL indexing into Pinecone (fetch → embeddings → upsert → verify → cleanup) + typed
read-only semantic search (`RetrievalService.search`) + minimal grounded RAG answering
(`DocumentationAnswerService.answer`) + short-term in-memory conversation memory
(`ConversationAnswerService`) + minimal Telegram bot MVP (`TelegramBotService`,
`python-telegram-bot`) + typed read-only PyPI JSON lookup (`PyPILookupService`,
`lookup_pypi_package`, `scripts/pypi_lookup.py`) + real LangChain tool-calling agent
(`LangChainToolCallingAgent`, `scripts/ask_agent.py`) implemented. Telegram integration of
the new agent and vector memory are not implemented yet.**

## Планируемые возможности

- приём URL страниц документации;
- загрузка и очистка HTML;
- разбиение текста на чанки;
- создание embeddings;
- хранение данных в Pinecone;
- семантический поиск;
- генерация grounded-ответов по найденному контексту;
- контролируемая пользовательская память;
- интерфейс через Telegram.

## Архитектурный статус

Реализованы: URL ingestion, HTML parsing, чанкинг, создание embeddings и запись чанков в
Pinecone (`DocumentIndexingService.index_url`), типизированный read-only семантический поиск
по уже проиндексированным чанкам (`RetrievalService.search`, см. ниже), минимальный
grounded RAG answering поверх retrieval (`DocumentationAnswerService.answer`, см. ниже),
короткая process-local разговорная память по `session_id` (`ConversationAnswerService`, см.
ниже), минимальный Telegram bot MVP (`TelegramBotService`, см. ниже) поверх того же
`ConversationAnswerService`, а также typed read-only PyPI lookup (`PyPILookupService`,
`lookup_pypi_package`, см. ниже) и полноценный LangChain tool-calling agent
(`LangChainToolCallingAgent`, см. ниже). Telegram-бот на этом этапе по-прежнему использует
существующий `ConversationAnswerService`, а не новый agent layer.

На данном этапе реализовано:

- типизированная конфигурация приложения (`AppSettings`, `pydantic-settings`);
- управление Pinecone-индексом (`PineconeStore.ensure_index`);
- live-интеграционный smoke-test: OpenAI embedding → Pinecone upsert → query → cleanup
  (`PineconeStore.smoke_test`);
- URL ingestion и chunking pipeline (`UrlIngestionService.process_url`): валидация URL,
  ограниченная по размеру и redirect'ам загрузка HTML, извлечение и нормализация текста,
  детерминированные ID документа/чанков и разбиение на чанки;
- индексация URL в Pinecone (`DocumentIndexingService.index_url`): URL → chunks (через
  `UrlIngestionService`) → batch OpenAI embeddings → batch Pinecone upsert → bounded
  fetch-verification → удаление устаревших версий той же страницы;
- типизированный read-only семантический поиск (`RetrievalService.search`): query text →
  OpenAI query embedding → Pinecone similarity query → строго провалидированные
  `RetrievedChunk` (см. ниже);
- минимальный grounded RAG answering (`DocumentationAnswerService.answer`): question →
  `RetrievalService.search` → grounded LLM prompt → одна text-ответ chat completion →
  `GroundedAnswerResult` с детерминированным списком источников (см. ниже);
- типизированный read-only PyPI lookup (`PyPILookupService.lookup`, `lookup_pypi_package`,
  см. ниже): package name → реальный GET `https://pypi.org/pypi/{package_name}/json` →
  `PyPIPackageInfo` c безопасным domain error mapping;
- real LangChain tool-calling orchestration (`LangChainToolCallingAgent`,
  `scripts/ask_agent.py`, см. ниже): natural-language request → autonomous single-tool
  selection между `documentation_search` и `pypi_lookup` → детерминированный
  пользовательский ответ из authoritative tool result → `LangChainAgentResult`;
- короткая process-local разговорная память по `session_id` (`ConversationAnswerService`,
  `InMemoryConversationMemory`, см. ниже): последние сообщения диалога подставляются в промпт
  для continuity, но никогда не считаются документальным фактом;
- минимальный Telegram bot MVP (`TelegramBotService`, `python-telegram-bot`, см. ниже):
  `/start`, `/reset` и обычные текстовые вопросы поверх того же `ConversationAnswerService`
  (`chat_id` используется как `session_id`).

`tools.py` теперь содержит тонкие typed adapters `answer_documentation_question(...)` и
`lookup_pypi_package(...)`; реальная LangChain agent orchestration реализована отдельно в
`src/ai_docs_agent/langchain_agent.py`.

### URL ingestion и chunking pipeline

`src/ai_docs_agent/url_ingestion.py` определяет `UrlIngestionService.process_url(url)`,
реализующий pipeline:

```
URL → validation (scheme/host/SSRF guard) → bounded-redirect HTTP fetch →
bounded response size → HTML extraction → normalized text →
deterministic document ID → RecursiveCharacterTextSplitter → deterministic chunks
```

Результат — `UrlProcessingResult` с готовыми `DocumentChunk` в памяти. Ничего не
записывается в Pinecone и не отправляется в OpenAI на этом этапе.

**SSRF guard — best-effort защита, а не абсолютная гарантия безопасности.** Проверяются
схема (`http`/`https`), отсутствие credentials в URL, `localhost`/`*.localhost`, и все
resolved-адреса hostname (через injectable DNS resolver) на loopback/private/link-local/
multicast/reserved/unspecified диапазоны — при каждом redirect повторно. Hostname перед
всеми этими проверками (и перед вызовом resolver'а) нормализуется: lowercase и удаление
конечных точек — это закрывает обход вида `localhost.`/`LOCALHOST.`/`app.localhost.`, при
котором literal-проверка ранее выполнялась до нормализации. Это снижает риск базовых
SSRF-сценариев, но не заменяет сетевые egress-ограничения на инфраструктурном уровне
(например, DNS rebinding между проверкой и фактическим TCP-соединением остаётся
теоретически возможным).

Поддерживаемые `Content-Type`: `text/html`, `application/xhtml+xml` (с необязательным
`charset`). Остальные типы отклоняются (`UnsupportedContentTypeError`). Неизвестный или
невалидный `charset` (например, `charset=definitely-not-a-codec`) не приводит к raw
`LookupError` — тело ответа при этом не читается силой в случайной кодировке, а fetch
завершается controlled-ошибкой `ContentExtractionError` (с исходным `LookupError` в
`__cause__`, без утечки HTML body).

Извлечение текста нормализует обычную prose (схлопывание пробелов/пустых строк), но
сохраняет форматирование блоков `<pre>`/`<pre><code>` (значимые leading-отступы и
внутренняя индентация не удаляются) — это важно для технической документации с
примерами кода/конфигов.

Ограничения (настраиваются через `.env`, см. ниже):

- `URL_FETCH_TIMEOUT_SECONDS` — таймаут HTTP-запроса;
- `URL_MAX_RESPONSE_BYTES` — лимит размера тела ответа (проверяется по `Content-Length` до
  чтения тела **и** по фактически прочитанным байтам во время потокового чтения — чтение
  прекращается сразу при превышении);
- `URL_MAX_REDIRECTS` — максимум HTTP-redirect'ов (redirect обрабатывается вручную,
  `follow_redirects=False`; каждый redirect-target заново проходит полную URL/DNS
  валидацию);
- `URL_MIN_TEXT_CHARS` — минимальная длина извлечённого текста;
- `CHUNK_SIZE` / `CHUNK_OVERLAP` — параметры `RecursiveCharacterTextSplitter`.

Document ID и chunk ID детерминированы (SHA-256 от `final_url` + content hash) — одна и та
же страница с тем же содержимым всегда даёт одни и те же ID; изменившийся контент даёт
новый document ID.

Предпросмотр pipeline без записи куда-либо:

```
python scripts/url_preview.py "https://example.com/docs"
```

### URL indexing в Pinecone

`src/ai_docs_agent/indexing.py` определяет `DocumentIndexingService.index_url(url, *,
namespace=None)`, реализующий полный pipeline:

```
URL → UrlIngestionService.process_url → DocumentChunk[] →
batch OpenAI embeddings (PineconeStore.embed_documents) →
batch Pinecone upsert (PineconeStore.upsert_vectors) →
bounded fetch-verification (PineconeStore.fetch_existing_ids) →
удаление устаревших версий той же страницы (PineconeStore.delete_vectors_by_filter)
```

Сервис не дублирует URL validation/HTML extraction/chunking (переиспользует
`UrlIngestionService`) и не создаёт отдельный Pinecone/OpenAI client layer (переиспользует
`PineconeStore`, расширенный низкоуровневыми методами `embed_documents`, `upsert_vectors`,
`fetch_existing_ids`, `delete_vectors_by_filter`).

**Namespace.** По умолчанию используется `PINECONE_DOCUMENTS_NAMESPACE` (`documentation`);
можно передать явный namespace (`index_url(url, namespace=...)` / `--namespace` в CLI) — он
проверяется на пустоту **до** любого сетевого вызова.

**Batching.** Embeddings создаются батчами по `EMBEDDING_BATCH_SIZE`, upsert выполняется
батчами по `PINECONE_UPSERT_BATCH_SIZE`, verification (fetch) — батчами по
`PINECONE_FETCH_BATCH_SIZE` (максимум 1000 за один Pinecone `fetch`). Порядок chunk ↔
embedding ↔ upsert-record сохраняется даже при разных размерах батчей на разных этапах.
`embed_documents` не обращается к Pinecone control-plane — только к embeddings client.

**Один переиспользуемый Pinecone index handle.** В рамках одного `PineconeStore` instance
control-plane readiness (`ensure_index()` + получение index handle) выполняется лениво только
при первом data-plane вызове (upsert/fetch/delete) и кешируется; последующие upsert/fetch/delete
на этом же instance переиспользуют закешированный handle без повторных `has_index`/
`describe_index`. Если data-plane вызов падает с SDK-ошибкой, кеш инвалидируется, и следующий
самостоятельный вызов заново проверяет readiness. Кеш принадлежит конкретному instance —
глобального singleton нет.

**Верификация (cumulative).** После upsert сервис ограниченным polling (bounded, без реальных
задержек в тестах) проверяет через `fetch`, что все ожидаемые chunk ID появились в индексе —
подтверждённые ID накапливаются между раундами (а не пересчитываются заново на каждом раунде),
поэтому если один ID виден на раунде 1, а другой — только на раунде 2, верификация всё равно
успешно завершается. Таймаут и интервал берутся из `PINECONE_INDEX_VERIFY_TIMEOUT_SECONDS` /
`PINECONE_INDEX_VERIFY_POLL_INTERVAL_SECONDS`. При таймауте выбрасывается
`DocumentVerificationError` с cumulative числом найденных/ожидаемых записей, namespace и
document ID (без секретов).

**Cleanup устаревших записей.** Если `PINECONE_REPLACE_OLD_SOURCE_VERSIONS=true` (по
умолчанию), после успешного upsert и verification сервис удаляет из того же namespace все
записи того же `source_url`, которые либо принадлежат другой версии документа (другой
`document_id`), либо являются устаревшими чанками **текущего** документа с `chunk_index` вне
текущего `chunk_count` (например, если после изменения конфигурации чанкинга разбиение
страницы уменьшилось с 5 чанков до 4, а текст/`content_hash` не изменились — chunk 4 из
старого разбиения всё равно будет удалён). Текущие чанки (`chunk_index < chunk_count`) и
записи с другим `source_url` не затрагиваются; повторная индексация неизменённой страницы
остаётся идемпотентной. Document ID и chunk ID детерминированы (SHA-256 от `final_url` +
content hash), поэтому повторный запуск безопасно перезаписывает те же записи.

Если cleanup завершился ошибкой **после** успешного upsert и verification, ошибка не
скрывает уже выполненную успешную индексацию: `DocumentIndexingResult` возвращается с
`old_versions_cleanup_succeeded=False`, а CLI-скрипт завершается с exit code `2`.

**Известное ограничение: между batch-операциями upsert нет транзакционности.** Если сетевая
ошибка происходит между Pinecone upsert-батчами, часть новых записей текущей версии страницы
может быть уже записана. Так как chunk ID детерминированы, безопасный повторный запуск
`index_url` для того же URL перезапишет те же записи (upsert идемпотентен по ID) — данные не
задваиваются, но частично записанная попытка не считается успешной (выбрасывается
`DocumentUpsertError`) и не запускает cleanup.

**Известное ограничение: race между параллельными writer'ами.** `index_url` не использует
distributed locking — если два процесса одновременно индексируют один и тот же `source_url`
(например, старую и новую версию контента), их upsert/verification/cleanup шаги могут
чередоваться, и cleanup одного запуска потенциально может удалить записи, которые только что
записал другой. Для одного writer'а на URL за раз (типичный сценарий для этого этапа) это не
проблема.

Индексация URL (live, требует реальных OpenAI/Pinecone ключей):

```
python scripts/index_url.py "https://example.com/docs"
python scripts/index_url.py "https://example.com/docs" --namespace documentation
```

### Semantic search (retrieval)

`src/ai_docs_agent/retrieval.py` определяет `RetrievalService.search(query, *, top_k=None,
namespace=None) -> RetrievalResult`, реализующий read-only pipeline:

```
query text → validation → OpenAI query embedding (PineconeStore.embed_query) →
Pinecone similarity query (PineconeStore.query_similar, всегда с filter
{"kind": {"$eq": "documentation_chunk"}}) → строгая проверка metadata каждого match →
RetrievedChunk[] → RetrievalResult
```

Сервис переиспользует существующие `PineconeStore` (расширенный низкоуровневыми методами
`embed_query` и `query_similar`) и настройки индекса/namespace/embedding-модели/dimension —
отдельных retrieval-специфичных Pinecone/embedding настроек нет, кроме `RETRIEVAL_TOP_K`.

**Обязательный фильтр.** Каждый запрос всегда ограничен `filter={"kind": {"$eq":
"documentation_chunk"}}` — это защищает от попадания в результаты не-документных записей
(например, тестовых векторов из `PineconeStore.smoke_test`, которые помечены
`kind="integration_smoke_test"` и обычно живут в отдельном `PINECONE_SMOKE_NAMESPACE`).
Фильтр внутренний и не настраивается через `search()` или CLI.

**top_k.** По умолчанию — `RETRIEVAL_TOP_K` (`1..50`, по умолчанию `5`); можно передать
`top_k` явно (`search(query, top_k=...)` / `--top-k` в CLI). Namespace по умолчанию —
`PINECONE_DOCUMENTS_NAMESPACE`, можно переопределить (`namespace=...` / `--namespace`).
`len(RetrievalResult.matches)` может быть меньше `top_k` (обычный случай для небольшого
индекса или редкого запроса) — это не ошибка; пустой результат тоже считается успешным.

**Score.** `score` — непрозрачное (opaque) численное значение, возвращаемое Pinecone
как есть; используется только для сохранения порядка (Pinecone возвращает matches уже
отсортированными от наиболее похожего к наименее похожему). Сам `RetrievalService` **не**
применяет порог, ранжирование или дедупликацию по score — он возвращает raw matches как есть.
Минимальный relevance gate применяется уровнем выше, в `DocumentationAnswerService` (см. ниже).

**Устаревшие/дублирующиеся чанки.** Retrieval не выполняет собственной очистки — за
отсутствие устаревших версий страницы в индексе отвечает cleanup на этапе индексации
(`DocumentIndexingService.index_url`, см. выше), который по умолчанию включён
(`PINECONE_REPLACE_OLD_SOURCE_VERSIONS=true`), но является best-effort (может быть отключён,
или не завершиться, если запись была прервана, или произойти гонка при параллельной
переиндексации той же страницы). Поэтому в редких случаях результаты поиска теоретически
могут содержать устаревшие записи — это ограничение унаследовано от indexing-этапа, а не
специфично для retrieval.

Поиск по индексу (live, требует реальных OpenAI/Pinecone ключей; строго read-only — не
выполняет upsert/delete/переиндексацию):

```
python scripts/search_query.py "how do I configure the client?"
python scripts/search_query.py "how do I configure the client?" --top-k 3 --namespace documentation
```

### Grounded RAG answering

`src/ai_docs_agent/agent.py` определяет `DocumentationAnswerService.answer(question, *,
top_k=None, namespace=None) -> GroundedAnswerResult`, реализующий минимальный grounded
answering pipeline:

```
question → direct RetrievalService.search() →
если accepted-контекста нет и есть recent history: один bounded contextual retry
с standalone retrieval query, выведенным из history + current question →
retrieved RetrievedChunk[] → компактный grounded system+user prompt
(текст чанков — untrusted data, не instructions) →
один plain-text chat completion (OpenAI Chat Completions API) →
GroundedAnswerResult { answer, sources, retrieved_chunk_count }
```

Сервис переиспользует `RetrievalService` (не дублирует retrieval-логику или обязательный
`documentation_chunk` filter) и принимает chat-клиент через dependency injection
(`ChatClient` protocol; production-реализация — `OpenAIChatClient`, тонкая обёртка над
установленным `openai` SDK, `chat.completions.create`). Конструктор сервиса и импорт модуля
не выполняют сетевых вызовов.

**Источники — из метаданных, не от модели.** `AnswerSource` строится напрямую из
`RetrievedChunk` (`title`, `final_url`/`source_url`, `document_id`, `chunk_index`,
`chunk_count`) — модель никогда не генерирует URL или список источников сама; промпт прямо
запрещает придумывать источники. Источники сохраняют порядок первого появления и
дедуплицируются по итоговому URL (несколько чанков одной страницы → один источник); это
document-level источники, не claim-level citations (не привязаны к конкретному предложению
ответа).

**Conversation-aware retrieval retry.** Для обычного standalone-вопроса сервис сохраняет
существующий direct retrieval path. Если direct retrieval не дал accepted-контекста
(`score >= 0.25`) и recent conversation history есть, сервис может сделать **ровно одну**
bounded contextual retry: сначала с помощью history выводится компактный standalone retrieval
query (например, resolving alias/ссылку вида `Резак` → `RecursiveCharacterTextSplitter`), затем
по нему выполняется второй `RetrievalService.search()`. Если rewrite не делает запрос
текстуально более standalone, сервис все равно использует history для одного history-augmented
contextual query, чтобы alias/ссылки из текущего диалога не терялись в live path. History
помогает только сформулировать retrieval query; она не становится documentation context, не
попадает в `AnswerSource` и не может сама по себе обосновать ответ.

**Fallback без контекста.** Если ни direct retrieval, ни этот один contextual retry не дали
accepted-контекста, chat model для финального ответа не вызывается вообще — сервис
детерминированно возвращает `GroundedAnswerResult` с `retrieved_chunk_count=0`, пустым
`sources` и фиксированным текстом:

```
В базе знаний не найдено достаточно информации для ответа на этот вопрос.
```

**Ошибки.** `AnswerRetrievalError` — сбой на этапе `RetrievalService.search()` (embedding/
query/malformed metadata); `AnswerGenerationError` — сбой chat-клиента **или** пустой/
whitespace-only ответ модели. Оба наследуются от `AnswerServiceError`; исходное исключение
сохраняется в `__cause__`, публичное сообщение не содержит ключей/секретов/сырых деталей
исключения.

**Известное ограничение grounding'а (best-effort, не гарантия).** Grounding в текущем MVP
обеспечивается только промптом (`_SYSTEM_PROMPT` явно требует closed-book-ответ строго по
предоставленному контексту и запрещает домысливать) и содержимым retrieved-чанков — отдельной
верификации фактов нет. Так как генерация — это один свободный (free-form) LLM-вызов без
structured output, модель иногда может добавить правдоподобную, но не подтверждённую явно в
контексте деталь из своих pretrained-знаний (это наблюдалось на живой проверке). Детерминированная
claim-level верификация фактов, structured citations или дополнительный verification pass
(второй LLM-вызов/critic model) не реализованы и отнесены к будущим версиям.

Задать вопрос по индексу (live, требует реальных OpenAI/Pinecone ключей; строго read-only —
не выполняет upsert/delete/переиндексацию):

```
python scripts/ask_docs.py "how do I configure the client?"
python scripts/ask_docs.py "how do I configure the client?" --top-k 3 --namespace documentation
```

### PyPI JSON lookup

`src/ai_docs_agent/pypi.py` определяет `PyPILookupService.lookup(package_name) ->
PyPIPackageInfo`, выполняющий один реальный read-only GET:

```
GET https://pypi.org/pypi/{package_name}/json
```

Сервис заранее валидирует package name (буквы/цифры/`-`/`_`/`.` допускаются; пустые,
URL-like, query/path-injection и другие unsafe значения отклоняются до HTTP-запроса), затем
делает один `httpx`-запрос с явным timeout и возвращает типизированный результат как минимум с
полями:

- `package_name` — canonical name, который вернул PyPI;
- `latest_version`;
- `summary` (`None`, если отсутствует/`null`);
- `requires_python` (`None`, если отсутствует/`null`);
- `pypi_url`;
- `project_url` (`None`, если у пакета нет отдельного project/home URL).

Поддерживаемые error categories: `invalid_package_name`, `package_not_found`, `timeout`,
`network_error`, `malformed_response`, `upstream_http_error`. Публичный adapter
`lookup_pypi_package(...)` в `src/ai_docs_agent/tools.py` — тонкая обёртка над этим сервисом.

CLI для live read-only проверки:

```
python scripts/pypi_lookup.py httpx
```

Скрипт печатает стабильное summary с placeholder `not specified` для отсутствующих optional
полей. Unit-тесты для сервиса/CLI используют только `httpx.MockTransport` и fake-сервисы — без
реальных сетевых вызовов.

### LangChain tool-calling agent

`src/ai_docs_agent/langchain_agent.py` определяет `LangChainToolCallingAgent`, использующий
реальный `langchain.agents.create_agent(...)` поверх двух thin tools:

- `documentation_search(question: str)` → grounded answer + sources через существующий
  `DocumentationAnswerService`;
- `pypi_lookup(package_name: str)` → typed package metadata через существующий
  `PyPILookupService`.

Agent сам выбирает один из этих инструментов по natural-language запросу. Tool descriptions и
system instruction разделяют:

- текущие/последние package metadata вопросы → `pypi_lookup`;
- технические вопросы по OpenAI/Pinecone/LangChain и индексированной документации →
  `documentation_search`.

User-facing результат — `LangChainAgentResult`: итоговый текст ответа, `AnswerSource[]`,
`tools_used`, `tool_call_count`, `used_no_tool`, `outcome` (`success` / `safe_fallback`) и
безопасная `failure_category`. Для PyPI-ответов источник содержит как минимум реальный PyPI
URL; для документации сохраняются источники из `GroundedAnswerResult`.

**Bounded execution.** На этом этапе agent ограничен одним model step для выбора tool и одним
tool step. Пользовательский ответ рендерится напрямую из authoritative tool result, а не из
второго model pass, поэтому:

- текущая версия пакета не может "тихо" прийти из pretrained knowledge;
- documentation sources берутся только из tool output;
- запрос не уходит в loop с повторными tool calls.

**Безопасность tool output.** Tools сериализуют только краткий safe result: answer/status/sources
для документации и package metadata/status для PyPI. Raw traceback'и, chunk bodies, vectors,
секреты и upstream response bodies в tool output не попадают.

CLI для live read-only проверки:

```
python scripts/ask_agent.py "Какая последняя версия пакета httpx на PyPI?"
python scripts/ask_agent.py "Что такое embeddings в OpenAI API?"
```

Скрипт печатает итоговый ответ, sources и строку `Tools used: ...`. Exit code `0` означает
либо полноценный успешный ответ, либо безопасный handled fallback; `1` используется только для
startup/unhandled orchestration failure.

Unit-тесты для agent/CLI полностью mock-based: используется fake tool-calling chat model и
fake сервисы, без реальных OpenAI/Pinecone/PyPI/Telegram вызовов. Live CLI checks выполняются
отдельно вручную и не входят в pytest.

### Conversation memory

`src/ai_docs_agent/memory.py` определяет короткую, **process-local** разговорную память,
изолированную по `session_id`:

- `InMemoryConversationMemory` — обычный `dict[str, list[ConversationMessage]]` в памяти
  процесса: хранит **последние 10** сообщений (`user`/`assistant`) на сессию, старые
  сообщения при превышении лимита отбрасываются первыми (FIFO); `get_history()` возвращает
  неизменяемый `tuple[ConversationMessage, ...]` (снимок, а не живую ссылку на внутренний
  список), сессии друг от друга полностью изолированы, `clear(session_id)` удаляет только
  одну сессию;
- `ConversationAnswerService` — тонкая обёртка вокруг `DocumentationAnswerService`: читает
  историю сессии → вызывает `answer(question, history=..., ...)` → **только при успешном
  результате** добавляет в память нормализованный вопрос и полученный ответ. Если retrieval
  или генерация упали с ошибкой, память сессии остаётся без изменений; fallback-ответ
  "нет данных в базе" — успешный результат и тоже сохраняется в историю. Эта же history
  используется для contextual follow-up retrieval, но не считается documentary evidence.

**Память — process-local и не переживает перезапуск процесса.** Это намеренное ограничение
MVP для этого этапа домашнего задания: ничего не пишется на диск/в БД/по сети. Персистентное
хранилище (например, SQLite) — запланированное улучшение v2 (см. "Planned improvements"
ниже).

**История помогает разрешать ссылки, но не является источником фактов.** История диалога
подставляется в промпт только чтобы помочь модели понять, к чему относятся слова вида "он"/
"эта библиотека", и сохранить continuity между репликами. Промпт явно требует не считать
факт, упомянутый только в истории (а не в retrieved-чанках), подтверждённым документальным
фактом; retrieved-чанки остаются единственным источником фактов для ответа. История (как и
retrieved-текст) считается untrusted data и не может переопределить system-инструкции. В
`AnswerSource`/список источников история никогда не попадает.

Пример использования из Python (без сети — с fake-сервисами; для реального ответа
`DocumentationAnswerService` должен быть сконфигурирован обычным образом, см. выше):

```python
from ai_docs_agent.agent import DocumentationAnswerService
from ai_docs_agent.config import get_settings
from ai_docs_agent.memory import ConversationAnswerService

answer_service = DocumentationAnswerService(get_settings())
conversation = ConversationAnswerService(answer_service)

first = conversation.answer("session-123", "What is LangChain?")
second = conversation.answer("session-123", "How do I configure it?")  # "it" resolved via history

conversation.reset("session-123")  # clears only this session's history
```

### Telegram bot

`src/ai_docs_agent/telegram_bot.py` определяет минимальный Telegram MVP поверх
`ConversationAnswerService`, на библиотеке [`python-telegram-bot`](https://python-telegram-bot.org/)
(она уже требует `httpx`, который и так является зависимостью проекта — новых конфликтующих
зависимостей не добавлено):

```
Telegram-сообщение → str(chat_id) как session_id →
ConversationAnswerService.answer(session_id, question) →
GroundedAnswerResult → детерминированный текст с ответом и источниками → Telegram-ответ
```

**Команды и обычные сообщения:**

- `/start` — короткое приветствие: бот отвечает по индексированной документации, помнит
  до последних 10 сообщений (`user`/`assistant`) в рамках текущего диалога, `/reset` очищает
  историю, после перезапуска бота история теряется;
- `/reset` — вызывает `ConversationAnswerService.reset(str(chat_id))` и отвечает
  `История текущего диалога очищена.` (очистка уже пустой сессии тоже завершается успешно);
  после этого alias/continuity context текущего чата больше не доступен ни для generation,
  ни для contextual follow-up retrieval;
- обычный текст трактуется как вопрос по документации: `ConversationAnswerService.answer()`
  вызывается с настройками по умолчанию (без пользовательских `top_k`/namespace/фильтров —
  задать их через чат нельзя ни в каком виде, как и URL для индексации). Бот всегда использует
  `PINECONE_DOCUMENTS_NAMESPACE` из текущей конфигурации, поэтому live-namespace должен
  совпадать с namespace, в который ранее индексировалась документация.

**Изоляция по чатам.** `session_id = str(chat_id)` — разные чаты полностью изолированы друг
от друга; `/reset` очищает только текущий чат. В память попадают только текст вопроса и текст
ответа (через существующий `ConversationAnswerService`/`InMemoryConversationMemory`) — объекты
Telegram-сообщений, username/имя/телефон и источники/чанки никогда не сохраняются.

**Форматирование ответа** (plain text, без Markdown/HTML-экранирования):

```
<ответ>

Источники:
1. <title> — <url>
2. ...
```

Без источников — `Источники: не найдены`. Источники строятся только из
`GroundedAnswerResult.sources` (те же метаданные, что и в `ask_docs.py`), а не парсятся из
текста ответа модели.

**Разбиение длинных сообщений.** `split_telegram_message()` — небольшой детерминированный
helper: делит текст на части по ≤4000 символов, предпочитая границу по `\n`, сохраняет порядок
и не теряет ни одного символа (склейка частей равна исходному тексту), каждая часть непустая.

**Ошибки.** `AnswerRetrievalError`/`AnswerGenerationError` (и любой другой `AnswerServiceError`)
приводят к единому сообщению `Не удалось подготовить ответ. Попробуйте повторить запрос
позже.` — без traceback'а, деталей исключения, ключей или namespace. Сбой самой отправки в
Telegram (сетевая ошибка и т.п.) перехватывается на границе обработчика и не приводит к
падению бота.

**Operational logging.** Live entry point пишет один безопасный startup `INFO`-лог с
`PINECONE_INDEX_NAME`, `PINECONE_DOCUMENTS_NAMESPACE`, embedding model, `RETRIEVAL_TOP_K` и
score threshold (`0.25`), а для обычных вопросов — privacy-safe request/retrieval diagnostics:
короткий стабильный hash от `chat_id`, длину вопроса, raw/accepted candidate counts, top scores,
результат (`grounded` или `no_context`) и elapsed time. Текст вопроса, raw `chat_id`,
токен Telegram, API keys, векторы и полные chunk bodies в эти логи не попадают.

**Остановка.** Нормальное завершение polling (включая `Ctrl+C`) логируется как один краткий
`INFO`-лог `Telegram bot stopped`; genuine startup/runtime failures по-прежнему завершаются
с error exit code и безопасным сообщением в консоль.

Запуск бота (live, требует реальных `TELEGRAM_BOT_TOKEN`/OpenAI/Pinecone ключей; блокирует
процесс long polling'ом до `Ctrl+C`):

```
python scripts/run_telegram_bot.py
```

### Typed configuration

`src/ai_docs_agent/config.py` определяет `AppSettings` (pydantic-settings): загружает
переменные окружения (и опционально `.env`), не выполняет сетевых обращений при импорте,
хранит API-ключи как `SecretStr` (не раскрываются через `repr`/`str`), игнорирует неизвестные
переменные и валидирует значения (dimension > 0, timeout > 0, poll interval > 0 и не больше
timeout, непустые имена индекса/модели, метрика пока только `cosine`). Используйте
`get_settings()` — кэшированную фабрику настроек.

### Pinecone integration smoke-test

`src/ai_docs_agent/pinecone_store.py` определяет `PineconeStore`:

- `ensure_index()` — проверяет существование индекса, при необходимости создаёт
  serverless-индекс (см. `PINECONE_CREATE_IF_MISSING` ниже) и проверяет, что dimension и
  metric существующего индекса совпадают с конфигурацией;
- `smoke_test()` — создаёт embedding для фиксированного тестового текста, upsert’ит один
  вектор в отдельный namespace, ждёт (bounded polling) появления результата в query,
  проверяет совпадение ID и в `finally` всегда пытается удалить тестовый вектор.

#### `PINECONE_CREATE_IF_MISSING`

Если `false` (по умолчанию) и индекс с именем `PINECONE_INDEX_NAME` не существует —
`ensure_index()` выбрасывает `PineconeIndexNotFoundError` (без раскрытия ключей). Если
`true` — индекс будет создан как serverless-индекс с настроенными dimension/metric/cloud/
region. Существующий индекс никогда не пересоздаётся автоматически.

Пустой `OPENAI_BASE_URL` (или строка из пробелов) нормализуется в `None` на границе
конфигурации — это означает использование стандартного OpenAI endpoint, а не пустой URL.

`TELEGRAM_BOT_TOKEN` обязателен и не должен быть пустым для реального запуска бота
(`scripts/run_telegram_bot.py`); хранится как `SecretStr` и никогда не появляется в
error-сообщениях, логах, `repr()` или примерах в этом README.

### Unit tests vs. live smoke-test

- `pytest` (`tests/test_config.py`, `tests/test_models.py`, `tests/test_pinecone_store.py`,
  `tests/test_pinecone_smoke_script.py`, `tests/test_url_ingestion.py`,
  `tests/test_url_preview_script.py`, `tests/test_indexing.py`,
  `tests/test_index_url_script.py`, `tests/test_retrieval.py`,
  `tests/test_search_query_script.py`, `tests/test_agent.py`,
  `tests/test_ask_docs_script.py`, `tests/test_memory.py`, `tests/test_telegram_bot.py`,
  `tests/test_run_telegram_bot_script.py`) — быстрые unit-тесты на fake-объектах, без сети,
  без реального DNS, без реальных ключей, без реальных задержек, без long polling'а.
  HTTP fake'ается через `httpx.MockTransport`, DNS — через injectable resolver, Pinecone/
  OpenAI — через dependency-injection-friendly fakes (`DocumentIndexingService`,
  `RetrievalService`, `DocumentationAnswerService` и `PineconeStore` принимают
  fake-сервисы/клиенты и injectable clock/sleep; `ConversationAnswerService` принимает
  fake `DocumentationAnswerService`). `InMemoryConversationMemory` не требует fake'ов —
  это обычный in-process dict без внешних зависимостей. `TelegramBotService` тестируется
  через fake-объекты `Update`/`Message`/`Chat`/`Bot` (без реального Telegram API).
- `scripts/pinecone_smoke_test.py` — **live**-скрипт, выполняющий реальные вызовы OpenAI и
  Pinecone. Требует настоящих `OPENAI_API_KEY` и `PINECONE_API_KEY` и создаёт/удаляет один
  реальный вектор в Pinecone. Не запускается автоматически и не входит в тестовый набор.
  Считается полностью успешным (exit code `0`, `Pinecone smoke test OK`) только если
  удаление тестового вектора тоже прошло успешно; если pipeline прошёл, но cleanup не
  удался, скрипт печатает `Pinecone smoke test FAILED: cleanup did not complete` и
  возвращает exit code `2` (домен/выполнение — `1`).
- `scripts/url_preview.py` — выполняет реальный HTTP-запрос к переданному URL (но не
  OpenAI/Pinecone) — тоже не входит в автоматический тестовый набор; тестируется только его
  чистая функция форматирования и `main()` с внедрённым fake-сервисом.
- `scripts/index_url.py` — **live**-скрипт, выполняющий реальный HTTP-запрос к URL и реальные
  вызовы OpenAI/Pinecone (embeddings + upsert + verification + cleanup). Требует настоящих
  ключей, не входит в автоматический тестовый набор; тестируется только чистая функция
  форматирования (`format_index_report`) и `main()` с внедрённым fake-сервисом. Exit codes:
  `0` — indexing + verification + запрошенный cleanup успешны; `2` — indexing и verification
  успешны, но cleanup не удался; `1` — домен/выполнение (включая ошибки URL ingestion).
- `scripts/search_query.py` — **live**-скрипт, выполняющий реальные вызовы OpenAI (query
  embedding) и Pinecone (query). Строго read-only: никогда не выполняет upsert/delete/
  переиндексацию. Требует настоящих ключей, не входит в автоматический тестовый набор;
  тестируется только чистая функция форматирования (`format_search_report`) и `main()` с
  внедрённым fake-сервисом. Exit codes: `0` — поиск успешен (в т.ч. с пустым результатом);
  `1` — домен/выполнение.
- `scripts/ask_docs.py` — **live**-скрипт, выполняющий реальные вызовы OpenAI (query
  embedding + chat completion) и Pinecone (query). Строго read-only: никогда не выполняет
  upsert/delete/переиндексацию. Требует настоящих ключей, не входит в автоматический
  тестовый набор; тестируется только чистая функция форматирования (`format_answer_report`)
  и `main()` с внедрённым fake-сервисом. Exit code `0` — ответ получен успешно (в т.ч.
  fallback без контекста); `1` — домен/выполнение.
- `scripts/ask_agent.py` — **live**-скрипт нового LangChain tool-calling layer: выполняет
  реальные read-only вызовы OpenAI/Pinecone и, когда agent выбирает PyPI tool, один реальный
  read-only GET к PyPI JSON API. Не выполняет index/delete/mutate операций. Тестируется через
  `main(service=fake_service)` и pure formatting function `format_agent_report`; unit-тесты
  используют только fake tool-calling model и fake сервисы. Exit code `0` — success или safe
  fallback; `1` — startup/unhandled orchestration failure.
- `scripts/pypi_lookup.py` — **live**-скрипт, выполняющий один реальный read-only GET к
  PyPI JSON API (`/pypi/{package_name}/json`). Не требует OpenAI/Pinecone/Telegram ключей и не
  изменяет внешнее состояние. Тестируется только через `main(service=fake_service)` и pure
  formatting function `format_pypi_report`; unit-тесты используют только mocks/fakes. Exit code
  `0` — lookup успешен; `1` — invalid input, package not found, network/upstream или malformed
  response.
- `scripts/run_telegram_bot.py` — **live**-скрипт: строит Telegram `Application`
  (`build_application()`) и запускает long polling (`Application.run_polling()`), блокируя
  процесс до остановки. Требует настоящих `TELEGRAM_BOT_TOKEN`/OpenAI/Pinecone ключей, не
  входит в автоматический тестовый набор; тестируется через внедрённую fake-фабрику
  приложения (`main(application_factory=...)`), без реального Telegram/OpenAI/Pinecone.
  На старте пишет безопасный runtime summary в standard-library logging, а при штатной
  остановке (включая `Ctrl+C`) — один `INFO`-лог `Telegram bot stopped`. Exit code `0` —
  polling запущен и штатно завершился; `1` — ошибка конфигурации или запуска (сообщение не
  содержит токен/секреты).

## Требования

- Python >= 3.11

## Создание и активация виртуального окружения (Windows)

```
python -m venv .venv
.venv\Scripts\activate
```

## Установка проекта (editable, с dev-зависимостями)

```
pip install -e ".[dev]"
```

## Переменные окружения

Скопируйте `.env.example` в `.env` и заполните реальные значения (API-ключи и Telegram
token) локально:

```
copy .env.example .env
```

**`.env` не должен попадать в Git** — файл уже входит в `.gitignore`; не коммитьте его и не
вставляйте реальные ключи в issues, PR или документацию.

## Тесты (unit, без сети)

```
pytest
```

## Линтинг (Ruff)

```
ruff check src tests scripts
```

## Version smoke test

```
python scripts/smoke_test.py
```

## Pinecone integration smoke test (live, требует реальных ключей)

```
python scripts/pinecone_smoke_test.py
```

## URL ingestion preview (live HTTP-запрос к переданному URL, без OpenAI/Pinecone)

```
python scripts/url_preview.py "https://example.com/docs"
```

## URL indexing в Pinecone (live, требует реальных OpenAI/Pinecone ключей)

```
python scripts/index_url.py "https://example.com/docs"
python scripts/index_url.py "https://example.com/docs" --namespace documentation
```

## Semantic search / retrieval (live, read-only, требует реальных OpenAI/Pinecone ключей)

```
python scripts/search_query.py "how do I configure the client?"
python scripts/search_query.py "how do I configure the client?" --top-k 3 --namespace documentation
```

## Grounded RAG answering (live, read-only, требует реальных OpenAI/Pinecone ключей)

```
python scripts/ask_docs.py "how do I configure the client?"
python scripts/ask_docs.py "how do I configure the client?" --top-k 3 --namespace documentation
```

## PyPI package lookup (live, read-only, без OpenAI/Pinecone)

```
python scripts/pypi_lookup.py httpx
```

## LangChain tool-calling agent CLI (live, read-only)

```
python scripts/ask_agent.py "Какая последняя версия пакета httpx на PyPI?"
python scripts/ask_agent.py "Что такое embeddings в OpenAI API?"
```

## Telegram bot (live, требует реальных TELEGRAM_BOT_TOKEN/OpenAI/Pinecone ключей)

Запускает long polling и блокирует процесс до `Ctrl+C`:

```
python scripts/run_telegram_bot.py
```

На старте в логи попадает безопасное summary текущего retrieval/runtime-конфига; при
`Ctrl+C` бот завершает polling, выполняет обычный cleanup и пишет один `INFO`-лог
`Telegram bot stopped`.

## Planned improvements / Следующая версия

Текущие ограничения этого этапа (Stage 4G — LangChain tool-calling agent):

- разговорная память (`InMemoryConversationMemory`) — **process-local и не персистентная**:
  хранится только в памяти процесса (последние 10 сообщений на `session_id`, включая
  сообщения через Telegram) и полностью теряется при перезапуске процесса/бота;
  персистентное хранилище (например, SQLite) — запланированное улучшение v2;
- новый `LangChainToolCallingAgent` пока не интегрирован в Telegram-бот: Telegram на этом
  этапе продолжает использовать существующий `ConversationAnswerService`; интеграция агента —
  следующий stage;
- vector memory пока не реализована;
- Telegram bot — MVP без admin-доступа, без управления документами/индексацией через чат
  (URL indexing через Telegram не реализовано и не планируется в этом виде), без streaming
  ответов, без reranking, без claim-level верификации;
- нет reranking результатов retrieval — используется порядок, возвращённый Pinecone, как есть;
  answer layer поверх raw matches применяет только минимальный relevance gate `score >= 0.25`;
- источники (`AnswerSource`) — document-level ссылки (страница + чанк), а не claim-level
  citations (не привязаны к конкретному предложению или факту в ответе модели);
- нет детерминированной claim-level верификации фактов и structured citations, нет
  дополнительного verification pass (второго LLM-вызова/critic model) — grounding
  обеспечивается только промптом и retrieved-контекстом, поэтому модель иногда может
  добавить правдоподобную, но явно не подтверждённую контекстом деталь.

## Безопасность

Файл `.env` и любые секреты/API-ключи не должны попадать в Git. Используйте `.env.example`
как шаблон и создавайте локальный `.env` только у себя, вне репозитория.
