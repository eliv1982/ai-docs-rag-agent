# AI Docs RAG Agent

Telegram RAG-агент для работы с технической документацией.

**Статус: typed configuration + Pinecone index management/smoke-test + URL ingestion/chunking
+ URL indexing into Pinecone (fetch → embeddings → upsert → verify → cleanup) + typed
read-only semantic search (`RetrievalService.search`) + minimal grounded RAG answering
(`DocumentationAnswerService.answer`) + short-term in-memory conversation memory
(`ConversationAnswerService`) implemented. Agent/tools и Telegram-интерфейс ещё не
реализованы.**

## Планируемые возможности

- приём URL страниц документации;
- загрузка и очистка HTML;
- разбиение текста на чанки;
- создание embeddings;
- хранение данных в Pinecone;
- семантический поиск;
- генерация grounded-ответов по найденному контексту;
- обращение к PyPI через реальный GET API tool;
- контролируемая пользовательская память;
- интерфейс через Telegram.

## Архитектурный статус

Реализованы: URL ingestion, HTML parsing, чанкинг, создание embeddings и запись чанков в
Pinecone (`DocumentIndexingService.index_url`), типизированный read-only семантический поиск
по уже проиндексированным чанкам (`RetrievalService.search`, см. ниже), минимальный
grounded RAG answering поверх retrieval (`DocumentationAnswerService.answer`, см. ниже), а
также короткая process-local разговорная память по `session_id`
(`ConversationAnswerService`, см. ниже). LangChain agent, tools и Telegram-интерфейс пока не
реализованы.

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
- короткая process-local разговорная память по `session_id` (`ConversationAnswerService`,
  `InMemoryConversationMemory`, см. ниже): последние сообщения диалога подставляются в промпт
  для continuity, но никогда не считаются документальным фактом.

Остальные модули (`tools.py`, `telegram_bot.py`) по-прежнему содержат только module
docstring. **Agent orchestration, tools (включая PyPI GET API tool) и Telegram-интерфейс
ещё не реализованы.**

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
отсортированными от наиболее похожего к наименее похожему). **Порог по score не
применяется** — на этом этапе никакая фильтрация/ранжирование/дедупликация результатов
не производится, порядок и содержимое, полученные от Pinecone, сохраняются как есть.

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
question → RetrievalService.search() → retrieved RetrievedChunk[] →
компактный grounded system+user prompt (текст чанков — untrusted data, не instructions) →
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

**Fallback без контекста.** Если `RetrievalService.search()` не вернул ни одного чанка, chat
model не вызывается вообще — сервис детерминированно возвращает `GroundedAnswerResult` с
`retrieved_chunk_count=0`, пустым `sources` и фиксированным текстом:

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
  "нет данных в базе" — успешный результат и тоже сохраняется в историю.

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

### Unit tests vs. live smoke-test

- `pytest` (`tests/test_config.py`, `tests/test_models.py`, `tests/test_pinecone_store.py`,
  `tests/test_pinecone_smoke_script.py`, `tests/test_url_ingestion.py`,
  `tests/test_url_preview_script.py`, `tests/test_indexing.py`,
  `tests/test_index_url_script.py`, `tests/test_retrieval.py`,
  `tests/test_search_query_script.py`, `tests/test_agent.py`,
  `tests/test_ask_docs_script.py`, `tests/test_memory.py`) — быстрые unit-тесты на
  fake-объектах, без сети, без реального DNS, без реальных ключей, без реальных задержек.
  HTTP fake'ается через `httpx.MockTransport`, DNS — через injectable resolver, Pinecone/
  OpenAI — через dependency-injection-friendly fakes (`DocumentIndexingService`,
  `RetrievalService`, `DocumentationAnswerService` и `PineconeStore` принимают
  fake-сервисы/клиенты и injectable clock/sleep; `ConversationAnswerService` принимает
  fake `DocumentationAnswerService`). `InMemoryConversationMemory` не требует fake'ов —
  это обычный in-process dict без внешних зависимостей.
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

## Planned improvements / Следующая версия

Текущие ограничения этого этапа (Stage 4D — minimal in-memory conversation memory):

- разговорная память (`InMemoryConversationMemory`) — **process-local и не персистентная**:
  хранится только в памяти процесса (последние 10 сообщений на `session_id`) и полностью
  теряется при перезапуске процесса; персистентное хранилище (например, SQLite) —
  запланированное улучшение v2;
- нет Telegram-интерфейса;
- нет score threshold/reranking результатов retrieval — используется порядок, возвращённый
  Pinecone, как есть;
- источники (`AnswerSource`) — document-level ссылки (страница + чанк), а не claim-level
  citations (не привязаны к конкретному предложению или факту в ответе модели);
- нет детерминированной claim-level верификации фактов и structured citations, нет
  дополнительного verification pass (второго LLM-вызова/critic model) — grounding
  обеспечивается только промптом и retrieved-контекстом, поэтому модель иногда может
  добавить правдоподобную, но явно не подтверждённую контекстом деталь.

## Безопасность

Файл `.env` и любые секреты/API-ключи не должны попадать в Git. Используйте `.env.example`
как шаблон и создавайте локальный `.env` только у себя, вне репозитория.
