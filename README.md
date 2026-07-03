# AI Docs RAG Agent

Telegram RAG-агент для работы с технической документацией.

**Статус: typed configuration + Pinecone index management/smoke-test + URL ingestion/chunking
+ URL indexing into Pinecone (fetch → embeddings → upsert → verify → cleanup) + typed
read-only semantic search (`RetrievalService.search`) implemented. RAG generation, agent/tools
и Telegram-интерфейс ещё не реализованы.**

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
Pinecone (`DocumentIndexingService.index_url`), а также типизированный read-only
семантический поиск по уже проиндексированным чанкам (`RetrievalService.search`, см. ниже).
Генерация ответов через LLM, LangChain agent, tools и Telegram-интерфейс пока не реализованы.

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
  `RetrievedChunk` (см. ниже).

Остальные модули (`memory.py`, `tools.py`, `agent.py`, `telegram_bot.py`) по-прежнему
содержат только module docstring. **RAG generation (сборка промпта и генерация ответа через
LLM), пользовательские citations, agent orchestration, tools, память и Telegram-интерфейс
ещё не реализованы** — retrieval лишь находит и типизирует релевантные чанки.

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
  `tests/test_search_query_script.py`) — быстрые unit-тесты на fake-объектах, без сети, без
  реального DNS, без реальных ключей, без реальных задержек. HTTP fake'ается через
  `httpx.MockTransport`, DNS — через injectable resolver, Pinecone/OpenAI — через
  dependency-injection-friendly fakes (`DocumentIndexingService`, `RetrievalService` и
  `PineconeStore` принимают fake-сервисы/клиенты и injectable clock/sleep).
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

## Безопасность

Файл `.env` и любые секреты/API-ключи не должны попадать в Git. Используйте `.env.example`
как шаблон и создавайте локальный `.env` только у себя, вне репозитория.
