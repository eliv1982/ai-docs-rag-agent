# AI Docs RAG Agent

Telegram RAG-агент для работы с технической документацией.

**Статус: typed configuration + Pinecone index management and integration smoke-test.**

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

Большая часть функциональности выше пока не реализована: URL ingestion, HTML parsing,
чанкинг, полноценный `PineconeVectorStore`, retrieval, генерация ответов через LLM,
LangChain agent, tools и Telegram-интерфейс отсутствуют.

На данном этапе реализовано:

- типизированная конфигурация приложения (`AppSettings`, `pydantic-settings`);
- управление Pinecone-индексом (`PineconeStore.ensure_index`);
- live-интеграционный smoke-test: OpenAI embedding → Pinecone upsert → query → cleanup
  (`PineconeStore.smoke_test`).

Остальные модули (`url_ingestion.py`, `retrieval.py`, `memory.py`, `tools.py`, `agent.py`,
`telegram_bot.py`) по-прежнему содержат только module docstring.

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

- `pytest` (`tests/test_config.py`, `tests/test_pinecone_store.py`,
  `tests/test_pinecone_smoke_script.py`) — быстрые unit-тесты на fake-объектах, без сети,
  без реальных ключей, без реальных задержек.
- `scripts/pinecone_smoke_test.py` — **live**-скрипт, выполняющий реальные вызовы OpenAI и
  Pinecone. Требует настоящих `OPENAI_API_KEY` и `PINECONE_API_KEY` и создаёт/удаляет один
  реальный вектор в Pinecone. Не запускается автоматически и не входит в тестовый набор.
  Считается полностью успешным (exit code `0`, `Pinecone smoke test OK`) только если
  удаление тестового вектора тоже прошло успешно; если pipeline прошёл, но cleanup не
  удался, скрипт печатает `Pinecone smoke test FAILED: cleanup did not complete` и
  возвращает exit code `2` (домен/выполнение — `1`).

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

## Безопасность

Файл `.env` и любые секреты/API-ключи не должны попадать в Git. Используйте `.env.example`
как шаблон и создавайте локальный `.env` только у себя, вне репозитория.
