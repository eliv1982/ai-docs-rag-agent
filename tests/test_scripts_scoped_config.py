"""CLI configuration scoping and redacted configuration failures.

Each script must (a) load only the settings group its feature uses and (b) turn a
configuration failure into a concise message that never echoes a configured
value. The environment is scrubbed and the working directory is an empty
temporary directory so a developer's real .env is never read. Services are
replaced by spies, so no network, DNS, OpenAI, Pinecone, Telegram or PyPI call
ever happens; the two keyless CLIs are additionally run for real in a
subprocess with an empty environment and offline-failing arguments.
"""

import contextlib
import importlib.util
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from pydantic import ValidationError

from ai_docs_agent.config import (
    AgentSettings,
    AnswerSettings,
    AppSettings,
    IndexingSettings,
    PyPISettings,
    UrlFetchSettings,
    UserMemorySettings,
    VectorStoreSettings,
    get_settings,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO_ROOT / "scripts"
_KEYS = {"OPENAI_API_KEY": "sk-test-openai", "PINECONE_API_KEY": "pc-test-key"}
_CHAT = {**_KEYS, "OPENAI_CHAT_MODEL": "gpt-4o-mini"}
_MEMORY = {**_KEYS, "USER_MEMORY_HASH_SECRET": "unit-test-user-memory-secret"}
_URL = "https://docs.example.com/p"

# (script, service attribute patched in the script, argv, settings group, env it needs)
_SCRIPT_CASES: list[tuple[str, str, list[str], type, dict[str, str]]] = [
    ("url_preview", "UrlIngestionService", [_URL], UrlFetchSettings, {}),
    ("pypi_lookup", "PyPILookupService", ["httpx"], PyPISettings, {}),
    ("pinecone_smoke_test", "PineconeStore", [], VectorStoreSettings, _KEYS),
    ("search_query", "RetrievalService", ["query"], VectorStoreSettings, _KEYS),
    ("index_url", "DocumentIndexingService", [_URL], IndexingSettings, _KEYS),
    ("ask_docs", "DocumentationAnswerService", ["question"], AnswerSettings, _CHAT),
    ("ask_agent", "LangChainToolCallingAgent", ["question"], AgentSettings, _CHAT),
    ("user_memory", "UserMemoryService", ["remember", "user", "text"], UserMemorySettings, _MEMORY),
]
_SCRIPT_IDS = [case[0] for case in _SCRIPT_CASES]


def _load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"{name}_scoped_script", _SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _setting_env_var_names() -> set[str]:
    return {
        str(field.validation_alias) for field in AppSettings.model_fields.values()
    }


@pytest.fixture
def clean_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[pytest.MonkeyPatch]:
    for name in _setting_env_var_names():
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield monkeypatch
    get_settings.cache_clear()


class _SpyService:
    """Stands in for a real service class; records how the script constructed it."""

    instances: list["_SpyService"]

    def __init__(self, settings: Any, *args: Any, **kwargs: Any) -> None:
        self.settings = settings
        type(self).instances.append(self)

    def __getattr__(self, name: str) -> Any:
        def fail(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("spy service stops the run before any real work")

        return fail


def _install_spy(monkeypatch: pytest.MonkeyPatch, module: ModuleType, attr: str) -> type:
    spy = type("Spy", (_SpyService,), {"instances": []})
    monkeypatch.setattr(module, attr, spy)
    return spy


def _run_main(module: ModuleType, argv: list[str]) -> int:
    # pinecone_smoke_test.main() takes no arguments; every other script takes argv.
    return module.main() if module.__name__.startswith("pinecone_smoke") else module.main(argv)


# --- scoping: each script loads only its own group ---------------------------------


@pytest.mark.parametrize(("script", "attr", "argv", "group", "env"), _SCRIPT_CASES, ids=_SCRIPT_IDS)
def test_script_loads_only_its_own_settings_group(
    clean_env: pytest.MonkeyPatch,
    script: str,
    attr: str,
    argv: list[str],
    group: type,
    env: dict[str, str],
) -> None:
    for name, value in env.items():
        clean_env.setenv(name, value)
    module = _load_script(script)
    spy = _install_spy(clean_env, module, attr)

    with contextlib.suppress(Exception):
        _run_main(module, argv)

    assert len(spy.instances) == 1
    settings = spy.instances[0].settings
    assert type(settings) is group
    # No script outside the Telegram runtime may need the Telegram token.
    assert not hasattr(settings, "telegram_bot_token")


@pytest.mark.parametrize("script", ["url_preview", "pypi_lookup"])
def test_keyless_scripts_run_in_a_subprocess_with_an_empty_environment(
    tmp_path: Path, script: str
) -> None:
    """Real invocation: no secrets, no .env; fails on the (offline) input, not on config."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key.upper() not in _setting_env_var_names()
    }
    env["PYTHONPATH"] = str(_REPO_ROOT / "src")
    argument = "ftp://example.com/page" if script == "url_preview" else "not a package!"

    completed = subprocess.run(
        [sys.executable, str(_SCRIPTS / f"{script}.py"), argument],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    output = completed.stdout + completed.stderr
    assert "Invalid configuration" not in output
    assert "Traceback" not in output
    assert completed.returncode == 1
    # The domain-level rejection proves the script got past configuration.
    assert "FAILED" in output


# --- redaction: configuration failures never echo values ------------------------------

_FAKE_TOKEN = "Zr3Kq8Wm1Xv5Tb"


def _assert_no_leak(text: str) -> None:
    assert _FAKE_TOKEN not in text
    for i in range(len(_FAKE_TOKEN) - 5):
        assert _FAKE_TOKEN[i : i + 6] not in text
    assert "input_value" not in text
    assert "Traceback" not in text


@pytest.mark.parametrize(("script", "attr", "argv", "group", "env"), _SCRIPT_CASES, ids=_SCRIPT_IDS)
def test_script_reports_missing_configuration_without_building_the_service(
    clean_env: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    script: str,
    attr: str,
    argv: list[str],
    group: type,
    env: dict[str, str],
) -> None:
    for name, value in env.items():
        clean_env.setenv(name, value)
    # Make the script's own group invalid via a secret-looking value, so any echo of
    # supplied values would show up in the output.
    own_invalid = {
        "url_preview": ("CHUNK_SIZE", f"size-{_FAKE_TOKEN}"),
        "pypi_lookup": ("PYPI_TIMEOUT_SECONDS", f"t-{_FAKE_TOKEN}"),
    }.get(script, ("PINECONE_DIMENSION", f"sk-{_FAKE_TOKEN}"))
    clean_env.setenv(*own_invalid)
    module = _load_script(script)
    spy = _install_spy(clean_env, module, attr)

    exit_code = _run_main(module, argv)

    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert exit_code == 1
    assert spy.instances == []
    assert "FAILED" in output
    assert "Invalid configuration" in output
    _assert_no_leak(output)


def test_run_telegram_bot_prints_redacted_text_for_a_raw_validation_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script("run_telegram_bot")

    def failing_factory() -> None:
        AppSettings(
            _env_file=None,
            openai_api_key=f"sk-{_FAKE_TOKEN}",
            pinecone_api_key="pc-test-key",
            openai_chat_model="gpt-4o-mini",
            telegram_bot_token="test-telegram-token",
            user_memory_hash_secret="unit-test-user-memory-secret",
            pinecone_dimension=f"dim-{_FAKE_TOKEN}",
        )

    # Sanity check: the raw error really would have leaked the value.
    with pytest.raises(ValidationError) as raw:
        failing_factory()
    assert _FAKE_TOKEN in str(raw.value)

    exit_code = module.main(application_factory=failing_factory)

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "Telegram bot FAILED to start" in output
    assert "pinecone_dimension" in output.lower()
    _assert_no_leak(output)


def test_run_telegram_bot_prints_redacted_text_for_missing_runtime_configuration(
    clean_env: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    clean_env.setenv("OPENAI_API_KEY", f"sk-{_FAKE_TOKEN}")
    clean_env.setenv("PINECONE_DIMENSION", f"dim-{_FAKE_TOKEN}")
    module = _load_script("run_telegram_bot")

    exit_code = module.main()

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "TELEGRAM_BOT_TOKEN" in output
    assert "PINECONE_DIMENSION" in output
    _assert_no_leak(output)
