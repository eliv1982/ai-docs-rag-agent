"""Feature-scoped configuration and redacted configuration errors.

Every test isolates itself from the developer's shell environment and from any
real .env file: all setting variables are removed and the working directory is
an empty temporary directory. No network access.
"""

import traceback
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_docs_agent.config import (
    AgentSettings,
    AnswerSettings,
    AppSettings,
    ConfigurationError,
    IndexingSettings,
    PyPISettings,
    UrlFetchSettings,
    UserMemorySettings,
    VectorStoreSettings,
    get_settings,
    load_settings,
)

_ENV_EXAMPLE = Path(__file__).resolve().parents[1] / ".env.example"

_SECRET_ENV = {
    "OPENAI_API_KEY": "sk-test-openai",
    "PINECONE_API_KEY": "pc-test-key",
    "OPENAI_CHAT_MODEL": "gpt-4o-mini",
    "TELEGRAM_BOT_TOKEN": "test-telegram-token",
    "USER_MEMORY_HASH_SECRET": "unit-test-user-memory-secret",
}


def _all_env_var_names() -> set[str]:
    names: set[str] = set()
    for field in AppSettings.model_fields.values():
        alias = field.validation_alias
        assert isinstance(alias, str)
        names.add(alias)
    return names


@pytest.fixture
def clean_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[pytest.MonkeyPatch]:
    """No setting variables in the environment and no .env file in the cwd."""
    for name in _all_env_var_names():
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield monkeypatch
    get_settings.cache_clear()


def _missing_names(exc_info: pytest.ExceptionInfo[ConfigurationError]) -> set[str]:
    return {name for name, _reason in exc_info.value.problems}


# --- keyless commands -----------------------------------------------------------


@pytest.mark.parametrize("settings_cls", [UrlFetchSettings, PyPISettings])
def test_keyless_groups_load_with_no_secrets_at_all(
    clean_env: pytest.MonkeyPatch, settings_cls: type
) -> None:
    settings = load_settings(settings_cls)

    assert not any(
        name.endswith(("_api_key", "_token", "_secret")) for name in type(settings).model_fields
    )


def test_keyless_groups_ignore_invalid_unrelated_settings(
    clean_env: pytest.MonkeyPatch,
) -> None:
    clean_env.setenv("PINECONE_DIMENSION", "not-a-number")
    clean_env.setenv("RETRIEVAL_TOP_K", "999")

    assert load_settings(UrlFetchSettings).chunk_size == 1200
    assert load_settings(PyPISettings).pypi_base_url == "https://pypi.org"


def test_keyless_groups_still_validate_their_own_settings(
    clean_env: pytest.MonkeyPatch,
) -> None:
    clean_env.setenv("CHUNK_OVERLAP", "5000")
    clean_env.setenv("PYPI_TIMEOUT_SECONDS", "0")

    with pytest.raises(ConfigurationError) as url_error:
        load_settings(UrlFetchSettings)
    with pytest.raises(ConfigurationError) as pypi_error:
        load_settings(PyPISettings)

    assert "chunk_overlap" in str(url_error.value)
    assert _missing_names(pypi_error) == {"PYPI_TIMEOUT_SECONDS"}


# --- partially keyed commands -----------------------------------------------------


@pytest.mark.parametrize(
    "settings_cls", [VectorStoreSettings, IndexingSettings]
)
def test_vector_store_commands_need_only_openai_and_pinecone_keys(
    clean_env: pytest.MonkeyPatch, settings_cls: type
) -> None:
    clean_env.setenv("OPENAI_API_KEY", _SECRET_ENV["OPENAI_API_KEY"])
    clean_env.setenv("PINECONE_API_KEY", _SECRET_ENV["PINECONE_API_KEY"])

    settings = load_settings(settings_cls)

    assert not hasattr(settings, "telegram_bot_token")
    assert not hasattr(settings, "user_memory_hash_secret")
    assert not hasattr(settings, "openai_chat_model")


def test_vector_store_commands_report_exactly_the_missing_keys(
    clean_env: pytest.MonkeyPatch,
) -> None:
    clean_env.setenv("TELEGRAM_BOT_TOKEN", "unrelated-but-present")

    with pytest.raises(ConfigurationError) as exc_info:
        load_settings(VectorStoreSettings)

    assert _missing_names(exc_info) == {"OPENAI_API_KEY", "PINECONE_API_KEY"}


def test_answer_group_adds_only_the_chat_model(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("OPENAI_API_KEY", _SECRET_ENV["OPENAI_API_KEY"])
    clean_env.setenv("PINECONE_API_KEY", _SECRET_ENV["PINECONE_API_KEY"])

    with pytest.raises(ConfigurationError) as exc_info:
        load_settings(AnswerSettings)
    assert _missing_names(exc_info) == {"OPENAI_CHAT_MODEL"}

    clean_env.setenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    answer_settings = load_settings(AnswerSettings)
    agent_settings = load_settings(AgentSettings)
    assert answer_settings.openai_chat_model == "gpt-4o-mini"
    assert agent_settings.pypi_base_url == "https://pypi.org"
    assert not hasattr(agent_settings, "telegram_bot_token")
    assert not hasattr(agent_settings, "user_memory_hash_secret")


def test_user_memory_group_needs_the_memory_secret_but_not_telegram_or_chat_model(
    clean_env: pytest.MonkeyPatch,
) -> None:
    clean_env.setenv("OPENAI_API_KEY", _SECRET_ENV["OPENAI_API_KEY"])
    clean_env.setenv("PINECONE_API_KEY", _SECRET_ENV["PINECONE_API_KEY"])

    with pytest.raises(ConfigurationError) as exc_info:
        load_settings(UserMemorySettings)
    assert _missing_names(exc_info) == {"USER_MEMORY_HASH_SECRET"}

    clean_env.setenv("USER_MEMORY_HASH_SECRET", _SECRET_ENV["USER_MEMORY_HASH_SECRET"])
    settings = load_settings(UserMemorySettings)
    assert not hasattr(settings, "telegram_bot_token")
    assert not hasattr(settings, "openai_chat_model")


def test_memory_prefix_must_still_differ_from_documents_namespace(
    clean_env: pytest.MonkeyPatch,
) -> None:
    clean_env.setenv("OPENAI_API_KEY", _SECRET_ENV["OPENAI_API_KEY"])
    clean_env.setenv("PINECONE_API_KEY", _SECRET_ENV["PINECONE_API_KEY"])
    clean_env.setenv("USER_MEMORY_HASH_SECRET", _SECRET_ENV["USER_MEMORY_HASH_SECRET"])
    clean_env.setenv("USER_MEMORY_NAMESPACE_PREFIX", "documentation")

    with pytest.raises(ConfigurationError) as exc_info:
        load_settings(UserMemorySettings)

    assert "must differ from pinecone_documents_namespace" in str(exc_info.value)


def test_telegram_runtime_still_requires_its_complete_configuration(
    clean_env: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ConfigurationError) as exc_info:
        get_settings()

    assert _missing_names(exc_info) == {
        "OPENAI_API_KEY",
        "PINECONE_API_KEY",
        "OPENAI_CHAT_MODEL",
        "TELEGRAM_BOT_TOKEN",
        "USER_MEMORY_HASH_SECRET",
    }


@pytest.mark.parametrize("name", ["OPENAI_API_KEY", "PINECONE_API_KEY"])
def test_blank_required_api_key_is_a_configuration_error(
    clean_env: pytest.MonkeyPatch, name: str
) -> None:
    clean_env.setenv("OPENAI_API_KEY", _SECRET_ENV["OPENAI_API_KEY"])
    clean_env.setenv("PINECONE_API_KEY", _SECRET_ENV["PINECONE_API_KEY"])
    clean_env.setenv(name, "   ")

    with pytest.raises(ConfigurationError) as exc_info:
        load_settings(VectorStoreSettings)

    assert _missing_names(exc_info) == {name}


def test_openai_timeout_defaults_and_validation(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("OPENAI_API_KEY", _SECRET_ENV["OPENAI_API_KEY"])
    clean_env.setenv("PINECONE_API_KEY", _SECRET_ENV["PINECONE_API_KEY"])

    assert load_settings(VectorStoreSettings).openai_timeout_seconds == 60

    clean_env.setenv("OPENAI_TIMEOUT_SECONDS", "12.5")
    assert load_settings(VectorStoreSettings).openai_timeout_seconds == 12.5

    clean_env.setenv("OPENAI_TIMEOUT_SECONDS", "0")
    with pytest.raises(ConfigurationError):
        load_settings(VectorStoreSettings)


# --- .env.example placeholders ------------------------------------------------------


def test_env_example_placeholders_do_not_break_keyless_commands(
    clean_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / ".env").write_text(_ENV_EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")

    assert load_settings(UrlFetchSettings).chunk_size == 1200
    assert load_settings(PyPISettings).pypi_timeout_seconds == 10


def test_env_example_blank_secrets_fail_only_commands_that_need_them(
    clean_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / ".env").write_text(_ENV_EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(ConfigurationError) as vector_error:
        load_settings(VectorStoreSettings)
    with pytest.raises(ConfigurationError) as app_error:
        load_settings(AppSettings)

    assert _missing_names(vector_error) == {"OPENAI_API_KEY", "PINECONE_API_KEY"}
    assert _missing_names(app_error) == {
        "OPENAI_API_KEY",
        "PINECONE_API_KEY",
        "TELEGRAM_BOT_TOKEN",
        "USER_MEMORY_HASH_SECRET",
    }


def test_env_example_non_secret_values_are_valid_once_secrets_are_filled(
    clean_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / ".env").write_text(_ENV_EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    for name, value in _SECRET_ENV.items():
        clean_env.setenv(name, value)

    settings = load_settings(AppSettings)

    assert settings.openai_timeout_seconds == 60
    assert settings.openai_base_url is None


# --- redacted configuration errors ----------------------------------------------------

# Random-looking tokens that can never occur in a human-written validation message.
_FAKE_VALUES = {
    "OPENAI_API_KEY": "sk-zq7Kx9PLm2WvTa",
    "PINECONE_API_KEY": "pcsk-Bv4Ntr8QeY1uJh",
    "OPENAI_CHAT_MODEL": "gpt-4o-mini",
    "TELEGRAM_BOT_TOKEN": "7391845026:AAHjw3Zr0pLxQv9",
    "USER_MEMORY_HASH_SECRET": "pepper-Xc5Fm1Rd8Gk3Ls",
    # Deliberately invalid values that look like secrets:
    "PINECONE_DIMENSION": "sk-Hn6Wq2Dz9VbPe",
    "PINECONE_METRIC": "metric-Yt3Jc7Ub1Mo5Ia",
    "OPENAI_TIMEOUT_SECONDS": "timeout-Rk8Sg4Nx2Lf6Ow",
    "USER_MEMORY_NAMESPACE_PREFIX": "bad prefix Qw9Ez5Tp3Hv7Cn",
    "URL_MAX_REDIRECTS": "redir-Fb2Ud6Ka8Zy4Xs",
    "RETRIEVAL_TOP_K": "topk-Mj1Ge5Ri9Lt3Pv",
}


def _fragments(value: str, size: int = 6) -> list[str]:
    token = value.rsplit(" ", 1)[-1].rsplit("-", 1)[-1]
    return [token[i : i + size] for i in range(len(token) - size + 1)]


def _assert_no_value_fragments(text: str) -> None:
    for name, value in _FAKE_VALUES.items():
        assert value not in text, name
        for fragment in _fragments(value):
            assert fragment not in text, (name, fragment)
    assert "input_value" not in text
    assert "input_type" not in text


def test_configuration_errors_name_variables_but_never_echo_values(
    clean_env: pytest.MonkeyPatch,
) -> None:
    for name, value in _FAKE_VALUES.items():
        clean_env.setenv(name, value)

    with pytest.raises(ConfigurationError) as exc_info:
        load_settings(AppSettings)

    rendered = str(exc_info.value)
    # Every deliberately invalid variable is named...
    for name in (
        "PINECONE_DIMENSION",
        "PINECONE_METRIC",
        "OPENAI_TIMEOUT_SECONDS",
        "USER_MEMORY_NAMESPACE_PREFIX",
        "URL_MAX_REDIRECTS",
        "RETRIEVAL_TOP_K",
    ):
        assert name in rendered
    # ...but no supplied value (or fragment of one) is.
    _assert_no_value_fragments(rendered)
    assert "Set the variables above" in rendered


def test_blank_secret_error_is_safe_and_actionable(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("OPENAI_API_KEY", "   ")
    clean_env.setenv("PINECONE_API_KEY", _FAKE_VALUES["PINECONE_API_KEY"])

    with pytest.raises(ConfigurationError) as exc_info:
        load_settings(VectorStoreSettings)

    rendered = str(exc_info.value)
    assert "OPENAI_API_KEY: must not be empty." in rendered
    _assert_no_value_fragments(rendered)


def test_embedded_url_credentials_are_never_echoed(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("PYPI_BASE_URL", "https://user:pw-Kd3Lq8Zx1Wm6@pypi.org")

    with pytest.raises(ConfigurationError) as exc_info:
        load_settings(PyPISettings)

    rendered = str(exc_info.value)
    assert "PYPI_BASE_URL" in rendered
    assert "pw-Kd3Lq8Zx1Wm6" not in rendered
    assert "Kd3Lq8" not in rendered


def test_traceback_of_a_configuration_error_does_not_chain_the_raw_validation_error(
    clean_env: pytest.MonkeyPatch,
) -> None:
    clean_env.setenv("PINECONE_DIMENSION", _FAKE_VALUES["PINECONE_DIMENSION"])
    clean_env.setenv("OPENAI_API_KEY", _FAKE_VALUES["OPENAI_API_KEY"])
    clean_env.setenv("PINECONE_API_KEY", _FAKE_VALUES["PINECONE_API_KEY"])

    with pytest.raises(ConfigurationError) as exc_info:
        load_settings(VectorStoreSettings)

    printed = "".join(traceback.format_exception(exc_info.value))
    assert exc_info.value.__suppress_context__ is True
    assert "ValidationError" not in printed
    _assert_no_value_fragments(printed)


def test_from_validation_error_redacts_a_raw_validation_error() -> None:
    with pytest.raises(ValidationError) as exc_info:
        VectorStoreSettings(
            _env_file=None,
            openai_api_key=_FAKE_VALUES["OPENAI_API_KEY"],
            pinecone_api_key=_FAKE_VALUES["PINECONE_API_KEY"],
            pinecone_dimension=_FAKE_VALUES["PINECONE_DIMENSION"],
        )
    # The raw error really does contain the supplied value...
    assert _FAKE_VALUES["PINECONE_DIMENSION"] in str(exc_info.value)

    redacted = str(ConfigurationError.from_validation_error(exc_info.value))

    # ...and the redacted form does not (keyword construction reports the field
    # name; env loading reports the variable name).
    assert "pinecone_dimension" in redacted.lower()
    _assert_no_value_fragments(redacted)
