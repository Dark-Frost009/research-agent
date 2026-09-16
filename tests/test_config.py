"""Tests for Research Agent application settings.

These tests verify our configuration contract:
- sensible defaults
- environment-variable loading
- secret handling
- research-budget validation
- timeout validation

They deliberately avoid testing pydantic-settings internals.
"""

import pytest
from pydantic import SecretStr, ValidationError

from research_agent.config import Settings


def _settings(**kwargs) -> Settings:
    """Build Settings without reading a developer's local .env file."""
    return Settings(_env_file=None, **kwargs)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_default_settings():
    settings = _settings()

    assert settings.app_env == "development"
    assert settings.log_level == "INFO"

    assert settings.llm_provider is None
    assert settings.llm_model is None
    assert settings.llm_api_key is None

    assert settings.search_provider == "tavily"
    assert settings.tavily_api_key is None

    assert settings.max_research_iterations == 2
    assert settings.max_search_queries == 8
    assert settings.max_search_results_per_query == 5
    assert settings.max_sources == 12

    assert settings.request_timeout_seconds == 15.0


# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------


def test_environment_variables_override_defaults(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_MODEL", "example-model")

    monkeypatch.setenv("SEARCH_PROVIDER", "custom")

    monkeypatch.setenv("MAX_RESEARCH_ITERATIONS", "4")
    monkeypatch.setenv("MAX_SEARCH_QUERIES", "10")
    monkeypatch.setenv("MAX_SEARCH_RESULTS_PER_QUERY", "7")
    monkeypatch.setenv("MAX_SOURCES", "20")

    monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", "30")

    settings = _settings()

    assert settings.app_env == "production"
    assert settings.log_level == "DEBUG"

    assert settings.llm_provider == "groq"
    assert settings.llm_model == "example-model"

    assert settings.search_provider == "custom"

    assert settings.max_research_iterations == 4
    assert settings.max_search_queries == 10
    assert settings.max_search_results_per_query == 7
    assert settings.max_sources == 20

    assert settings.request_timeout_seconds == 30.0


def test_environment_variable_names_are_case_insensitive(monkeypatch):
    monkeypatch.setenv("max_search_queries", "11")

    settings = _settings()

    assert settings.max_search_queries == 11


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------


def test_api_keys_are_stored_as_secret_strings(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "llm-secret-value")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-secret-value")

    settings = _settings()

    assert isinstance(settings.llm_api_key, SecretStr)
    assert isinstance(settings.tavily_api_key, SecretStr)

    assert settings.llm_api_key.get_secret_value() == "llm-secret-value"
    assert settings.tavily_api_key.get_secret_value() == "tavily-secret-value"


def test_secret_values_are_not_exposed_in_settings_repr(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "super-secret-llm-key")
    monkeypatch.setenv("TAVILY_API_KEY", "super-secret-tavily-key")

    settings = _settings()
    rendered = repr(settings)

    assert "super-secret-llm-key" not in rendered
    assert "super-secret-tavily-key" not in rendered


# ---------------------------------------------------------------------------
# Research-budget validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field_name",
    [
        "max_research_iterations",
        "max_search_queries",
        "max_search_results_per_query",
        "max_sources",
    ],
)
def test_research_budget_fields_reject_zero(field_name):
    with pytest.raises(ValidationError):
        _settings(**{field_name: 0})


@pytest.mark.parametrize(
    "field_name",
    [
        "max_research_iterations",
        "max_search_queries",
        "max_search_results_per_query",
        "max_sources",
    ],
)
def test_research_budget_fields_reject_negative_values(field_name):
    with pytest.raises(ValidationError):
        _settings(**{field_name: -1})


@pytest.mark.parametrize(
    "field_name",
    [
        "max_research_iterations",
        "max_search_queries",
        "max_search_results_per_query",
        "max_sources",
    ],
)
def test_research_budget_fields_accept_one(field_name):
    settings = _settings(**{field_name: 1})

    assert getattr(settings, field_name) == 1


# ---------------------------------------------------------------------------
# Network timeout validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [0, -1, -0.1])
def test_request_timeout_rejects_non_positive_values(value):
    with pytest.raises(ValidationError):
        _settings(request_timeout_seconds=value)


@pytest.mark.parametrize("value", [0.1, 1, 30.5])
def test_request_timeout_accepts_positive_values(value):
    settings = _settings(request_timeout_seconds=value)

    assert settings.request_timeout_seconds == value