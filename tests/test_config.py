"""Tests for Research Agent application settings.

These tests verify our configuration contract:

- sensible defaults

- environment-variable loading

- secret handling

- explicit whole-run research budgets

- backward-compatible legacy budget environment names

- backward-compatible and role-specific LLM model selection

- finalization LLM reserve validation

- timeout validation

- extraction-limit validation

They deliberately avoid testing pydantic-settings internals.

"""

import pytest

from pydantic import (

    SecretStr,

    ValidationError,

)

from research_agent.config import Settings

def _settings(**kwargs) -> Settings:

    """Build Settings without reading a developer's local .env file."""

    return Settings(

        _env_file=None,

        **kwargs,

    )

# ---------------------------------------------------------------------------

# Defaults

# ---------------------------------------------------------------------------

def test_default_settings():

    settings = _settings()

    assert settings.app_env == "development"

    assert settings.log_level == "INFO"

    assert settings.public_demo_only is False

    assert settings.llm_provider is None

    assert settings.llm_model is None

    assert settings.llm_worker_model is None

    assert settings.llm_final_model is None

    assert settings.worker_llm_model is None

    assert settings.final_llm_model is None

    assert settings.llm_api_key is None

    assert settings.search_provider == "tavily"

    assert settings.tavily_api_key is None

    assert settings.max_research_iterations == 2

    assert settings.max_sub_questions == 5

    assert settings.max_search_queries_per_run == 8

    assert (

        settings.max_search_queries_per_iteration

        == 5

    )

    assert settings.max_search_results_per_query == 5

    assert settings.max_sources_per_run == 12

    assert settings.max_source_fetches_per_run == 12

    assert settings.max_llm_calls_per_run == 64

    assert settings.finalization_llm_reserve == 2

    assert settings.request_timeout_seconds == 15.0

def test_legacy_python_properties_reflect_explicit_whole_run_fields():

    settings = _settings(

        max_search_queries_per_run=11,

        max_sources_per_run=15,

    )

    assert settings.max_search_queries == 11

    assert settings.max_sources == 15

def test_role_models_fall_back_to_legacy_llm_model():

    settings = _settings(

        llm_model="legacy-model",

        llm_worker_model=None,

        llm_final_model=None,

    )

    assert settings.worker_llm_model == "legacy-model"

    assert settings.final_llm_model == "legacy-model"

def test_worker_model_overrides_legacy_llm_model():

    settings = _settings(

        llm_model="legacy-model",

        llm_worker_model="worker-model",

        llm_final_model=None,

    )

    assert settings.worker_llm_model == "worker-model"

    assert settings.final_llm_model == "legacy-model"

def test_final_model_overrides_legacy_llm_model():

    settings = _settings(

        llm_model="legacy-model",

        llm_worker_model=None,

        llm_final_model="final-model",

    )

    assert settings.worker_llm_model == "legacy-model"

    assert settings.final_llm_model == "final-model"

def test_role_specific_models_override_legacy_model_independently():

    settings = _settings(

        llm_model="legacy-model",

        llm_worker_model="worker-model",

        llm_final_model="final-model",

    )

    assert settings.worker_llm_model == "worker-model"

    assert settings.final_llm_model == "final-model"

# ---------------------------------------------------------------------------

# Environment variables

# ---------------------------------------------------------------------------

def test_environment_variables_override_defaults(

    monkeypatch,

):

    monkeypatch.setenv(

        "APP_ENV",

        "production",

    )

    monkeypatch.setenv(

        "LOG_LEVEL",

        "DEBUG",

    )

    monkeypatch.setenv(
        "PUBLIC_DEMO_ONLY",
        "true",
    )

    monkeypatch.setenv(

        "LLM_PROVIDER",

        "groq",

    )

    monkeypatch.setenv(

        "LLM_MODEL",

        "example-model",

    )

    monkeypatch.setenv(

        "SEARCH_PROVIDER",

        "custom",

    )

    monkeypatch.setenv(

        "MAX_RESEARCH_ITERATIONS",

        "4",

    )

    monkeypatch.setenv(

        "MAX_SUB_QUESTIONS",

        "6",

    )

    monkeypatch.setenv(

        "MAX_SEARCH_QUERIES_PER_RUN",

        "10",

    )

    monkeypatch.setenv(

        "MAX_SEARCH_QUERIES_PER_ITERATION",

        "4",

    )

    monkeypatch.setenv(

        "MAX_SEARCH_RESULTS_PER_QUERY",

        "7",

    )

    monkeypatch.setenv(

        "MAX_SOURCES_PER_RUN",

        "20",

    )

    monkeypatch.setenv(

        "MAX_SOURCE_FETCHES_PER_RUN",

        "18",

    )

    monkeypatch.setenv(

        "MAX_LLM_CALLS_PER_RUN",

        "70",

    )

    monkeypatch.setenv(

        "FINALIZATION_LLM_RESERVE",

        "3",

    )

    monkeypatch.setenv(

        "REQUEST_TIMEOUT_SECONDS",

        "30",

    )

    settings = _settings()

    assert settings.app_env == "production"

    assert settings.log_level == "DEBUG"

    assert settings.public_demo_only is True

    assert settings.llm_provider == "groq"

    assert settings.llm_model == "example-model"

    assert settings.search_provider == "custom"

    assert settings.max_research_iterations == 4

    assert settings.max_sub_questions == 6

    assert settings.max_search_queries_per_run == 10

    assert (

        settings.max_search_queries_per_iteration

        == 4

    )

    assert settings.max_search_results_per_query == 7

    assert settings.max_sources_per_run == 20

    assert settings.max_source_fetches_per_run == 18

    assert settings.max_llm_calls_per_run == 70

    assert settings.finalization_llm_reserve == 3

    assert settings.request_timeout_seconds == 30.0

def test_environment_variable_names_are_case_insensitive(

    monkeypatch,

):

    monkeypatch.setenv(

        "max_search_queries_per_run",

        "11",

    )

    settings = _settings()

    assert settings.max_search_queries_per_run == 11

def test_legacy_search_budget_environment_name_is_supported(

    monkeypatch,

):

    monkeypatch.setenv(

        "MAX_SEARCH_QUERIES",

        "11",

    )

    settings = _settings()

    assert settings.max_search_queries_per_run == 11

def test_legacy_source_budget_environment_name_is_supported(

    monkeypatch,

):

    monkeypatch.setenv(

        "MAX_SOURCES",

        "14",

    )

    settings = _settings()

    assert settings.max_sources_per_run == 14

def test_legacy_python_input_name_for_search_budget_is_supported():

    settings = _settings(

        max_search_queries=9,

    )

    assert settings.max_search_queries_per_run == 9

def test_legacy_python_input_name_for_source_budget_is_supported():

    settings = _settings(

        max_sources=13,

    )

    assert settings.max_sources_per_run == 13

# ---------------------------------------------------------------------------

# Secrets

# ---------------------------------------------------------------------------

def test_api_keys_are_stored_as_secret_strings(

    monkeypatch,

):

    monkeypatch.setenv(

        "LLM_API_KEY",

        "llm-secret-value",

    )

    monkeypatch.setenv(

        "TAVILY_API_KEY",

        "tavily-secret-value",

    )

    settings = _settings()

    assert isinstance(

        settings.llm_api_key,

        SecretStr,

    )

    assert isinstance(

        settings.tavily_api_key,

        SecretStr,

    )

    assert (

        settings.llm_api_key.get_secret_value()

        == "llm-secret-value"

    )

    assert (

        settings.tavily_api_key.get_secret_value()

        == "tavily-secret-value"

    )

def test_secret_values_are_not_exposed_in_settings_repr(

    monkeypatch,

):

    monkeypatch.setenv(

        "LLM_API_KEY",

        "super-secret-llm-key",

    )

    monkeypatch.setenv(

        "TAVILY_API_KEY",

        "super-secret-tavily-key",

    )

    settings = _settings()

    rendered = repr(

        settings

    )

    assert "super-secret-llm-key" not in rendered

    assert "super-secret-tavily-key" not in rendered

# ---------------------------------------------------------------------------

# Research-budget validation

# ---------------------------------------------------------------------------

@pytest.mark.parametrize(

    "field_name",

    [

        "max_research_iterations",

        "max_sub_questions",

        "max_search_queries_per_run",

        "max_search_queries_per_iteration",

        "max_search_results_per_query",

        "max_sources_per_run",

        "max_source_fetches_per_run",

        "max_llm_calls_per_run",

    ],

)

def test_positive_research_budget_fields_reject_zero(

    field_name,

):

    with pytest.raises(

        ValidationError

    ):

        _settings(

            **{

                field_name: 0,

            }

        )

@pytest.mark.parametrize(

    "field_name",

    [

        "max_research_iterations",

        "max_sub_questions",

        "max_search_queries_per_run",

        "max_search_queries_per_iteration",

        "max_search_results_per_query",

        "max_sources_per_run",

        "max_source_fetches_per_run",

        "max_llm_calls_per_run",

    ],

)

def test_positive_research_budget_fields_reject_negative_values(

    field_name,

):

    with pytest.raises(

        ValidationError

    ):

        _settings(

            **{

                field_name: -1,

            }

        )

@pytest.mark.parametrize(

    "field_name",

    [

        "max_research_iterations",

        "max_sub_questions",

        "max_search_queries_per_run",

        "max_search_queries_per_iteration",

        "max_search_results_per_query",

        "max_sources_per_run",

        "max_source_fetches_per_run",

        "max_llm_calls_per_run",

    ],

)

def test_positive_research_budget_fields_accept_one(

    field_name,

):

    kwargs = {

        field_name: 1,

    }

    if field_name == "max_llm_calls_per_run":

        kwargs["finalization_llm_reserve"] = 1

    settings = _settings(

        **kwargs

    )

    assert getattr(

        settings,

        field_name,

    ) == 1

def test_finalization_llm_reserve_accepts_zero():

    settings = _settings(

        finalization_llm_reserve=0,

    )

    assert settings.finalization_llm_reserve == 0

def test_finalization_llm_reserve_rejects_negative_value():

    with pytest.raises(

        ValidationError

    ):

        _settings(

            finalization_llm_reserve=-1,

        )

def test_finalization_llm_reserve_may_equal_total_llm_budget():

    settings = _settings(

        max_llm_calls_per_run=2,

        finalization_llm_reserve=2,

    )

    assert settings.finalization_llm_reserve == 2

def test_finalization_llm_reserve_must_not_exceed_total_llm_budget():

    with pytest.raises(

        ValidationError,

        match="must not exceed",

    ):

        _settings(

            max_llm_calls_per_run=2,

            finalization_llm_reserve=3,

        )

# ---------------------------------------------------------------------------

# Network timeout validation

# ---------------------------------------------------------------------------

@pytest.mark.parametrize(

    "value",

    [

        0,

        -1,

        -0.1,

    ],

)

def test_request_timeout_rejects_non_positive_values(

    value,

):

    with pytest.raises(

        ValidationError

    ):

        _settings(

            request_timeout_seconds=value,

        )

@pytest.mark.parametrize(

    "value",

    [

        0.1,

        1,

        30.5,

    ],

)

def test_request_timeout_accepts_positive_values(

    value,

):

    settings = _settings(

        request_timeout_seconds=value,

    )

    assert settings.request_timeout_seconds == value

# ---------------------------------------------------------------------------

# Extraction limits

# ---------------------------------------------------------------------------

def test_extraction_defaults():

    settings = _settings()

    assert settings.max_response_bytes == (

        2 * 1024 * 1024

    )

    assert settings.max_text_chars == 100_000

    assert settings.max_redirects == 5

def test_extraction_limits_can_be_loaded_from_environment(

    monkeypatch,

):

    monkeypatch.setenv(

        "MAX_RESPONSE_BYTES",

        "500000",

    )

    monkeypatch.setenv(

        "MAX_TEXT_CHARS",

        "25000",

    )

    monkeypatch.setenv(

        "MAX_REDIRECTS",

        "3",

    )

    settings = _settings()

    assert settings.max_response_bytes == 500000

    assert settings.max_text_chars == 25000

    assert settings.max_redirects == 3

@pytest.mark.parametrize(

    "field_name",

    [

        "max_response_bytes",

        "max_text_chars",

    ],

)

def test_positive_extraction_limits_reject_zero(

    field_name,

):

    with pytest.raises(

        ValidationError

    ):

        _settings(

            **{

                field_name: 0,

            }

        )

@pytest.mark.parametrize(

    "field_name",

    [

        "max_response_bytes",

        "max_text_chars",

        "max_redirects",

    ],

)

def test_extraction_limits_reject_negative_values(

    field_name,

):

    with pytest.raises(

        ValidationError

    ):

        _settings(

            **{

                field_name: -1,

            }

        )

def test_zero_redirects_is_valid():

    settings = _settings(

        max_redirects=0,

    )

    assert settings.max_redirects == 0