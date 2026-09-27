"""Session-only BYOK boundary, separate from the local history application.

This is a pre-deployment foundation. The in-process capacity gate is not a
distributed rate limiter and must not be advertised as one.
"""
from contextlib import contextmanager
from threading import BoundedSemaphore

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from research_agent.config import Settings
from research_agent.ui_service import run_question


class PublicConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='PUBLIC_', env_file=None, extra='ignore')
    enabled: bool = False
    auth_preview: bool = False
    gemini_model: str = Field(default='', max_length=150)


class VisitorSettings(Settings):
    """Only explicit constructor values; never read host secrets or .env."""

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings,
                                   dotenv_settings, file_secret_settings):
        return (init_settings,)


class PublicInputError(ValueError):
    pass


class ServiceBusyError(RuntimeError):
    pass


# Shared across sessions in one Python process. No credentials or reports live here.
_capacity = BoundedSemaphore(2)


@contextmanager
def research_slot():
    if not _capacity.acquire(blocking=False):
        raise ServiceBusyError('The service is busy. Please try again shortly.')
    try:
        yield
    finally:
        _capacity.release()


def visitor_settings(gemini_key: str, tavily_key: str, model: str) -> VisitorSettings:
    for name, value in [('Gemini', gemini_key), ('Tavily', tavily_key)]:
        if not isinstance(value, str) or not value.strip():
            raise PublicInputError(f'Enter your {name} API key.')
        if len(value) > 512 or any(c.isspace() for c in value.strip()):
            raise PublicInputError(f'Check your {name} API key format.')
    if not model.strip() or len(model) > 150:
        raise PublicInputError('The service operator must configure the research model.')
    return VisitorSettings(
        app_env='public', llm_provider='gemini', llm_model=model.strip(),
        llm_api_key=SecretStr(gemini_key.strip()),
        tavily_api_key=SecretStr(tavily_key.strip()), search_provider='tavily',
        max_research_iterations=1, max_search_queries_per_run=2,
        max_search_queries_per_iteration=2, max_sub_questions=2,
        max_search_results_per_query=3, max_sources_per_run=4,
        max_source_fetches_per_run=4, max_llm_calls_per_run=12,
        finalization_llm_reserve=2, request_timeout_seconds=15,
        max_response_bytes=1_048_576, max_text_chars=30_000, max_redirects=3,
    )


def run_visitor_question(question, on_progress, *, gemini_key, tavily_key, model):
    if not isinstance(question, str) or not 1 <= len(question.strip()) <= 2000:
        raise PublicInputError('Enter a question between 1 and 2,000 characters.')
    settings = visitor_settings(gemini_key, tavily_key, model)
    with research_slot():
        # Already fixed to Quick limits. Avoid converting back to environment-
        # reading Settings in apply_depth; preserve the explicit-only settings.
        return run_question(question.strip(), on_progress, settings=settings)
