"""Application configuration for the Research Agent.

Settings are loaded from environment variables and an optional local
.env file.

Secrets and static operational limits belong here rather than in
LangGraph state.

Whole-run budget settings use explicit names so it is clear whether a
limit applies to one iteration or to the complete research run.

For compatibility with earlier project configuration:

    MAX_SEARCH_QUERIES
    MAX_SOURCES

are still accepted as environment/input aliases for:

    max_search_queries_per_run
    max_sources_per_run

The legacy LLM_MODEL setting is also retained. New deployments may
configure separate worker and finalization models with:

    LLM_WORKER_MODEL
    LLM_FINAL_MODEL

When either role-specific model is not configured, it falls back to
LLM_MODEL.

New code should use the explicit field names.
"""

from typing import Optional

from pydantic import (
    AliasChoices,
    Field,
    SecretStr,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


class Settings(BaseSettings):
    """Configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    app_env: str = "development"
    log_level: str = "INFO"
    public_demo_only: bool = False

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------

    llm_provider: Optional[str] = None

    # Legacy single-model setting.
    #
    # Existing .env files using LLM_MODEL continue to work. When a
    # role-specific model is not configured, that role falls back to
    # this value.
    llm_model: Optional[str] = None

    # Model used for optional/high-volume research work such as
    # planning, evidence extraction, and critique.
    llm_worker_model: Optional[str] = None

    # Model used for the protected finalization path such as synthesis
    # and semantic verification.
    llm_final_model: Optional[str] = None

    llm_api_key: Optional[SecretStr] = None

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    search_provider: str = "tavily"
    tavily_api_key: Optional[SecretStr] = None

    # ------------------------------------------------------------------
    # Research budgets
    # ------------------------------------------------------------------

    # Two means two total research iterations, including the initial one.
    max_research_iterations: int = Field(
        default=2,
        ge=1,
    )

    # Planner shaping limit.
    max_sub_questions: int = Field(
        default=5,
        ge=1,
    )

    # Whole-run search-query budget.
    #
    # MAX_SEARCH_QUERIES remains accepted for backward compatibility.
    max_search_queries_per_run: int = Field(
        default=8,
        ge=1,
        validation_alias=AliasChoices(
            "max_search_queries_per_run",
            "MAX_SEARCH_QUERIES_PER_RUN",
            "max_search_queries",
            "MAX_SEARCH_QUERIES",
        ),
    )

    # Local shaping limit for one research iteration.
    max_search_queries_per_iteration: int = Field(
        default=5,
        ge=1,
    )

    # Provider result shaping limit. This is not itself a whole-run
    # side-effect counter; it limits results returned by each search.
    max_search_results_per_query: int = Field(
        default=5,
        ge=1,
    )

    # Maximum number of unique Sources retained across the whole run.
    #
    # MAX_SOURCES remains accepted for backward compatibility.
    max_sources_per_run: int = Field(
        default=12,
        ge=1,
        validation_alias=AliasChoices(
            "max_sources_per_run",
            "MAX_SOURCES_PER_RUN",
            "max_sources",
            "MAX_SOURCES",
        ),
    )

    # Maximum number of source-fetch attempts across the whole run.
    #
    # A failed fetch still consumes one attempt once execution has been
    # authorized.
    max_source_fetches_per_run: int = Field(
        default=12,
        ge=1,
    )

    # Whole-run LLM-call ceiling.
    #
    # The default leaves room for the current research pipeline while
    # preventing an uncontrolled evidence-extraction fan-out.
    max_llm_calls_per_run: int = Field(
        default=64,
        ge=1,
    )

    # Protected LLM capacity for finalization:
    #
    # 1. synthesis
    # 2. A1.2 semantic verification
    finalization_llm_reserve: int = Field(
        default=2,
        ge=0,
    )

    # ------------------------------------------------------------------
    # Network / extraction
    # ------------------------------------------------------------------

    request_timeout_seconds: float = Field(
        default=15.0,
        gt=0,
    )

    max_response_bytes: int = Field(
        default=2 * 1024 * 1024,
        ge=1,
    )

    max_text_chars: int = Field(
        default=100_000,
        ge=1,
    )

    max_redirects: int = Field(
        default=5,
        ge=0,
    )

    @model_validator(mode="after")
    def validate_llm_finalization_reserve(
        self,
    ) -> "Settings":
        """Ensure protected finalization capacity fits inside the LLM budget."""

        if (
            self.finalization_llm_reserve
            > self.max_llm_calls_per_run
        ):
            raise ValueError(
                "finalization_llm_reserve must not exceed "
                "max_llm_calls_per_run."
            )

        return self

    # ------------------------------------------------------------------
    # Effective LLM model selection
    # ------------------------------------------------------------------

    @property
    def worker_llm_model(self) -> Optional[str]:
        """Return the configured model for research-stage LLM work."""

        return self.llm_worker_model or self.llm_model

    @property
    def final_llm_model(self) -> Optional[str]:
        """Return the configured model for finalization-stage LLM work."""

        return self.llm_final_model or self.llm_model

    # ------------------------------------------------------------------
    # Transitional compatibility properties
    # ------------------------------------------------------------------

    @property
    def max_search_queries(self) -> int:
        """Deprecated compatibility view of the whole-run search budget.

        New code should use ``max_search_queries_per_run``.
        """

        return self.max_search_queries_per_run

    @property
    def max_sources(self) -> int:
        """Deprecated compatibility view of the whole-run source budget.

        New code should use ``max_sources_per_run``.
        """

        return self.max_sources_per_run