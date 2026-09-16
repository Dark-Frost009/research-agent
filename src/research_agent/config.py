"""Application configuration for the Research Agent.

Settings are loaded from environment variables and an optional local
.env file. Secrets and operational limits belong here rather than in
LangGraph state.
"""

from typing import Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_env: str = "development"
    log_level: str = "INFO"

    # LLM
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_api_key: Optional[SecretStr] = None

    # Search
    search_provider: str = "tavily"
    tavily_api_key: Optional[SecretStr] = None

    # Research budgets
    max_research_iterations: int = Field(default=2, ge=1)
    max_search_queries: int = Field(default=8, ge=1)
    max_search_results_per_query: int = Field(default=5, ge=1)
    max_sources: int = Field(default=12, ge=1)

    # Network
    request_timeout_seconds: float = Field(default=15.0, gt=0)