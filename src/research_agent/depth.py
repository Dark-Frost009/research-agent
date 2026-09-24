"""Research depth presets, bounded by the operator's configured ceilings."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from research_agent.config import Settings

DepthName = Literal["Quick", "Standard", "Thorough"]
DEPTH_NAMES = ("Quick", "Standard", "Thorough")


class DepthConfigurationError(ValueError):
    """The configured ceilings cannot support the selected depth safely."""


class ResearchDepth(BaseModel):
    """Only non-secret limits; safe to keep alongside a completed/partial run."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: DepthName
    rounds: int = Field(ge=1)
    searches: int = Field(ge=1)
    sources: int = Field(ge=1)
    fetches: int = Field(ge=1)
    ai_calls: int = Field(ge=3)
    sub_questions: int = Field(ge=1)
    searches_per_round: int = Field(ge=1)
    results_per_search: int = Field(ge=1)
    finalization_reserve: int = Field(ge=2)

    @model_validator(mode="after")
    def validate_reserve(self):
        if self.ai_calls <= self.finalization_reserve:
            raise ValueError("Research must have capacity beyond the finalization reserve.")
        return self

    @property
    def summary(self) -> str:
        return (f"{self.name} · up to {self.rounds} research rounds, {self.searches} searches, "
                f"{self.sources} sources, {self.fetches} page reads, and {self.ai_calls} AI calls.")


_FIELDS = {
    "rounds": "max_research_iterations",
    "searches": "max_search_queries_per_run",
    "sources": "max_sources_per_run",
    "fetches": "max_source_fetches_per_run",
    "ai_calls": "max_llm_calls_per_run",
    "sub_questions": "max_sub_questions",
    "searches_per_round": "max_search_queries_per_iteration",
    "results_per_search": "max_search_results_per_query",
}
_PRESETS = {
    "Quick": (1, 2, 4, 4, 12, 2, 2, 3),
    "Standard": (2, 5, 8, 8, 32, 3, 3, 4),
    "Thorough": (3, 8, 12, 12, 64, 5, 5, 5),
}


def apply_depth(settings: Settings, name: DepthName) -> tuple[Settings, ResearchDepth]:
    """Return fresh settings without modifying .env, secrets, or network policy."""
    if name not in _PRESETS:
        raise DepthConfigurationError("Choose Quick, Standard, or Thorough research.")
    limits = {
        field: min(cap, getattr(settings, setting))
        for (field, setting), cap in zip(_FIELDS.items(), _PRESETS[name], strict=True)
    }
    # Preserve the configured reserve, including intentionally larger reserves.
    # Never quietly enable a mode that weakens synthesis + semantic verification.
    if settings.finalization_llm_reserve < 2 or limits["ai_calls"] <= settings.finalization_llm_reserve:
        raise DepthConfigurationError(
            "Research depth needs a finalization reserve of at least two AI calls "
            "and an AI-call limit larger than that reserve. Check the research budget settings."
        )
    chosen = ResearchDepth(name=name, **limits, finalization_reserve=settings.finalization_llm_reserve)
    values = settings.model_dump()
    values.update({setting: limits[field] for field, setting in _FIELDS.items()})
    return Settings.model_validate(values), chosen
