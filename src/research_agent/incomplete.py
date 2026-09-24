"""Recoverable research data; deliberately has no report or draft-answer field."""
from datetime import datetime, timezone
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from research_agent.models.schemas import Evidence, Source
from research_agent.depth import ResearchDepth
from research_agent.run_summary import RunSummary


STOP_REASONS = {
    "quota": "The AI service's rate or quota limit was reached.",
    "unavailable": "The AI service was temporarily unavailable.",
    "provider": "An AI service request failed.",
    "response": "The AI response did not pass the required checks.",
    "unexpected": "The research run stopped unexpectedly.",
}


class IncompleteResearch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    status: Literal["incomplete"] = "incomplete"
    question: str = Field(min_length=1)
    created_at: AwareDatetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_stage: str
    reason: Literal["quota", "unavailable", "provider", "response", "unexpected"]
    sources: list[Source]
    evidence: list[Evidence]
    iterations: int = Field(ge=0)
    searches: int = Field(ge=0)
    warning_count: int = Field(ge=0)
    issues: tuple[str, ...] = ()
    depth: ResearchDepth | None = None
    summary: RunSummary | None = None

    @model_validator(mode="after")
    def validate_links(self):
        if self.summary is not None and self.summary.outcome != "incomplete":
            raise ValueError("An incomplete run cannot claim a completed outcome.")
        source_ids = {source.id for source in self.sources}
        if len(source_ids) != len(self.sources):
            raise ValueError("Duplicate source identifiers.")
        if len({item.id for item in self.evidence}) != len(self.evidence):
            raise ValueError("Duplicate evidence identifiers.")
        if any(item.source_id not in source_ids for item in self.evidence):
            raise ValueError("Saved evidence has no source.")
        return self

    @property
    def stop_message(self) -> str:
        return STOP_REASONS[self.reason]

    def json_export(self) -> str:
        return self.model_dump_json(indent=2)


class ResearchInterrupted(RuntimeError):
    """An unsuccessful run with an explicitly limited snapshot for recovery."""

    def __init__(self, partial: IncompleteResearch):
        self.partial = partial
        super().__init__(partial.stop_message)
