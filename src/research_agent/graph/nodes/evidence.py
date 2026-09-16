"""Structured LLM output contract for evidence extraction.

The LLM may propose only evidence excerpts and relevance notes.

Trusted internal fields such as Evidence IDs, source IDs, and
sub-question IDs are assigned later by Python.

Actual evidence extraction behavior and deterministic excerpt grounding
will be added separately.
"""

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class _EvidenceSchema(BaseModel):
    """Strict base model for LLM-generated evidence output."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )


class EvidenceCandidate(_EvidenceSchema):
    """One piece of evidence proposed by the LLM."""

    excerpt: str = Field(
        min_length=1,
    )

    relevance_note: str | None = Field(
        default=None,
        min_length=1,
    )


class EvidenceResponse(_EvidenceSchema):
    """Structured response returned by the evidence-extraction LLM."""

    evidence: list[EvidenceCandidate] = Field(
        default_factory=list,
    )