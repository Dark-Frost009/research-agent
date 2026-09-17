"""Structured LLM output contract for grounded research synthesis.

The LLM receives controlled evidence handles such as E1 and E2 instead
of trusted internal Evidence IDs.

Actual handle validation and conversion into Citation domain objects are
performed later by Python.
"""

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class _SynthesisSchema(BaseModel):
    """Strict base model for synthesis-generated structured output."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )


class SynthesizedClaim(_SynthesisSchema):
    """One factual claim and the controlled evidence handles supporting it."""

    claim_text: str = Field(
        min_length=1,
    )

    evidence_handles: list[str] = Field(
        min_length=1,
    )


class SynthesisResponse(_SynthesisSchema):
    """Structured draft returned by the synthesis LLM."""

    content: str = Field(
        min_length=1,
    )

    claims: list[SynthesizedClaim] = Field(
        default_factory=list,
    )