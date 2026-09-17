"""Grounded synthesis and deterministic citation construction.

The synthesis LLM sees only controlled evidence handles such as E1 and E2.
It never receives or generates trusted internal Evidence IDs.

Python is responsible for:
- validating every evidence handle
- ensuring claims actually appear in the generated content
- rejecting duplicate claims and duplicate handles
- resolving controlled handles to trusted Evidence IDs
- assigning Citation IDs

No LLM-generated internal identifiers are trusted.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from research_agent.llm.client import (
    LLMClient,
    LLMResponseError,
)
from research_agent.models.schemas import (
    Citation,
    Evidence,
)
from research_agent.prompts.synthesis import (
    SYNTHESIS_SYSTEM_PROMPT,
    build_evidence_catalog,
    build_synthesis_user_prompt,
)


class _SynthesisSchema(BaseModel):
    """Strict base model for synthesis-generated structured output."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )


class SynthesizedClaim(_SynthesisSchema):
    """One factual claim and its controlled evidence handles."""

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


class SynthesisValidationError(LLMResponseError):
    """Raised when synthesis output violates grounding invariants."""


CitationIDFactory = Callable[[], str]


def _default_citation_id_factory() -> str:
    """Create a trusted internal Citation ID."""

    return f"cit_{uuid4().hex}"


@dataclass(frozen=True)
class SynthesisResult:
    """Persistent grounded output from synthesis."""

    content: str
    citations: list[Citation]


class Synthesizer:
    """Create grounded draft content and trusted Citation objects."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        citation_id_factory: CitationIDFactory = _default_citation_id_factory,
    ) -> None:
        self._llm = llm
        self._citation_id_factory = citation_id_factory

    def synthesize(
        self,
        *,
        original_question: str,
        evidence: list[Evidence],
    ) -> SynthesisResult:
        """Synthesize one grounded research draft."""

        clean_question = self._validate_question(
            original_question
        )

        evidence_catalog, handle_map = build_evidence_catalog(
            evidence
        )

        # There is no factual source material to synthesize.
        # Avoid an unnecessary LLM call and avoid inviting unsupported claims.
        if not evidence:
            return SynthesisResult(
                content=(
                    "The available evidence is insufficient "
                    "to answer the research question."
                ),
                citations=[],
            )

        user_prompt = build_synthesis_user_prompt(
            original_question=clean_question,
            evidence_catalog=evidence_catalog,
        )

        response = self._llm.generate_structured(
            system_prompt=SYNTHESIS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=SynthesisResponse,
        )

        if not isinstance(
            response,
            SynthesisResponse,
        ):
            raise LLMResponseError(
                "Synthesis LLM returned an unexpected response type."
            )

        self._validate_response(
            response=response,
            handle_map=handle_map,
        )

        # All LLM-controlled data has now been validated.
        # Generate trusted citation IDs only after validation succeeds.
        citation_ids = self._generate_citation_ids(
            count=len(response.claims),
        )

        citations: list[Citation] = []

        for citation_id, claim in zip(
            citation_ids,
            response.claims,
            strict=True,
        ):
            evidence_ids = [
                handle_map[handle]
                for handle in claim.evidence_handles
            ]

            citations.append(
                Citation(
                    id=citation_id,
                    claim_text=claim.claim_text,
                    evidence_ids=evidence_ids,
                )
            )

        return SynthesisResult(
            content=response.content,
            citations=citations,
        )

    @staticmethod
    def _validate_question(
        original_question: str,
    ) -> str:
        if not isinstance(
            original_question,
            str,
        ):
            raise TypeError(
                "original_question must be a string."
            )

        clean_question = original_question.strip()

        if not clean_question:
            raise ValueError(
                "original_question must not be blank."
            )

        return clean_question

    @staticmethod
    def _validate_response(
        *,
        response: SynthesisResponse,
        handle_map: dict[str, str],
    ) -> None:
        """Validate all LLM-controlled synthesis relationships."""

        seen_claims: set[str] = set()

        for claim in response.claims:
            if claim.claim_text not in response.content:
                raise SynthesisValidationError(
                    "Synthesis claim_text does not appear verbatim "
                    "in the content field."
                )

            if claim.claim_text in seen_claims:
                raise SynthesisValidationError(
                    "Synthesis LLM returned a duplicate claim."
                )

            seen_claims.add(
                claim.claim_text
            )

            seen_handles: set[str] = set()

            for handle in claim.evidence_handles:
                if handle in seen_handles:
                    raise SynthesisValidationError(
                        "Synthesis claim contains a duplicate "
                        "evidence handle."
                    )

                if handle not in handle_map:
                    raise SynthesisValidationError(
                        "Synthesis claim references an unknown "
                        "evidence handle."
                    )

                seen_handles.add(
                    handle
                )

    def _generate_citation_ids(
        self,
        *,
        count: int,
    ) -> list[str]:
        """Generate unique trusted Citation IDs."""

        generated: list[str] = []
        seen: set[str] = set()

        for _ in range(count):
            citation_id = self._citation_id_factory()

            if not isinstance(
                citation_id,
                str,
            ):
                raise RuntimeError(
                    "Citation ID factory must return a string."
                )

            citation_id = citation_id.strip()

            if not citation_id:
                raise RuntimeError(
                    "Citation ID factory returned a blank ID."
                )

            if citation_id in seen:
                raise RuntimeError(
                    "Citation ID factory returned a duplicate ID."
                )

            seen.add(
                citation_id
            )

            generated.append(
                citation_id
            )

        return generated