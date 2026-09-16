"""Grounded evidence extraction from transient webpage content.

The LLM may propose evidence excerpts and relevance notes only.

Python remains responsible for:
- validating that proposed excerpts actually exist in the fetched page
- assigning Evidence IDs
- assigning source IDs
- assigning sub-question IDs

Fetched webpage text remains transient and is never stored directly in
ResearchState.
"""

from __future__ import annotations

from collections.abc import Callable
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from research_agent.graph.nodes.source_fetcher import (
    SourceFetchResult,
)
from research_agent.llm.client import (
    LLMClient,
    LLMResponseError,
)
from research_agent.models.schemas import (
    Evidence,
    SubQuestion,
)
from research_agent.prompts.evidence import (
    EVIDENCE_SYSTEM_PROMPT,
    build_evidence_user_prompt,
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


class EvidenceGroundingError(LLMResponseError):
    """Raised when proposed evidence is not grounded in the webpage."""


IDFactory = Callable[[], str]


def _default_id_factory() -> str:
    """Create a trusted internal Evidence ID."""

    return f"ev_{uuid4().hex}"


class EvidenceExtractor:
    """Extract grounded Evidence objects from one successfully fetched page."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        id_factory: IDFactory = _default_id_factory,
    ) -> None:
        self._llm = llm
        self._id_factory = id_factory

    def extract(
        self,
        *,
        sub_question: SubQuestion,
        fetch_result: SourceFetchResult,
    ) -> list[Evidence]:
        """Extract evidence relevant to one research sub-question."""

        if not isinstance(
            sub_question,
            SubQuestion,
        ):
            raise TypeError(
                "sub_question must be a SubQuestion."
            )

        if not isinstance(
            fetch_result,
            SourceFetchResult,
        ):
            raise TypeError(
                "fetch_result must be a SourceFetchResult."
            )

        if not fetch_result.succeeded:
            raise ValueError(
                "Evidence can only be extracted from a successful fetch."
            )

        page = fetch_result.page

        if page is None:
            raise ValueError(
                "Successful fetch result must contain a page."
            )

        # A successfully fetched page can still contain no meaningful text.
        # In that case there is nothing to send to the LLM or extract.
        if not page.text.strip():
            return []

        user_prompt = build_evidence_user_prompt(
            sub_question=sub_question.question,
            webpage_text=page.text,
        )

        response = self._llm.generate_structured(
            system_prompt=EVIDENCE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=EvidenceResponse,
        )

        if not isinstance(
            response,
            EvidenceResponse,
        ):
            raise LLMResponseError(
                "Evidence LLM returned an unexpected response type."
            )

        self._validate_grounding(
            response=response,
            webpage_text=page.text,
        )

        generated_ids: set[str] = set()
        evidence_items: list[Evidence] = []

        for candidate in response.evidence:
            evidence_id = self._id_factory()

            if not isinstance(
                evidence_id,
                str,
            ):
                raise RuntimeError(
                    "Evidence ID factory must return a string."
                )

            evidence_id = evidence_id.strip()

            if not evidence_id:
                raise RuntimeError(
                    "Evidence ID factory returned a blank ID."
                )

            if evidence_id in generated_ids:
                raise RuntimeError(
                    "Evidence ID factory returned a duplicate ID."
                )

            generated_ids.add(
                evidence_id
            )

            evidence_items.append(
                Evidence(
                    id=evidence_id,
                    source_id=fetch_result.source.id,
                    sub_question_id=sub_question.id,
                    excerpt=candidate.excerpt,
                    relevance_note=candidate.relevance_note,
                )
            )

        return evidence_items

    @staticmethod
    def _validate_grounding(
        *,
        response: EvidenceResponse,
        webpage_text: str,
    ) -> None:
        """Verify every proposed excerpt exists verbatim in the page."""

        seen_excerpts: set[str] = set()

        for candidate in response.evidence:
            excerpt = candidate.excerpt

            if excerpt not in webpage_text:
                raise EvidenceGroundingError(
                    "LLM proposed an evidence excerpt that does not "
                    "exist verbatim in the fetched webpage."
                )

            if excerpt in seen_excerpts:
                raise LLMResponseError(
                    "Evidence LLM returned duplicate excerpts."
                )

            seen_excerpts.add(
                excerpt
            )