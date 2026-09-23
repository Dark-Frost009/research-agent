"""Evidence-sufficiency critic with whole-run LLM budget authorization.

The critic evaluates accumulated grounded Evidence before finalization.

It does not:

- draft the final answer;
- perform graph routing;
- start a new research iteration;
- independently mutate whole-run budget state.

Budget flow:

    prepare_critique_call(...)
        ↓
    authorize at most one optional-research LLM call
        ↓
    graph persists CritiqueCall.llm_calls_used
        ↓
    Critic.critique(...)

The protected finalization reserve therefore cannot be consumed by critique.

Zero Evidence is handled deterministically without an LLM call. If optional
research LLM capacity is exhausted, critique also returns a deterministic
non-sufficient result so orchestration can proceed toward finalization rather
than performing an untracked provider call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from research_agent.graph.budget import (
    BudgetAuthorization,
    BudgetPolicy,
    BudgetUsage,
)
from research_agent.llm.client import (
    LLMClient,
    LLMResponseError,
)
from research_agent.models.schemas import (
    CritiqueResult,
    Evidence,
)
from research_agent.prompts.critic import (
    CRITIC_SYSTEM_PROMPT,
    build_critic_user_prompt,
)


# ---------------------------------------------------------------------------
# Structured LLM response
# ---------------------------------------------------------------------------


class _CriticSchema(BaseModel):
    """Strict base model for critic-generated structured output."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )


NonEmptyText = Annotated[
    str,
    Field(
        min_length=1,
    ),
]


class CriticResponse(_CriticSchema):
    """Structured response returned by the critic LLM."""

    sufficient: bool

    gaps: list[
        NonEmptyText
    ] = Field(
        default_factory=list,
    )

    follow_up_questions: list[
        NonEmptyText
    ] = Field(
        default_factory=list,
    )

    reasoning: NonEmptyText | None = None


# ---------------------------------------------------------------------------
# Authorized call contract
# ---------------------------------------------------------------------------


@dataclass(
    frozen=True,
)
class CritiqueCall:
    """One parent-authorized evidence-sufficiency critique operation."""

    original_question: str
    evidence: tuple[Evidence, ...]
    authorization: BudgetAuthorization

    def __post_init__(
        self,
    ) -> None:
        if not isinstance(
            self.original_question,
            str,
        ):
            raise TypeError(
                "original_question must be a string."
            )

        if not self.original_question.strip():
            raise ValueError(
                "original_question must not be blank."
            )

        if not isinstance(
            self.evidence,
            tuple,
        ):
            raise TypeError(
                "evidence must be a tuple."
            )

        seen_evidence_ids: set[
            str
        ] = set()

        for item in self.evidence:
            if not isinstance(
                item,
                Evidence,
            ):
                raise TypeError(
                    "evidence must contain only "
                    "Evidence objects."
                )

            if item.id in seen_evidence_ids:
                raise ValueError(
                    "evidence must not contain duplicate "
                    "Evidence IDs."
                )

            seen_evidence_ids.add(
                item.id
            )

        if not isinstance(
            self.authorization,
            BudgetAuthorization,
        ):
            raise TypeError(
                "authorization must be a "
                "BudgetAuthorization object."
            )

        if (
            self.authorization.resource
            != "llm_calls"
        ):
            raise ValueError(
                "Critique authorization must be for llm_calls."
            )

        if (
            self.authorization.llm_purpose
            != "optional_research"
        ):
            raise ValueError(
                "Critique authorization must be for "
                "optional_research."
            )

        expected_requested = (
            1
            if self.requires_llm
            else 0
        )

        if (
            self.authorization.requested
            != expected_requested
        ):
            raise ValueError(
                "Critique authorization requested count "
                "does not match critique requirements."
            )

        if (
            self.authorization.authorized
            > expected_requested
        ):
            raise ValueError(
                "Critique authorization cannot exceed "
                "the requested count."
            )

    @property
    def requires_llm(
        self,
    ) -> bool:
        """Whether evidence exists and semantic critique is required."""

        return bool(
            self.evidence
        )

    @property
    def authorized(
        self,
    ) -> bool:
        """Whether the one required critic LLM call was authorized."""

        return (
            self.requires_llm
            and self.authorization.authorized
            == 1
        )

    @property
    def llm_calls_used(
        self,
    ) -> int:
        """Whole-run LLM usage delta committed by this call."""

        if not self.requires_llm:
            return 0

        return (
            1
            if self.authorized
            else 0
        )


# ---------------------------------------------------------------------------
# Pure preparation
# ---------------------------------------------------------------------------


def prepare_critique_call(
    *,
    original_question: str,
    evidence: list[Evidence],
    usage: BudgetUsage,
    budget_policy: BudgetPolicy,
) -> CritiqueCall:
    """Prepare and authorize one evidence-sufficiency critique.

    Critique is optional research work and therefore must preserve the
    protected finalization LLM reserve.

    Non-empty Evidence requests one optional-research LLM call.

    Zero Evidence requests zero calls because insufficiency can be determined
    without contacting the provider.
    """

    if not isinstance(
        original_question,
        str,
    ):
        raise TypeError(
            "original_question must be a string."
        )

    clean_question = (
        original_question.strip()
    )

    if not clean_question:
        raise ValueError(
            "original_question must not be blank."
        )

    if not isinstance(
        evidence,
        list,
    ):
        raise TypeError(
            "evidence must be a list."
        )

    seen_evidence_ids: set[
        str
    ] = set()

    validated_evidence: list[
        Evidence
    ] = []

    for item in evidence:
        if not isinstance(
            item,
            Evidence,
        ):
            raise TypeError(
                "evidence must contain only Evidence objects."
            )

        if item.id in seen_evidence_ids:
            raise ValueError(
                "evidence must not contain duplicate "
                "Evidence IDs."
            )

        seen_evidence_ids.add(
            item.id
        )

        validated_evidence.append(
            item
        )

    if not isinstance(
        usage,
        BudgetUsage,
    ):
        raise TypeError(
            "usage must be a BudgetUsage object."
        )

    if not isinstance(
        budget_policy,
        BudgetPolicy,
    ):
        raise TypeError(
            "budget_policy must be a BudgetPolicy object."
        )

    requested = (
        1
        if validated_evidence
        else 0
    )

    authorization = (
        budget_policy.authorize_llm_calls(
            usage=usage,
            requested=requested,
            purpose="optional_research",
        )
    )

    return CritiqueCall(
        original_question=clean_question,
        evidence=tuple(
            validated_evidence
        ),
        authorization=authorization,
    )


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------


class Critic:
    """Evaluate whether accumulated grounded Evidence is sufficient."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        max_follow_up_questions: int,
    ) -> None:
        if (
            isinstance(
                max_follow_up_questions,
                bool,
            )
            or not isinstance(
                max_follow_up_questions,
                int,
            )
        ):
            raise TypeError(
                "max_follow_up_questions must be an integer."
            )

        if max_follow_up_questions < 1:
            raise ValueError(
                "max_follow_up_questions must be at least 1."
            )

        self._llm = llm
        self._max_follow_up_questions = (
            max_follow_up_questions
        )

    def critique(
        self,
        call: CritiqueCall,
    ) -> CritiqueResult:
        """Execute one already-authorized evidence critique."""

        if not isinstance(
            call,
            CritiqueCall,
        ):
            raise TypeError(
                "call must be a CritiqueCall object."
            )

        # Complete evidence absence can be judged deterministically.
        #
        # Repeating the original question as the follow-up is intentional in
        # this one special case: there is no evidence from which to derive a
        # narrower evidence gap yet.
        if not call.requires_llm:
            return CritiqueResult(
                sufficient=False,
                gaps=[
                    (
                        "No grounded evidence is available "
                        "to answer the research question."
                    )
                ],
                follow_up_questions=[
                    call.original_question
                ],
                reasoning=(
                    "Additional research is required before "
                    "a grounded answer can be produced."
                ),
            )

        # Budget exhaustion is expected control flow.
        #
        # We must not make an untracked provider call. Returning no follow-up
        # questions also prevents orchestration from inventing critique-driven
        # research work that the critic was unable to identify.
        if not call.authorized:
            return CritiqueResult(
                sufficient=False,
                gaps=[
                    (
                        "Evidence sufficiency could not be "
                        "evaluated because optional-research "
                        "LLM budget is unavailable."
                    )
                ],
                follow_up_questions=[],
                reasoning=(
                    "No additional critique LLM call was "
                    "performed. Continue toward finalization "
                    "using the evidence already collected."
                ),
            )

        user_prompt = (
            build_critic_user_prompt(
                original_question=(
                    call.original_question
                ),
                evidence=list(
                    call.evidence
                ),
                max_follow_up_questions=(
                    self._max_follow_up_questions
                ),
            )
        )

        response = (
            self._llm.generate_structured(
                system_prompt=(
                    CRITIC_SYSTEM_PROMPT
                ),
                user_prompt=user_prompt,
                response_model=(
                    CriticResponse
                ),
            )
        )

        if not isinstance(
            response,
            CriticResponse,
        ):
            raise LLMResponseError(
                "Critic LLM returned an "
                "unexpected response type."
            )

        self._validate_response(
            response=response,
            call=call,
        )

        return CritiqueResult(
            sufficient=(
                response.sufficient
            ),
            gaps=list(
                response.gaps
            ),
            follow_up_questions=list(
                response.follow_up_questions
            ),
            reasoning=(
                response.reasoning
            ),
        )

    @staticmethod
    def _normalize_text_key(
        value: str,
    ) -> str:
        """Normalize model text for deterministic duplicate detection."""

        return " ".join(
            value.casefold().split()
        )

    def _validate_response(
        self,
        *,
        response: CriticResponse,
        call: CritiqueCall,
    ) -> None:
        """Validate semantic relationships before trusting model output."""

        if response.sufficient:
            if response.gaps:
                raise LLMResponseError(
                    "Critic marked evidence sufficient "
                    "but also returned evidence gaps."
                )

            if response.follow_up_questions:
                raise LLMResponseError(
                    "Critic marked evidence sufficient "
                    "but also returned follow-up questions."
                )

        else:
            if not response.gaps:
                raise LLMResponseError(
                    "Critic marked evidence insufficient "
                    "without identifying any evidence gaps."
                )

            if not response.follow_up_questions:
                raise LLMResponseError(
                    "Critic marked evidence insufficient "
                    "without proposing follow-up questions."
                )

        if (
            len(
                response.follow_up_questions
            )
            > self._max_follow_up_questions
        ):
            raise LLMResponseError(
                "Critic returned more follow-up questions "
                "than allowed."
            )

        gap_keys: set[
            str
        ] = set()

        for gap in response.gaps:
            key = self._normalize_text_key(
                gap
            )

            if key in gap_keys:
                raise LLMResponseError(
                    "Critic returned duplicate evidence gaps."
                )

            gap_keys.add(
                key
            )

        follow_up_keys: set[
            str
        ] = set()

        original_question_key = (
            self._normalize_text_key(
                call.original_question
            )
        )

        for follow_up in (
            response.follow_up_questions
        ):
            key = self._normalize_text_key(
                follow_up
            )

            if key in follow_up_keys:
                raise LLMResponseError(
                    "Critic returned duplicate "
                    "follow-up questions."
                )

            # Evidence exists on every LLM-backed critique path.
            # Simply returning the original question would not identify a
            # critique-derived gap and could create an unproductive loop.
            if key == original_question_key:
                raise LLMResponseError(
                    "Critic follow-up questions must not "
                    "repeat the original research question "
                    "when grounded evidence already exists."
                )

            follow_up_keys.add(
                key
            )