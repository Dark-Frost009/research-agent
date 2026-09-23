"""Batch-budgeted grounded evidence extraction.

The LLM may propose evidence excerpts and relevance notes only.

Python remains responsible for:

- validating that proposed excerpts actually exist in the fetched page
- assigning Evidence IDs
- assigning source IDs
- assigning sub-question IDs
- centrally authorizing LLM usage before provider execution

Fetched webpage text remains transient and is never stored directly in
ResearchState.

Evidence extraction uses a parent-allocation design.

Phase 1 - describe work:

    SubQuestion + successful SourceFetchResult
        ↓
    EvidenceRequest

Phase 2 - centrally authorize the full batch:

    EvidenceRequest[]
        ↓
    prepare_evidence_batch(...)
        ↓
    count nonblank webpages requiring LLM work
        ↓
    BudgetPolicy.authorize_llm_calls(
        requested=N,
        purpose="optional_research",
    )
        ↓
    deterministic authorized prefix
        ↓
    EvidenceBatch

Phase 3 - provider side effects:

    EvidenceBatch.calls
        ↓
    EvidenceExtractor.extract(EvidenceCall)
        ↓
    LLM

The EvidenceBatch exposes ``llm_calls_used`` so orchestration can persist one
aggregate usage delta before any workers are dispatched.

Workers never independently consult the whole-run budget. They receive an
EvidenceCall that already contains their individual permit.

This prevents future parallel evidence fan-out from oversubscribing the LLM
budget by authorizing several calls from the same stale BudgetUsage snapshot.

Evidence extraction is optional research and therefore must not consume the
protected finalization reserve.

Grounding remains atomic and fail-closed: every proposed excerpt is validated
before any trusted Evidence ID is created.
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

from research_agent.graph.budget import (
    BudgetAuthorization,
    BudgetPolicy,
    BudgetUsage,
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

    evidence: list[
        EvidenceCandidate
    ] = Field(
        default_factory=list,
    )


class EvidenceGroundingError(
    LLMResponseError
):
    """Raised when proposed evidence is not grounded in the webpage."""


IDFactory = Callable[[], str]


def _default_id_factory() -> str:
    """Create a trusted internal Evidence ID."""

    return f"ev_{uuid4().hex}"


def _validate_evidence_inputs(
    *,
    sub_question: SubQuestion,
    fetch_result: SourceFetchResult,
) -> None:
    """Validate trusted inputs before authorization or execution."""

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
            "Evidence can only be extracted "
            "from a successful fetch."
        )

    if fetch_result.page is None:
        raise ValueError(
            "Successful fetch result must contain a page."
        )


@dataclass(frozen=True)
class EvidenceRequest:
    """One validated evidence-extraction work item before budgeting.

    A nonblank webpage requires one optional-research LLM call.

    A blank webpage requires no LLM call and deterministically produces no
    Evidence.
    """

    sub_question: SubQuestion
    fetch_result: SourceFetchResult

    def __post_init__(
        self,
    ) -> None:
        _validate_evidence_inputs(
            sub_question=self.sub_question,
            fetch_result=self.fetch_result,
        )

    @property
    def requires_llm(
        self,
    ) -> bool:
        """Whether this request contains webpage text requiring extraction."""

        page = self.fetch_result.page

        if page is None:
            return False

        return bool(
            page.text.strip()
        )


@dataclass(frozen=True)
class EvidenceCall:
    """One worker-level evidence call after parent budget allocation.

    The contained authorization is a child permit derived from the parent
    EvidenceBatch allocation.

    EvidenceExtractor never consults BudgetPolicy itself.
    """

    request: EvidenceRequest
    authorization: BudgetAuthorization

    def __post_init__(
        self,
    ) -> None:
        if not isinstance(
            self.request,
            EvidenceRequest,
        ):
            raise TypeError(
                "request must be an EvidenceRequest object."
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
                "EvidenceCall authorization must "
                "be for llm_calls."
            )

        if (
            self.authorization.llm_purpose
            != "optional_research"
        ):
            raise ValueError(
                "EvidenceCall authorization must "
                "be for optional_research."
            )

        expected_requested = (
            1
            if self.request.requires_llm
            else 0
        )

        if (
            self.authorization.requested
            != expected_requested
        ):
            raise ValueError(
                "EvidenceCall requested LLM count "
                "does not match webpage content."
            )

        if self.request.requires_llm:
            if (
                self.authorization.authorized
                not in {
                    0,
                    1,
                }
            ):
                raise ValueError(
                    "EvidenceCall may authorize "
                    "either zero or one LLM call."
                )

        else:
            if (
                self.authorization.authorized
                != 0
            ):
                raise ValueError(
                    "Blank webpage content must not "
                    "authorize an LLM call."
                )

    @property
    def sub_question(
        self,
    ) -> SubQuestion:
        """Return the trusted SubQuestion for this worker call."""

        return self.request.sub_question

    @property
    def fetch_result(
        self,
    ) -> SourceFetchResult:
        """Return the successful fetch result for this worker call."""

        return self.request.fetch_result

    @property
    def requires_llm(
        self,
    ) -> bool:
        """Whether this worker call requires an LLM request."""

        return self.request.requires_llm

    @property
    def authorized(
        self,
    ) -> bool:
        """Whether this worker may execute its LLM request."""

        return (
            self.requires_llm
            and self.authorization.authorized == 1
        )

    @property
    def skipped(
        self,
    ) -> int:
        """Number of LLM calls skipped for this individual work item."""

        return self.authorization.skipped


@dataclass(frozen=True)
class EvidenceBatch:
    """One centrally authorized collection of evidence-extraction work.

    ``authorization`` is the only whole-run budget decision for the batch.

    Orchestration must persist ``llm_calls_used`` once before dispatching
    workers. Individual EvidenceCall objects must not independently increment
    the global LLM counter.
    """

    calls: tuple[
        EvidenceCall,
        ...
    ]

    authorization: BudgetAuthorization

    def __post_init__(
        self,
    ) -> None:
        if not isinstance(
            self.calls,
            tuple,
        ):
            raise TypeError(
                "calls must be a tuple."
            )

        for call in self.calls:
            if not isinstance(
                call,
                EvidenceCall,
            ):
                raise TypeError(
                    "calls must contain only "
                    "EvidenceCall objects."
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
                "EvidenceBatch authorization must "
                "be for llm_calls."
            )

        if (
            self.authorization.llm_purpose
            != "optional_research"
        ):
            raise ValueError(
                "EvidenceBatch authorization must "
                "be for optional_research."
            )

        expected_requested = sum(
            1
            for call in self.calls
            if call.requires_llm
        )

        if (
            self.authorization.requested
            != expected_requested
        ):
            raise ValueError(
                "EvidenceBatch requested LLM count "
                "does not match its calls."
            )

        expected_authorized = sum(
            1
            for call in self.calls
            if call.authorized
        )

        if (
            self.authorization.authorized
            != expected_authorized
        ):
            raise ValueError(
                "EvidenceBatch authorized LLM count "
                "does not match its worker permits."
            )

        # Budget allocation must preserve deterministic input order.
        # Once a nonblank work item is denied, no later nonblank work item
        # may be authorized.
        blocked_nonblank_seen = False

        for call in self.calls:
            if not call.requires_llm:
                continue

            if not call.authorized:
                blocked_nonblank_seen = True
                continue

            if blocked_nonblank_seen:
                raise ValueError(
                    "EvidenceBatch worker permits must "
                    "form a deterministic prefix."
                )

    @property
    def llm_calls_used(
        self,
    ) -> int:
        """Aggregate usage delta to persist once before fan-out."""

        return self.authorization.authorized

    @property
    def requested(
        self,
    ) -> int:
        """Number of nonblank evidence requests requiring LLM work."""

        return self.authorization.requested

    @property
    def authorized(
        self,
    ) -> int:
        """Number of evidence LLM calls authorized for execution."""

        return self.authorization.authorized

    @property
    def skipped(
        self,
    ) -> int:
        """Number of requested evidence LLM calls denied by the budget."""

        return self.authorization.skipped


def prepare_evidence_batch(
    *,
    requests: list[EvidenceRequest],
    usage: BudgetUsage,
    budget_policy: BudgetPolicy,
) -> EvidenceBatch:
    """Centrally authorize a deterministic batch of evidence work.

    The whole batch consults BudgetPolicy exactly once.

    Only nonblank webpages request LLM capacity.

    When capacity is partial, the first N nonblank requests in caller-supplied
    order receive worker permits. Later nonblank requests receive blocked
    permits.

    Blank webpages always receive zero-call permits and cost no LLM budget.

    This function performs no provider side effect and does not mutate its
    inputs.
    """

    if not isinstance(
        requests,
        list,
    ):
        raise TypeError(
            "requests must be a list."
        )

    for request in requests:
        if not isinstance(
            request,
            EvidenceRequest,
        ):
            raise TypeError(
                "requests must contain only "
                "EvidenceRequest objects."
            )

    if not isinstance(
        budget_policy,
        BudgetPolicy,
    ):
        raise TypeError(
            "budget_policy must be a BudgetPolicy object."
        )

    requested = sum(
        1
        for request in requests
        if request.requires_llm
    )

    authorization = (
        budget_policy.authorize_llm_calls(
            usage=usage,
            requested=requested,
            purpose="optional_research",
        )
    )

    permits_remaining = (
        authorization.authorized
    )

    calls: list[
        EvidenceCall
    ] = []

    for request in requests:
        if not request.requires_llm:
            child_authorization = (
                BudgetAuthorization(
                    resource="llm_calls",
                    requested=0,
                    authorized=0,
                    llm_purpose=(
                        "optional_research"
                    ),
                )
            )

        elif permits_remaining > 0:
            child_authorization = (
                BudgetAuthorization(
                    resource="llm_calls",
                    requested=1,
                    authorized=1,
                    llm_purpose=(
                        "optional_research"
                    ),
                )
            )

            permits_remaining -= 1

        else:
            child_authorization = (
                BudgetAuthorization(
                    resource="llm_calls",
                    requested=1,
                    authorized=0,
                    reason=(
                        authorization.reason
                        or "optional evidence LLM "
                        "budget not authorized"
                    ),
                    llm_purpose=(
                        "optional_research"
                    ),
                )
            )

        calls.append(
            EvidenceCall(
                request=request,
                authorization=(
                    child_authorization
                ),
            )
        )

    return EvidenceBatch(
        calls=tuple(
            calls
        ),
        authorization=authorization,
    )


class EvidenceExtractor:
    """Extract grounded Evidence from one parent-authorized worker call."""

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
        call: EvidenceCall,
    ) -> list[Evidence]:
        """Execute one already-authorized evidence-extraction operation."""

        if not isinstance(
            call,
            EvidenceCall,
        ):
            raise TypeError(
                "call must be an EvidenceCall object."
            )

        page = (
            call.fetch_result.page
        )

        if page is None:
            raise ValueError(
                "Successful fetch result must contain a page."
            )

        # A successfully fetched page can contain no meaningful text.
        # This is deterministic and costs zero LLM calls.
        if not call.requires_llm:
            return []

        # Budget exhaustion is expected control flow.
        if not call.authorized:
            return []

        user_prompt = (
            build_evidence_user_prompt(
                sub_question=(
                    call.sub_question.question
                ),
                webpage_text=page.text,
            )
        )

        response = (
            self._llm.generate_structured(
                system_prompt=(
                    EVIDENCE_SYSTEM_PROMPT
                ),
                user_prompt=user_prompt,
                response_model=(
                    EvidenceResponse
                ),
            )
        )

        if not isinstance(
            response,
            EvidenceResponse,
        ):
            raise LLMResponseError(
                "Evidence LLM returned an "
                "unexpected response type."
            )

        # Atomic fail-closed boundary:
        # validate every candidate before generating trusted IDs.
        self._validate_grounding(
            response=response,
            webpage_text=page.text,
        )

        generated_ids: set[
            str
        ] = set()

        evidence_items: list[
            Evidence
        ] = []

        for candidate in response.evidence:
            evidence_id = (
                self._id_factory()
            )

            if not isinstance(
                evidence_id,
                str,
            ):
                raise RuntimeError(
                    "Evidence ID factory must "
                    "return a string."
                )

            evidence_id = (
                evidence_id.strip()
            )

            if not evidence_id:
                raise RuntimeError(
                    "Evidence ID factory returned "
                    "a blank ID."
                )

            if (
                evidence_id
                in generated_ids
            ):
                raise RuntimeError(
                    "Evidence ID factory returned "
                    "a duplicate ID."
                )

            generated_ids.add(
                evidence_id
            )

            evidence_items.append(
                Evidence(
                    id=evidence_id,
                    source_id=(
                        call.fetch_result.source.id
                    ),
                    sub_question_id=(
                        call.sub_question.id
                    ),
                    excerpt=(
                        candidate.excerpt
                    ),
                    relevance_note=(
                        candidate.relevance_note
                    ),
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

        seen_excerpts: set[
            str
        ] = set()

        for candidate in response.evidence:
            excerpt = (
                candidate.excerpt
            )

            if (
                excerpt
                not in webpage_text
            ):
                raise EvidenceGroundingError(
                    "LLM proposed an evidence "
                    "excerpt that does not exist "
                    "verbatim in the fetched webpage."
                )

            if excerpt in seen_excerpts:
                raise LLMResponseError(
                    "Evidence LLM returned "
                    "duplicate excerpts."
                )

            seen_excerpts.add(
                excerpt
            )