"""Grounded synthesis and deterministic citation construction.

The synthesis LLM sees only controlled evidence handles such as E1 and E2.
It never receives or generates trusted internal Evidence IDs.

Python is responsible for:

- validating every evidence handle
- validating that each supporting quote really comes from the specific
  Evidence excerpt referenced by that handle
- ensuring claims actually appear in the generated content
- rejecting duplicate claims and duplicate handles
- resolving controlled handles to trusted Evidence IDs
- assigning Citation IDs

A1.1 proves deterministic supporting-quote provenance.

A1.2 then performs a separate adversarial semantic-verification pass before
trusted Citation IDs or Citation objects are created. That pass checks:

- whether each declared claim is semantically supported by its grounded quotes
- whether the full synthesized answer contains undeclared factual assertions

Supporting quotes and verifier diagnostics are transient validation data.
They are never persisted inside trusted Citation domain objects.

No LLM-generated internal identifiers are trusted.

B1 adds atomic whole-run finalization budgeting.

For non-empty Evidence, synthesis and semantic verification are mandatory as
one two-call finalization bundle:

    prepare_finalization_call(...)
        ↓
    reserve exactly two finalization LLM calls
        ↓
    orchestration persists FinalizationCall.llm_calls_used
        ↓
    Synthesizer.synthesize(...)
        ↓
    synthesis LLM
        ↓
    deterministic A1.1 validation
        ↓
    semantic verifier
        ↓
    trusted Citation construction

A partial BudgetPolicy authorization of one out of two calls does not permit
either provider call. The pair is atomic.

For empty Evidence, zero LLM calls are required and a deterministic
insufficient-evidence response is returned.
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
from research_agent.graph.nodes.synthesis_verifier import (
    SynthesisVerifier,
    VerificationClaim,
    VerificationEvidenceSupport,
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
    assign_evidence_handles,
    build_evidence_catalog,
    build_synthesis_user_prompt,
)


_FINALIZATION_LLM_CALLS = 2


class _SynthesisSchema(BaseModel):
    """Strict base model for synthesis-generated structured output."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )


class ClaimEvidenceSupport(
    _SynthesisSchema
):
    """One evidence handle cited by a claim and its supporting quote.

    Both fields are proposed by the LLM.

    Python later verifies that:

    - evidence_handle refers to a real controlled Evidence handle
    - supporting_quote occurs verbatim inside that specific Evidence excerpt
    """

    evidence_handle: str = Field(
        min_length=1,
    )

    supporting_quote: str = Field(
        min_length=1,
    )


class SynthesizedClaim(
    _SynthesisSchema
):
    """One factual claim and its proposed evidence support."""

    claim_text: str = Field(
        min_length=1,
    )

    evidence_support: list[
        ClaimEvidenceSupport
    ] = Field(
        min_length=1,
    )


class SynthesisResponse(
    _SynthesisSchema
):
    """Structured draft returned by the synthesis LLM."""

    content: str = Field(
        min_length=1,
    )

    claims: list[
        SynthesizedClaim
    ] = Field(
        default_factory=list,
    )


class SynthesisValidationError(
    LLMResponseError
):
    """Raised when synthesis output violates grounding invariants."""


CitationIDFactory = Callable[
    [],
    str,
]


def _default_citation_id_factory() -> str:
    """Create a trusted internal Citation ID."""

    return f"cit_{uuid4().hex}"


def _validate_question(
    original_question: str,
) -> str:
    """Validate and normalize one research question."""

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

    return clean_question


def _validate_evidence_list(
    evidence: list[Evidence],
) -> None:
    """Validate trusted Evidence before any budget is reserved."""

    if not isinstance(
        evidence,
        list,
    ):
        raise TypeError(
            "evidence must be a list."
        )

    for item in evidence:
        if not isinstance(
            item,
            Evidence,
        ):
            raise TypeError(
                "evidence must contain only "
                "Evidence objects."
            )

    # Reuse the canonical handle assignment validator so duplicate trusted
    # Evidence IDs are rejected before finalization capacity is reserved.
    assign_evidence_handles(
        evidence
    )


@dataclass(frozen=True)
class FinalizationCall:
    """Prepared atomic synthesis + semantic-verification operation.

    This object contains the BudgetPolicy decision but performs no external
    side effect.

    For non-empty Evidence:

        authorization.requested == 2

    The underlying policy may partially authorize the request, for example
    one of two calls when only one slot remains.

    Such a partial authorization is intentionally NOT executable. Synthesis
    and verification form one atomic pair.

    Therefore ``llm_calls_used`` is:

    - 2 when the complete non-empty finalization bundle is authorized
    - 0 when the bundle is partial or denied
    - 0 for the deterministic zero-Evidence path

    Orchestration must persist ``llm_calls_used`` before calling
    ``Synthesizer.synthesize``.
    """

    original_question: str

    evidence: tuple[
        Evidence,
        ...
    ]

    authorization: BudgetAuthorization

    def __post_init__(
        self,
    ) -> None:
        _validate_question(
            self.original_question
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

            if (
                item.id
                in seen_evidence_ids
            ):
                raise ValueError(
                    "Duplicate Evidence IDs "
                    "are not allowed."
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
                "FinalizationCall authorization "
                "must be for llm_calls."
            )

        if (
            self.authorization.llm_purpose
            != "finalization"
        ):
            raise ValueError(
                "FinalizationCall authorization "
                "must be for finalization."
            )

        expected_requested = (
            _FINALIZATION_LLM_CALLS
            if self.evidence
            else 0
        )

        if (
            self.authorization.requested
            != expected_requested
        ):
            raise ValueError(
                "FinalizationCall requested LLM "
                "count does not match Evidence."
            )

        if not self.evidence:
            if (
                self.authorization.authorized
                != 0
            ):
                raise ValueError(
                    "Zero-Evidence finalization "
                    "must authorize zero LLM calls."
                )

        elif (
            self.authorization.authorized
            not in {
                0,
                1,
                2,
            }
        ):
            raise ValueError(
                "FinalizationCall may authorize "
                "zero, one, or two LLM calls."
            )

    @property
    def requires_llm(
        self,
    ) -> bool:
        """Whether finalization requires the two provider calls."""

        return bool(
            self.evidence
        )

    @property
    def fully_authorized(
        self,
    ) -> bool:
        """Whether the atomic finalization pair may execute."""

        if not self.requires_llm:
            return True

        return (
            self.authorization.authorized
            == _FINALIZATION_LLM_CALLS
        )

    @property
    def llm_calls_used(
        self,
    ) -> int:
        """Usage delta orchestration must persist before execution.

        Partial finalization authorization deliberately commits zero because
        neither member of the mandatory two-call pair may execute.
        """

        if (
            self.requires_llm
            and self.fully_authorized
        ):
            return _FINALIZATION_LLM_CALLS

        return 0

    @property
    def skipped(
        self,
    ) -> int:
        """Number of finalization calls not executable atomically."""

        if not self.requires_llm:
            return 0

        return (
            _FINALIZATION_LLM_CALLS
            - self.llm_calls_used
        )


def prepare_finalization_call(
    *,
    original_question: str,
    evidence: list[Evidence],
    usage: BudgetUsage,
    budget_policy: BudgetPolicy,
) -> FinalizationCall:
    """Authorize finalization without performing an LLM call.

    Non-empty Evidence always requests exactly two finalization calls:

    1. synthesis
    2. semantic verification

    Empty Evidence requests zero calls.

    A partial 1/2 policy authorization remains visible inside
    ``FinalizationCall.authorization`` for observability, but its committed
    ``llm_calls_used`` delta is zero because the pair is atomic.

    This function performs no provider side effect and mutates no inputs.
    """

    clean_question = (
        _validate_question(
            original_question
        )
    )

    _validate_evidence_list(
        evidence
    )

    if not isinstance(
        budget_policy,
        BudgetPolicy,
    ):
        raise TypeError(
            "budget_policy must be a "
            "BudgetPolicy object."
        )

    requested = (
        _FINALIZATION_LLM_CALLS
        if evidence
        else 0
    )

    authorization = (
        budget_policy.authorize_llm_calls(
            usage=usage,
            requested=requested,
            purpose="finalization",
        )
    )

    return FinalizationCall(
        original_question=clean_question,
        evidence=tuple(
            evidence
        ),
        authorization=authorization,
    )


@dataclass(frozen=True)
class SynthesisResult:
    """Persistent grounded and semantically verified synthesis output."""

    content: str
    citations: list[
        Citation
    ]


class Synthesizer:
    """Create grounded, verified content and trusted Citation objects."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        verifier: SynthesisVerifier
        | None = None,
        citation_id_factory: CitationIDFactory = (
            _default_citation_id_factory
        ),
    ) -> None:
        self._llm = llm

        self._verifier = (
            verifier
            if verifier is not None
            else SynthesisVerifier(
                llm=llm,
            )
        )

        self._citation_id_factory = (
            citation_id_factory
        )

    def synthesize(
        self,
        call: FinalizationCall,
    ) -> SynthesisResult:
        """Execute one already-prepared finalization operation.

        Budget authorization does not happen here.

        Orchestration must persist ``call.llm_calls_used`` before invoking
        this method.

        Zero Evidence returns a deterministic insufficient-evidence result and
        consumes zero LLM calls.

        Non-empty Evidence requires the complete atomic two-call reservation.
        If BudgetPolicy authorized fewer than two calls, neither provider call
        executes and a deterministic budget-unavailable result is returned.

        For an executable finalization:

        A1.1 first guarantees deterministic provenance relationships:

        - every evidence handle refers to real Evidence
        - every supporting quote occurs verbatim in the referenced excerpt
        - every declared claim appears verbatim somewhere in the content

        A1.2 then performs a separate adversarial semantic-verification pass:

        - every declared claim must be semantically supported
        - the complete answer must have factual-claim coverage

        Trusted Citation IDs are generated only after both layers succeed.
        """

        if not isinstance(
            call,
            FinalizationCall,
        ):
            raise TypeError(
                "call must be a "
                "FinalizationCall object."
            )

        evidence = list(
            call.evidence
        )

        # Deterministic zero-Evidence path.
        if not evidence:
            return SynthesisResult(
                content=(
                    "The available evidence is insufficient "
                    "to answer the research question."
                ),
                citations=[],
            )

        # Finalization is an atomic two-call operation.
        #
        # A partial policy authorization such as 1/2 must not allow synthesis
        # to consume that one call and strand mandatory semantic verification.
        if not call.fully_authorized:
            return SynthesisResult(
                content=(
                    "The research answer could not be finalized "
                    "within the available LLM budget."
                ),
                citations=[],
            )

        clean_question = (
            _validate_question(
                call.original_question
            )
        )

        # Canonical Python-controlled E1/E2/... assignment.
        #
        # This gives synthesis validation access to both the trusted
        # Evidence.id and the grounded Evidence.excerpt.
        evidence_lookup = (
            assign_evidence_handles(
                evidence
            )
        )

        evidence_catalog, _ = (
            build_evidence_catalog(
                evidence
            )
        )

        user_prompt = (
            build_synthesis_user_prompt(
                original_question=clean_question,
                evidence_catalog=evidence_catalog,
            )
        )

        response = (
            self._llm.generate_structured(
                system_prompt=(
                    SYNTHESIS_SYSTEM_PROMPT
                ),
                user_prompt=user_prompt,
                response_model=(
                    SynthesisResponse
                ),
            )
        )

        if not isinstance(
            response,
            SynthesisResponse,
        ):
            raise LLMResponseError(
                "Synthesis LLM returned an "
                "unexpected response type."
            )

        # A1.1:
        # Validate the complete synthesis response before generating trusted
        # Citation IDs or Citation objects.
        self._validate_response(
            response=response,
            evidence_lookup=evidence_lookup,
        )

        # Build a transient A1.2 view only after deterministic provenance
        # validation has succeeded.
        #
        # The semantic verifier sees controlled E handles and already-grounded
        # supporting quotes, but never trusted Evidence IDs, Source IDs,
        # Citation IDs, relevance notes, or raw webpage text.
        verification_claims = (
            self._build_verification_claims(
                response.claims
            )
        )

        # A1.2:
        #
        # Run the verifier even when response.claims is empty.
        #
        # This is intentional. The complete content still needs factual-claim
        # coverage checking so a synthesis model cannot bypass verification by
        # emitting factual content while declaring claims=[].
        #
        # The SAME parent two-call authorization is supplied to the verifier.
        # It does not perform a second budget decision.
        self._verifier.verify(
            content=response.content,
            claims=verification_claims,
            finalization_authorization=(
                call.authorization
            ),
        )

        # Trusted Citation IDs are generated only after BOTH:
        #
        # 1. deterministic A1.1 provenance validation
        # 2. adversarial A1.2 semantic verification
        #
        # Any failure before this point therefore creates zero trusted
        # Citation IDs and zero Citation objects.
        citation_ids = (
            self._generate_citation_ids(
                count=len(
                    response.claims
                ),
            )
        )

        citations: list[
            Citation
        ] = []

        for (
            citation_id,
            claim,
        ) in zip(
            citation_ids,
            response.claims,
            strict=True,
        ):
            evidence_ids = [
                evidence_lookup[
                    support.evidence_handle
                ].id
                for support
                in claim.evidence_support
            ]

            citations.append(
                Citation(
                    id=citation_id,
                    claim_text=(
                        claim.claim_text
                    ),
                    evidence_ids=(
                        evidence_ids
                    ),
                )
            )

        return SynthesisResult(
            content=response.content,
            citations=citations,
        )

    @staticmethod
    def _validate_response(
        *,
        response: SynthesisResponse,
        evidence_lookup: dict[
            str,
            Evidence,
        ],
    ) -> None:
        """Validate all deterministic synthesis relationships.

        This A1.1 validation is atomic: any invalid claim, handle, or
        supporting quote rejects the entire SynthesisResponse before A1.2
        verification or Citation-ID generation occurs.

        The supporting-quote check proves provenance only. It does not prove
        semantic entailment between the evidence and the claim.
        """

        seen_claims: set[
            str
        ] = set()

        for claim in response.claims:
            if (
                claim.claim_text
                not in response.content
            ):
                raise SynthesisValidationError(
                    "Synthesis claim_text does not "
                    "appear verbatim in the content field."
                )

            if (
                claim.claim_text
                in seen_claims
            ):
                raise SynthesisValidationError(
                    "Synthesis LLM returned "
                    "a duplicate claim."
                )

            seen_claims.add(
                claim.claim_text
            )

            seen_handles: set[
                str
            ] = set()

            for support in (
                claim.evidence_support
            ):
                handle = (
                    support.evidence_handle
                )

                if handle in seen_handles:
                    raise SynthesisValidationError(
                        "Synthesis claim contains "
                        "a duplicate evidence handle."
                    )

                if (
                    handle
                    not in evidence_lookup
                ):
                    raise SynthesisValidationError(
                        "Synthesis claim references "
                        "an unknown evidence handle."
                    )

                evidence_item = (
                    evidence_lookup[
                        handle
                    ]
                )

                if (
                    support.supporting_quote
                    not in evidence_item.excerpt
                ):
                    raise SynthesisValidationError(
                        "Synthesis supporting_quote "
                        "does not appear verbatim in "
                        "the Evidence excerpt referenced "
                        "by its evidence handle."
                    )

                seen_handles.add(
                    handle
                )

    @staticmethod
    def _build_verification_claims(
        claims: list[
            SynthesizedClaim
        ],
    ) -> list[
        VerificationClaim
    ]:
        """Create transient A1.2 verifier input from validated claims.

        The returned verifier objects deliberately contain only:

        - claim text
        - controlled evidence handles
        - already-grounded supporting quotes

        They do not expose:

        - trusted Evidence IDs
        - Source IDs
        - Citation IDs
        - relevance notes
        - raw webpage text
        """

        return [
            VerificationClaim(
                claim_text=(
                    claim.claim_text
                ),
                evidence_support=[
                    VerificationEvidenceSupport(
                        evidence_handle=(
                            support.evidence_handle
                        ),
                        supporting_quote=(
                            support.supporting_quote
                        ),
                    )
                    for support
                    in claim.evidence_support
                ],
            )
            for claim
            in claims
        ]

    def _generate_citation_ids(
        self,
        *,
        count: int,
    ) -> list[str]:
        """Generate unique trusted Citation IDs."""

        generated: list[
            str
        ] = []

        seen: set[
            str
        ] = set()

        for _ in range(
            count
        ):
            citation_id = (
                self._citation_id_factory()
            )

            if not isinstance(
                citation_id,
                str,
            ):
                raise RuntimeError(
                    "Citation ID factory must "
                    "return a string."
                )

            citation_id = (
                citation_id.strip()
            )

            if not citation_id:
                raise RuntimeError(
                    "Citation ID factory returned "
                    "a blank ID."
                )

            if (
                citation_id
                in seen
            ):
                raise RuntimeError(
                    "Citation ID factory returned "
                    "a duplicate ID."
                )

            seen.add(
                citation_id
            )

            generated.append(
                citation_id
            )

        return generated