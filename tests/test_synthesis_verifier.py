"""Tests for adversarial synthesis semantic verification.

These tests cover A1.2 in isolation.

The verifier is responsible for two probabilistic judgments:

1. semantic support of every declared claim
2. factual-claim coverage of the complete synthesized answer

Python then validates the verifier's structured output deterministically.

B1 additionally requires semantic verification to run only under an
already-reserved two-call finalization authorization.

These tests deliberately do not involve Citation construction. Integration
with Synthesizer is tested separately in tests/test_synthesis.py.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from research_agent.graph.budget import (
    BudgetAuthorization,
)
from research_agent.graph.nodes.synthesis_verifier import (
    ClaimSupportVerdict,
    SynthesisVerificationError,
    SynthesisVerificationResponse,
    SynthesisVerifier,
    VerificationClaim,
    VerificationEvidenceSupport,
)
from research_agent.llm.client import LLMResponseError
from research_agent.prompts.synthesis_verification import (
    build_synthesis_verification_prompt,
)


# ---------------------------------------------------------------------------
# Test doubles and helpers
# ---------------------------------------------------------------------------


class FakeLLM:
    """Minimal LLM test double returning one configured response."""

    def __init__(
        self,
        response: Any,
    ) -> None:
        self.response = response
        self.calls: list[
            dict[str, Any]
        ] = []

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[Any],
    ) -> Any:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "response_model": response_model,
            }
        )

        return self.response

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        raise AssertionError(
            "Semantic verification must use "
            "generate_structured()."
        )


def _support(
    *,
    handle: str = "E1",
    quote: str = (
        "Mortality was lower in the treatment group."
    ),
) -> VerificationEvidenceSupport:
    return VerificationEvidenceSupport(
        evidence_handle=handle,
        supporting_quote=quote,
    )


def _claim(
    *,
    text: str = (
        "Mortality was lower in the treatment group."
    ),
    support: list[
        VerificationEvidenceSupport
    ]
    | None = None,
) -> VerificationClaim:
    return VerificationClaim(
        claim_text=text,
        evidence_support=(
            support
            if support is not None
            else [
                _support()
            ]
        ),
    )


def _verdict(
    *,
    handle: str = "C1",
    supported: bool = True,
    reason: str | None = None,
) -> ClaimSupportVerdict:
    return ClaimSupportVerdict(
        claim_handle=handle,
        supported=supported,
        reason=reason,
    )


def _response(
    *,
    coverage_complete: bool = True,
    uncovered_factual_claims: list[str]
    | None = None,
    claim_verdicts: list[
        ClaimSupportVerdict
    ]
    | None = None,
) -> SynthesisVerificationResponse:
    return SynthesisVerificationResponse(
        coverage_complete=coverage_complete,
        uncovered_factual_claims=(
            uncovered_factual_claims
            if uncovered_factual_claims is not None
            else []
        ),
        claim_verdicts=(
            claim_verdicts
            if claim_verdicts is not None
            else [
                _verdict()
            ]
        ),
    )


def _finalization_authorization(
    *,
    requested: int = 2,
    authorized: int = 2,
    purpose: str = "finalization",
) -> BudgetAuthorization:
    return BudgetAuthorization(
        resource="llm_calls",
        requested=requested,
        authorized=authorized,
        reason=(
            None
            if requested == authorized
            else "finalization capacity limited"
        ),
        llm_purpose=purpose,
    )


def _verify(
    verifier: SynthesisVerifier,
    *,
    content: str,
    claims: list[VerificationClaim],
    authorization: BudgetAuthorization
    | None = None,
) -> SynthesisVerificationResponse:
    """Call verifier with the normal valid B1 reservation."""

    return verifier.verify(
        content=content,
        claims=claims,
        finalization_authorization=(
            authorization
            if authorization is not None
            else _finalization_authorization()
        ),
    )


# ---------------------------------------------------------------------------
# Verification models
# ---------------------------------------------------------------------------


def test_verification_evidence_support_accepts_valid_values():
    support = VerificationEvidenceSupport(
        evidence_handle="E1",
        supporting_quote=(
            "A real grounded quote."
        ),
    )

    assert support.evidence_handle == "E1"
    assert (
        support.supporting_quote
        == "A real grounded quote."
    )


def test_verification_evidence_support_strips_whitespace():
    support = VerificationEvidenceSupport(
        evidence_handle="  E1  ",
        supporting_quote=(
            "  A real grounded quote.  "
        ),
    )

    assert support.evidence_handle == "E1"

    assert (
        support.supporting_quote
        == "A real grounded quote."
    )


@pytest.mark.parametrize(
    "value",
    [
        "",
        " ",
        "   ",
        "\n",
        "\t",
    ],
)
def test_verification_evidence_handle_must_not_be_blank(
    value: str,
):
    with pytest.raises(
        ValidationError
    ):
        VerificationEvidenceSupport(
            evidence_handle=value,
            supporting_quote="Quote.",
        )


@pytest.mark.parametrize(
    "value",
    [
        "",
        " ",
        "   ",
        "\n",
        "\t",
    ],
)
def test_verification_supporting_quote_must_not_be_blank(
    value: str,
):
    with pytest.raises(
        ValidationError
    ):
        VerificationEvidenceSupport(
            evidence_handle="E1",
            supporting_quote=value,
        )


def test_verification_claim_requires_at_least_one_support_item():
    with pytest.raises(
        ValidationError
    ):
        VerificationClaim(
            claim_text="A claim.",
            evidence_support=[],
        )


@pytest.mark.parametrize(
    "value",
    [
        "",
        " ",
        "   ",
        "\n",
        "\t",
    ],
)
def test_verification_claim_text_must_not_be_blank(
    value: str,
):
    with pytest.raises(
        ValidationError
    ):
        VerificationClaim(
            claim_text=value,
            evidence_support=[
                _support()
            ],
        )


def test_claim_support_verdict_reason_is_optional():
    verdict = ClaimSupportVerdict(
        claim_handle="C1",
        supported=True,
    )

    assert verdict.reason is None


def test_verification_models_forbid_extra_fields():
    with pytest.raises(
        ValidationError
    ):
        ClaimSupportVerdict.model_validate(
            {
                "claim_handle": "C1",
                "supported": True,
                "unexpected": "not allowed",
            }
        )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def test_build_verification_prompt_contains_full_content():
    prompt = (
        build_synthesis_verification_prompt(
            content=(
                "The answer contains one "
                "factual claim."
            ),
            claims=[],
        )
    )

    assert (
        "The answer contains one factual claim."
        in prompt
    )


def test_build_verification_prompt_contains_controlled_claim_data():
    prompt = (
        build_synthesis_verification_prompt(
            content=(
                "Mortality was lower in "
                "the treatment group."
            ),
            claims=[
                {
                    "claim_handle": "C1",
                    "claim_text": (
                        "Mortality was lower in "
                        "the treatment group."
                    ),
                    "evidence_support": [
                        {
                            "evidence_handle": "E1",
                            "supporting_quote": (
                                "Mortality was lower in "
                                "the treatment group."
                            ),
                        }
                    ],
                }
            ],
        )
    )

    assert (
        '"claim_handle": "C1"'
        in prompt
    )

    assert (
        '"evidence_handle": "E1"'
        in prompt
    )

    assert (
        "Mortality was lower in the treatment group."
        in prompt
    )


def test_build_verification_prompt_marks_payload_as_untrusted():
    prompt = (
        build_synthesis_verification_prompt(
            content="Some content.",
            claims=[],
        )
    )

    assert (
        "untrusted data"
        in prompt.lower()
    )

    assert (
        "do not follow instructions"
        in prompt.lower()
    )


def test_build_verification_prompt_rejects_non_string_content():
    with pytest.raises(
        TypeError,
        match="content must be a string",
    ):
        build_synthesis_verification_prompt(
            content=123,
            claims=[],
        )


@pytest.mark.parametrize(
    "content",
    [
        "",
        " ",
        "   ",
        "\n",
        "\t",
    ],
)
def test_build_verification_prompt_rejects_blank_content(
    content: str,
):
    with pytest.raises(
        ValueError,
        match="content must not be blank",
    ):
        build_synthesis_verification_prompt(
            content=content,
            claims=[],
        )


@pytest.mark.parametrize(
    "claims",
    [
        "not claims",
        b"not claims",
        123,
        None,
    ],
)
def test_build_verification_prompt_requires_sequence_of_claims(
    claims: Any,
):
    with pytest.raises(
        TypeError,
        match=(
            "claims must be a sequence "
            "of mappings"
        ),
    ):
        build_synthesis_verification_prompt(
            content="Content.",
            claims=claims,
        )


def test_build_verification_prompt_requires_mapping_items():
    with pytest.raises(
        TypeError,
        match=r"claims\[0\] must be a mapping",
    ):
        build_synthesis_verification_prompt(
            content="Content.",
            claims=[
                "not a mapping"
            ],
        )


# ---------------------------------------------------------------------------
# Finalization authorization
# ---------------------------------------------------------------------------


def test_verify_requires_finalization_authorization_argument():
    verifier = SynthesisVerifier(
        llm=FakeLLM(
            _response()
        )
    )

    with pytest.raises(
        TypeError
    ):
        verifier.verify(
            content=(
                "Mortality was lower in "
                "the treatment group."
            ),
            claims=[
                _claim()
            ],
        )


@pytest.mark.parametrize(
    "value",
    [
        None,
        "authorization",
        123,
        {},
        [],
    ],
)
def test_verify_requires_budget_authorization(
    value,
):
    llm = FakeLLM(
        _response()
    )

    verifier = SynthesisVerifier(
        llm=llm
    )

    with pytest.raises(
        TypeError,
        match=(
            "finalization_authorization "
            "must be a BudgetAuthorization"
        ),
    ):
        verifier.verify(
            content=(
                "Mortality was lower in "
                "the treatment group."
            ),
            claims=[
                _claim()
            ],
            finalization_authorization=value,
        )

    assert llm.calls == []


def test_verify_rejects_non_llm_budget_resource():
    llm = FakeLLM(
        _response()
    )

    verifier = SynthesisVerifier(
        llm=llm
    )

    authorization = (
        BudgetAuthorization(
            resource="search_queries",
            requested=2,
            authorized=2,
        )
    )

    with pytest.raises(
        ValueError,
        match="for llm_calls",
    ):
        verifier.verify(
            content=(
                "Mortality was lower in "
                "the treatment group."
            ),
            claims=[
                _claim()
            ],
            finalization_authorization=(
                authorization
            ),
        )

    assert llm.calls == []


def test_verify_rejects_optional_research_authorization():
    llm = FakeLLM(
        _response()
    )

    verifier = SynthesisVerifier(
        llm=llm
    )

    authorization = (
        _finalization_authorization(
            purpose="optional_research",
        )
    )

    with pytest.raises(
        ValueError,
        match="for finalization",
    ):
        verifier.verify(
            content=(
                "Mortality was lower in "
                "the treatment group."
            ),
            claims=[
                _claim()
            ],
            finalization_authorization=(
                authorization
            ),
        )

    assert llm.calls == []


def test_verify_requires_exactly_two_requested_finalization_calls():
    llm = FakeLLM(
        _response()
    )

    verifier = SynthesisVerifier(
        llm=llm
    )

    authorization = (
        _finalization_authorization(
            requested=1,
            authorized=1,
        )
    )

    with pytest.raises(
        ValueError,
        match="exactly two LLM calls",
    ):
        verifier.verify(
            content=(
                "Mortality was lower in "
                "the treatment group."
            ),
            claims=[
                _claim()
            ],
            finalization_authorization=(
                authorization
            ),
        )

    assert llm.calls == []


def test_verify_rejects_partial_finalization_reservation():
    llm = FakeLLM(
        _response()
    )

    verifier = SynthesisVerifier(
        llm=llm
    )

    authorization = (
        _finalization_authorization(
            requested=2,
            authorized=1,
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "complete two-call "
            "finalization reservation"
        ),
    ):
        verifier.verify(
            content=(
                "Mortality was lower in "
                "the treatment group."
            ),
            claims=[
                _claim()
            ],
            finalization_authorization=(
                authorization
            ),
        )

    assert llm.calls == []


def test_valid_finalization_authorization_allows_verification():
    response = _response()

    llm = FakeLLM(
        response
    )

    verifier = SynthesisVerifier(
        llm=llm
    )

    result = _verify(
        verifier,
        content=(
            "Mortality was lower in "
            "the treatment group."
        ),
        claims=[
            _claim()
        ],
    )

    assert result is response
    assert len(llm.calls) == 1


# ---------------------------------------------------------------------------
# Successful verification
# ---------------------------------------------------------------------------


def test_verify_uses_generate_structured():
    response = _response()

    llm = FakeLLM(
        response
    )

    verifier = SynthesisVerifier(
        llm=llm
    )

    _verify(
        verifier,
        content=(
            "Mortality was lower in "
            "the treatment group."
        ),
        claims=[
            _claim()
        ],
    )

    assert len(
        llm.calls
    ) == 1


def test_verify_requests_synthesis_verification_response_model():
    response = _response()

    llm = FakeLLM(
        response
    )

    verifier = SynthesisVerifier(
        llm=llm
    )

    _verify(
        verifier,
        content=(
            "Mortality was lower in "
            "the treatment group."
        ),
        claims=[
            _claim()
        ],
    )

    assert (
        llm.calls[
            0
        ][
            "response_model"
        ]
        is SynthesisVerificationResponse
    )


def test_verify_assigns_python_controlled_claim_handle():
    response = _response()

    llm = FakeLLM(
        response
    )

    verifier = SynthesisVerifier(
        llm=llm
    )

    _verify(
        verifier,
        content=(
            "Mortality was lower in "
            "the treatment group."
        ),
        claims=[
            _claim()
        ],
    )

    prompt = (
        llm.calls[
            0
        ][
            "user_prompt"
        ]
    )

    assert (
        '"claim_handle": "C1"'
        in prompt
    )


def test_verify_assigns_multiple_claim_handles_in_order():
    response = _response(
        claim_verdicts=[
            _verdict(
                handle="C1",
            ),
            _verdict(
                handle="C2",
            ),
        ]
    )

    llm = FakeLLM(
        response
    )

    verifier = SynthesisVerifier(
        llm=llm
    )

    _verify(
        verifier,
        content=(
            "Claim one. Claim two."
        ),
        claims=[
            _claim(
                text="Claim one.",
            ),
            _claim(
                text="Claim two.",
                support=[
                    _support(
                        handle="E2",
                        quote="Quote two.",
                    )
                ],
            ),
        ],
    )

    prompt = (
        llm.calls[
            0
        ][
            "user_prompt"
        ]
    )

    assert (
        '"claim_handle": "C1"'
        in prompt
    )

    assert (
        '"claim_handle": "C2"'
        in prompt
    )


def test_verify_accepts_supported_claim_with_complete_coverage():
    response = _response(
        coverage_complete=True,
        claim_verdicts=[
            _verdict(
                handle="C1",
                supported=True,
                reason=(
                    "The quote directly "
                    "supports the claim."
                ),
            )
        ],
    )

    verifier = SynthesisVerifier(
        llm=FakeLLM(
            response
        )
    )

    result = _verify(
        verifier,
        content=(
            "Mortality was lower in "
            "the treatment group."
        ),
        claims=[
            _claim()
        ],
    )

    assert result is response


def test_verifier_reason_remains_transient_response_data():
    response = _response(
        claim_verdicts=[
            _verdict(
                reason=(
                    "Diagnostic semantic "
                    "explanation."
                ),
            )
        ]
    )

    verifier = SynthesisVerifier(
        llm=FakeLLM(
            response
        )
    )

    result = _verify(
        verifier,
        content=(
            "Mortality was lower in "
            "the treatment group."
        ),
        claims=[
            _claim()
        ],
    )

    assert (
        result.claim_verdicts[
            0
        ].reason
        == (
            "Diagnostic semantic "
            "explanation."
        )
    )


def test_empty_declared_claims_still_runs_verifier_for_coverage():
    response = (
        SynthesisVerificationResponse(
            coverage_complete=True,
            uncovered_factual_claims=[],
            claim_verdicts=[],
        )
    )

    llm = FakeLLM(
        response
    )

    verifier = SynthesisVerifier(
        llm=llm
    )

    result = _verify(
        verifier,
        content=(
            "The available evidence "
            "is inconclusive."
        ),
        claims=[],
    )

    assert result is response
    assert len(llm.calls) == 1

    assert (
        '"claims": []'
        in llm.calls[
            0
        ][
            "user_prompt"
        ]
    )


def test_verdict_order_does_not_need_to_match_claim_order():
    response = _response(
        claim_verdicts=[
            _verdict(
                handle="C2",
            ),
            _verdict(
                handle="C1",
            ),
        ]
    )

    verifier = SynthesisVerifier(
        llm=FakeLLM(
            response
        )
    )

    result = _verify(
        verifier,
        content=(
            "Claim one. Claim two."
        ),
        claims=[
            _claim(
                text="Claim one.",
            ),
            _claim(
                text="Claim two.",
            ),
        ],
    )

    assert result is response


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_verify_requires_string_content():
    verifier = SynthesisVerifier(
        llm=FakeLLM(
            _response()
        )
    )

    with pytest.raises(
        TypeError,
        match="content must be a string",
    ):
        _verify(
            verifier,
            content=123,
            claims=[
                _claim()
            ],
        )


@pytest.mark.parametrize(
    "content",
    [
        "",
        " ",
        "   ",
        "\n",
        "\t",
    ],
)
def test_verify_rejects_blank_content(
    content: str,
):
    verifier = SynthesisVerifier(
        llm=FakeLLM(
            _response()
        )
    )

    with pytest.raises(
        ValueError,
        match="content must not be blank",
    ):
        _verify(
            verifier,
            content=content,
            claims=[
                _claim()
            ],
        )


def test_verify_requires_claim_list():
    verifier = SynthesisVerifier(
        llm=FakeLLM(
            _response()
        )
    )

    with pytest.raises(
        TypeError,
        match="claims must be a list",
    ):
        _verify(
            verifier,
            content="Content.",
            claims=(
                _claim(),
            ),
        )


def test_verify_requires_verification_claim_objects():
    verifier = SynthesisVerifier(
        llm=FakeLLM(
            _response()
        )
    )

    with pytest.raises(
        TypeError,
        match=(
            "claims must contain only "
            "VerificationClaim objects"
        ),
    ):
        _verify(
            verifier,
            content="Content.",
            claims=[
                "not a claim"
            ],
        )


# ---------------------------------------------------------------------------
# Verifier response type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "invalid_response",
    [
        None,
        "not a response",
        123,
        {},
        [],
    ],
)
def test_verify_rejects_unexpected_response_type(
    invalid_response: Any,
):
    verifier = SynthesisVerifier(
        llm=FakeLLM(
            invalid_response
        )
    )

    with pytest.raises(
        LLMResponseError,
        match="unexpected response type",
    ):
        _verify(
            verifier,
            content=(
                "Mortality was lower in "
                "the treatment group."
            ),
            claims=[
                _claim()
            ],
        )


# ---------------------------------------------------------------------------
# Claim-handle invariants
# ---------------------------------------------------------------------------


def test_duplicate_claim_handle_is_rejected():
    response = _response(
        claim_verdicts=[
            _verdict(
                handle="C1",
            ),
            _verdict(
                handle="C1",
            ),
        ]
    )

    verifier = SynthesisVerifier(
        llm=FakeLLM(
            response
        )
    )

    with pytest.raises(
        SynthesisVerificationError,
        match="duplicate claim handle",
    ):
        _verify(
            verifier,
            content=(
                "Claim one. Claim two."
            ),
            claims=[
                _claim(
                    text="Claim one.",
                ),
                _claim(
                    text="Claim two.",
                ),
            ],
        )


def test_unknown_claim_handle_is_rejected():
    response = _response(
        claim_verdicts=[
            _verdict(
                handle="C99",
            )
        ]
    )

    verifier = SynthesisVerifier(
        llm=FakeLLM(
            response
        )
    )

    with pytest.raises(
        SynthesisVerificationError,
        match="unknown claim handle",
    ):
        _verify(
            verifier,
            content="Claim one.",
            claims=[
                _claim(
                    text="Claim one.",
                )
            ],
        )


def test_missing_claim_handle_is_rejected():
    response = _response(
        claim_verdicts=[
            _verdict(
                handle="C1",
            )
        ]
    )

    verifier = SynthesisVerifier(
        llm=FakeLLM(
            response
        )
    )

    with pytest.raises(
        SynthesisVerificationError,
        match=(
            "omitted one or more "
            "claim handles"
        ),
    ):
        _verify(
            verifier,
            content=(
                "Claim one. Claim two."
            ),
            claims=[
                _claim(
                    text="Claim one.",
                ),
                _claim(
                    text="Claim two.",
                ),
            ],
        )


def test_verdict_for_empty_expected_claim_set_is_rejected():
    response = (
        SynthesisVerificationResponse(
            coverage_complete=True,
            uncovered_factual_claims=[],
            claim_verdicts=[
                _verdict(
                    handle="C1",
                )
            ],
        )
    )

    verifier = SynthesisVerifier(
        llm=FakeLLM(
            response
        )
    )

    with pytest.raises(
        SynthesisVerificationError,
        match="unknown claim handle",
    ):
        _verify(
            verifier,
            content=(
                "The available evidence "
                "is inconclusive."
            ),
            claims=[],
        )


# ---------------------------------------------------------------------------
# Semantic-support rejection
# ---------------------------------------------------------------------------


def test_unsupported_claim_rejects_entire_verification():
    response = _response(
        coverage_complete=True,
        claim_verdicts=[
            _verdict(
                supported=False,
                reason=(
                    "The quote describes "
                    "enrollment and does not "
                    "support the mortality claim."
                ),
            )
        ],
    )

    verifier = SynthesisVerifier(
        llm=FakeLLM(
            response
        )
    )

    with pytest.raises(
        SynthesisVerificationError,
        match="semantically unsupported",
    ):
        _verify(
            verifier,
            content=(
                "The treatment reduced "
                "mortality by 40%."
            ),
            claims=[
                _claim(
                    text=(
                        "The treatment reduced "
                        "mortality by 40%."
                    ),
                    support=[
                        _support(
                            quote=(
                                "The study enrolled "
                                "500 participants."
                            ),
                        )
                    ],
                )
            ],
        )


def test_one_unsupported_claim_rejects_all_claims():
    response = _response(
        coverage_complete=True,
        claim_verdicts=[
            _verdict(
                handle="C1",
                supported=True,
            ),
            _verdict(
                handle="C2",
                supported=False,
                reason=(
                    "Unsupported specificity."
                ),
            ),
        ],
    )

    verifier = SynthesisVerifier(
        llm=FakeLLM(
            response
        )
    )

    with pytest.raises(
        SynthesisVerificationError,
        match="semantically unsupported",
    ):
        _verify(
            verifier,
            content=(
                "Claim one. Claim two."
            ),
            claims=[
                _claim(
                    text="Claim one.",
                ),
                _claim(
                    text="Claim two.",
                ),
            ],
        )


# ---------------------------------------------------------------------------
# Factual-coverage invariants
# ---------------------------------------------------------------------------


def test_complete_coverage_must_not_report_uncovered_claims():
    response = _response(
        coverage_complete=True,
        uncovered_factual_claims=[
            "Extra factual assertion.",
        ],
    )

    verifier = SynthesisVerifier(
        llm=FakeLLM(
            response
        )
    )

    with pytest.raises(
        SynthesisVerificationError,
        match="complete coverage",
    ):
        _verify(
            verifier,
            content=(
                "Mortality was lower in "
                "the treatment group. "
                "Extra factual assertion."
            ),
            claims=[
                _claim()
            ],
        )


def test_incomplete_coverage_requires_uncovered_claim():
    response = _response(
        coverage_complete=False,
        uncovered_factual_claims=[],
    )

    verifier = SynthesisVerifier(
        llm=FakeLLM(
            response
        )
    )

    with pytest.raises(
        SynthesisVerificationError,
        match="incomplete coverage",
    ):
        _verify(
            verifier,
            content=(
                "Mortality was lower in "
                "the treatment group."
            ),
            claims=[
                _claim()
            ],
        )


def test_blank_uncovered_factual_claim_is_rejected():
    response = _response(
        coverage_complete=False,
        uncovered_factual_claims=[
            "   ",
        ],
    )

    verifier = SynthesisVerifier(
        llm=FakeLLM(
            response
        )
    )

    with pytest.raises(
        SynthesisVerificationError,
        match="blank uncovered factual claim",
    ):
        _verify(
            verifier,
            content=(
                "Mortality was lower in "
                "the treatment group."
            ),
            claims=[
                _claim()
            ],
        )


def test_duplicate_uncovered_factual_claim_is_rejected():
    uncovered = (
        "The treatment reduced "
        "mortality by 40%."
    )

    response = _response(
        coverage_complete=False,
        uncovered_factual_claims=[
            uncovered,
            uncovered,
        ],
    )

    verifier = SynthesisVerifier(
        llm=FakeLLM(
            response
        )
    )

    with pytest.raises(
        SynthesisVerificationError,
        match=(
            "duplicate uncovered "
            "factual claim"
        ),
    ):
        _verify(
            verifier,
            content=(
                f"{uncovered} "
                "Mortality was lower in "
                "the treatment group."
            ),
            claims=[
                _claim()
            ],
        )


def test_hallucinated_uncovered_claim_is_rejected():
    response = _response(
        coverage_complete=False,
        uncovered_factual_claims=[
            (
                "The company earned "
                "$900 billion."
            ),
        ],
    )

    verifier = SynthesisVerifier(
        llm=FakeLLM(
            response
        )
    )

    with pytest.raises(
        SynthesisVerificationError,
        match="does not appear verbatim",
    ):
        _verify(
            verifier,
            content=(
                "Mortality was lower in "
                "the treatment group."
            ),
            claims=[
                _claim()
            ],
        )


def test_real_uncovered_factual_claim_rejects_verification():
    uncovered = (
        "The treatment reduced "
        "mortality by 40%."
    )

    response = _response(
        coverage_complete=False,
        uncovered_factual_claims=[
            uncovered,
        ],
    )

    verifier = SynthesisVerifier(
        llm=FakeLLM(
            response
        )
    )

    with pytest.raises(
        SynthesisVerificationError,
        match=(
            "factual claim coverage "
            "is incomplete"
        ),
    ):
        _verify(
            verifier,
            content=(
                f"{uncovered} "
                "The study enrolled "
                "500 participants."
            ),
            claims=[
                _claim(
                    text=(
                        "The study enrolled "
                        "500 participants."
                    ),
                    support=[
                        _support(
                            quote=(
                                "The study enrolled "
                                "500 participants."
                            )
                        )
                    ],
                )
            ],
        )


def test_stronger_content_can_be_reported_as_uncovered():
    stronger_claim = (
        "The treatment reduced "
        "mortality by 40%."
    )

    response = _response(
        coverage_complete=False,
        uncovered_factual_claims=[
            stronger_claim,
        ],
    )

    verifier = SynthesisVerifier(
        llm=FakeLLM(
            response
        )
    )

    with pytest.raises(
        SynthesisVerificationError,
        match=(
            "factual claim coverage "
            "is incomplete"
        ),
    ):
        _verify(
            verifier,
            content=stronger_claim,
            claims=[
                _claim(
                    text=(
                        "The treatment "
                        "reduced mortality"
                    ),
                    support=[
                        _support(
                            quote=(
                                "Mortality was lower "
                                "in the treatment group."
                            )
                        )
                    ],
                )
            ],
        )


def test_empty_claim_list_with_uncovered_fact_is_rejected():
    uncovered = (
        "The treatment reduced "
        "mortality by 40%."
    )

    response = (
        SynthesisVerificationResponse(
            coverage_complete=False,
            uncovered_factual_claims=[
                uncovered,
            ],
            claim_verdicts=[],
        )
    )

    verifier = SynthesisVerifier(
        llm=FakeLLM(
            response
        )
    )

    with pytest.raises(
        SynthesisVerificationError,
        match=(
            "factual claim coverage "
            "is incomplete"
        ),
    ):
        _verify(
            verifier,
            content=uncovered,
            claims=[],
        )