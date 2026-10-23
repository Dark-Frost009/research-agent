"""Tests for recording WHY a synthesized answer was rejected.

A rejected answer is discarded; only a fixed code from
``research_agent.graph.rejection`` is kept. These tests check that each
failing check produces the right code, that codes can only come from the fixed
list, and that the run summary stores the code without any model text.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from research_agent.graph.nodes.synthesis import (
    SynthesisResponse,
    SynthesisValidationError,
    Synthesizer,
)
from research_agent.graph.nodes.synthesis_verifier import (
    SynthesisVerificationError,
    SynthesisVerifier,
)
from research_agent.graph.rejection import (
    REJECTION_REASONS,
    require_rejection_reason,
)
from research_agent.run_summary import BudgetCounter, RunSummary
from test_synthesis import FakeSynthesisLLM, _evidence, _synthesize
from test_synthesis_verifier import (
    FakeLLM,
    _claim,
    _response,
    _support,
    _verdict,
    _verify,
)


# ---------------------------------------------------------------------------
# The fixed list of codes
# ---------------------------------------------------------------------------


def test_the_codes_are_exactly_the_documented_five():
    assert REJECTION_REASONS == {
        "quote_mismatch",
        "synthesis_format",
        "verifier_format",
        "unsupported_claim",
        "incomplete_coverage",
    }


@pytest.mark.parametrize("bad", ["", "QUOTE_MISMATCH", "free text from a model", None, 3])
def test_unknown_codes_are_a_programming_error(bad):
    with pytest.raises(ValueError):
        require_rejection_reason(bad)


@pytest.mark.parametrize("error_type", [SynthesisValidationError, SynthesisVerificationError])
def test_a_rejection_error_must_carry_a_known_code(error_type):
    # Forgetting the code fails at the raise site, not silently later.
    with pytest.raises(TypeError):
        error_type("message")

    with pytest.raises(ValueError):
        error_type("message", reason="made_up")

    error = error_type("message", reason="quote_mismatch")

    assert error.reason == "quote_mismatch"
    assert str(error) == "message"


# ---------------------------------------------------------------------------
# Deterministic synthesis checks
# ---------------------------------------------------------------------------


def _reject_synthesis(response: SynthesisResponse, excerpt: str) -> SynthesisValidationError:
    synthesizer = Synthesizer(llm=FakeSynthesisLLM(response))

    with pytest.raises(SynthesisValidationError) as caught:
        _synthesize(
            synthesizer,
            original_question="What happened?",
            evidence=[_evidence(id="ev_one", excerpt=excerpt)],
        )

    return caught.value


def _one_claim(content: str, claim: str, handle: str, quote: str) -> SynthesisResponse:
    return SynthesisResponse(
        content=content,
        claims=[
            {
                "claim_text": claim,
                "evidence_support": [
                    {"evidence_handle": handle, "supporting_quote": quote}
                ],
            }
        ],
    )


def test_a_quote_missing_from_its_evidence_is_quote_mismatch():
    error = _reject_synthesis(
        _one_claim("Answer.", "Answer.", "E1", "A sentence the page never said."),
        excerpt="The page said something else.",
    )

    assert error.reason == "quote_mismatch"


def test_an_unknown_evidence_handle_is_synthesis_format():
    error = _reject_synthesis(
        _one_claim("Answer.", "Answer.", "E99", "Grounded evidence."),
        excerpt="Grounded evidence.",
    )

    assert error.reason == "synthesis_format"


def test_a_claim_missing_from_the_answer_text_is_synthesis_format():
    error = _reject_synthesis(
        _one_claim("Different answer.", "Answer.", "E1", "Grounded evidence."),
        excerpt="Grounded evidence.",
    )

    assert error.reason == "synthesis_format"


# ---------------------------------------------------------------------------
# Semantic verification
# ---------------------------------------------------------------------------


def _reject_verification(response, *, content, claims) -> SynthesisVerificationError:
    verifier = SynthesisVerifier(llm=FakeLLM(response))

    with pytest.raises(SynthesisVerificationError) as caught:
        _verify(verifier, content=content, claims=claims)

    return caught.value


def test_a_malformed_verifier_reply_is_verifier_format():
    error = _reject_verification(
        _response(claim_verdicts=[_verdict(handle="C99")]),
        content="Claim one.",
        claims=[_claim(text="Claim one.")],
    )

    assert error.reason == "verifier_format"


def test_an_unexpected_verifier_reply_type_is_verifier_format():
    error = _reject_verification(
        "not a verification response",
        content="Claim one.",
        claims=[_claim(text="Claim one.")],
    )

    assert error.reason == "verifier_format"


def test_a_claim_judged_unsupported_is_unsupported_claim():
    claim = "The treatment reduced mortality by 40%."
    error = _reject_verification(
        _response(claim_verdicts=[_verdict(supported=False, reason="Does not follow.")]),
        content=claim,
        claims=[_claim(text=claim, support=[_support(quote="The study enrolled 500 participants.")])],
    )

    assert error.reason == "unsupported_claim"


def test_uncited_factual_content_is_incomplete_coverage():
    uncovered = "The treatment reduced mortality by 40%."
    cited = "The study enrolled 500 participants."
    error = _reject_verification(
        _response(coverage_complete=False, uncovered_factual_claims=[uncovered]),
        content=f"{uncovered} {cited}",
        claims=[_claim(text=cited, support=[_support(quote=cited)])],
    )

    assert error.reason == "incomplete_coverage"


def test_unsupported_wins_when_a_reply_has_both_problems():
    # The first failing check decides the code (documented in rejection.py).
    uncovered = "The treatment reduced mortality by 40%."
    cited = "The study enrolled 500 participants."
    error = _reject_verification(
        _response(
            coverage_complete=False,
            uncovered_factual_claims=[uncovered],
            claim_verdicts=[_verdict(supported=False)],
        ),
        content=f"{uncovered} {cited}",
        claims=[_claim(text=cited, support=[_support(quote=cited)])],
    )

    assert error.reason == "unsupported_claim"


# ---------------------------------------------------------------------------
# Run summary
# ---------------------------------------------------------------------------


def _summary(**changes) -> RunSummary:
    counter = BudgetCounter(used=1, limit=2)
    values = dict(
        rounds=counter,
        searches=counter,
        sources=counter,
        page_reads=counter,
        ai_calls=counter,
        end_reason="sufficient",
        outcome="verification_rejected",
    )
    values.update(changes)

    return RunSummary(**values)


def test_a_summary_saved_before_this_field_existed_still_loads():
    summary = _summary()

    assert summary.rejection_reason is None


def test_only_a_rejected_answer_may_have_a_reason():
    with pytest.raises(ValidationError):
        _summary(outcome="verified", rejection_reason="quote_mismatch")


def test_a_summary_rejects_codes_outside_the_fixed_list():
    with pytest.raises(ValidationError):
        _summary(rejection_reason="the model said it was wrong")


@pytest.mark.parametrize("reason", sorted(REJECTION_REASONS))
def test_every_code_has_a_plain_language_explanation(reason):
    plain = _summary().outcome_message
    explained = _summary(rejection_reason=reason).outcome_message

    assert explained.startswith(plain)
    assert len(explained) > len(plain)
    assert reason not in explained
