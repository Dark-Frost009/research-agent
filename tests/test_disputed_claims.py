"""Tests for marking claims that other evidence contradicts (phase C1).

The verifier now also sees every evidence excerpt, by its controlled E handle,
and may name excerpts that directly contradict a claim. Python checks each
named handle; a contradicted claim is kept but recorded as disputed on its
Citation (``disputed_by``). These tests use the CUBIC case from live
evaluation: one source says "binary search increase", another contradicts it.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from research_agent.graph.nodes.synthesis import (
    SynthesisResponse,
    Synthesizer,
)
from research_agent.graph.nodes.synthesis_verifier import (
    ClaimSupportVerdict,
    SynthesisVerificationError,
    SynthesisVerificationResponse,
    SynthesisVerifier,
)
from research_agent.models.schemas import Citation, Evidence, ResearchReport, Source
from research_agent.prompts.synthesis_verification import (
    build_synthesis_verification_prompt,
)
from test_synthesis import FakeSynthesisLLM, _evidence, _synthesize
from test_synthesis_verifier import (
    FakeLLM,
    _claim,
    _finalization_authorization,
    _response,
    _support,
    _verdict,
    _verify,
)

BLOG = "Cubic implements a binary search increase which can reach the available bandwidth much faster than Reno."
WIKI = "CUBIC replaced the binary search increase of its predecessor BIC with a cubic function of time."
CLAIM = "Cubic implements a binary search increase."


def _payload(prompt: str) -> dict:
    return json.loads(prompt.split("UNTRUSTED VERIFICATION PAYLOAD:", 1)[1])


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def test_the_prompt_carries_the_whole_catalog_as_untrusted_data():
    prompt = build_synthesis_verification_prompt(
        content="An answer.",
        claims=[],
        evidence_catalog={"E1": BLOG, "E2": WIKI},
    )

    assert _payload(prompt)["evidence_catalog"] == [
        {"evidence_handle": "E1", "excerpt": BLOG},
        {"evidence_handle": "E2", "excerpt": WIKI},
    ]
    assert "3. CONTRADICTION" in prompt
    assert "catalog excerpts" in prompt.split("IMPORTANT SECURITY")[1]


def test_without_a_catalog_the_prompt_has_an_empty_one():
    prompt = build_synthesis_verification_prompt(content="An answer.", claims=[])

    assert _payload(prompt)["evidence_catalog"] == []


@pytest.mark.parametrize("catalog", [["E1"], {"E1": 3}, {1: "x"}])
def test_a_malformed_catalog_is_a_programming_error(catalog):
    with pytest.raises(TypeError):
        build_synthesis_verification_prompt(
            content="An answer.",
            claims=[],
            evidence_catalog=catalog,
        )


# ---------------------------------------------------------------------------
# Verifier: deterministic checks on contradicted_by
# ---------------------------------------------------------------------------


def _verdict_for_c1(*, supported: bool = True, contradicted_by=None) -> ClaimSupportVerdict:
    return ClaimSupportVerdict(
        claim_handle="C1",
        supported=supported,
        contradicted_by=contradicted_by or [],
    )


def _verify_with(verdict: ClaimSupportVerdict, *, catalog):
    verifier = SynthesisVerifier(llm=FakeLLM(_response(claim_verdicts=[verdict])))

    return verifier.verify(
        content=CLAIM,
        claims=[_claim(text=CLAIM, support=[_support(handle="E1", quote="binary search increase")])],
        finalization_authorization=_finalization_authorization(),
        evidence_catalog=catalog,
    )


def test_a_valid_contradiction_is_accepted_and_returned():
    response = _verify_with(
        _verdict_for_c1(contradicted_by=["E2"]),
        catalog={"E1": BLOG, "E2": WIKI},
    )

    assert response.claim_verdicts[0].contradicted_by == ["E2"]


def test_the_verifier_sends_the_catalog_in_its_prompt():
    llm = FakeLLM(_response(claim_verdicts=[_verdict()]))
    SynthesisVerifier(llm=llm).verify(
        content=CLAIM,
        claims=[_claim(text=CLAIM, support=[_support(handle="E1", quote="binary search increase")])],
        finalization_authorization=_finalization_authorization(),
        evidence_catalog={"E1": BLOG, "E2": WIKI},
    )

    catalog = _payload(llm.calls[0]["user_prompt"])["evidence_catalog"]

    assert [item["evidence_handle"] for item in catalog] == ["E1", "E2"]


@pytest.mark.parametrize(
    "handles, catalog",
    [
        (["E9"], {"E1": BLOG, "E2": WIKI}),  # not in the catalog
        (["E1"], {"E1": BLOG, "E2": WIKI}),  # the claim's own evidence
        (["E2", "E2"], {"E1": BLOG, "E2": WIKI}),  # repeated
        (["E2"], None),  # no catalog was given at all
        ([""], {"E1": BLOG, "E2": WIKI}),  # blank
    ],
)
def test_an_invalid_contradiction_rejects_the_reply_as_malformed(handles, catalog):
    with pytest.raises(SynthesisVerificationError) as caught:
        _verify_with(_verdict_for_c1(contradicted_by=handles), catalog=catalog)

    assert caught.value.reason == "verifier_format"


def test_support_is_still_judged_first():
    # An unsupported claim is rejected as before, whatever contradicted_by says.
    with pytest.raises(SynthesisVerificationError) as caught:
        _verify_with(
            _verdict_for_c1(supported=False, contradicted_by=["E2"]),
            catalog={"E1": BLOG, "E2": WIKI},
        )

    assert caught.value.reason == "unsupported_claim"


@pytest.mark.parametrize("catalog", [{"E1": " "}, {" ": BLOG}])
def test_a_blank_catalog_entry_is_a_programming_error(catalog):
    with pytest.raises(ValueError):
        _verify_with(_verdict(), catalog=catalog)


def test_existing_callers_without_a_catalog_still_work():
    response = _verify(
        SynthesisVerifier(llm=FakeLLM(_response())),
        content="Mortality was lower in the treatment group.",
        claims=[_claim()],
    )

    assert response.claim_verdicts[0].contradicted_by == []


# ---------------------------------------------------------------------------
# Synthesis: a disputed claim is kept and recorded
# ---------------------------------------------------------------------------


def _cubic_synthesis(verification: SynthesisVerificationResponse):
    response = SynthesisResponse(
        content=f"{CLAIM} CUBIC grows its window as a cubic function of time.",
        claims=[
            {
                "claim_text": CLAIM,
                "evidence_support": [
                    {"evidence_handle": "E1", "supporting_quote": "binary search increase"}
                ],
            },
            {
                "claim_text": "CUBIC grows its window as a cubic function of time.",
                "evidence_support": [
                    {"evidence_handle": "E2", "supporting_quote": "a cubic function of time"}
                ],
            },
        ],
    )
    llm = FakeSynthesisLLM(response, verification_response=verification)
    result = _synthesize(
        Synthesizer(llm=llm),
        original_question="How does CUBIC differ from Reno?",
        evidence=[
            _evidence(id="ev_blog", source_id="src_blog", excerpt=BLOG),
            _evidence(id="ev_wiki", source_id="src_wiki", excerpt=WIKI),
        ],
    )

    return result, llm


def test_a_contradicted_claim_is_kept_and_marked_disputed():
    verification = SynthesisVerificationResponse(
        coverage_complete=True,
        claim_verdicts=[
            ClaimSupportVerdict(claim_handle="C1", supported=True, contradicted_by=["E2"]),
            ClaimSupportVerdict(claim_handle="C2", supported=True),
        ],
    )

    result, llm = _cubic_synthesis(verification)
    disputed, undisputed = result.citations

    # Kept, with its own support unchanged...
    assert disputed.claim_text == CLAIM
    assert disputed.evidence_ids == ["ev_blog"]
    # ...and the contradicting evidence recorded by its trusted Evidence ID.
    assert disputed.disputed_by == ["ev_wiki"]
    assert undisputed.disputed_by == []
    # The verifier saw both excerpts.
    catalog = _payload(llm.calls[-1]["user_prompt"])["evidence_catalog"]
    assert {item["excerpt"] for item in catalog} == {BLOG, WIKI}


def test_verdicts_are_matched_to_claims_by_handle_not_by_order():
    verification = SynthesisVerificationResponse(
        coverage_complete=True,
        claim_verdicts=[
            ClaimSupportVerdict(claim_handle="C2", supported=True),
            ClaimSupportVerdict(claim_handle="C1", supported=True, contradicted_by=["E2"]),
        ],
    )

    result, _ = _cubic_synthesis(verification)

    assert [citation.disputed_by for citation in result.citations] == [["ev_wiki"], []]


def test_a_contradiction_never_hides_or_rewrites_the_answer():
    verification = SynthesisVerificationResponse(
        coverage_complete=True,
        claim_verdicts=[
            ClaimSupportVerdict(claim_handle="C1", supported=True, contradicted_by=["E2"]),
            ClaimSupportVerdict(claim_handle="C2", supported=True),
        ],
    )

    result, _ = _cubic_synthesis(verification)

    assert CLAIM in result.content
    assert len(result.citations) == 2


# ---------------------------------------------------------------------------
# Domain model and saved reports
# ---------------------------------------------------------------------------


def test_citations_saved_before_this_field_existed_still_load():
    citation = Citation.model_validate(
        {"id": "cit_1", "claim_text": "A claim.", "evidence_ids": ["ev_1"]}
    )

    assert citation.disputed_by == []


def test_disputed_by_is_exported_with_the_citation():
    citation = Citation(id="cit_1", claim_text="A claim.", evidence_ids=["ev_1"], disputed_by=["ev_2"])

    assert citation.model_dump(mode="json")["disputed_by"] == ["ev_2"]


def _saved(disputed_by: list[str]):
    # The model that validates a report when it is saved to or read from history.
    from research_agent.history import _Snapshot

    source = Source(id="src_1", url="https://example.com/a", domain="example.com", fetch_status="success")
    evidence = [
        Evidence(id="ev_1", source_id="src_1", excerpt="One excerpt."),
        Evidence(id="ev_2", source_id="src_1", excerpt="Another excerpt."),
    ]
    report = ResearchReport(
        question="Q?",
        content="A claim.",
        citations=[
            Citation(id="cit_1", claim_text="A claim.", evidence_ids=["ev_1"], disputed_by=disputed_by)
        ],
    )

    return _Snapshot(
        report=report,
        evidence=evidence,
        sources=[source],
        iterations=1,
        searches=1,
        warning_count=0,
    )


def test_a_saved_report_may_record_a_dispute_with_real_evidence():
    assert _saved(["ev_2"]).report.citations[0].disputed_by == ["ev_2"]


def test_a_saved_report_disputed_by_missing_evidence_is_refused():
    with pytest.raises(ValidationError):
        _saved(["ev_missing"])
