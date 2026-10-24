"""Tests for measuring disputed claims in live evaluation (phase C3).

Built on the real CUBIC export recorded after A2. Claim 2 of that run is the
wrong "binary search increase" sentence; here it is marked as disputed by the
Wikipedia excerpt that is also in the run's evidence, exactly what the new
verifier check should produce on a live run.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from research_agent.live_compare import compare, format_report, load_batch
from research_agent.live_metrics import (
    ExpectationResult,
    RunMetrics,
    check_expectations,
    compute_run_metrics,
)
from research_agent.live_questions import load_questions
from research_agent.live_runner import DEFAULT_QUESTIONS, RunRecord, _describe
from test_live_compare import write_record

FIXTURES = Path(__file__).parent / "fixtures" / "live_runs"
AFTER = json.loads((FIXTURES / "cubic_after_a2.json").read_text(encoding="utf-8"))
QUESTIONS = list(load_questions(DEFAULT_QUESTIONS).questions)
CUBIC = next(question for question in QUESTIONS if question.id == "cubic_vs_reno")
WIKIPEDIA_EVIDENCE = "ev_78cd9c0ae7f44362bd335dceb2ceb201"
BINARY_SEARCH_CLAIM = 1  # index of the wrong sentence in the fixture's citations


def with_dispute(evidence_id: str = WIKIPEDIA_EVIDENCE, claim: int = BINARY_SEARCH_CLAIM) -> dict:
    data = copy.deepcopy(AFTER)
    data["report"]["citations"][claim]["disputed_by"] = [evidence_id]

    return data


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_a_run_without_disputes_counts_none():
    assert compute_run_metrics(AFTER).disputed_claims == 0


def test_disputed_claims_are_counted():
    metrics = compute_run_metrics(with_dispute())

    assert metrics.disputed_claims == 1
    assert metrics.chain_valid is True


def test_a_dispute_pointing_to_missing_evidence_breaks_the_chain():
    metrics = compute_run_metrics(with_dispute(evidence_id="ev_missing"))

    assert metrics.broken_links == 1
    assert metrics.chain_valid is False


def test_the_cubic_pitfall_counts_as_disputed_when_its_claim_is_disputed():
    result = check_expectations(with_dispute(), CUBIC.expectations)

    # The error is still in the answer...
    assert result.pitfalls_flagged == [CUBIC.expectations.pitfalls[0].name]
    # ...but the reader is warned.
    assert result.pitfalls_disputed == [CUBIC.expectations.pitfalls[0].name]


def test_a_dispute_on_a_different_claim_does_not_count_for_the_pitfall():
    result = check_expectations(with_dispute(claim=3), CUBIC.expectations)

    assert result.pitfalls_flagged and result.pitfalls_disputed == []


def test_result_files_saved_before_these_fields_existed_still_load():
    metrics = compute_run_metrics(AFTER).model_dump()
    expected = check_expectations(AFTER, CUBIC.expectations).model_dump()
    del metrics["disputed_claims"]
    del expected["pitfalls_disputed"]

    assert RunMetrics.model_validate(metrics).disputed_claims == 0
    assert ExpectationResult.model_validate(expected).pitfalls_disputed == []


# ---------------------------------------------------------------------------
# Comparison and runner output
# ---------------------------------------------------------------------------


def test_the_comparison_shows_disputes_as_answer_quality(tmp_path):
    a = tmp_path / "before"
    b = tmp_path / "after"
    write_record(a, "cubic_vs_reno", AFTER, commit="aaaaaaa")
    write_record(b, "cubic_vs_reno", with_dispute(), commit="bbbbbbb")

    batch_a, batch_b = load_batch(a), load_batch(b)
    result = compare(batch_a, batch_b, QUESTIONS)
    left, right = result.per_question["cubic_vs_reno"]

    assert (left["disputed_claims"].mean, right["disputed_claims"].mean) == (0, 1)
    assert (left["pitfalls_disputed"].mean, right["pitfalls_disputed"].mean) == (0, 1)

    lines = format_report(batch_a, batch_b, result)

    assert any(line.strip().startswith("pitfalls marked disputed") for line in lines)
    assert any(line.strip().startswith("disputed claims") for line in lines)


def test_pitfalls_marked_disputed_is_blank_when_no_pitfall_was_flagged(tmp_path):
    data = copy.deepcopy(AFTER)
    data["report"]["content"] = "Instead, the window size is a cubic function of the time since the last congestion event."
    data["report"]["citations"] = [data["report"]["citations"][3]]
    write_record(tmp_path / "b", "cubic_vs_reno", data)

    from research_agent.live_compare import score_run

    record = load_batch(tmp_path / "b").completed["cubic_vs_reno"][0]

    assert score_run(record, CUBIC)["pitfalls_disputed"] is None


def test_the_runner_line_shows_the_disputed_count(tmp_path):
    path = write_record(tmp_path / "b", "cubic_vs_reno", with_dispute())
    record = RunRecord.model_validate_json(path.read_text(encoding="utf-8"))

    assert "disputed 1" in _describe(record)
