"""Offline tests for the live-evaluation metrics (evaluation Phase 1).

The fixtures are two REAL evidence exports from live runs of the same
question ("How does the CUBIC congestion control algorithm in TCP differ
from Reno?"), one before and one after the Phase A2 synthesis changes. The
excerpt text in them is replaced by a short fingerprint so third-party page
text is not republished; equal excerpts still have equal fingerprints, which
is all these metrics rely on.

No test here calls a model, a search API, or the network.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from research_agent.live_metrics import (
    Expectations,
    Pitfall,
    check_expectations,
    compute_run_metrics,
)

_FIXTURES = Path(__file__).parent / "fixtures" / "live_runs"


def _load(name: str) -> dict:
    return json.loads(
        (_FIXTURES / name).read_text(encoding="utf-8")
    )


@pytest.fixture()
def before() -> dict:
    return _load("cubic_before_a2.json")


@pytest.fixture()
def after() -> dict:
    return _load("cubic_after_a2.json")


# ---------------------------------------------------------------------------
# compute_run_metrics on the two real runs
# ---------------------------------------------------------------------------


def test_run_before_a2_cited_only_one_source(before) -> None:
    m = compute_run_metrics(before)

    assert m.outcome == "verified"
    assert m.end_reason == "sufficient"
    assert m.claims == 4

    assert m.sources_with_evidence == 6
    assert m.sources_cited == 1
    assert m.source_coverage == pytest.approx(1 / 6, abs=1e-3)


def test_run_after_a2_cited_two_sources(after) -> None:
    m = compute_run_metrics(after)

    assert m.claims == 4
    assert m.sources_with_evidence == 4
    assert m.sources_cited == 2
    assert m.source_coverage == 0.5


def test_duplicate_claims_are_counted_per_shared_excerpt(
    before,
    after,
) -> None:
    # Before: claims 1+2 share one excerpt and claims 3+4 share another.
    m_before = compute_run_metrics(before)
    assert m_before.duplicate_claims == 2
    assert m_before.claim_duplication_ratio == 0.5

    # After: only claims 1+2 share an excerpt.
    m_after = compute_run_metrics(after)
    assert m_after.duplicate_claims == 1
    assert m_after.claim_duplication_ratio == 0.25


def test_same_excerpt_extracted_twice_counts_once_as_distinct(
    before,
    after,
) -> None:
    # Both runs extracted two witestlab excerpts twice (once per
    # sub-question), under different evidence IDs.
    m_before = compute_run_metrics(before)
    assert m_before.evidence_count == 17
    assert m_before.distinct_evidence_count == 15

    m_after = compute_run_metrics(after)
    assert m_after.evidence_count == 12
    assert m_after.distinct_evidence_count == 10


def test_distinct_evidence_cited(before, after) -> None:
    assert compute_run_metrics(before).distinct_evidence_cited == 2
    assert compute_run_metrics(after).distinct_evidence_cited == 3


def test_fetch_metrics(before, after) -> None:
    m_before = compute_run_metrics(before)
    assert m_before.sources_total == 8
    assert m_before.sources_read == 6
    assert m_before.sources_failed == 2
    assert m_before.fetch_failure_rate == 0.25

    m_after = compute_run_metrics(after)
    assert m_after.sources_read == 5
    assert m_after.sources_failed == 3
    assert m_after.fetch_failure_rate == 0.375


def test_evidence_per_read_source(before, after) -> None:
    assert compute_run_metrics(before).evidence_per_read_source == (
        pytest.approx(17 / 6, abs=1e-3)
    )
    assert compute_run_metrics(after).evidence_per_read_source == (
        pytest.approx(12 / 5, abs=1e-3)
    )


def test_budgets_are_read_from_the_summary(before, after) -> None:
    m_before = compute_run_metrics(before)

    assert m_before.budgets["rounds"].used == 1
    assert m_before.budgets["rounds"].limit == 2
    assert m_before.budgets["searches"].used == 3
    assert m_before.budgets["page_reads"].used == 8
    assert m_before.budgets["ai_calls"].used == 12
    assert m_before.budgets["ai_calls"].limit == 32

    assert compute_run_metrics(after).budgets["ai_calls"].used == 11


def test_issues_are_counted(before, after) -> None:
    assert compute_run_metrics(before).issues == 1
    assert compute_run_metrics(after).issues == 0


def test_real_runs_have_a_valid_citation_chain(before, after) -> None:
    for export in (before, after):
        m = compute_run_metrics(export)

        assert m.chain_valid
        assert m.broken_links == 0


# ---------------------------------------------------------------------------
# Broken chains (synthetic edits of a real export)
# ---------------------------------------------------------------------------


def test_citation_pointing_at_missing_evidence_is_a_broken_link(
    after,
) -> None:
    broken = copy.deepcopy(after)
    broken["report"]["citations"][0]["evidence_ids"] = ["ev_missing"]

    m = compute_run_metrics(broken)

    assert not m.chain_valid
    assert m.broken_links == 1


def test_citation_without_evidence_is_a_broken_link(after) -> None:
    broken = copy.deepcopy(after)
    broken["report"]["citations"][0]["evidence_ids"] = []

    m = compute_run_metrics(broken)

    assert not m.chain_valid
    assert m.broken_links == 1


def test_evidence_pointing_at_missing_source_is_a_broken_link(
    after,
) -> None:
    broken = copy.deepcopy(after)
    cited_id = broken["report"]["citations"][0]["evidence_ids"][0]

    for item in broken["evidence"]:
        if item["id"] == cited_id:
            item["source_id"] = "src_missing"

    m = compute_run_metrics(broken)

    assert not m.chain_valid
    assert m.broken_links >= 1


def test_no_citations_means_undefined_ratios_not_a_crash(after) -> None:
    empty = copy.deepcopy(after)
    empty["report"]["citations"] = []
    empty["summary"]["outcome"] = "no_verified_answer"

    m = compute_run_metrics(empty)

    assert m.claims == 0
    assert m.sources_cited == 0
    assert m.claim_duplication_ratio is None
    assert m.chain_valid
    assert m.outcome == "no_verified_answer"


def test_no_sources_read_means_undefined_rates(after) -> None:
    unread = copy.deepcopy(after)

    for source in unread["sources"]:
        source["fetch_status"] = "pending"

    m = compute_run_metrics(unread)

    assert m.fetch_failure_rate is None
    assert m.evidence_per_read_source is None


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "missing",
    ["report", "evidence", "sources", "summary"],
)
def test_missing_top_level_key_is_rejected(after, missing) -> None:
    broken = copy.deepcopy(after)
    del broken[missing]

    with pytest.raises(ValueError, match=missing):
        compute_run_metrics(broken)


@pytest.mark.parametrize("bad", [None, [], "text", 5])
def test_non_mapping_export_is_rejected(bad) -> None:
    with pytest.raises(ValueError, match="not a research evidence export"):
        compute_run_metrics(bad)


def test_malformed_report_is_rejected(after) -> None:
    broken = copy.deepcopy(after)
    broken["report"] = {"content": 5, "citations": []}

    with pytest.raises(ValueError, match="malformed 'report'"):
        compute_run_metrics(broken)


# ---------------------------------------------------------------------------
# check_expectations
# ---------------------------------------------------------------------------

_CUBIC = Expectations(
    must_mention=[
        ["cubic function"],
        ["linear"],
        ["BIC"],
    ],
    pitfalls=[
        Pitfall(
            name="binary search attributed to CUBIC",
            pattern=r"binary search",
            unless=r"\bBIC\b",
        )
    ],
)


def test_both_real_answers_mention_cubic_and_linear_but_not_bic(
    before,
    after,
) -> None:
    for export in (before, after):
        result = check_expectations(export, _CUBIC)

        assert result.facts_total == 3
        assert result.facts_found == 2
        assert result.missing_facts == [2]


def test_both_real_answers_trip_the_binary_search_pitfall(
    before,
    after,
) -> None:
    for export in (before, after):
        result = check_expectations(export, _CUBIC)

        assert result.pitfalls_flagged == [
            "binary search attributed to CUBIC"
        ]


def test_pitfall_is_not_flagged_when_the_sentence_names_bic(
    after,
) -> None:
    fixed = copy.deepcopy(after)
    fixed["report"]["content"] = (
        "Reno uses linear growth. "
        "BIC used a binary search increase. "
        "CUBIC replaced it with a cubic function of time."
    )

    result = check_expectations(fixed, _CUBIC)

    assert result.pitfalls_flagged == []
    assert result.missing_facts == []


def test_unless_applies_to_the_same_sentence_only(after) -> None:
    mixed = copy.deepcopy(after)
    mixed["report"]["content"] = (
        "BIC is the predecessor. "
        "Cubic implements a binary search increase."
    )

    result = check_expectations(mixed, _CUBIC)

    assert result.pitfalls_flagged == [
        "binary search attributed to CUBIC"
    ]


def test_fact_alternatives_are_case_insensitive(after) -> None:
    result = check_expectations(
        after,
        Expectations(must_mention=[["CUBIC FUNCTION", "something else"]]),
    )

    assert result.facts_found == 1


def test_no_expectations_means_nothing_flagged(after) -> None:
    result = check_expectations(after, Expectations())

    assert result.facts_total == 0
    assert result.facts_found == 0
    assert result.pitfalls_flagged == []
    assert result.abstain_ok is None


def test_confident_verified_answer_fails_an_abstain_expectation(
    after,
) -> None:
    # As if the question had no real answer.
    result = check_expectations(
        after,
        Expectations(expect_abstain=True),
    )

    assert result.abstain_ok is False


def test_unverified_outcome_passes_an_abstain_expectation(after) -> None:
    abstained = copy.deepcopy(after)
    abstained["summary"]["outcome"] = "no_verified_answer"

    result = check_expectations(
        abstained,
        Expectations(expect_abstain=True),
    )

    assert result.abstain_ok is True


def test_verified_answer_stating_a_limitation_passes_abstain(
    after,
) -> None:
    limited = copy.deepcopy(after)
    limited["report"]["content"] = (
        "The available evidence does not establish what this "
        "standard specified."
    )

    result = check_expectations(
        limited,
        Expectations(expect_abstain=True),
    )

    assert result.abstain_ok is True


# ---------------------------------------------------------------------------
# Expectations validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "facts",
    [[[]], [[""]], [["ok", "  "]]],
)
def test_blank_fact_phrases_are_rejected(facts) -> None:
    with pytest.raises(ValidationError):
        Expectations(must_mention=facts)


def test_invalid_pitfall_regex_is_rejected_at_construction() -> None:
    with pytest.raises(ValidationError):
        Pitfall(name="bad", pattern="(unclosed")

    with pytest.raises(ValidationError):
        Pitfall(name="bad", pattern="ok", unless="(unclosed")


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Expectations(must_mention=[], surprise=True)
