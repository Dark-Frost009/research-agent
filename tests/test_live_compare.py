"""Offline tests for comparing two live-evaluation batches.

The batches are built in temporary folders from the two real CUBIC exports
saved earlier (before and after the A2 change), so the numbers checked here
are the ones those real runs produced.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_agent.live_compare import (
    NOTE,
    CompareError,
    compare,
    format_report,
    load_batch,
    main,
    score_run,
    summarize,
)
from research_agent.live_metrics import check_expectations, compute_run_metrics
from research_agent.live_questions import load_questions
from research_agent.live_runner import DEFAULT_QUESTIONS, RunRecord

FIXTURES = Path(__file__).parent / "fixtures" / "live_runs"
QUESTIONS = list(load_questions(DEFAULT_QUESTIONS).questions)
BY_ID = {question.id: question for question in QUESTIONS}


def export(name: str) -> dict:
    return json.loads((FIXTURES / f"cubic_{name}_a2.json").read_text(encoding="utf-8"))


BEFORE = export("before")
AFTER = export("after")


def write_record(
    folder: Path,
    question_id: str,
    data: dict,
    *,
    run_index: int = 1,
    status: str = "completed",
    commit: str = "aaaaaaa",
    worker: str = "w",
    final: str = "f",
    depth: str = "Standard",
    limited: bool = False,
    tamper=None,
) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    completed = status == "completed"

    record = RunRecord(
        label=folder.name,
        question_id=question_id,
        run_index=run_index,
        started_at="2026-10-23T10:00:00Z",
        elapsed_seconds=1.0,
        depth=depth,
        worker_model=worker,
        final_model=final,
        git_commit=commit,
        status=status,
        interruption_reason=None if completed else "response",
        export=data,
        provider_limited=limited,
        metrics=compute_run_metrics(data) if completed else None,
        expectations=(
            check_expectations(data, BY_ID[question_id].expectations)
            if completed and question_id in BY_ID
            else (check_expectations(data, BY_ID["cubic_vs_reno"].expectations) if completed else None)
        ),
    )

    text = record.model_dump_json()

    if tamper is not None:
        raw = json.loads(text)
        tamper(raw)
        text = json.dumps(raw)

    suffix = ".limited.json" if limited else ".json"
    path = folder / f"{question_id}__{run_index}{suffix}"
    path.write_text(text, encoding="utf-8")

    return path


def two_batches(tmp_path):
    a = tmp_path / "baseline"
    b = tmp_path / "a2"
    write_record(a, "cubic_vs_reno", BEFORE, commit="d5612ba")
    write_record(b, "cubic_vs_reno", AFTER, commit="1f00044")

    return load_batch(a), load_batch(b)


# ---------------------------------------------------------------------------
# Loading a batch
# ---------------------------------------------------------------------------


def test_load_batch_skips_throttled_files_and_counts_interrupted_runs(tmp_path):
    folder = tmp_path / "b"
    write_record(folder, "cubic_vs_reno", BEFORE)
    write_record(folder, "cubic_vs_reno", AFTER, run_index=2, limited=True)
    write_record(folder, "rag_basics", {}, status="interrupted")

    batch = load_batch(folder)

    assert batch.completed_runs == 1
    assert batch.interrupted_runs == 1
    assert batch.limited_skipped == 1
    assert batch.commits == {"aaaaaaa"}
    assert batch.models == {"w / f"}
    assert batch.depths == {"Standard"}


def test_a_missing_folder_is_reported(tmp_path):
    with pytest.raises(CompareError):
        load_batch(tmp_path / "nope")


def test_an_empty_folder_is_reported(tmp_path):
    (tmp_path / "empty").mkdir()

    with pytest.raises(CompareError):
        load_batch(tmp_path / "empty")


def test_a_broken_file_is_named_without_echoing_its_content(tmp_path):
    folder = tmp_path / "b"
    folder.mkdir()
    (folder / "x__1.json").write_text('{"secret-looking": "sk-ABC"}', encoding="utf-8")

    with pytest.raises(CompareError) as error:
        load_batch(folder)

    assert "x__1.json" in str(error.value)
    assert "sk-ABC" not in str(error.value)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_scores_are_recomputed_and_stored_metrics_are_ignored(tmp_path):
    def tamper(raw):
        raw["metrics"]["claims"] = 99
        raw["metrics"]["sources_cited"] = 99

    path = write_record(tmp_path / "b", "cubic_vs_reno", AFTER, tamper=tamper)
    record = RunRecord.model_validate_json(path.read_text(encoding="utf-8"))

    score = score_run(record, BY_ID["cubic_vs_reno"])

    assert score["claims"] == 4
    assert score["sources_cited"] == 2


def test_the_real_before_and_after_runs_score_as_recorded(tmp_path):
    a, b = two_batches(tmp_path)
    result = compare(a, b, QUESTIONS)
    left, right = result.per_question["cubic_vs_reno"]

    assert (left["claims"].mean, right["claims"].mean) == (4, 4)
    assert (left["sources_cited"].mean, right["sources_cited"].mean) == (1, 2)
    # live_metrics rounds ratios to 4 decimal places: 1/6 and 2/4.
    assert left["source_coverage"].mean == 0.1667
    assert right["source_coverage"].mean == 0.5
    assert (left["duplicate_claims"].mean, right["duplicate_claims"].mean) == (2, 1)
    # The known wrong "binary search" sentence is in both runs.
    assert (left["pitfalls"].mean, right["pitfalls"].mean) == (1, 1)
    # Not an abstention question, so no abstention score.
    assert left["abstained"] is None


def test_abstention_is_scored_for_the_question_with_no_real_answer(tmp_path):
    data = json.loads(json.dumps(BEFORE))
    data["summary"]["outcome"] = "no_evidence"
    path = write_record(tmp_path / "b", "quantum_tcp_standard", data)
    record = RunRecord.model_validate_json(path.read_text(encoding="utf-8"))

    score = score_run(record, BY_ID["quantum_tcp_standard"])

    assert score["abstained"] == 1.0
    # "Verified" would be the wrong outcome here, so it is not scored at all
    # and cannot drag down (or inflate) the overall verified rate.
    assert score["verified"] is None


def test_summarize_ignores_undefined_values():
    assert summarize([None, None]) is None

    summary = summarize([1.0, None, 3.0])

    assert (summary.mean, summary.low, summary.high, summary.n) == (2.0, 1.0, 3.0, 2)


# ---------------------------------------------------------------------------
# Comparing
# ---------------------------------------------------------------------------


def test_repeats_give_a_mean_and_a_range(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    write_record(a, "cubic_vs_reno", BEFORE, run_index=1)
    write_record(a, "cubic_vs_reno", AFTER, run_index=2)
    write_record(b, "cubic_vs_reno", AFTER, commit="bbbbbbb")

    result = compare(load_batch(a), load_batch(b), QUESTIONS)
    cited = result.per_question["cubic_vs_reno"][0]["sources_cited"]

    assert (cited.mean, cited.low, cited.high, cited.n) == (1.5, 1, 2, 2)


def test_overall_averages_questions_not_runs(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"

    # Batch A: three runs citing 1 source for one question, one run citing 2
    # for another. Per run that averages 1.25; per question it is 1.5.
    for index in (1, 2, 3):
        write_record(a, "cubic_vs_reno", BEFORE, run_index=index)
    write_record(a, "linux_cubic_default", AFTER)

    write_record(b, "cubic_vs_reno", AFTER, commit="bbbbbbb")
    write_record(b, "linux_cubic_default", AFTER, commit="bbbbbbb")

    result = compare(load_batch(a), load_batch(b), QUESTIONS)
    mean_a, mean_b, questions = result.overall["sources_cited"]

    assert (mean_a, mean_b, questions) == (1.5, 2, 2)


def test_questions_answered_by_only_one_batch_are_listed_not_compared(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    write_record(a, "cubic_vs_reno", BEFORE)
    write_record(a, "rag_basics", AFTER)
    write_record(a, "retired_question", AFTER)
    write_record(b, "cubic_vs_reno", AFTER, commit="bbbbbbb")
    write_record(b, "python_313", AFTER, commit="bbbbbbb")

    result = compare(load_batch(a), load_batch(b), QUESTIONS)

    assert result.common == ["cubic_vs_reno"]
    assert result.only_a == ["rag_basics"]
    assert result.only_b == ["python_313"]
    assert result.unknown == ["retired_question"]


def test_interrupted_runs_stay_out_of_averages_but_show_in_completion(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    write_record(a, "cubic_vs_reno", BEFORE)
    write_record(a, "cubic_vs_reno", {}, run_index=2, status="interrupted")
    write_record(b, "cubic_vs_reno", AFTER, commit="bbbbbbb")

    batch_a, batch_b = load_batch(a), load_batch(b)
    result = compare(batch_a, batch_b, QUESTIONS)

    assert result.per_question["cubic_vs_reno"][0]["claims"].n == 1
    assert "1/2 runs finished" in "\n".join(format_report(batch_a, batch_b, result))


@pytest.mark.parametrize(
    "setup, expected",
    [
        (
            lambda a, b: (
                write_record(a, "cubic_vs_reno", BEFORE, commit="aaaaaaa"),
                write_record(a, "rag_basics", BEFORE, commit="ccccccc"),
            ),
            "Batch A mixes code versions",
        ),
        (
            lambda a, b: write_record(a, "cubic_vs_reno", BEFORE, commit="aaaaaaa+dirty"),
            "uncommitted edits",
        ),
        (
            lambda a, b: (
                write_record(a, "cubic_vs_reno", BEFORE, worker="m1"),
                write_record(a, "rag_basics", BEFORE, worker="m2"),
            ),
            "Batch A mixes models",
        ),
        (
            lambda a, b: write_record(a, "cubic_vs_reno", BEFORE, worker="other"),
            "different models",
        ),
        (
            lambda a, b: write_record(a, "cubic_vs_reno", BEFORE, depth="Quick"),
            "different depths",
        ),
        (
            lambda a, b: write_record(a, "cubic_vs_reno", BEFORE, commit="bbbbbbb"),
            "same code version",
        ),
    ],
)
def test_confounds_are_warned_about(tmp_path, setup, expected):
    a = tmp_path / "a"
    b = tmp_path / "b"
    write_record(b, "cubic_vs_reno", AFTER, commit="bbbbbbb")
    setup(a, b)

    result = compare(load_batch(a), load_batch(b), QUESTIONS)

    assert any(expected in warning for warning in result.warnings)


def test_a_clean_comparison_has_no_warnings(tmp_path):
    a, b = two_batches(tmp_path)

    assert compare(a, b, QUESTIONS).warnings == []


# ---------------------------------------------------------------------------
# Report and command line
# ---------------------------------------------------------------------------


def test_report_shows_ranges_overall_and_the_caution_note(tmp_path):
    a = tmp_path / "baseline"
    b = tmp_path / "a2"
    write_record(a, "cubic_vs_reno", BEFORE, run_index=1)
    write_record(a, "cubic_vs_reno", AFTER, run_index=2)
    write_record(b, "cubic_vs_reno", AFTER, commit="bbbbbbb")

    batch_a, batch_b = load_batch(a), load_batch(b)
    text = "\n".join(format_report(batch_a, batch_b, compare(batch_a, batch_b, QUESTIONS)))

    assert "A: baseline" in text and "B: a2" in text
    assert "cubic_vs_reno" in text
    assert "1.5 [1-2] (n=2)" in text
    assert "OVERALL" in text
    assert NOTE in text


def test_main_prints_the_report_and_returns_zero(tmp_path):
    two_batches(tmp_path)
    lines: list[str] = []

    code = main([str(tmp_path / "baseline"), str(tmp_path / "a2")], echo=lines.append)

    assert code == 0
    assert any("OVERALL" in line for line in lines)


def test_main_returns_one_when_nothing_is_in_common(tmp_path):
    write_record(tmp_path / "a", "cubic_vs_reno", BEFORE)
    write_record(tmp_path / "b", "rag_basics", AFTER, commit="bbbbbbb")
    lines: list[str] = []

    code = main([str(tmp_path / "a"), str(tmp_path / "b")], echo=lines.append)

    assert code == 1
    assert "nothing to compare" in "\n".join(lines)


def test_main_returns_one_for_a_missing_folder(tmp_path):
    two_batches(tmp_path)
    lines: list[str] = []

    code = main([str(tmp_path / "baseline"), str(tmp_path / "nope")], echo=lines.append)

    assert code == 1
    assert "Not a folder" in "\n".join(lines)


# ---------------------------------------------------------------------------
# Outcomes versus answer quality
# ---------------------------------------------------------------------------


def rejected() -> dict:
    """A run whose synthesis the verifier rejected: no answer, no citations."""

    data = json.loads(json.dumps(BEFORE))
    data["summary"]["outcome"] = "verification_rejected"
    data["report"]["citations"] = []
    data["report"]["content"] = (
        "The available evidence could not be verified strongly enough to produce a grounded answer."
    )

    return data


def test_a_rejected_run_counts_in_the_verified_rate_but_not_in_answer_quality(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    write_record(a, "cubic_vs_reno", AFTER, run_index=1)
    write_record(a, "cubic_vs_reno", rejected(), run_index=2)
    write_record(b, "cubic_vs_reno", AFTER, commit="bbbbbbb")

    left, right = compare(load_batch(a), load_batch(b), QUESTIONS).per_question["cubic_vs_reno"]

    # The outcome shows the rejection...
    assert left["verified"].mean == 0.5 and left["verified"].n == 2
    # ...but the rejected run does not drag answer quality down with zeros.
    assert (left["sources_cited"].mean, left["sources_cited"].n) == (2, 1)
    assert (left["facts_found"].mean, left["facts_found"].n) == (1, 1)
    assert (left["pitfalls"].mean, left["pitfalls"].n) == (1, 1)
    assert left["sources_cited"] == right["sources_cited"]
    # Cost still counts every finished run.
    assert left["ai_calls"].n == 2


def test_a_batch_with_only_rejected_runs_has_no_quality_numbers(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    write_record(a, "cubic_vs_reno", rejected())
    write_record(b, "cubic_vs_reno", AFTER, commit="bbbbbbb")

    batch_a, batch_b = load_batch(a), load_batch(b)
    result = compare(batch_a, batch_b, QUESTIONS)
    left, _ = result.per_question["cubic_vs_reno"]

    assert left["verified"].mean == 0
    assert left["claims"] is None
    # With no verified run in A, the question cannot enter the overall quality average.
    assert result.overall["claims"] is None
    assert result.overall["verified"] == (0, 1, 1)

    text = "\n".join(format_report(batch_a, batch_b, result))

    assert "-- outcome, all runs" in text
    assert "-- answer quality, verified runs only" in text


# ---------------------------------------------------------------------------
# Counting rejections by reason
# ---------------------------------------------------------------------------


def rejected_with(reason: str | None) -> dict:
    data = rejected()

    if reason is not None:
        data["summary"]["rejection_reason"] = reason

    return data


def test_rejections_are_counted_by_reason_code(tmp_path):
    folder = tmp_path / "b"
    write_record(folder, "cubic_vs_reno", rejected_with("quote_mismatch"), run_index=1)
    write_record(folder, "rag_basics", rejected_with("quote_mismatch"), run_index=1)
    write_record(folder, "python_313", rejected_with("unsupported_claim"), run_index=1)
    write_record(folder, "ethernet_mtu", AFTER, run_index=1)

    batch = load_batch(folder)

    assert dict(batch.rejections) == {"quote_mismatch": 2, "unsupported_claim": 1}


def test_runs_saved_before_reason_codes_existed_show_as_not_recorded(tmp_path):
    folder = tmp_path / "b"
    write_record(folder, "cubic_vs_reno", rejected_with(None))

    assert dict(load_batch(folder).rejections) == {"not recorded": 1}


def test_interrupted_and_throttled_runs_are_not_counted_as_rejections(tmp_path):
    folder = tmp_path / "b"
    write_record(folder, "cubic_vs_reno", AFTER)
    write_record(folder, "rag_basics", {}, status="interrupted")
    write_record(folder, "python_313", rejected_with("quote_mismatch"), limited=True)

    assert not load_batch(folder).rejections


def test_the_report_lists_rejections_per_batch_most_common_first(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    write_record(a, "cubic_vs_reno", rejected_with("unsupported_claim"), run_index=1)
    write_record(a, "cubic_vs_reno", rejected_with("quote_mismatch"), run_index=2)
    write_record(a, "cubic_vs_reno", rejected_with("quote_mismatch"), run_index=3)
    write_record(a, "cubic_vs_reno", AFTER, run_index=4)
    write_record(b, "cubic_vs_reno", AFTER, commit="bbbbbbb")

    batch_a, batch_b = load_batch(a), load_batch(b)
    lines = format_report(batch_a, batch_b, compare(batch_a, batch_b, QUESTIONS))

    assert "   rejections: 2 x quote_mismatch, 1 x unsupported_claim" in lines
    assert "   rejections: none" in lines
