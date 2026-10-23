"""Offline tests for the live evaluation runner.

Nothing here calls an AI model, a search provider or the network: the
function that would run a question is replaced by a scripted stand-in, and
the finished-run data is a real export saved earlier (``cubic_after_a2.json``).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from research_agent import live_runner
from research_agent.incomplete import IncompleteResearch, ResearchInterrupted
from research_agent.live_questions import load_questions
from research_agent.live_runner import (
    RunRecord,
    git_commit,
    is_provider_limited,
    main,
    plan_runs,
    result_path,
    run_batch,
)

FIXTURES = Path(__file__).parent / "fixtures" / "live_runs"
QUESTIONS = load_questions(live_runner.DEFAULT_QUESTIONS).questions
CUBIC = next(question for question in QUESTIONS if question.id == "cubic_vs_reno")
SETTINGS = SimpleNamespace(
    llm_provider="gemini",
    worker_llm_model="worker-x",
    final_llm_model="final-y",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeResult:
    """Stands in for ui_service.CompletedResearch (only json_export is used)."""

    def __init__(self, text: str):
        self._text = text

    def json_export(self) -> str:
        return self._text


def cubic_text() -> str:
    return (FIXTURES / "cubic_after_a2.json").read_text(encoding="utf-8")


def completed() -> FakeResult:
    return FakeResult(cubic_text())


def interrupted(reason: str) -> ResearchInterrupted:
    return ResearchInterrupted(
        IncompleteResearch(
            question="Q",
            last_stage="Searching the web",
            reason=reason,
            sources=[],
            evidence=[],
            iterations=0,
            searches=0,
            warning_count=0,
        )
    )


class ScriptedRun:
    """Replaces ui_service.run_question: returns or raises scripted outcomes."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, str | None]] = []

    def __call__(self, question, on_progress, *, depth=None, settings=None):
        self.calls.append((question, depth))
        outcome = self.outcomes.pop(0)

        if isinstance(outcome, BaseException):
            raise outcome

        return outcome


def go(planned, run_fn, *, max_runs=8):
    lines: list[str] = []

    result = run_batch(
        planned,
        label="t",
        depth="Standard",
        run_fn=run_fn,
        settings=SETTINGS,
        commit="abc123",
        max_runs=max_runs,
        echo=lines.append,
    )

    return result, lines


def three(tmp_path):
    return plan_runs(QUESTIONS[:3], 1, tmp_path / "batch")


@pytest.fixture(autouse=True)
def _no_real_depth(monkeypatch):
    # apply_depth needs real Settings; the runner only uses its result for
    # a preview and an early check.
    monkeypatch.setattr(
        live_runner,
        "apply_depth",
        lambda settings, depth: (settings, SimpleNamespace(ai_calls=32, searches=5)),
    )


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def test_plan_runs_does_run_one_of_every_question_before_run_two(tmp_path):
    planned = plan_runs(QUESTIONS[:3], 2, tmp_path)

    assert [(item.question.id, item.run_index) for item in planned] == [
        (QUESTIONS[0].id, 1),
        (QUESTIONS[1].id, 1),
        (QUESTIONS[2].id, 1),
        (QUESTIONS[0].id, 2),
        (QUESTIONS[1].id, 2),
        (QUESTIONS[2].id, 2),
    ]


def test_a_planned_run_is_done_only_if_its_normal_result_file_exists(tmp_path):
    planned = plan_runs(QUESTIONS[:2], 1, tmp_path)

    # A throttled run's .limited file must NOT count as a finished result.
    result_path(tmp_path, QUESTIONS[0].id, 1, limited=True).write_text("{}")
    result_path(tmp_path, QUESTIONS[1].id, 1).write_text("{}")

    assert [item.done for item in planned] == [False, True]


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def test_completed_run_is_saved_with_metrics_expectations_and_provenance(tmp_path):
    planned = plan_runs([CUBIC], 1, tmp_path / "batch")
    run = ScriptedRun(completed())

    result, _ = go(planned, run)

    assert run.calls == [(CUBIC.question, "Standard")]
    assert result.stopped_reason is None and result.remaining == 0

    saved = RunRecord.model_validate_json(result.written[0].read_text(encoding="utf-8"))

    assert saved.status == "completed"
    assert saved.question_id == "cubic_vs_reno" and saved.run_index == 1
    assert (saved.label, saved.depth, saved.git_commit) == ("t", "Standard", "abc123")
    assert (saved.llm_provider, saved.worker_model, saved.final_model) == (
        "gemini",
        "worker-x",
        "final-y",
    )
    assert saved.metrics is not None and saved.metrics.claims == 4
    assert saved.metrics.chain_valid is True
    assert saved.expectations is not None
    assert saved.expectations.facts_total == len(CUBIC.expectations.must_mention)
    # The known error in this recorded run is found by the scoring.
    assert saved.expectations.pitfalls_flagged == [
        CUBIC.expectations.pitfalls[0].name
    ]
    assert saved.export["report"]["question"].startswith("How does the CUBIC")


def test_saved_record_has_only_the_documented_fields(tmp_path):
    result, _ = go(plan_runs([CUBIC], 1, tmp_path), ScriptedRun(completed()))

    keys = set(json.loads(result.written[0].read_text(encoding="utf-8")))

    assert keys == set(RunRecord.model_fields)


def test_no_temporary_file_is_left_behind(tmp_path):
    go(plan_runs([CUBIC], 1, tmp_path), ScriptedRun(completed()))

    assert not [path for path in tmp_path.rglob("*") if path.name.endswith(".tmp")]


def test_max_runs_caps_how_many_runs_start(tmp_path):
    run = ScriptedRun(completed(), completed())

    result, _ = go(three(tmp_path), run, max_runs=2)

    assert len(run.calls) == 2
    assert len(result.written) == 2
    assert result.remaining == 1
    assert result.stopped_reason is None


def test_rerunning_the_same_batch_resumes_instead_of_repeating(tmp_path):
    first = ScriptedRun(completed(), completed())
    go(three(tmp_path), first, max_runs=2)

    second = ScriptedRun(completed())
    result, _ = go(three(tmp_path), second, max_runs=8)

    assert [call[0] for call in second.calls] == [QUESTIONS[2].question]
    assert result.remaining == 0


@pytest.mark.parametrize("reason", ["quota", "unavailable"])
def test_provider_quota_or_overload_stops_the_batch_and_saves_nothing_for_it(
    tmp_path, reason
):
    run = ScriptedRun(completed(), interrupted(reason), completed())

    result, lines = go(three(tmp_path), run)

    assert result.stopped_reason == reason
    # The third question was never started and the stopped one left no file.
    assert len(run.calls) == 2
    assert len(result.written) == 1
    assert result.remaining == 2
    assert any("not saved" in line for line in lines)


def test_other_interruptions_are_recorded_as_results_and_the_batch_continues(tmp_path):
    run = ScriptedRun(interrupted("response"), completed())

    result, _ = go(plan_runs(QUESTIONS[:2], 1, tmp_path), run)

    first = RunRecord.model_validate_json(result.written[0].read_text(encoding="utf-8"))

    assert first.status == "interrupted"
    assert first.interruption_reason == "response"
    assert first.metrics is None and first.expectations is None
    assert len(result.written) == 2
    assert result.stopped_reason is None


def test_a_run_throttled_by_the_provider_is_saved_aside_and_ends_the_batch(tmp_path):
    data = json.loads(cubic_text())
    data["issues"] = [
        "Gemini's rate or quota limit interrupted evidence extraction. "
        "Check the provider allowance before trying again."
    ]
    run = ScriptedRun(FakeResult(json.dumps(data)), completed())

    planned = plan_runs(QUESTIONS[:2], 1, tmp_path)
    result, _ = go(planned, run)

    assert result.stopped_reason == "limited"
    assert len(run.calls) == 1
    assert result.written[0].name.endswith(".limited.json")
    assert not planned[0].path.exists()
    assert result.remaining == 2

    saved = RunRecord.model_validate_json(result.written[0].read_text(encoding="utf-8"))
    assert saved.provider_limited is True


def test_an_unexpected_error_propagates_instead_of_being_hidden(tmp_path):
    with pytest.raises(RuntimeError):
        go(three(tmp_path), ScriptedRun(RuntimeError("boom")))


# ---------------------------------------------------------------------------
# Detecting provider trouble from a finished run's own issues
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error_type, expected",
    [
        ("LLMRateLimitError", True),
        ("LLMUnavailableError", True),
        ("EvidenceGroundingError", False),
        ("LLMResponseError", False),
        ("LLMProviderError", False),
    ],
)
def test_detection_matches_the_messages_the_app_really_produces(error_type, expected):
    # Uses the app's own function so a reworded message cannot quietly
    # switch this safeguard off.
    from research_agent.ui_service import summarize_issues

    issues = summarize_issues(
        [f"Evidence extraction failed for source src_1: {error_type}: details"]
    )

    assert issues, "the app produced no message for this error type"
    assert is_provider_limited({"issues": list(issues)}) is expected


def test_no_issues_means_not_limited():
    assert is_provider_limited({}) is False
    assert is_provider_limited({"issues": []}) is False
    assert is_provider_limited({"issues": None}) is False


# ---------------------------------------------------------------------------
# The record's own consistency rules
# ---------------------------------------------------------------------------


def _record_kwargs(**overrides):
    base = dict(
        label="t",
        question_id="q",
        run_index=1,
        started_at="2026-10-22T10:00:00Z",
        elapsed_seconds=1.0,
        depth="Standard",
        status="interrupted",
        interruption_reason="response",
        export={},
    )
    base.update(overrides)

    return base


def test_an_interrupted_record_is_valid_without_metrics():
    RunRecord(**_record_kwargs())


def test_an_interrupted_record_needs_a_reason():
    with pytest.raises(ValidationError):
        RunRecord(**_record_kwargs(interruption_reason=None))


def test_a_completed_record_needs_metrics_and_expectations():
    with pytest.raises(ValidationError):
        RunRecord(**_record_kwargs(status="completed", interruption_reason=None))


def test_a_completed_record_must_not_carry_an_interruption_reason(tmp_path):
    result, _ = go(plan_runs([CUBIC], 1, tmp_path), ScriptedRun(completed()))
    data = json.loads(result.written[0].read_text(encoding="utf-8"))
    data["interruption_reason"] = "quota"

    with pytest.raises(ValidationError):
        RunRecord.model_validate(data)


def test_an_interrupted_record_must_not_carry_metrics(tmp_path):
    result, _ = go(plan_runs([CUBIC], 1, tmp_path), ScriptedRun(completed()))
    data = json.loads(result.written[0].read_text(encoding="utf-8"))
    data["status"] = "interrupted"
    data["interruption_reason"] = "response"

    with pytest.raises(ValidationError):
        RunRecord.model_validate(data)


def test_a_record_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        RunRecord(**_record_kwargs(api_key="should never be stored"))


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def run_main(tmp_path, *args, run_fn=None, settings_loader=None, sleeps=None):
    lines: list[str] = []
    # Record pauses instead of really waiting, so the tests stay fast.
    sleeps = [] if sleeps is None else sleeps

    code = main(
        ["--out-dir", str(tmp_path), *args],
        run_fn=run_fn,
        settings_loader=settings_loader or (lambda: SETTINGS),
        commit_fn=lambda: "abc123",
        echo=lines.append,
        sleep=sleeps.append,
    )

    return code, lines


def test_dry_run_makes_no_calls_and_writes_nothing(tmp_path):
    run = ScriptedRun()

    code, lines = run_main(tmp_path, "--label", "t", "--dry-run", run_fn=run)

    assert code == 0
    assert run.calls == []
    assert list(tmp_path.rglob("*")) == []

    text = "\n".join(lines)

    assert "DRY RUN" in text
    assert "worker-x" in text and "final-y" in text
    # Every shipped question appears in the plan (default cap is 8).
    for question in QUESTIONS[:8]:
        assert question.id in text


def test_dry_run_shows_observed_use_and_the_budget_ceiling(tmp_path):
    _, lines = run_main(
        tmp_path, "--label", "t", "--dry-run", "--only", "cubic_vs_reno", "--runs", "2"
    )

    text = "\n".join(lines)

    # 2 runs x 12 observed, and at most 2 x 32 by the (stubbed) ceiling.
    assert "about 24 AI requests" in text
    assert "at most 64" in text


def test_dry_run_still_works_when_settings_cannot_be_loaded_and_leaks_nothing(tmp_path):
    def broken():
        raise RuntimeError("key sk-SECRET-123 is malformed")

    code, lines = run_main(
        tmp_path, "--label", "t", "--dry-run", settings_loader=broken
    )

    text = "\n".join(lines)

    assert code == 0
    assert "RuntimeError" in text
    assert "sk-SECRET-123" not in text


def test_a_real_run_refuses_to_start_without_settings(tmp_path):
    def broken():
        raise RuntimeError("key sk-SECRET-123 is malformed")

    run = ScriptedRun()
    code, lines = run_main(tmp_path, "--label", "t", run_fn=run, settings_loader=broken)

    assert code == 1
    assert run.calls == []
    assert "sk-SECRET-123" not in "\n".join(lines)


def test_an_exception_message_is_never_printed_and_finished_runs_stay_saved(tmp_path):
    run = ScriptedRun(completed(), RuntimeError("token sk-SECRET-456 rejected"))

    code, lines = run_main(
        tmp_path, "--label", "t", "--only", "cubic_vs_reno,rag_basics", run_fn=run
    )

    text = "\n".join(lines)

    assert code == 1
    assert "RuntimeError" in text
    assert "sk-SECRET-456" not in text
    assert len(list((tmp_path / "t").glob("*.json"))) == 1


def test_exit_code_is_two_when_the_provider_ends_the_batch(tmp_path):
    run = ScriptedRun(interrupted("quota"))

    code, _ = run_main(tmp_path, "--label", "t", run_fn=run)

    assert code == 2


def test_exit_code_is_zero_for_a_clean_batch_and_runs_use_the_chosen_depth(tmp_path):
    run = ScriptedRun(completed())

    code, _ = run_main(
        tmp_path, "--label", "t", "--only", "cubic_vs_reno", "--depth", "Quick", run_fn=run
    )

    assert code == 0
    assert run.calls == [(CUBIC.question, "Quick")]


def test_unknown_question_id_is_rejected_before_anything_runs(tmp_path):
    run = ScriptedRun()

    code, lines = run_main(tmp_path, "--label", "t", "--only", "nope", run_fn=run)

    assert code == 1
    assert run.calls == []
    assert "nope" in "\n".join(lines)


def test_only_selects_just_those_questions(tmp_path):
    run = ScriptedRun(completed(), completed())

    run_main(tmp_path, "--label", "t", "--only", "cubic_vs_reno, rag_basics", run_fn=run)

    assert [call[0] for call in run.calls] == [
        CUBIC.question,
        next(q for q in QUESTIONS if q.id == "rag_basics").question,
    ]


@pytest.mark.parametrize("label", ["", "../escape", "a/b", "a b", "-leading", "x" * 65])
def test_unsafe_labels_are_rejected(tmp_path, label):
    with pytest.raises(SystemExit):
        main(["--label", label, "--out-dir", str(tmp_path), "--dry-run"])


@pytest.mark.parametrize("flag", ["--runs", "--max-runs"])
@pytest.mark.parametrize("value", ["0", "-1", "two"])
def test_counts_must_be_positive_whole_numbers(tmp_path, flag, value):
    with pytest.raises(SystemExit):
        main(["--label", "t", "--out-dir", str(tmp_path), flag, value, "--dry-run"])


def test_a_bad_question_file_is_reported_not_run(tmp_path):
    bad = tmp_path / "q.json"
    bad.write_text('{"version": 1, "questions": []}', encoding="utf-8")

    code, lines = run_main(
        tmp_path, "--label", "t", "--questions", str(bad), "--dry-run"
    )

    assert code == 1
    assert "could not be loaded" in "\n".join(lines)


# ---------------------------------------------------------------------------
# Recording the code version
# ---------------------------------------------------------------------------


def _fake_git(monkeypatch, *, head, status="", head_code=0, error=None):
    def fake(command, **kwargs):
        if error is not None:
            raise error

        if "rev-parse" in command:
            return subprocess.CompletedProcess(command, head_code, head, "")

        return subprocess.CompletedProcess(command, 0, status, "")

    monkeypatch.setattr(live_runner.subprocess, "run", fake)


def test_git_commit_reports_a_clean_checkout(monkeypatch, tmp_path):
    _fake_git(monkeypatch, head="6f1de96aabbc\n")

    assert git_commit(tmp_path) == "6f1de96aabbc"


def test_git_commit_marks_uncommitted_edits_so_a_baseline_cannot_be_mislabelled(
    monkeypatch, tmp_path
):
    _fake_git(monkeypatch, head="6f1de96aabbc\n", status=" M app.py\n")

    assert git_commit(tmp_path) == "6f1de96aabbc+dirty"


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(head="", head_code=128),
        dict(head="not-a-hash\n"),
        dict(head="", error=FileNotFoundError("git")),
        dict(head="", error=subprocess.TimeoutExpired("git", 10)),
    ],
)
def test_git_commit_is_none_when_git_is_unavailable_or_odd(monkeypatch, tmp_path, kwargs):
    _fake_git(monkeypatch, **kwargs)

    assert git_commit(tmp_path) is None


# ---------------------------------------------------------------------------
# Pausing between runs (keeps consecutive runs under per-minute limits)
# ---------------------------------------------------------------------------


def _paused(planned, run_fn, pause):
    sleeps: list[float] = []

    result = run_batch(
        planned,
        label="t",
        depth="Standard",
        run_fn=run_fn,
        settings=SETTINGS,
        commit=None,
        max_runs=8,
        echo=lambda line: None,
        pause_seconds=pause,
        sleep=sleeps.append,
    )

    return result, sleeps


def test_the_runner_pauses_between_runs_but_not_before_the_first(tmp_path):
    _, sleeps = _paused(
        three(tmp_path), ScriptedRun(completed(), completed(), completed()), 60
    )

    assert sleeps == [60, 60]


def test_no_pause_after_the_provider_ends_the_batch(tmp_path):
    _, sleeps = _paused(
        three(tmp_path), ScriptedRun(completed(), interrupted("quota")), 60
    )

    # One pause before run 2, none after it stopped the batch.
    assert sleeps == [60]


def test_a_zero_pause_never_sleeps(tmp_path):
    _, sleeps = _paused(three(tmp_path), ScriptedRun(completed(), completed(), completed()), 0)

    assert sleeps == []


def test_the_command_line_pauses_60_seconds_by_default(tmp_path):
    sleeps: list[float] = []

    run_main(
        tmp_path,
        "--label",
        "t",
        "--only",
        "cubic_vs_reno,rag_basics",
        run_fn=ScriptedRun(completed(), completed()),
        sleeps=sleeps,
    )

    assert sleeps == [60]


def test_the_pause_can_be_turned_off(tmp_path):
    sleeps: list[float] = []

    run_main(
        tmp_path,
        "--label",
        "t",
        "--only",
        "cubic_vs_reno,rag_basics",
        "--pause-seconds",
        "0",
        run_fn=ScriptedRun(completed(), completed()),
        sleeps=sleeps,
    )

    assert sleeps == []


@pytest.mark.parametrize("value", ["-1", "soon"])
def test_the_pause_must_be_a_non_negative_whole_number(tmp_path, value):
    with pytest.raises(SystemExit):
        main(["--label", "t", "--out-dir", str(tmp_path), "--pause-seconds", value, "--dry-run"])


def test_dry_run_shows_the_pause_and_the_time_it_adds(tmp_path):
    _, lines = run_main(tmp_path, "--label", "t", "--dry-run")

    # 8 runs means 7 pauses of 60 seconds.
    assert "Pause between runs: 60s (adds about 7 min)." in lines
