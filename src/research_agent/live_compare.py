"""Compare two live-evaluation batches, for example a baseline and a change.

Usage (from the project root)::

    python -m research_agent.live_compare .local/live-evals/baseline-d5612ba \
        .local/live-evals/a2-1f00044

Design choices, each covered by an offline test:

* Both batches are re-scored with the CURRENT ``live_metrics`` code and
  ``questions.json``, from each run's saved export. Numbers stored at run time
  are ignored, so a change in scoring can never look like a change in the
  research pipeline.
* ``.limited.json`` files (runs the provider throttled) are skipped. Runs the
  pipeline did not finish are counted per batch but kept out of the averages,
  so a version that finishes less often still shows it.
* Each batch's code commit, models and depth are shown, with a warning when a
  batch mixes them or the two batches differ in model or depth. Those would be
  mixed into any difference.
* Per question: mean [min-max] (n) for each batch. Overall: the average of
  per-question averages over questions both batches answered, so a question
  with more runs cannot dominate.
* No significance test. With a few runs per question it would give false
  confidence; the ranges are shown instead.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from research_agent.live_metrics import check_expectations, compute_run_metrics
from research_agent.live_questions import LiveQuestion, load_questions
from research_agent.live_runner import DEFAULT_QUESTIONS, RunRecord

# (key, label) in display order.
METRICS: tuple[tuple[str, str], ...] = (
    ("verified", "verified rate"),
    ("chain_valid", "citation chain valid"),
    ("claims", "claims"),
    ("sources_cited", "sources cited"),
    ("source_coverage", "source coverage"),
    ("duplicate_claims", "duplicate claims"),
    ("facts_found", "facts found (share)"),
    ("pitfalls", "pitfalls flagged"),
    ("abstained", "abstained correctly"),
    ("ai_calls", "AI calls used"),
)

NOTE = (
    "Note: a few runs per question are not enough for a significance test. "
    "Treat a difference smaller than the min-max ranges as noise, and read the "
    "answers themselves before drawing a conclusion."
)


class CompareError(ValueError):
    """A batch folder cannot be compared (missing, empty or unreadable)."""


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


@dataclass
class Batch:
    path: Path
    completed: dict[str, list[RunRecord]] = field(default_factory=dict)
    interrupted: dict[str, int] = field(default_factory=dict)
    limited_skipped: int = 0
    commits: set[str] = field(default_factory=set)
    models: set[str] = field(default_factory=set)
    depths: set[str] = field(default_factory=set)

    @property
    def completed_runs(self) -> int:
        return sum(len(runs) for runs in self.completed.values())

    @property
    def interrupted_runs(self) -> int:
        return sum(self.interrupted.values())


def load_batch(path: Path) -> Batch:
    """Read every result file in one batch folder. Raises on a bad file."""

    path = Path(path)

    if not path.is_dir():
        raise CompareError(f"Not a folder: {path}")

    batch = Batch(path=path)

    for file in sorted(path.glob("*.json")):
        if file.name.endswith(".limited.json"):
            batch.limited_skipped += 1
            continue

        try:
            record = RunRecord.model_validate_json(file.read_text(encoding="utf-8"))
        except Exception as error:  # noqa: BLE001 - name the file, not its content
            raise CompareError(
                f"{file.name} is not a valid result file ({type(error).__name__})."
            ) from error

        batch.commits.add(record.git_commit or "unknown")
        batch.models.add(f"{record.worker_model} / {record.final_model}")
        batch.depths.add(record.depth)

        if record.status == "completed":
            batch.completed.setdefault(record.question_id, []).append(record)
        else:
            batch.interrupted[record.question_id] = (
                batch.interrupted.get(record.question_id, 0) + 1
            )

    if not batch.completed and not batch.interrupted:
        raise CompareError(f"No result files in {path}")

    return batch


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def score_run(record: RunRecord, question: LiveQuestion) -> dict[str, float | None]:
    """Re-score one finished run with the current metrics and expectations."""

    metrics = compute_run_metrics(record.export)
    expected = check_expectations(record.export, question.expectations)
    ai_calls = metrics.budgets.get("ai_calls")

    # For a question with no real answer, "verified" would be the wrong
    # outcome; that question is scored by "abstained" instead.
    if question.expectations.expect_abstain:
        verified = None
    else:
        verified = 1.0 if metrics.outcome == "verified" else 0.0

    return {
        "verified": verified,
        "chain_valid": 1.0 if metrics.chain_valid else 0.0,
        "claims": float(metrics.claims),
        "sources_cited": float(metrics.sources_cited),
        "source_coverage": metrics.source_coverage,
        "duplicate_claims": float(metrics.duplicate_claims),
        "facts_found": (
            expected.facts_found / expected.facts_total
            if expected.facts_total
            else None
        ),
        "pitfalls": float(len(expected.pitfalls_flagged)),
        "abstained": (
            None
            if expected.abstain_ok is None
            else (1.0 if expected.abstain_ok else 0.0)
        ),
        "ai_calls": float(ai_calls.used) if ai_calls is not None else None,
    }


@dataclass(frozen=True)
class Summary:
    mean: float
    low: float
    high: float
    n: int


def summarize(values: Sequence[float | None]) -> Summary | None:
    """Mean and range of the defined values, or None if there are none."""

    present = [value for value in values if value is not None]

    if not present:
        return None

    return Summary(
        mean=sum(present) / len(present),
        low=min(present),
        high=max(present),
        n=len(present),
    )


def question_summaries(
    batch: Batch,
    question: LiveQuestion,
) -> dict[str, Summary | None]:
    scores = [score_run(record, question) for record in batch.completed.get(question.id, [])]

    return {key: summarize([score[key] for score in scores]) for key, _ in METRICS}


# --------------------------------------------------------------------------
# Comparing
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Comparison:
    common: list[str]
    only_a: list[str]
    only_b: list[str]
    unknown: list[str]
    per_question: dict[str, tuple[dict[str, Summary | None], dict[str, Summary | None]]]
    # Average of per-question means over questions where BOTH batches have a value.
    overall: dict[str, tuple[float, float, int] | None]
    warnings: list[str]


def compare(a: Batch, b: Batch, questions: Sequence[LiveQuestion]) -> Comparison:
    by_id = {question.id: question for question in questions}

    answered_a = set(a.completed)
    answered_b = set(b.completed)

    unknown = sorted((answered_a | answered_b) - set(by_id))
    common = [q.id for q in questions if q.id in answered_a and q.id in answered_b]
    only_a = [q.id for q in questions if q.id in answered_a and q.id not in answered_b]
    only_b = [q.id for q in questions if q.id in answered_b and q.id not in answered_a]

    per_question = {
        question_id: (
            question_summaries(a, by_id[question_id]),
            question_summaries(b, by_id[question_id]),
        )
        for question_id in common
    }

    overall: dict[str, tuple[float, float, int] | None] = {}

    for key, _ in METRICS:
        pairs = [
            (left[key].mean, right[key].mean)
            for left, right in per_question.values()
            if left[key] is not None and right[key] is not None
        ]
        overall[key] = (
            (
                sum(pair[0] for pair in pairs) / len(pairs),
                sum(pair[1] for pair in pairs) / len(pairs),
                len(pairs),
            )
            if pairs
            else None
        )

    return Comparison(
        common=common,
        only_a=only_a,
        only_b=only_b,
        unknown=unknown,
        per_question=per_question,
        overall=overall,
        warnings=_warnings(a, b),
    )


def _warnings(a: Batch, b: Batch) -> list[str]:
    warnings: list[str] = []

    for name, batch in (("A", a), ("B", b)):
        if len(batch.commits) > 1:
            warnings.append(f"Batch {name} mixes code versions: {', '.join(sorted(batch.commits))}.")
        if any(commit.endswith("+dirty") for commit in batch.commits):
            warnings.append(f"Batch {name} ran with uncommitted edits (+dirty).")
        if len(batch.models) > 1:
            warnings.append(f"Batch {name} mixes models: {'; '.join(sorted(batch.models))}.")
        if len(batch.depths) > 1:
            warnings.append(f"Batch {name} mixes depths: {', '.join(sorted(batch.depths))}.")

    if a.models != b.models:
        warnings.append("The batches used different models, so model and code changes are mixed.")
    if a.depths != b.depths:
        warnings.append("The batches used different depths.")
    if a.commits == b.commits:
        warnings.append("Both batches ran the same code version.")

    return warnings


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def _number(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")

    return "0" if text == "-0" else text


def _cell(summary: Summary | None) -> str:
    if summary is None:
        return "-"

    if summary.low == summary.high:
        return f"{_number(summary.mean)} (n={summary.n})"

    return f"{_number(summary.mean)} [{_number(summary.low)}-{_number(summary.high)}] (n={summary.n})"


def _completion(batch: Batch) -> str:
    total = batch.completed_runs + batch.interrupted_runs

    return f"{batch.completed_runs}/{total} runs finished"


def format_report(a: Batch, b: Batch, result: Comparison) -> list[str]:
    lines: list[str] = []

    for name, batch in (("A", a), ("B", b)):
        lines.append(
            f"{name}: {batch.path.name} | code {', '.join(sorted(batch.commits))} | "
            f"models {'; '.join(sorted(batch.models))} | depth {', '.join(sorted(batch.depths))} | "
            f"{_completion(batch)} | {batch.limited_skipped} throttled file(s) skipped"
        )

    for warning in result.warnings:
        lines.append(f"WARNING: {warning}")

    if result.only_a:
        lines.append(f"Only in A (not compared): {', '.join(result.only_a)}")
    if result.only_b:
        lines.append(f"Only in B (not compared): {', '.join(result.only_b)}")
    if result.unknown:
        lines.append(f"Not in the question file (ignored): {', '.join(result.unknown)}")

    width = max(len(label) for _, label in METRICS) + 2

    for question_id in result.common:
        left, right = result.per_question[question_id]
        lines.append("")
        lines.append(f"{question_id}")
        lines.append(f"  {'':<{width}}{'A':<28}B")

        for key, label in METRICS:
            if left[key] is None and right[key] is None:
                continue
            lines.append(f"  {label:<{width}}{_cell(left[key]):<28}{_cell(right[key])}")

    lines.append("")
    lines.append(f"OVERALL (average of per-question means, {len(result.common)} question(s) in both)")
    lines.append(f"  {'':<{width}}{'A':<12}{'B':<12}questions")

    for key, label in METRICS:
        value = result.overall[key]
        if value is None:
            continue
        lines.append(f"  {label:<{width}}{_number(value[0]):<12}{_number(value[1]):<12}{value[2]}")

    lines.append("")
    lines.append(NOTE)

    return lines


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def main(
    argv: Sequence[str] | None = None,
    *,
    echo: Callable[[str], None] = print,
) -> int:
    """Return 0 on success and 1 if the batches cannot be compared."""

    parser = argparse.ArgumentParser(
        prog="python -m research_agent.live_compare",
        description="Compare two live-evaluation batch folders (A = before, B = after).",
    )
    parser.add_argument("batch_a", type=Path, help="Folder of batch A, e.g. the baseline.")
    parser.add_argument("batch_b", type=Path, help="Folder of batch B, e.g. the change.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    args = parser.parse_args(argv)

    try:
        questions = list(load_questions(args.questions).questions)
        a = load_batch(args.batch_a)
        b = load_batch(args.batch_b)
    except CompareError as error:
        echo(str(error))
        return 1
    except Exception as error:  # noqa: BLE001 - e.g. a broken question file
        echo(f"Could not load the inputs: {type(error).__name__}.")
        return 1

    result = compare(a, b, questions)

    if not result.common:
        echo("The two batches have no finished question in common; nothing to compare.")
        return 1

    for line in format_report(a, b, result):
        echo(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
