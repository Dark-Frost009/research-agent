"""A thin runner for live evaluation batches.

It runs the questions in ``evals/live/questions.json`` through the SAME
function the Streamlit app uses (``ui_service.run_question``), scores each
finished run with ``live_metrics`` and saves one JSON file per run under
``.local/live-evals/<label>/``. It adds no research logic of its own.

Safety properties, each covered by an offline test:

* ``--dry-run`` shows the plan and makes no AI or search request.
* ``--max-runs`` caps how many runs one invocation may start.
* A provider quota or overload stops the batch at once. That is not a
  pipeline result, so the interrupted run is not saved as one.
* A finished run whose own issues say the provider throttled it is saved
  under a ``.limited.json`` name and ends the batch: its low evidence yield
  would measure the quota, not the pipeline.
* Re-running the same command resumes: runs that already have a result file
  are skipped, so a batch can be spread over several days of free-tier quota.
* Only non-secret facts are recorded (provider and model names, git commit).
  Exception messages are never printed; only the exception class name is.

Usage (from the project root)::

    python -m research_agent.live_runner --label a2-6f1de96 --dry-run
    python -m research_agent.live_runner --label a2-6f1de96 --max-runs 8
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from research_agent.depth import (
    DEPTH_NAMES,
    DepthConfigurationError,
    apply_depth,
)
from research_agent.incomplete import ResearchInterrupted
from research_agent.live_metrics import (
    ExpectationResult,
    RunMetrics,
    check_expectations,
    compute_run_metrics,
)
from research_agent.live_questions import LiveQuestion, load_questions

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUESTIONS = PROJECT_ROOT / "evals" / "live" / "questions.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / ".local" / "live-evals"

# What an interrupted run says when the provider, not the pipeline, stopped it.
STOP_REASONS = frozenset({"quota", "unavailable"})

# The fixed wording ``ui_service.summarize_issues`` uses when evidence
# extraction was throttled or the model was overloaded. A test builds the real
# messages and checks they are still detected, so a reworded message cannot
# silently disable this check.
_LIMITED_MARKERS = ("rate or quota limit", "temporarily unavailable")

# Observed on the Standard depth in live runs (11 to 12). The budget ceiling
# is much higher; the dry run shows both.
OBSERVED_AI_CALLS_PER_RUN = 12

_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


# --------------------------------------------------------------------------
# One saved run
# --------------------------------------------------------------------------


class RunRecord(BaseModel):
    """Everything saved for one run. Contains no secrets by construction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1] = 1
    label: str
    question_id: str
    run_index: int = Field(ge=1)
    started_at: AwareDatetime
    elapsed_seconds: float = Field(ge=0)
    depth: str
    llm_provider: str | None = None
    worker_model: str | None = None
    final_model: str | None = None
    git_commit: str | None = None
    status: Literal["completed", "interrupted"]
    # For an interrupted run: why it stopped (never an exception message).
    interruption_reason: str | None = None
    # The app's "Download evidence (.json)" content, kept whole so a later
    # comparison can recompute everything from the file alone.
    export: dict[str, Any]
    # True when the run's own issues say the provider throttled it.
    provider_limited: bool = False
    metrics: RunMetrics | None = None
    expectations: ExpectationResult | None = None

    @model_validator(mode="after")
    def _consistent(self) -> RunRecord:
        completed = self.status == "completed"

        if completed != (self.metrics is not None):
            raise ValueError("Only completed runs carry metrics.")

        if completed != (self.expectations is not None):
            raise ValueError("Only completed runs carry expectation results.")

        if completed == (self.interruption_reason is not None):
            raise ValueError(
                "An interrupted run needs a reason and a completed run must not have one."
            )

        return self


def is_provider_limited(export: dict[str, Any]) -> bool:
    """True if the run's own issues say the provider throttled or overloaded it."""

    return any(
        marker in str(issue)
        for issue in export.get("issues") or []
        for marker in _LIMITED_MARKERS
    )


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PlannedRun:
    question: LiveQuestion
    run_index: int
    path: Path

    @property
    def done(self) -> bool:
        return self.path.exists()


def result_path(
    batch_dir: Path,
    question_id: str,
    run_index: int,
    *,
    limited: bool = False,
) -> Path:
    suffix = ".limited.json" if limited else ".json"

    return batch_dir / f"{question_id}__{run_index}{suffix}"


def plan_runs(
    questions: Sequence[LiveQuestion],
    runs_per_question: int,
    batch_dir: Path,
) -> list[PlannedRun]:
    """Run 1 of every question first, then run 2, and so on.

    If quota ends the batch early, the runs that exist then cover every
    question once before any question gets a second run.
    """

    return [
        PlannedRun(
            question=question,
            run_index=index,
            path=result_path(batch_dir, question.id, index),
        )
        for index in range(1, runs_per_question + 1)
        for question in questions
    ]


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BatchResult:
    written: tuple[Path, ...]
    # Set when the provider ended the batch early ("quota", "unavailable"),
    # otherwise None.
    stopped_reason: str | None
    # Planned runs that still have no result after this invocation.
    remaining: int


RunFn = Callable[..., Any]


def _write_json_atomically(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _ignore_progress(_stage: str) -> None:
    return None


def _describe(record: RunRecord) -> str:
    if record.metrics is None or record.expectations is None:
        return f"interrupted ({record.interruption_reason})"

    flagged = record.expectations.pitfalls_flagged

    parts = [
        record.metrics.outcome,
        f"claims {record.metrics.claims}",
        f"sources cited {record.metrics.sources_cited}/{record.metrics.sources_with_evidence}",
        f"facts {record.expectations.facts_found}/{record.expectations.facts_total}",
        f"pitfalls {', '.join(flagged) if flagged else 'none'}",
    ]

    if record.provider_limited:
        parts.append("PROVIDER LIMITED")

    return " · ".join(parts)


def run_batch(
    planned: Sequence[PlannedRun],
    *,
    label: str,
    depth: str,
    run_fn: RunFn,
    settings: Any,
    commit: str | None,
    max_runs: int,
    echo: Callable[[str], None] = print,
) -> BatchResult:
    pending = [item for item in planned if not item.done]
    selected = pending[:max_runs]
    written: list[Path] = []

    worker_model = getattr(settings, "worker_llm_model", None)
    final_model = getattr(settings, "final_llm_model", None)
    provider = getattr(settings, "llm_provider", None)

    def finish(stopped: str | None) -> BatchResult:
        remaining = len([item for item in planned if not item.done])

        return BatchResult(tuple(written), stopped, remaining)

    for number, item in enumerate(selected, start=1):
        echo(f"[{number}/{len(selected)}] {item.question.id} run {item.run_index}")

        started = datetime.now(timezone.utc)
        began = time.monotonic()

        try:
            result = run_fn(
                item.question.question,
                _ignore_progress,
                depth=depth,
                settings=settings,
            )
        except ResearchInterrupted as interrupted:
            reason = interrupted.partial.reason

            if reason in STOP_REASONS:
                echo(
                    f"    stopped: the AI provider reported '{reason}'. "
                    "This run was not saved. Re-run the same command later to resume."
                )

                return finish(reason)

            export = json.loads(interrupted.partial.json_export())
            completed = False
        else:
            export = json.loads(result.json_export())
            completed = True
            reason = None

        elapsed = time.monotonic() - began
        limited = completed and is_provider_limited(export)

        record = RunRecord(
            label=label,
            question_id=item.question.id,
            run_index=item.run_index,
            started_at=started,
            elapsed_seconds=elapsed,
            depth=depth,
            llm_provider=provider,
            worker_model=worker_model,
            final_model=final_model,
            git_commit=commit,
            status="completed" if completed else "interrupted",
            interruption_reason=reason,
            export=export,
            provider_limited=limited,
            metrics=compute_run_metrics(export) if completed else None,
            expectations=(
                check_expectations(export, item.question.expectations)
                if completed
                else None
            ),
        )

        path = result_path(
            item.path.parent,
            item.question.id,
            item.run_index,
            limited=limited,
        )

        _write_json_atomically(
            path,
            record.model_dump_json(indent=2),
        )
        written.append(path)

        echo(f"    {_describe(record)}")

        if limited:
            echo(
                "    stopped: this run was throttled by the provider, so it was saved "
                "as a .limited file and is not counted. Re-run later to repeat it."
            )

            return finish("limited")

    return finish(None)


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def _label(value: str) -> str:
    if not _LABEL.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "use letters, digits, dot, underscore or hyphen (at most 64 characters)."
        )

    return value


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a whole number.") from error

    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1.")

    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m research_agent.live_runner",
        description="Run the live evaluation questions and save one result file per run.",
    )

    parser.add_argument(
        "--label",
        type=_label,
        required=True,
        help="Name of this batch, e.g. baseline-d5612ba. Results go in <out-dir>/<label>/.",
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--only",
        help="Comma-separated question ids to run (default: all).",
    )
    parser.add_argument(
        "--runs",
        type=_positive_int,
        default=1,
        help="Runs per question (default 1).",
    )
    parser.add_argument(
        "--max-runs",
        type=_positive_int,
        default=8,
        help="Most runs this invocation may start (default 8).",
    )
    parser.add_argument(
        "--depth",
        choices=DEPTH_NAMES,
        default="Standard",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the plan and exit without any AI or search request.",
    )

    return parser


def _default_settings() -> Any:
    from research_agent.config import Settings

    return Settings(_env_file=PROJECT_ROOT / ".env")


def _default_run_fn() -> RunFn:
    # Imported here so planning and tests do not load the whole graph stack.
    from research_agent.ui_service import run_question

    return run_question


def git_commit(root: Path = PROJECT_ROOT) -> str | None:
    """Short commit id of the checked-out code, with +dirty if it has edits."""

    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    value = head.stdout.strip()

    if head.returncode != 0 or not re.fullmatch(r"[0-9a-f]{7,40}", value):
        return None

    return value + ("+dirty" if status.stdout.strip() else "")


def main(
    argv: Sequence[str] | None = None,
    *,
    run_fn: RunFn | None = None,
    settings_loader: Callable[[], Any] | None = None,
    commit_fn: Callable[[], str | None] | None = None,
    echo: Callable[[str], None] = print,
) -> int:
    """Return 0 on success, 2 if the provider ended the batch, 1 on an error."""

    args = build_parser().parse_args(argv)

    try:
        question_set = load_questions(args.questions)
    except Exception as error:  # noqa: BLE001 - report any bad file plainly
        echo(f"The question file could not be loaded: {type(error).__name__}: {error}")
        return 1

    questions = list(question_set.questions)

    if args.only:
        wanted = [part.strip() for part in args.only.split(",") if part.strip()]
        known = {question.id for question in questions}
        unknown = [name for name in wanted if name not in known]

        if unknown:
            echo(f"Unknown question id(s): {', '.join(unknown)}")
            return 1

        questions = [question for question in questions if question.id in set(wanted)]

    batch_dir = Path(args.out_dir) / args.label
    planned = plan_runs(questions, args.runs, batch_dir)

    try:
        settings = (settings_loader or _default_settings)()
    except Exception as error:  # noqa: BLE001 - class name only, never the message
        settings = None
        settings_error = type(error).__name__
    else:
        settings_error = None

    if args.dry_run:
        return _dry_run(args, planned, settings, settings_error, echo)

    if settings is None:
        echo(
            f"Settings could not be loaded ({settings_error}). "
            "Check the project's .env file."
        )
        return 1

    try:
        apply_depth(settings, args.depth)
    except DepthConfigurationError as error:
        echo(str(error))
        return 1

    commit = (commit_fn or git_commit)()

    echo(
        f"Batch '{args.label}' at depth {args.depth}, code {commit or 'unknown'}, "
        f"output {batch_dir}"
    )

    try:
        result = run_batch(
            planned,
            label=args.label,
            depth=args.depth,
            run_fn=run_fn or _default_run_fn(),
            settings=settings,
            commit=commit,
            max_runs=args.max_runs,
            echo=echo,
        )
    except Exception as error:  # noqa: BLE001 - class name only, never the message
        echo(
            f"Batch aborted: {type(error).__name__}. Finished runs are saved; "
            "re-run the same command to resume."
        )
        return 1

    echo(
        f"Saved {len(result.written)} file(s). "
        f"{result.remaining} planned run(s) still have no result."
    )

    return 2 if result.stopped_reason else 0


def _dry_run(
    args: argparse.Namespace,
    planned: Sequence[PlannedRun],
    settings: Any,
    settings_error: str | None,
    echo: Callable[[str], None],
) -> int:
    pending = [item for item in planned if not item.done]
    selected = pending[: args.max_runs]

    echo("DRY RUN: no AI or search request will be made.")
    echo(f"Label: {args.label}   Depth: {args.depth}")
    echo(f"Output folder: {Path(args.out_dir) / args.label}")

    ai_ceiling: int | None = None
    search_ceiling: int | None = None

    if settings is None:
        echo(f"Settings: not loaded ({settings_error}); models and limits unknown.")
    else:
        echo(
            f"Provider: {getattr(settings, 'llm_provider', None)}   "
            f"worker model: {getattr(settings, 'worker_llm_model', None)}   "
            f"final model: {getattr(settings, 'final_llm_model', None)}"
        )

        try:
            _, chosen = apply_depth(settings, args.depth)
        except DepthConfigurationError as error:
            echo(str(error))
            return 1

        ai_ceiling = chosen.ai_calls
        search_ceiling = chosen.searches

    echo(
        f"Planned: {len(planned)} run(s); {len(planned) - len(pending)} already have "
        f"a result; {len(pending)} pending; this invocation would start "
        f"{len(selected)} (cap {args.max_runs})."
    )

    estimate = f"about {len(selected) * OBSERVED_AI_CALLS_PER_RUN} AI requests"

    if ai_ceiling is not None and search_ceiling is not None:
        estimate += (
            f" (observed ~{OBSERVED_AI_CALLS_PER_RUN} per run; the budget allows up to "
            f"{ai_ceiling} per run, so at most {len(selected) * ai_ceiling}) and at "
            f"most {len(selected) * search_ceiling} searches"
        )

    echo(f"Expected use: {estimate}. Check your provider's daily allowance first.")

    for number, item in enumerate(selected, start=1):
        echo(f"  {number}. {item.question.id} run {item.run_index}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
