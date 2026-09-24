"""Command-line entry point for the production Research Agent."""

from __future__ import annotations

import argparse
import sys
from research_agent.graph.execution import research_run_config

from research_agent.bootstrap import (
    build_research_application,
)
from research_agent.graph.state import (
    ResearchState,
)
from research_agent.models.schemas import (
    ResearchReport,
)


def build_initial_state(
    question: str,
) -> ResearchState:
    """Create the complete initial state for one research run."""

    if not isinstance(
        question,
        str,
    ):
        raise TypeError(
            "question must be a string."
        )

    clean_question = question.strip()

    if not clean_question:
        raise ValueError(
            "question must not be blank."
        )

    return {
        "original_question": clean_question,
        "sub_questions": [],
        "search_results": [],
        "sources": [],
        "evidence": [],
        "draft_content": None,
        "citations": [],
        "critique": None,
        "iteration_count": 0,
        "search_queries_used": 0,
        "source_fetches_used": 0,
        "llm_calls_used": 0,
        "final_report": None,
        "errors": [],
    }


def run_research(
    question: str,
) -> ResearchReport:
    """Execute one complete production research run."""

    application = (
        build_research_application()
    )

    result = application.graph.invoke(
        build_initial_state(
            question
        ),
        context=application.context,
        config=research_run_config(application.context.budget_policy.limits),
    )

    report = result.get(
        "final_report"
    )

    if not isinstance(
        report,
        ResearchReport,
    ):
        raise RuntimeError(
            "Research graph completed without "
            "a valid ResearchReport."
        )

    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research-agent",
        description=(
            "Run the bounded LangGraph "
            "research agent."
        ),
    )

    parser.add_argument(
        "question",
        help=(
            "Research question to investigate."
        ),
    )

    return parser


def _print_report(
    report: ResearchReport,
) -> None:
    """Render a ResearchReport for terminal users."""

    print()
    print("RESEARCH REPORT")
    print("=" * 60)
    print()
    print(report.content)

    if report.citations:
        print()
        print("CITATIONS")
        print("-" * 60)

        for citation in report.citations:
            evidence_ids = ", ".join(
                citation.evidence_ids
            )

            print(
                f"[{citation.id}] "
                f"{citation.claim_text}"
            )
            print(
                f"  Evidence: "
                f"{evidence_ids}"
            )


def main() -> int:
    """Run the command-line application."""

    parser = _build_parser()

    args = parser.parse_args()

    try:
        report = run_research(
            args.question
        )
    except Exception as exc:
        print(
            f"Research failed: {exc}",
            file=sys.stderr,
        )
        return 1

    _print_report(
        report
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
