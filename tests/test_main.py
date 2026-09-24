"""Tests for the production Research Agent CLI boundary."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from test_production_graph import _isolated_context

import research_agent.main as main_module
from research_agent.models.schemas import (
    Citation,
    ResearchReport,
)


QUESTION = "What is the evidence for this topic?"


def _report(
    *,
    citations: list[Citation] | None = None,
) -> ResearchReport:
    return ResearchReport(
        question=QUESTION,
        content="Grounded final answer.",
        citations=(
            []
            if citations is None
            else list(citations)
        ),
    )


def test_build_initial_state_builds_complete_state() -> None:
    state = main_module.build_initial_state(
        f"  {QUESTION}  "
    )

    assert state == {
        "original_question": QUESTION,
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


@pytest.mark.parametrize(
    "value",
    [
        None,
        123,
        [],
        {},
        True,
    ],
)
def test_build_initial_state_requires_string(
    value,
) -> None:
    with pytest.raises(
        TypeError,
        match="question must be a string",
    ):
        main_module.build_initial_state(
            value
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
def test_build_initial_state_rejects_blank_question(
    value: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="question must not be blank",
    ):
        main_module.build_initial_state(
            value
        )


def test_run_research_invokes_graph_with_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report()

    seen = {}

    class FakeGraph:
        def invoke(
            self,
            state,
            *,
            context,
            config,
        ):
            seen["state"] = state
            seen["context"] = context
            seen["config"] = config

            return {
                "final_report": report
            }

    fake_context = _isolated_context()

    application = SimpleNamespace(
        graph=FakeGraph(),
        context=fake_context,
    )

    monkeypatch.setattr(
        main_module,
        "build_research_application",
        lambda: application,
    )

    result = main_module.run_research(
        f"  {QUESTION}  "
    )

    assert result is report
    assert seen["config"]["recursion_limit"] >= 28

    assert seen["context"] is (
        fake_context
    )

    assert seen["state"][
        "original_question"
    ] == QUESTION

    assert seen["state"][
        "iteration_count"
    ] == 0


@pytest.mark.parametrize(
    "graph_result",
    [
        {},
        {"final_report": None},
        {"final_report": "not-report"},
        {"final_report": 123},
    ],
)
def test_run_research_requires_valid_final_report(
    monkeypatch: pytest.MonkeyPatch,
    graph_result,
) -> None:
    class FakeGraph:
        def invoke(
            self,
            state,
            *,
            context,
            config,
        ):
            return graph_result

    application = SimpleNamespace(
        graph=FakeGraph(),
        context=_isolated_context(),
    )

    monkeypatch.setattr(
        main_module,
        "build_research_application",
        lambda: application,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Research graph completed without "
            "a valid ResearchReport"
        ),
    ):
        main_module.run_research(
            QUESTION
        )


def test_print_report_without_citations(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main_module._print_report(
        _report()
    )

    output = capsys.readouterr()

    assert "RESEARCH REPORT" in (
        output.out
    )

    assert "Grounded final answer." in (
        output.out
    )

    assert "CITATIONS" not in (
        output.out
    )

    assert output.err == ""


def test_print_report_with_citations(
    capsys: pytest.CaptureFixture[str],
) -> None:
    citation = Citation(
        id="cit-1",
        claim_text=(
            "Grounded final answer."
        ),
        evidence_ids=[
            "ev-1",
            "ev-2",
        ],
    )

    main_module._print_report(
        _report(
            citations=[
                citation
            ]
        )
    )

    output = capsys.readouterr()

    assert "CITATIONS" in (
        output.out
    )

    assert "[cit-1]" in (
        output.out
    )

    assert (
        "Grounded final answer."
        in output.out
    )

    assert "ev-1, ev-2" in (
        output.out
    )

    assert output.err == ""


def test_main_returns_zero_and_prints_report(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = _report()

    seen = {}

    def fake_run_research(
        question: str,
    ) -> ResearchReport:
        seen["question"] = question
        return report

    monkeypatch.setattr(
        main_module,
        "run_research",
        fake_run_research,
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "research-agent",
            QUESTION,
        ],
    )

    result = main_module.main()

    output = capsys.readouterr()

    assert result == 0

    assert seen[
        "question"
    ] == QUESTION

    assert "RESEARCH REPORT" in (
        output.out
    )

    assert (
        "Grounded final answer."
        in output.out
    )

    assert output.err == ""


def test_main_returns_one_when_research_fails(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(
        question: str,
    ) -> ResearchReport:
        raise RuntimeError(
            "simulated failure"
        )

    monkeypatch.setattr(
        main_module,
        "run_research",
        fail,
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "research-agent",
            QUESTION,
        ],
    )

    result = main_module.main()

    output = capsys.readouterr()

    assert result == 1

    assert output.out == ""

    assert (
        "Research failed: "
        "simulated failure"
        in output.err
    )
