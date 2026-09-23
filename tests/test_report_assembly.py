"""Tests for deterministic final ResearchReport assembly."""

from __future__ import annotations

import pytest

from research_agent.graph.nodes.report_assembly import (
    assemble_final_report,
)
from research_agent.models.schemas import (
    Citation,
    ResearchReport,
)


def _citation(
    *,
    identifier: str = "cit-1",
) -> Citation:
    return Citation(
        id=identifier,
        claim_text="The evidence supports this claim.",
        evidence_ids=["ev-1"],
    )


def _state(
    *,
    original_question: object = "What does the evidence show?",
    draft_content: object = "The evidence supports this claim.",
    citations: object | None = None,
    final_report: object = None,
) -> dict:
    return {
        "original_question": original_question,
        "draft_content": draft_content,
        "citations": (
            [_citation()]
            if citations is None
            else citations
        ),
        "final_report": final_report,
    }


def test_assemble_final_report_returns_report_delta() -> None:
    citation = _citation()

    update = assemble_final_report(
        _state(
            citations=[citation],
        )
    )

    assert set(update) == {
        "final_report"
    }

    report = update["final_report"]

    assert isinstance(
        report,
        ResearchReport,
    )

    assert report.question == (
        "What does the evidence show?"
    )

    assert report.content == (
        "The evidence supports this claim."
    )

    assert report.citations == [
        citation
    ]


def test_assemble_final_report_does_not_mutate_input_state() -> None:
    state = _state()
    before = dict(state)

    assemble_final_report(
        state
    )

    assert state == before


def test_assemble_final_report_copies_citation_list() -> None:
    citations = [
        _citation(
            identifier="cit-1"
        ),
        _citation(
            identifier="cit-2"
        ),
    ]

    state = _state(
        citations=citations
    )

    update = assemble_final_report(
        state
    )

    report = update[
        "final_report"
    ]

    assert report.citations == citations
    assert report.citations is not citations


def test_assemble_final_report_created_at_is_timezone_aware() -> None:
    update = assemble_final_report(
        _state()
    )

    created_at = update[
        "final_report"
    ].created_at

    assert created_at.tzinfo is not None
    assert (
        created_at.utcoffset()
        is not None
    )


def test_assemble_final_report_allows_empty_citation_list() -> None:
    update = assemble_final_report(
        _state(
            citations=[]
        )
    )

    report = update[
        "final_report"
    ]

    assert report.citations == []


def test_assemble_final_report_rejects_existing_report() -> None:
    existing = ResearchReport(
        question="Existing question?",
        content="Existing report.",
        citations=[],
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Final report is already assembled"
        ),
    ):
        assemble_final_report(
            _state(
                final_report=existing
            )
        )


def test_assemble_final_report_rejects_invalid_existing_report_type() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "state final_report must be a "
            "ResearchReport object or None"
        ),
    ):
        assemble_final_report(
            _state(
                final_report="not-a-report"
            )
        )


def test_assemble_final_report_rejects_non_string_question() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "state original_question must be a string"
        ),
    ):
        assemble_final_report(
            _state(
                original_question=123
            )
        )


@pytest.mark.parametrize(
    "question",
    [
        "",
        "   ",
        "\n\t",
    ],
)
def test_assemble_final_report_rejects_blank_question(
    question: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "state original_question must not be blank"
        ),
    ):
        assemble_final_report(
            _state(
                original_question=question
            )
        )


def test_assemble_final_report_requires_finalized_draft_content() -> None:
    with pytest.raises(
        RuntimeError,
        match=(
            "Final report assembly requires finalized "
            "draft_content"
        ),
    ):
        assemble_final_report(
            _state(
                draft_content=None
            )
        )


def test_assemble_final_report_rejects_non_string_draft_content() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "state draft_content must be a string or None"
        ),
    ):
        assemble_final_report(
            _state(
                draft_content=123
            )
        )


@pytest.mark.parametrize(
    "content",
    [
        "",
        "   ",
        "\n\t",
    ],
)
def test_assemble_final_report_rejects_blank_draft_content(
    content: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "state draft_content must not be blank"
        ),
    ):
        assemble_final_report(
            _state(
                draft_content=content
            )
        )


def test_assemble_final_report_requires_citation_list() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "state citations must be a list"
        ),
    ):
        assemble_final_report(
            _state(
                citations="not-a-list"
            )
        )


def test_assemble_final_report_rejects_invalid_citation_items() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "state citations must contain only "
            "Citation objects"
        ),
    ):
        assemble_final_report(
            _state(
                citations=[
                    _citation(),
                    object(),
                ]
            )
        )