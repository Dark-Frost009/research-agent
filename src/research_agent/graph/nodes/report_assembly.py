"""Deterministic final ResearchReport assembly.

Final synthesis and semantic verification are completed before this node runs.

This module performs no LLM calls, network access, budget reservation, or
other provider side effects. It only converts already-durable finalization
state into the public ResearchReport model.
"""

from __future__ import annotations

from research_agent.graph.state import (
    ResearchState,
)
from research_agent.models.schemas import (
    Citation,
    ResearchReport,
)


def assemble_final_report(
    state: ResearchState,
) -> dict[str, ResearchReport]:
    """Build the final ResearchReport from validated durable state.

    Required inputs:

    - ``original_question``
    - ``draft_content``
    - ``citations``

    The node fails closed when finalization output is missing or malformed.

    An existing ``final_report`` is rejected rather than silently replaced.
    This prevents accidental graph replay from producing a second report with
    a different ``created_at`` timestamp.
    """

    existing_report = state[
        "final_report"
    ]

    if existing_report is not None:
        if not isinstance(
            existing_report,
            ResearchReport,
        ):
            raise TypeError(
                "state final_report must be a "
                "ResearchReport object or None."
            )

        raise RuntimeError(
            "Final report is already assembled."
        )

    original_question = state[
        "original_question"
    ]

    if not isinstance(
        original_question,
        str,
    ):
        raise TypeError(
            "state original_question must be a string."
        )

    if not original_question.strip():
        raise ValueError(
            "state original_question must not be blank."
        )

    draft_content = state[
        "draft_content"
    ]

    if draft_content is None:
        raise RuntimeError(
            "Final report assembly requires finalized "
            "draft_content."
        )

    if not isinstance(
        draft_content,
        str,
    ):
        raise TypeError(
            "state draft_content must be a string or None."
        )

    if not draft_content.strip():
        raise ValueError(
            "state draft_content must not be blank."
        )

    citations = state[
        "citations"
    ]

    if not isinstance(
        citations,
        list,
    ):
        raise TypeError(
            "state citations must be a list."
        )

    for citation in citations:
        if not isinstance(
            citation,
            Citation,
        ):
            raise TypeError(
                "state citations must contain only "
                "Citation objects."
            )

    report = ResearchReport(
        question=original_question,
        content=draft_content,
        citations=list(
            citations
        ),
    )

    return {
        "final_report": report
    }