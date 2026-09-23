"""Unit tests for research_agent.models.schemas.

These tests check our own schema contract and the guardrails we added
on top of Pydantic (shared config, min_length invariants, AwareDatetime
enforcement, numeric floors) - not Pydantic's own correctness. No
network access, mocks, LLM calls, or LangGraph involved.
"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from research_agent.models.schemas import (
    Citation,
    CritiqueResult,
    Evidence,
    ResearchReport,
    RunMetrics,
    SearchResult,
    Source,
    SubQuestion,
)


# ---------------------------------------------------------------------------
# Happy-path construction (requirement 1), including relevant defaults
# ---------------------------------------------------------------------------


def test_subquestion_happy_path():
    sub_question = SubQuestion(
        id="sq_1",
        question="What is the boiling point of water at sea level?",
        rationale="Establishes a baseline fact needed for later comparisons.",
    )
    assert sub_question.id == "sq_1"
    assert sub_question.created_at_iteration == 0  # default


def test_search_result_happy_path():
    result = SearchResult(
        sub_question_id="sq_1",
        query="boiling point of water sea level",
        title="Boiling point - Wikipedia",
        url="https://en.wikipedia.org/wiki/Boiling_point",
        snippet="At sea level, water boils at 100 C.",
        rank=1,
        provider="tavily",
    )
    assert result.provider == "tavily"
    assert result.retrieved_at.utcoffset() is not None  # default factory is aware


def test_source_happy_path():
    source = Source(
        id="src_abc123",
        url="https://en.wikipedia.org/wiki/Boiling_point",
        title="Boiling point",
        domain="en.wikipedia.org",
    )
    assert source.fetch_status == "pending"  # default
    assert source.fetched_at is None
    assert source.final_url is None


def test_evidence_happy_path():
    evidence = Evidence(
        id="ev_1",
        source_id="src_abc123",
        sub_question_id="sq_1",
        excerpt="Water boils at 100 degrees Celsius at sea level.",
        relevance_note="Directly answers the sub-question.",
    )
    assert evidence.source_id == "src_abc123"


def test_citation_happy_path():
    citation = Citation(
        id="cite_1",
        claim_text="Water boils at 100 C at sea level.",
        evidence_ids=["ev_1"],
    )
    assert citation.evidence_ids == ["ev_1"]


def test_critique_result_happy_path():
    critique = CritiqueResult(
        sufficient=False,
        reasoning="Missing altitude effects.",
    )
    assert critique.sufficient is False
    assert critique.gaps == []  # default
    assert critique.follow_up_questions == []  # default


def test_research_report_happy_path():
    report = ResearchReport(
        question="What is the boiling point of water at sea level?",
        content="Water boils at 100 C at sea level [cite_1].",
        citations=[
            Citation(
                id="cite_1",
                claim_text="Boils at 100 C.",
                evidence_ids=["ev_1"],
            )
        ],
    )
    assert len(report.citations) == 1
    assert report.created_at.tzinfo is not None  # default factory is aware


def test_run_metrics_happy_path():
    metrics = RunMetrics(
        run_id="run_1",
        iterations_used=1,
        total_search_queries=3,
        total_sources=2,
        total_evidence_extracted=4,
    )
    assert metrics.total_search_queries == 3
    assert metrics.errors == []  # default


# ---------------------------------------------------------------------------
# Shared schema configuration (requirement 2)
# ---------------------------------------------------------------------------


def test_subquestion_strips_whitespace():
    sub_question = SubQuestion(
        id="  sq_1  ",
        question="  What is X?  ",
    )
    assert sub_question.id == "sq_1"
    assert sub_question.question == "What is X?"


def test_source_strips_whitespace():
    source = Source(
        id="  src_1  ",
        url="  https://example.com  ",
        final_url="  https://example.com/final  ",
        domain="  example.com  ",
    )

    assert source.id == "src_1"
    assert source.url == "https://example.com"
    assert source.final_url == "https://example.com/final"
    assert source.domain == "example.com"


def test_subquestion_rejects_unexpected_fields():
    with pytest.raises(ValidationError):
        SubQuestion(
            id="sq_1",
            question="What is X?",
            unexpected_field="oops",
        )


def test_source_rejects_unexpected_fields():
    with pytest.raises(ValidationError):
        Source(
            id="src_1",
            url="https://example.com",
            domain="example.com",
            unexpected_field="oops",
        )


# ---------------------------------------------------------------------------
# Required non-empty fields (requirement 3)
# ---------------------------------------------------------------------------


def _valid_subquestion_kwargs():
    return dict(
        id="sq_1",
        question="What is X?",
    )


def _valid_search_result_kwargs():
    return dict(
        sub_question_id="sq_1",
        query="X query",
        title="X title",
        url="https://example.com",
        provider="tavily",
    )


def _valid_source_kwargs():
    return dict(
        id="src_1",
        url="https://example.com",
        domain="example.com",
    )


def _valid_evidence_kwargs():
    return dict(
        id="ev_1",
        source_id="src_1",
        excerpt="X is true.",
    )


def _valid_citation_kwargs():
    return dict(
        id="cite_1",
        claim_text="X is true.",
        evidence_ids=["ev_1"],
    )


def _valid_research_report_kwargs():
    return dict(
        question="What is X?",
        content="X is true.",
    )


def _valid_run_metrics_kwargs():
    return dict(
        run_id="run_1",
        iterations_used=1,
        total_search_queries=1,
        total_sources=1,
        total_evidence_extracted=1,
    )


REQUIRED_NON_EMPTY_FIELDS = [
    (SubQuestion, _valid_subquestion_kwargs, "id"),
    (SubQuestion, _valid_subquestion_kwargs, "question"),
    (SearchResult, _valid_search_result_kwargs, "sub_question_id"),
    (SearchResult, _valid_search_result_kwargs, "query"),
    (SearchResult, _valid_search_result_kwargs, "title"),
    (SearchResult, _valid_search_result_kwargs, "url"),
    (SearchResult, _valid_search_result_kwargs, "provider"),
    (Source, _valid_source_kwargs, "id"),
    (Source, _valid_source_kwargs, "url"),
    (Source, _valid_source_kwargs, "domain"),
    (Evidence, _valid_evidence_kwargs, "id"),
    (Evidence, _valid_evidence_kwargs, "source_id"),
    (Evidence, _valid_evidence_kwargs, "excerpt"),
    (Citation, _valid_citation_kwargs, "id"),
    (Citation, _valid_citation_kwargs, "claim_text"),
    (ResearchReport, _valid_research_report_kwargs, "question"),
    (ResearchReport, _valid_research_report_kwargs, "content"),
    (RunMetrics, _valid_run_metrics_kwargs, "run_id"),
]


@pytest.mark.parametrize(
    "model_cls, valid_kwargs_factory, field_name",
    REQUIRED_NON_EMPTY_FIELDS,
    ids=[
        f"{cls.__name__}.{field}"
        for cls, _, field in REQUIRED_NON_EMPTY_FIELDS
    ],
)
def test_empty_string_rejected_for_required_fields(
    model_cls,
    valid_kwargs_factory,
    field_name,
):
    kwargs = valid_kwargs_factory()
    kwargs[field_name] = ""

    with pytest.raises(ValidationError):
        model_cls(**kwargs)


# ---------------------------------------------------------------------------
# SubQuestion (requirement 4)
# ---------------------------------------------------------------------------


def test_subquestion_negative_iteration_rejected():
    with pytest.raises(ValidationError):
        SubQuestion(
            id="sq_1",
            question="What is X?",
            created_at_iteration=-1,
        )


# ---------------------------------------------------------------------------
# SearchResult (requirement 5)
# ---------------------------------------------------------------------------


def test_search_result_naive_retrieved_at_rejected():
    with pytest.raises(ValidationError):
        SearchResult(
            **_valid_search_result_kwargs(),
            retrieved_at=datetime(2024, 1, 1),
        )


@pytest.mark.parametrize(
    "rank",
    [
        1,
        5,
    ],
)
def test_search_result_rank_accepts_positive_values(rank):
    result = SearchResult(
        **_valid_search_result_kwargs(),
        rank=rank,
    )
    assert result.rank == rank


@pytest.mark.parametrize(
    "rank",
    [
        0,
        -1,
    ],
)
def test_search_result_rank_rejects_non_positive_values(rank):
    with pytest.raises(ValidationError):
        SearchResult(
            **_valid_search_result_kwargs(),
            rank=rank,
        )


# ---------------------------------------------------------------------------
# Source (requirement 6)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status",
    [
        "pending",
        "success",
        "failed",
        "skipped",
    ],
)
def test_source_accepts_known_fetch_statuses(status):
    source = Source(
        **_valid_source_kwargs(),
        fetch_status=status,
    )
    assert source.fetch_status == status


def test_source_rejects_unknown_fetch_status():
    with pytest.raises(ValidationError):
        Source(
            **_valid_source_kwargs(),
            fetch_status="unknown",
        )


def test_source_fetched_at_none_is_valid():
    source = Source(
        **_valid_source_kwargs(),
        fetched_at=None,
    )
    assert source.fetched_at is None


def test_source_naive_fetched_at_rejected():
    with pytest.raises(ValidationError):
        Source(
            **_valid_source_kwargs(),
            fetched_at=datetime(2024, 1, 1),
        )


def test_source_aware_fetched_at_is_valid():
    aware = datetime(
        2024,
        1,
        1,
        tzinfo=timezone.utc,
    )

    source = Source(
        **_valid_source_kwargs(),
        fetched_at=aware,
    )

    assert source.fetched_at == aware


def test_source_final_url_none_is_valid():
    source = Source(
        **_valid_source_kwargs(),
        final_url=None,
    )

    assert source.final_url is None


def test_source_final_url_accepts_non_empty_value():
    source = Source(
        **_valid_source_kwargs(),
        final_url="https://example.com/final",
    )

    assert source.final_url == "https://example.com/final"


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
def test_source_final_url_rejects_blank_value(value):
    with pytest.raises(ValidationError):
        Source(
            **_valid_source_kwargs(),
            final_url=value,
        )


# ---------------------------------------------------------------------------
# Citation (requirement 7)
# ---------------------------------------------------------------------------


def test_citation_requires_at_least_one_evidence_id():
    with pytest.raises(ValidationError):
        Citation(
            id="cite_1",
            claim_text="X is true.",
            evidence_ids=[],
        )


# ---------------------------------------------------------------------------
# ResearchReport (requirement 8)
# ---------------------------------------------------------------------------


def test_research_report_naive_created_at_rejected():
    with pytest.raises(ValidationError):
        ResearchReport(
            **_valid_research_report_kwargs(),
            created_at=datetime(2024, 1, 1),
        )


# ---------------------------------------------------------------------------
# RunMetrics (requirement 9)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field_name, negative_value",
    [
        ("iterations_used", -1),
        ("total_search_queries", -1),
        ("total_sources", -1),
        ("total_evidence_extracted", -1),
        ("total_llm_calls", -1),
        ("duration_seconds", -1.0),
    ],
)
def test_run_metrics_rejects_negative_values(
    field_name,
    negative_value,
):
    kwargs = _valid_run_metrics_kwargs()
    kwargs[field_name] = negative_value

    with pytest.raises(ValidationError):
        RunMetrics(**kwargs)


def test_run_metrics_zero_is_valid_everywhere():
    metrics = RunMetrics(
        run_id="run_1",
        iterations_used=0,
        total_search_queries=0,
        total_sources=0,
        total_evidence_extracted=0,
        total_llm_calls=0,
        duration_seconds=0.0,
    )

    assert metrics.iterations_used == 0
    assert metrics.total_llm_calls == 0
    assert metrics.duration_seconds == 0.0