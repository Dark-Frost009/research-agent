"""Tests for the ResearchState source reducer.

These tests focus on our reducer contract rather than LangGraph itself.

The key properties we care about are:
- deterministic deduplication by Source.id
- fetch-status priority
- metadata-completeness preference
- non-mutation
- idempotence
- commutativity
- associativity

Those properties matter because merge_sources may eventually be used
when parallel graph branches contribute source updates.
"""

from datetime import datetime, timezone

import pytest

from research_agent.graph.state import merge_sources
from research_agent.models.schemas import Source


def _source(
    source_id: str,
    *,
    status: str = "pending",
    title: str | None = None,
    content_type: str | None = None,
    fetched_at: datetime | None = None,
    url: str | None = None,
    domain: str = "example.com",
) -> Source:
    """Build a small valid Source for reducer tests."""
    return Source(
        id=source_id,
        url=url or f"https://example.com/{source_id}",
        title=title,
        domain=domain,
        content_type=content_type,
        fetch_status=status,
        fetched_at=fetched_at,
    )


# ---------------------------------------------------------------------------
# Basic merging
# ---------------------------------------------------------------------------


def test_merge_sources_empty_lists_returns_empty_list():
    assert merge_sources([], []) == []


def test_merge_sources_combines_distinct_ids():
    source_a = _source("src_a")
    source_b = _source("src_b")

    result = merge_sources([source_a], [source_b])

    assert result == [source_a, source_b]


def test_merge_sources_sorts_output_by_source_id():
    source_b = _source("src_b")
    source_a = _source("src_a")
    source_c = _source("src_c")

    result = merge_sources([source_b], [source_c, source_a])

    assert [source.id for source in result] == ["src_a", "src_b", "src_c"]


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def test_duplicate_source_id_produces_one_source():
    pending = _source("src_same", status="pending")
    success = _source("src_same", status="success")

    result = merge_sources([pending], [success])

    assert len(result) == 1
    assert result[0].id == "src_same"
    assert result[0].fetch_status == "success"


def test_duplicates_inside_existing_are_resolved():
    pending = _source("src_same", status="pending")
    success = _source("src_same", status="success")

    result = merge_sources([pending, success], [])

    assert len(result) == 1
    assert result[0].fetch_status == "success"


def test_duplicates_inside_incoming_are_resolved():
    skipped = _source("src_same", status="skipped")
    failed = _source("src_same", status="failed")

    result = merge_sources([], [skipped, failed])

    assert len(result) == 1
    assert result[0].fetch_status == "failed"


# ---------------------------------------------------------------------------
# Fetch-status priority
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "first_status, second_status, expected_status",
    [
        ("pending", "success", "success"),
        ("pending", "failed", "failed"),
        ("pending", "skipped", "skipped"),
        ("failed", "success", "success"),
        ("skipped", "success", "success"),
        ("success", "failed", "success"),
        ("success", "pending", "success"),
        ("failed", "skipped", "failed"),
    ],
)
def test_fetch_status_priority(first_status, second_status, expected_status):
    first = _source("src_same", status=first_status)
    second = _source("src_same", status=second_status)

    result = merge_sources([first], [second])

    assert len(result) == 1
    assert result[0].fetch_status == expected_status


@pytest.mark.parametrize(
    "low_status, high_status",
    [
        ("pending", "skipped"),
        ("pending", "failed"),
        ("pending", "success"),
        ("skipped", "failed"),
        ("skipped", "success"),
        ("failed", "success"),
    ],
)
def test_fetch_status_priority_is_independent_of_argument_order(
    low_status,
    high_status,
):
    low = _source("src_same", status=low_status)
    high = _source("src_same", status=high_status)

    forward = merge_sources([low], [high])
    reverse = merge_sources([high], [low])

    assert forward == reverse
    assert forward[0].fetch_status == high_status


# ---------------------------------------------------------------------------
# Metadata completeness
# ---------------------------------------------------------------------------


def test_more_complete_source_wins_when_status_is_equal():
    less_complete = _source(
        "src_same",
        status="success",
        title="Example",
    )

    more_complete = _source(
        "src_same",
        status="success",
        title="Example",
        content_type="text/html",
        fetched_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    result = merge_sources([less_complete], [more_complete])

    assert result == [more_complete]


def test_metadata_completeness_is_independent_of_argument_order():
    less_complete = _source(
        "src_same",
        status="failed",
        title="Example",
    )

    more_complete = _source(
        "src_same",
        status="failed",
        title="Example",
        content_type="text/html",
    )

    forward = merge_sources([less_complete], [more_complete])
    reverse = merge_sources([more_complete], [less_complete])

    assert forward == reverse
    assert forward == [more_complete]


# ---------------------------------------------------------------------------
# Deterministic final tie-break
# ---------------------------------------------------------------------------


def test_equal_priority_and_completeness_resolve_deterministically():
    source_a = _source(
        "src_same",
        status="success",
        title="Alpha",
        content_type="text/html",
    )

    source_b = _source(
        "src_same",
        status="success",
        title="Beta",
        content_type="text/html",
    )

    forward = merge_sources([source_a], [source_b])
    reverse = merge_sources([source_b], [source_a])

    assert forward == reverse
    assert len(forward) == 1
    assert forward[0] in (source_a, source_b)


# ---------------------------------------------------------------------------
# Non-mutation
# ---------------------------------------------------------------------------


def test_merge_sources_does_not_mutate_input_lists_or_sources():
    existing_source = _source("src_a", status="pending", title="Existing")
    incoming_source = _source("src_a", status="success", title="Incoming")

    existing = [existing_source]
    incoming = [incoming_source]

    existing_before = [source.model_copy(deep=True) for source in existing]
    incoming_before = [source.model_copy(deep=True) for source in incoming]

    result = merge_sources(existing, incoming)

    assert existing == existing_before
    assert incoming == incoming_before

    assert result is not existing
    assert result is not incoming


# ---------------------------------------------------------------------------
# Reducer algebraic properties
# ---------------------------------------------------------------------------


def test_merge_sources_is_idempotent():
    sources = [
        _source("src_b", status="pending"),
        _source("src_a", status="failed"),
        _source("src_a", status="success"),
    ]

    normalized = merge_sources([], sources)
    repeated = merge_sources(sources, sources)

    assert repeated == normalized


def test_merge_sources_is_commutative():
    fetched_at = datetime(2026, 1, 1, tzinfo=timezone.utc)

    group_a = [
        _source("src_a", status="pending"),
        _source("src_b", status="failed", title="B"),
    ]

    group_b = [
        _source(
            "src_a",
            status="success",
            title="A",
            content_type="text/html",
            fetched_at=fetched_at,
        ),
        _source("src_c", status="skipped"),
    ]

    assert merge_sources(group_a, group_b) == merge_sources(group_b, group_a)


def test_merge_sources_is_associative():
    fetched_at = datetime(2026, 1, 1, tzinfo=timezone.utc)

    group_a = [
        _source("src_a", status="pending"),
        _source("src_b", status="failed"),
    ]

    group_b = [
        _source(
            "src_a",
            status="success",
            title="A",
        ),
        _source("src_c", status="skipped"),
    ]

    group_c = [
        _source(
            "src_a",
            status="success",
            title="A",
            content_type="text/html",
            fetched_at=fetched_at,
        ),
        _source("src_b", status="success"),
        _source("src_d", status="pending"),
    ]

    left_grouped = merge_sources(
        merge_sources(group_a, group_b),
        group_c,
    )

    right_grouped = merge_sources(
        group_a,
        merge_sources(group_b, group_c),
    )

    assert left_grouped == right_grouped