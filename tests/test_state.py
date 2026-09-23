"""Tests for the ResearchState reducers.

These tests focus on our reducer contracts rather than LangGraph itself.

For Source merging, the key properties are:

- deterministic deduplication by Source.id
- fetch-status priority
- metadata-completeness preference
- non-mutation
- idempotence
- commutativity
- associativity

For whole-run usage counters, the key properties are:

- additive delta semantics
- non-negative integer validation
- commutativity
- associativity
- deliberate non-idempotence

Those properties matter because parallel graph branches may eventually
contribute Source updates and usage deltas into shared ResearchState.
"""

from datetime import datetime, timezone
from typing import get_args, get_type_hints

import pytest

from research_agent.graph.state import (
    ResearchState,
    add_usage,
    merge_sources,
)
from research_agent.models.schemas import Source


def _source(
    source_id: str,
    *,
    status: str = "pending",
    title: str | None = None,
    content_type: str | None = None,
    fetched_at: datetime | None = None,
    final_url: str | None = None,
    url: str | None = None,
    domain: str = "example.com",
) -> Source:
    """Build a small valid Source for reducer tests."""
    return Source(
        id=source_id,
        url=url or f"https://example.com/{source_id}",
        final_url=final_url,
        title=title,
        domain=domain,
        content_type=content_type,
        fetch_status=status,
        fetched_at=fetched_at,
    )


# ---------------------------------------------------------------------------
# Whole-run usage reducer
# ---------------------------------------------------------------------------


def test_add_usage_adds_incoming_delta_to_existing_total():
    assert add_usage(
        5,
        2,
    ) == 7


def test_add_usage_accepts_zero_delta():
    assert add_usage(
        5,
        0,
    ) == 5


def test_add_usage_accepts_zero_existing_total():
    assert add_usage(
        0,
        3,
    ) == 3


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        None,
        1.5,
        "1",
        [],
        {},
    ],
)
def test_add_usage_rejects_invalid_existing_type(
    value,
):
    with pytest.raises(TypeError):
        add_usage(
            value,
            1,
        )


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        None,
        1.5,
        "1",
        [],
        {},
    ],
)
def test_add_usage_rejects_invalid_incoming_type(
    value,
):
    with pytest.raises(TypeError):
        add_usage(
            1,
            value,
        )


def test_add_usage_rejects_negative_existing_total():
    with pytest.raises(ValueError):
        add_usage(
            -1,
            1,
        )


def test_add_usage_rejects_negative_incoming_delta():
    with pytest.raises(ValueError):
        add_usage(
            5,
            -1,
        )


def test_add_usage_is_commutative_for_valid_usage_values():
    assert add_usage(
        3,
        7,
    ) == add_usage(
        7,
        3,
    )


def test_add_usage_is_associative_for_valid_usage_values():
    left_grouped = add_usage(
        add_usage(
            2,
            3,
        ),
        4,
    )

    right_grouped = add_usage(
        2,
        add_usage(
            3,
            4,
        ),
    )

    assert left_grouped == right_grouped
    assert left_grouped == 9


def test_add_usage_is_deliberately_not_idempotent():
    """Duplicate logical updates would double-charge the budget."""

    assert add_usage(
        2,
        2,
    ) == 4

    assert add_usage(
        2,
        2,
    ) != 2


@pytest.mark.parametrize(
    "field_name",
    [
        "iteration_count",
        "search_queries_used",
        "source_fetches_used",
        "llm_calls_used",
    ],
)
def test_research_state_usage_counters_use_add_usage_reducer(
    field_name,
):
    hints = get_type_hints(
        ResearchState,
        include_extras=True,
    )

    field_args = get_args(
        hints[field_name]
    )

    assert field_args[0] is int
    assert field_args[1] is add_usage


def test_research_state_does_not_store_sources_used_counter():
    hints = get_type_hints(
        ResearchState,
        include_extras=True,
    )

    assert "sources_used" not in hints


def test_research_state_sources_still_use_merge_sources_reducer():
    hints = get_type_hints(
        ResearchState,
        include_extras=True,
    )

    field_args = get_args(
        hints["sources"]
    )

    assert field_args[0] == list[Source]
    assert field_args[1] is merge_sources


# ---------------------------------------------------------------------------
# Basic merging
# ---------------------------------------------------------------------------


def test_merge_sources_empty_lists_returns_empty_list():
    assert merge_sources([], []) == []


def test_merge_sources_combines_distinct_ids():
    source_a = _source("src_a")
    source_b = _source("src_b")

    result = merge_sources(
        [source_a],
        [source_b],
    )

    assert result == [
        source_a,
        source_b,
    ]


def test_merge_sources_sorts_output_by_source_id():
    source_b = _source("src_b")
    source_a = _source("src_a")
    source_c = _source("src_c")

    result = merge_sources(
        [source_b],
        [
            source_c,
            source_a,
        ],
    )

    assert [
        source.id
        for source in result
    ] == [
        "src_a",
        "src_b",
        "src_c",
    ]


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def test_duplicate_source_id_produces_one_source():
    pending = _source(
        "src_same",
        status="pending",
    )

    success = _source(
        "src_same",
        status="success",
    )

    result = merge_sources(
        [pending],
        [success],
    )

    assert len(result) == 1
    assert result[0].id == "src_same"
    assert result[0].fetch_status == "success"


def test_duplicates_inside_existing_are_resolved():
    pending = _source(
        "src_same",
        status="pending",
    )

    success = _source(
        "src_same",
        status="success",
    )

    result = merge_sources(
        [
            pending,
            success,
        ],
        [],
    )

    assert len(result) == 1
    assert result[0].fetch_status == "success"


def test_duplicates_inside_incoming_are_resolved():
    skipped = _source(
        "src_same",
        status="skipped",
    )

    failed = _source(
        "src_same",
        status="failed",
    )

    result = merge_sources(
        [],
        [
            skipped,
            failed,
        ],
    )

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
def test_fetch_status_priority(
    first_status,
    second_status,
    expected_status,
):
    first = _source(
        "src_same",
        status=first_status,
    )

    second = _source(
        "src_same",
        status=second_status,
    )

    result = merge_sources(
        [first],
        [second],
    )

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
    low = _source(
        "src_same",
        status=low_status,
    )

    high = _source(
        "src_same",
        status=high_status,
    )

    forward = merge_sources(
        [low],
        [high],
    )

    reverse = merge_sources(
        [high],
        [low],
    )

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
        fetched_at=datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
    )

    result = merge_sources(
        [less_complete],
        [more_complete],
    )

    assert result == [
        more_complete,
    ]


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

    forward = merge_sources(
        [less_complete],
        [more_complete],
    )

    reverse = merge_sources(
        [more_complete],
        [less_complete],
    )

    assert forward == reverse
    assert forward == [
        more_complete,
    ]


def test_final_url_counts_toward_metadata_completeness():
    less_complete = _source(
        "src_same",
        status="success",
        title="Example",
    )

    more_complete = _source(
        "src_same",
        status="success",
        title="Example",
        final_url="https://example.com/final",
    )

    result = merge_sources(
        [less_complete],
        [more_complete],
    )

    assert result == [
        more_complete,
    ]


def test_final_url_completeness_is_independent_of_argument_order():
    without_final_url = _source(
        "src_same",
        status="success",
        title="Example",
    )

    with_final_url = _source(
        "src_same",
        status="success",
        title="Example",
        final_url="https://example.com/final",
    )

    forward = merge_sources(
        [without_final_url],
        [with_final_url],
    )

    reverse = merge_sources(
        [with_final_url],
        [without_final_url],
    )

    assert forward == reverse
    assert forward == [
        with_final_url,
    ]


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

    forward = merge_sources(
        [source_a],
        [source_b],
    )

    reverse = merge_sources(
        [source_b],
        [source_a],
    )

    assert forward == reverse
    assert len(forward) == 1
    assert forward[0] in (
        source_a,
        source_b,
    )


def test_different_final_urls_resolve_deterministically():
    source_a = _source(
        "src_same",
        status="success",
        title="Example",
        content_type="text/html",
        final_url="https://example.com/a",
    )

    source_b = _source(
        "src_same",
        status="success",
        title="Example",
        content_type="text/html",
        final_url="https://example.com/b",
    )

    forward = merge_sources(
        [source_a],
        [source_b],
    )

    reverse = merge_sources(
        [source_b],
        [source_a],
    )

    assert forward == reverse
    assert forward == [
        source_b,
    ]


# ---------------------------------------------------------------------------
# Non-mutation
# ---------------------------------------------------------------------------


def test_merge_sources_does_not_mutate_input_lists_or_sources():
    existing_source = _source(
        "src_a",
        status="pending",
        title="Existing",
    )

    incoming_source = _source(
        "src_a",
        status="success",
        title="Incoming",
        final_url="https://example.com/final",
    )

    existing = [
        existing_source,
    ]

    incoming = [
        incoming_source,
    ]

    existing_before = [
        source.model_copy(deep=True)
        for source in existing
    ]

    incoming_before = [
        source.model_copy(deep=True)
        for source in incoming
    ]

    result = merge_sources(
        existing,
        incoming,
    )

    assert existing == existing_before
    assert incoming == incoming_before

    assert result is not existing
    assert result is not incoming


# ---------------------------------------------------------------------------
# Reducer algebraic properties
# ---------------------------------------------------------------------------


def test_merge_sources_is_idempotent():
    sources = [
        _source(
            "src_b",
            status="pending",
        ),
        _source(
            "src_a",
            status="failed",
        ),
        _source(
            "src_a",
            status="success",
            final_url="https://example.com/final",
        ),
    ]

    normalized = merge_sources(
        [],
        sources,
    )

    repeated = merge_sources(
        sources,
        sources,
    )

    assert repeated == normalized


def test_merge_sources_is_commutative():
    fetched_at = datetime(
        2026,
        1,
        1,
        tzinfo=timezone.utc,
    )

    group_a = [
        _source(
            "src_a",
            status="pending",
        ),
        _source(
            "src_b",
            status="failed",
            title="B",
        ),
    ]

    group_b = [
        _source(
            "src_a",
            status="success",
            title="A",
            content_type="text/html",
            fetched_at=fetched_at,
            final_url="https://example.com/src_a/final",
        ),
        _source(
            "src_c",
            status="skipped",
        ),
    ]

    assert merge_sources(
        group_a,
        group_b,
    ) == merge_sources(
        group_b,
        group_a,
    )


def test_merge_sources_is_associative():
    fetched_at = datetime(
        2026,
        1,
        1,
        tzinfo=timezone.utc,
    )

    group_a = [
        _source(
            "src_a",
            status="pending",
        ),
        _source(
            "src_b",
            status="failed",
        ),
    ]

    group_b = [
        _source(
            "src_a",
            status="success",
            title="A",
        ),
        _source(
            "src_c",
            status="skipped",
        ),
    ]

    group_c = [
        _source(
            "src_a",
            status="success",
            title="A",
            content_type="text/html",
            fetched_at=fetched_at,
            final_url="https://example.com/src_a/final",
        ),
        _source(
            "src_b",
            status="success",
            final_url="https://example.com/src_b/final",
        ),
        _source(
            "src_d",
            status="pending",
        ),
    ]

    left_grouped = merge_sources(
        merge_sources(
            group_a,
            group_b,
        ),
        group_c,
    )

    right_grouped = merge_sources(
        group_a,
        merge_sources(
            group_b,
            group_c,
        ),
    )

    assert left_grouped == right_grouped


def test_merge_sources_remains_associative_when_final_url_differs():
    source_a = _source(
        "src_same",
        status="success",
        final_url="https://example.com/a",
    )

    source_b = _source(
        "src_same",
        status="success",
        final_url="https://example.com/b",
    )

    source_c = _source(
        "src_same",
        status="success",
        final_url="https://example.com/c",
    )

    left_grouped = merge_sources(
        merge_sources(
            [source_a],
            [source_b],
        ),
        [source_c],
    )

    right_grouped = merge_sources(
        [source_a],
        merge_sources(
            [source_b],
            [source_c],
        ),
    )

    assert left_grouped == right_grouped