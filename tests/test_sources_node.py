"""Tests for deterministic SearchResult-to-Source conversion and budgeting.

These tests cover three responsibilities:

1. URL normalization and deterministic Source identity.
2. SourceNode conversion of SearchResults into unique Source candidates.
3. prepare_source_batch() whole-run source-budget authorization.

SourceNode deliberately owns no source budget.

Whole-run source usage is derived from the unique Sources already present
in ResearchState rather than from a separate additive counter.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from research_agent.graph.budget import (
    BudgetAuthorization,
    BudgetLimits,
    BudgetPolicy,
)
from research_agent.graph.nodes.sources import (
    SourceBatch,
    SourceNode,
    build_source_id,
    normalize_source_url,
    prepare_source_batch,
)
from research_agent.models.schemas import (
    SearchResult,
    Source,
)


def _result(
    *,
    url: str,
    title: str = "Example result",
    sub_question_id: str = "sq_one",
    query: str = "Research question",
    rank: int = 1,
) -> SearchResult:
    return SearchResult(
        sub_question_id=sub_question_id,
        query=query,
        title=title,
        url=url,
        snippet="Example snippet",
        rank=rank,
        provider="fake",
    )


def _source(
    *,
    url: str,
    title: str = "Example source",
    status: str = "pending",
) -> Source:
    normalized = normalize_source_url(
        url
    )

    from urllib.parse import urlsplit

    hostname = urlsplit(
        normalized
    ).hostname

    assert hostname is not None

    return Source(
        id=build_source_id(
            normalized
        ),
        url=normalized,
        title=title,
        domain=hostname,
        fetch_status=status,
    )


def _policy(
    *,
    max_sources_per_run: int = 12,
) -> BudgetPolicy:
    return BudgetPolicy(
        limits=BudgetLimits(
            max_research_iterations=2,
            max_search_queries_per_run=8,
            max_search_queries_per_iteration=5,
            max_sources_per_run=max_sources_per_run,
            max_source_fetches_per_run=12,
            max_llm_calls_per_run=64,
            finalization_llm_reserve=2,
        )
    )


def _source_authorization(
    *,
    requested: int,
    authorized: int,
    reason: str | None = None,
) -> BudgetAuthorization:
    return BudgetAuthorization(
        resource="sources",
        requested=requested,
        authorized=authorized,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# URL normalization
# ---------------------------------------------------------------------------


def test_normalize_source_url_strips_outer_whitespace():
    result = normalize_source_url(
        "  https://example.com/article  "
    )

    assert result == (
        "https://example.com/article"
    )


def test_normalize_source_url_lowercases_scheme_and_hostname():
    result = normalize_source_url(
        "HTTPS://EXAMPLE.COM/article"
    )

    assert result == (
        "https://example.com/article"
    )


def test_normalize_source_url_adds_root_path_when_missing():
    result = normalize_source_url(
        "https://example.com"
    )

    assert result == (
        "https://example.com/"
    )


def test_normalize_source_url_removes_fragment():
    result = normalize_source_url(
        "https://example.com/article#section-two"
    )

    assert result == (
        "https://example.com/article"
    )


def test_normalize_source_url_preserves_query_parameters():
    result = normalize_source_url(
        "https://example.com/report?year=2026"
    )

    assert result == (
        "https://example.com/report?year=2026"
    )


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "http://example.com:80/article",
            "http://example.com/article",
        ),
        (
            "https://example.com:443/article",
            "https://example.com/article",
        ),
    ],
)
def test_normalize_source_url_removes_default_ports(
    url,
    expected,
):
    assert normalize_source_url(
        url
    ) == expected


def test_normalize_source_url_preserves_non_default_port():
    result = normalize_source_url(
        "https://example.com:8443/article"
    )

    assert result == (
        "https://example.com:8443/article"
    )


def test_normalize_source_url_normalizes_trailing_hostname_dot():
    result = normalize_source_url(
        "https://Example.COM./article"
    )

    assert result == (
        "https://example.com/article"
    )


def test_normalize_source_url_supports_ipv6_host():
    result = normalize_source_url(
        "https://[2001:db8::1]:8443/article"
    )

    assert result == (
        "https://[2001:db8::1]:8443/article"
    )


@pytest.mark.parametrize(
    "value",
    [
        None,
        123,
        [],
        {},
    ],
)
def test_normalize_source_url_requires_string(
    value,
):
    with pytest.raises(TypeError):
        normalize_source_url(
            value
        )


@pytest.mark.parametrize(
    "value",
    [
        "",
        " ",
        "   ",
        "\n",
    ],
)
def test_normalize_source_url_rejects_blank_value(
    value,
):
    with pytest.raises(ValueError):
        normalize_source_url(
            value
        )


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/file",
        "file:///tmp/file",
        "mailto:test@example.com",
        "javascript:alert(1)",
    ],
)
def test_normalize_source_url_rejects_unsupported_scheme(
    url,
):
    with pytest.raises(ValueError):
        normalize_source_url(
            url
        )


@pytest.mark.parametrize(
    "url",
    [
        "https:///article",
        "https://",
    ],
)
def test_normalize_source_url_requires_hostname(
    url,
):
    with pytest.raises(ValueError):
        normalize_source_url(
            url
        )


@pytest.mark.parametrize(
    "url",
    [
        "https://user@example.com/article",
        "https://user:pass@example.com/article",
    ],
)
def test_normalize_source_url_rejects_embedded_credentials(
    url,
):
    with pytest.raises(ValueError):
        normalize_source_url(
            url
        )


def test_normalize_source_url_rejects_invalid_port():
    with pytest.raises(ValueError):
        normalize_source_url(
            "https://example.com:not-a-port/article"
        )


# ---------------------------------------------------------------------------
# Source IDs
# ---------------------------------------------------------------------------


def test_build_source_id_is_deterministic():
    first = build_source_id(
        "https://example.com/article"
    )

    second = build_source_id(
        "https://example.com/article"
    )

    assert first == second


def test_equivalent_urls_generate_same_source_id():
    first = build_source_id(
        "HTTPS://EXAMPLE.COM:443/article#section"
    )

    second = build_source_id(
        "https://example.com/article"
    )

    assert first == second


def test_different_urls_generate_different_source_ids():
    first = build_source_id(
        "https://example.com/article-a"
    )

    second = build_source_id(
        "https://example.com/article-b"
    )

    assert first != second


def test_source_id_has_expected_prefix():
    result = build_source_id(
        "https://example.com/article"
    )

    assert result.startswith(
        "src_"
    )


def test_source_id_has_stable_length():
    result = build_source_id(
        "https://example.com/article"
    )

    assert len(result) == (
        len("src_") + 24
    )


# ---------------------------------------------------------------------------
# SourceNode collection
# ---------------------------------------------------------------------------


def test_empty_search_results_return_empty_sources():
    node = SourceNode()

    assert node.collect(
        []
    ) == []


def test_search_result_becomes_source():
    node = SourceNode()

    result = node.collect(
        [
            _result(
                url="https://example.com/article",
                title="Research article",
            )
        ]
    )

    assert len(result) == 1

    source = result[0]

    assert isinstance(
        source,
        Source,
    )

    assert source.url == (
        "https://example.com/article"
    )

    assert source.title == (
        "Research article"
    )

    assert source.domain == (
        "example.com"
    )

    assert source.fetch_status == (
        "pending"
    )


def test_new_source_final_url_is_none():
    node = SourceNode()

    result = node.collect(
        [
            _result(
                url="https://example.com/article",
            )
        ]
    )

    assert result[0].final_url is None


def test_source_id_is_derived_from_normalized_url():
    node = SourceNode()

    result = node.collect(
        [
            _result(
                url=(
                    "HTTPS://EXAMPLE.COM:443/"
                    "article#section"
                ),
            )
        ]
    )

    assert result[0].id == build_source_id(
        "https://example.com/article"
    )


def test_source_url_is_normalized_before_storage():
    node = SourceNode()

    result = node.collect(
        [
            _result(
                url=(
                    "  HTTPS://EXAMPLE.COM:443/"
                    "article#section  "
                ),
            )
        ]
    )

    assert result[0].url == (
        "https://example.com/article"
    )


def test_duplicate_urls_are_removed():
    node = SourceNode()

    result = node.collect(
        [
            _result(
                url="https://example.com/article",
                title="First",
            ),
            _result(
                url=(
                    "HTTPS://EXAMPLE.COM:443/"
                    "article#fragment"
                ),
                title="Second",
            ),
        ]
    )

    assert len(result) == 1


def test_first_duplicate_result_wins():
    node = SourceNode()

    result = node.collect(
        [
            _result(
                url="https://example.com/article",
                title="First title",
            ),
            _result(
                url=(
                    "https://example.com/"
                    "article#fragment"
                ),
                title="Second title",
            ),
        ]
    )

    assert result[0].title == (
        "First title"
    )


def test_query_parameters_keep_sources_distinct():
    node = SourceNode()

    result = node.collect(
        [
            _result(
                url=(
                    "https://example.com/report"
                    "?year=2025"
                )
            ),
            _result(
                url=(
                    "https://example.com/report"
                    "?year=2026"
                )
            ),
        ]
    )

    assert len(result) == 2


def test_sources_preserve_first_seen_order():
    node = SourceNode()

    result = node.collect(
        [
            _result(
                url="https://a.example/article",
                title="A",
            ),
            _result(
                url="https://b.example/article",
                title="B",
            ),
            _result(
                url="https://c.example/article",
                title="C",
            ),
        ]
    )

    assert [
        source.title
        for source in result
    ] == [
        "A",
        "B",
        "C",
    ]


def test_source_node_no_longer_truncates_candidates():
    """Whole-run source budgeting belongs to prepare_source_batch()."""

    node = SourceNode()

    result = node.collect(
        [
            _result(
                url="https://a.example/article",
                title="A",
            ),
            _result(
                url="https://b.example/article",
                title="B",
            ),
            _result(
                url="https://c.example/article",
                title="C",
            ),
        ]
    )

    assert [
        source.title
        for source in result
    ] == [
        "A",
        "B",
        "C",
    ]


def test_duplicate_does_not_create_second_candidate():
    node = SourceNode()

    result = node.collect(
        [
            _result(
                url="https://a.example/article",
                title="A",
            ),
            _result(
                url=(
                    "https://a.example/"
                    "article#fragment"
                ),
                title="Duplicate A",
            ),
            _result(
                url="https://b.example/article",
                title="B",
            ),
        ]
    )

    assert len(result) == 2

    assert [
        source.title
        for source in result
    ] == [
        "A",
        "B",
    ]


def test_source_preserves_normalized_title():
    node = SourceNode()

    result = node.collect(
        [
            _result(
                url="https://example.com/article",
                title="   Research article   ",
            )
        ]
    )

    assert result[0].title == (
        "Research article"
    )


@pytest.mark.parametrize(
    "value",
    [
        None,
        (),
        {},
        "not a list",
    ],
)
def test_collect_requires_list(
    value,
):
    node = SourceNode()

    with pytest.raises(TypeError):
        node.collect(
            value
        )


def test_every_search_result_must_be_domain_model():
    node = SourceNode()

    with pytest.raises(TypeError):
        node.collect(
            [
                "not a SearchResult",
            ]
        )


def test_batch_is_validated_before_source_creation():
    node = SourceNode()

    valid = _result(
        url="https://example.com/article",
    )

    with pytest.raises(TypeError):
        node.collect(
            [
                valid,
                "invalid",
            ]
        )


def test_invalid_later_item_is_rejected_before_url_conversion():
    """The complete type batch is checked before conversion starts."""

    node = SourceNode()

    valid = _result(
        url="https://example.com/article",
    )

    with pytest.raises(TypeError):
        node.collect(
            [
                valid,
                "invalid",
            ]
        )


# ---------------------------------------------------------------------------
# SourceBatch contract
# ---------------------------------------------------------------------------


def test_source_batch_accepts_valid_authorized_sources():
    source = _source(
        url="https://example.com/article",
    )

    batch = SourceBatch(
        sources=(
            source,
        ),
        authorization=_source_authorization(
            requested=1,
            authorized=1,
        ),
    )

    assert batch.sources == (
        source,
    )

    assert batch.requested == 1
    assert batch.authorized == 1
    assert batch.skipped == 0


def test_source_batch_is_frozen():
    batch = SourceBatch(
        sources=(),
        authorization=_source_authorization(
            requested=0,
            authorized=0,
        ),
    )

    with pytest.raises(
        FrozenInstanceError
    ):
        batch.sources = ()


def test_source_batch_requires_tuple():
    with pytest.raises(TypeError):
        SourceBatch(
            sources=[],
            authorization=_source_authorization(
                requested=0,
                authorized=0,
            ),
        )


@pytest.mark.parametrize(
    "value",
    [
        None,
        "authorization",
        1,
        [],
        {},
    ],
)
def test_source_batch_requires_budget_authorization(
    value,
):
    with pytest.raises(TypeError):
        SourceBatch(
            sources=(),
            authorization=value,
        )


def test_source_batch_requires_source_authorization():
    with pytest.raises(
        ValueError,
        match="sources",
    ):
        SourceBatch(
            sources=(),
            authorization=BudgetAuthorization(
                resource="search_queries",
                requested=0,
                authorized=0,
            ),
        )


def test_source_batch_requires_source_models():
    with pytest.raises(TypeError):
        SourceBatch(
            sources=(
                "not a Source",
            ),
            authorization=_source_authorization(
                requested=1,
                authorized=1,
            ),
        )


def test_source_batch_size_must_match_authorized_count():
    source = _source(
        url="https://example.com/article",
    )

    with pytest.raises(
        ValueError,
        match="authorized",
    ):
        SourceBatch(
            sources=(
                source,
            ),
            authorization=_source_authorization(
                requested=2,
                authorized=2,
            ),
        )


def test_partial_source_batch_exposes_skipped_count():
    source = _source(
        url="https://example.com/article",
    )

    batch = SourceBatch(
        sources=(
            source,
        ),
        authorization=_source_authorization(
            requested=3,
            authorized=1,
            reason="source budget reached",
        ),
    )

    assert batch.requested == 3
    assert batch.authorized == 1
    assert batch.skipped == 2


# ---------------------------------------------------------------------------
# Whole-run source authorization
# ---------------------------------------------------------------------------


def test_empty_candidates_produce_empty_source_batch():
    batch = prepare_source_batch(
        candidates=[],
        existing_sources=[],
        budget_policy=_policy(),
    )

    assert batch.sources == ()
    assert batch.requested == 0
    assert batch.authorized == 0
    assert batch.skipped == 0


def test_new_sources_are_fully_authorized_with_capacity():
    candidates = [
        _source(
            url="https://a.example/article",
            title="A",
        ),
        _source(
            url="https://b.example/article",
            title="B",
        ),
    ]

    batch = prepare_source_batch(
        candidates=candidates,
        existing_sources=[],
        budget_policy=_policy(
            max_sources_per_run=5,
        ),
    )

    assert batch.sources == tuple(
        candidates
    )

    assert batch.requested == 2
    assert batch.authorized == 2
    assert batch.skipped == 0
    assert batch.authorization.reason is None


def test_source_budget_uses_existing_unique_sources():
    existing = [
        _source(
            url="https://existing-a.example/article",
        ),
        _source(
            url="https://existing-b.example/article",
        ),
    ]

    candidates = [
        _source(
            url="https://new-a.example/article",
            title="New A",
        ),
        _source(
            url="https://new-b.example/article",
            title="New B",
        ),
    ]

    batch = prepare_source_batch(
        candidates=candidates,
        existing_sources=existing,
        budget_policy=_policy(
            max_sources_per_run=3,
        ),
    )

    # Two existing Sources leave exactly one whole-run slot.
    assert [
        source.title
        for source in batch.sources
    ] == [
        "New A",
    ]

    assert batch.requested == 2
    assert batch.authorized == 1
    assert batch.skipped == 1

    assert batch.authorization.reason == (
        "whole-run source budget reached"
    )


def test_exhausted_source_budget_returns_empty_batch():
    existing = [
        _source(
            url="https://a.example/article",
        ),
        _source(
            url="https://b.example/article",
        ),
    ]

    candidate = _source(
        url="https://c.example/article",
    )

    batch = prepare_source_batch(
        candidates=[
            candidate,
        ],
        existing_sources=existing,
        budget_policy=_policy(
            max_sources_per_run=2,
        ),
    )

    assert batch.sources == ()
    assert batch.requested == 1
    assert batch.authorized == 0
    assert batch.skipped == 1
    assert batch.authorization.exhausted is True


def test_existing_source_does_not_consume_another_budget_slot():
    existing = _source(
        url="https://example.com/article",
        title="Existing",
        status="success",
    )

    rediscovered = _source(
        url="https://example.com/article",
        title="Rediscovered",
        status="pending",
    )

    new_source = _source(
        url="https://new.example/article",
        title="New",
    )

    batch = prepare_source_batch(
        candidates=[
            rediscovered,
            new_source,
        ],
        existing_sources=[
            existing,
        ],
        budget_policy=_policy(
            max_sources_per_run=2,
        ),
    )

    assert batch.sources == (
        new_source,
    )

    # Only the genuinely new Source reaches the budget policy.
    assert batch.requested == 1
    assert batch.authorized == 1
    assert batch.skipped == 0


def test_existing_source_is_filtered_even_when_budget_has_capacity():
    existing = _source(
        url="https://example.com/article",
        status="failed",
    )

    rediscovered = _source(
        url="https://example.com/article",
        status="pending",
    )

    batch = prepare_source_batch(
        candidates=[
            rediscovered,
        ],
        existing_sources=[
            existing,
        ],
        budget_policy=_policy(
            max_sources_per_run=10,
        ),
    )

    assert batch.sources == ()
    assert batch.requested == 0
    assert batch.authorized == 0


def test_existing_source_status_does_not_change_identity_accounting():
    """Unique-source budget is based on Source.id, not fetch status."""

    url = "https://example.com/article"

    for status in (
        "pending",
        "success",
        "failed",
        "skipped",
    ):
        existing = _source(
            url=url,
            status=status,
        )

        rediscovered = _source(
            url=url,
            status="pending",
        )

        batch = prepare_source_batch(
            candidates=[
                rediscovered,
            ],
            existing_sources=[
                existing,
            ],
            budget_policy=_policy(
                max_sources_per_run=5,
            ),
        )

        assert batch.sources == ()
        assert batch.requested == 0


def test_duplicate_candidates_consume_one_source_slot():
    first = _source(
        url="https://example.com/article",
        title="First",
    )

    duplicate = Source(
        id=first.id,
        url=first.url,
        title="Duplicate",
        domain=first.domain,
        fetch_status="pending",
    )

    batch = prepare_source_batch(
        candidates=[
            first,
            duplicate,
        ],
        existing_sources=[],
        budget_policy=_policy(
            max_sources_per_run=1,
        ),
    )

    assert batch.sources == (
        first,
    )

    assert batch.requested == 1
    assert batch.authorized == 1
    assert batch.skipped == 0


def test_first_duplicate_candidate_wins():
    first = _source(
        url="https://example.com/article",
        title="First",
    )

    duplicate = Source(
        id=first.id,
        url=first.url,
        title="Second",
        domain=first.domain,
        fetch_status="pending",
    )

    batch = prepare_source_batch(
        candidates=[
            first,
            duplicate,
        ],
        existing_sources=[],
        budget_policy=_policy(),
    )

    assert batch.sources == (
        first,
    )

    assert batch.sources[0].title == (
        "First"
    )


def test_authorization_preserves_candidate_order():
    candidates = [
        _source(
            url="https://c.example/article",
            title="C",
        ),
        _source(
            url="https://a.example/article",
            title="A",
        ),
        _source(
            url="https://b.example/article",
            title="B",
        ),
    ]

    batch = prepare_source_batch(
        candidates=candidates,
        existing_sources=[],
        budget_policy=_policy(
            max_sources_per_run=2,
        ),
    )

    assert [
        source.title
        for source in batch.sources
    ] == [
        "C",
        "A",
    ]


def test_budget_layer_never_reranks_candidates():
    candidates = [
        _source(
            url="https://z.example/article",
            title="First",
        ),
        _source(
            url="https://a.example/article",
            title="Second",
        ),
        _source(
            url="https://m.example/article",
            title="Third",
        ),
    ]

    batch = prepare_source_batch(
        candidates=candidates,
        existing_sources=[],
        budget_policy=_policy(
            max_sources_per_run=2,
        ),
    )

    assert batch.sources == (
        candidates[0],
        candidates[1],
    )


def test_duplicate_existing_source_ids_count_once():
    existing = _source(
        url="https://existing.example/article",
    )

    duplicate_existing = Source(
        id=existing.id,
        url=existing.url,
        title="Duplicate existing",
        domain=existing.domain,
        fetch_status="success",
    )

    new_a = _source(
        url="https://new-a.example/article",
        title="A",
    )

    new_b = _source(
        url="https://new-b.example/article",
        title="B",
    )

    batch = prepare_source_batch(
        candidates=[
            new_a,
            new_b,
        ],
        existing_sources=[
            existing,
            duplicate_existing,
        ],
        budget_policy=_policy(
            max_sources_per_run=2,
        ),
    )

    # Existing duplicate IDs represent one unique whole-run Source,
    # leaving one slot for new work.
    assert batch.sources == (
        new_a,
    )

    assert batch.authorized == 1


def test_prepare_source_batch_does_not_mutate_candidate_list():
    candidates = [
        _source(
            url="https://a.example/article",
        ),
        _source(
            url="https://b.example/article",
        ),
    ]

    before = list(
        candidates
    )

    prepare_source_batch(
        candidates=candidates,
        existing_sources=[],
        budget_policy=_policy(
            max_sources_per_run=1,
        ),
    )

    assert candidates == before


def test_prepare_source_batch_does_not_mutate_existing_list():
    existing = [
        _source(
            url="https://existing.example/article",
        ),
    ]

    before = list(
        existing
    )

    prepare_source_batch(
        candidates=[],
        existing_sources=existing,
        budget_policy=_policy(),
    )

    assert existing == before


@pytest.mark.parametrize(
    "value",
    [
        None,
        (),
        {},
        "not a list",
    ],
)
def test_prepare_source_batch_requires_candidate_list(
    value,
):
    with pytest.raises(TypeError):
        prepare_source_batch(
            candidates=value,
            existing_sources=[],
            budget_policy=_policy(),
        )


def test_prepare_source_batch_requires_candidate_source_models():
    with pytest.raises(TypeError):
        prepare_source_batch(
            candidates=[
                "not a Source",
            ],
            existing_sources=[],
            budget_policy=_policy(),
        )


@pytest.mark.parametrize(
    "value",
    [
        None,
        (),
        {},
        "not a list",
    ],
)
def test_prepare_source_batch_requires_existing_source_list(
    value,
):
    with pytest.raises(TypeError):
        prepare_source_batch(
            candidates=[],
            existing_sources=value,
            budget_policy=_policy(),
        )


def test_prepare_source_batch_requires_existing_source_models():
    with pytest.raises(TypeError):
        prepare_source_batch(
            candidates=[],
            existing_sources=[
                "not a Source",
            ],
            budget_policy=_policy(),
        )


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        [],
        "policy",
        123,
    ],
)
def test_prepare_source_batch_requires_budget_policy(
    value,
):
    with pytest.raises(TypeError):
        prepare_source_batch(
            candidates=[],
            existing_sources=[],
            budget_policy=value,
        )


# ---------------------------------------------------------------------------
# SourceNode + budget integration
# ---------------------------------------------------------------------------


def test_collected_sources_can_be_passed_directly_to_budget_layer():
    node = SourceNode()

    candidates = node.collect(
        [
            _result(
                url="https://a.example/article",
                title="A",
            ),
            _result(
                url="https://b.example/article",
                title="B",
            ),
            _result(
                url="https://c.example/article",
                title="C",
            ),
        ]
    )

    batch = prepare_source_batch(
        candidates=candidates,
        existing_sources=[],
        budget_policy=_policy(
            max_sources_per_run=2,
        ),
    )

    assert [
        source.title
        for source in batch.sources
    ] == [
        "A",
        "B",
    ]

    assert batch.requested == 3
    assert batch.authorized == 2
    assert batch.skipped == 1


def test_rediscovered_collected_source_is_not_readded_to_state():
    node = SourceNode()

    existing = node.collect(
        [
            _result(
                url="https://example.com/article",
                title="Original",
            )
        ]
    )[0]

    candidates = node.collect(
        [
            _result(
                url=(
                    "HTTPS://EXAMPLE.COM:443/"
                    "article#again"
                ),
                title="Rediscovered",
            ),
            _result(
                url="https://new.example/article",
                title="New",
            ),
        ]
    )

    batch = prepare_source_batch(
        candidates=candidates,
        existing_sources=[
            existing,
        ],
        budget_policy=_policy(
            max_sources_per_run=2,
        ),
    )

    assert len(
        batch.sources
    ) == 1

    assert batch.sources[0].title == (
        "New"
    )

    assert batch.requested == 1
    assert batch.authorized == 1