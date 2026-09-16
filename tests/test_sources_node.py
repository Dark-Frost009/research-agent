"""Tests for deterministic SearchResult-to-Source conversion."""

import pytest

from research_agent.graph.nodes.sources import (
    SourceNode,
    build_source_id,
    normalize_source_url,
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


# ---------------------------------------------------------------------------
# URL normalization
# ---------------------------------------------------------------------------


def test_normalize_source_url_strips_outer_whitespace():
    result = normalize_source_url(
        "  https://example.com/article  "
    )

    assert result == "https://example.com/article"


def test_normalize_source_url_lowercases_scheme_and_hostname():
    result = normalize_source_url(
        "HTTPS://EXAMPLE.COM/article"
    )

    assert result == "https://example.com/article"


def test_normalize_source_url_adds_root_path_when_missing():
    result = normalize_source_url(
        "https://example.com"
    )

    assert result == "https://example.com/"


def test_normalize_source_url_removes_fragment():
    result = normalize_source_url(
        "https://example.com/article#section-two"
    )

    assert result == "https://example.com/article"


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
    assert normalize_source_url(url) == expected


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
def test_normalize_source_url_requires_string(value):
    with pytest.raises(TypeError):
        normalize_source_url(value)


@pytest.mark.parametrize(
    "value",
    [
        "",
        " ",
        "   ",
        "\n",
    ],
)
def test_normalize_source_url_rejects_blank_value(value):
    with pytest.raises(ValueError):
        normalize_source_url(value)


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/file",
        "file:///tmp/file",
        "mailto:test@example.com",
        "javascript:alert(1)",
    ],
)
def test_normalize_source_url_rejects_unsupported_scheme(url):
    with pytest.raises(ValueError):
        normalize_source_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "https:///article",
        "https://",
    ],
)
def test_normalize_source_url_requires_hostname(url):
    with pytest.raises(ValueError):
        normalize_source_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://user@example.com/article",
        "https://user:pass@example.com/article",
    ],
)
def test_normalize_source_url_rejects_embedded_credentials(url):
    with pytest.raises(ValueError):
        normalize_source_url(url)


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

    assert result.startswith("src_")


def test_source_id_has_stable_length():
    result = build_source_id(
        "https://example.com/article"
    )

    assert len(result) == len("src_") + 24


# ---------------------------------------------------------------------------
# SourceNode constructor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        None,
        1.5,
        "5",
        [],
        {},
        True,
        False,
    ],
)
def test_max_sources_must_be_integer(value):
    with pytest.raises(TypeError):
        SourceNode(
            max_sources=value,
        )


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        -10,
    ],
)
def test_max_sources_must_be_positive(value):
    with pytest.raises(ValueError):
        SourceNode(
            max_sources=value,
        )


# ---------------------------------------------------------------------------
# Source collection
# ---------------------------------------------------------------------------


def test_empty_search_results_return_empty_sources():
    node = SourceNode(
        max_sources=5,
    )

    assert node.collect([]) == []


def test_search_result_becomes_source():
    node = SourceNode(
        max_sources=5,
    )

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

    assert isinstance(source, Source)
    assert source.url == (
        "https://example.com/article"
    )
    assert source.title == "Research article"
    assert source.domain == "example.com"
    assert source.fetch_status == "pending"


def test_source_id_is_derived_from_normalized_url():
    node = SourceNode(
        max_sources=5,
    )

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


def test_duplicate_urls_are_removed():
    node = SourceNode(
        max_sources=5,
    )

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
    node = SourceNode(
        max_sources=5,
    )

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

    assert result[0].title == "First title"


def test_query_parameters_keep_sources_distinct():
    node = SourceNode(
        max_sources=5,
    )

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
    node = SourceNode(
        max_sources=5,
    )

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


def test_source_limit_is_enforced():
    node = SourceNode(
        max_sources=2,
    )

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

    assert len(result) == 2

    assert [
        source.title
        for source in result
    ] == [
        "A",
        "B",
    ]


def test_duplicate_does_not_consume_source_budget():
    node = SourceNode(
        max_sources=2,
    )

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
    node = SourceNode(
        max_sources=5,
    )

    result = node.collect(
        [
            _result(
                url="https://example.com/article",
                title="   Research article   ",
            )
        ]
    )

    assert result[0].title == "Research article"


@pytest.mark.parametrize(
    "value",
    [
        None,
        (),
        {},
        "not a list",
    ],
)
def test_collect_requires_list(value):
    node = SourceNode(
        max_sources=5,
    )

    with pytest.raises(TypeError):
        node.collect(value)


def test_every_search_result_must_be_domain_model():
    node = SourceNode(
        max_sources=5,
    )

    with pytest.raises(TypeError):
        node.collect(
            [
                "not a SearchResult",
            ]
        )


def test_batch_is_validated_before_source_creation():
    node = SourceNode(
        max_sources=5,
    )

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