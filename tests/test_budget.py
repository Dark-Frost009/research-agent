"""Tests for deterministic whole-run research budget policy.

These tests exercise the central B1 budget layer in isolation.

The budget layer is intentionally pure:

- it does not mutate LangGraph state
- it does not call external services
- it does not own hidden usage counters
- it authorizes complete or partial deterministic work
- budget exhaustion is normal control flow
- optional LLM work cannot consume protected finalization capacity

Later graph/node integration tests will verify that authorized usage is
recorded before external side effects occur.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from research_agent.graph.budget import (
    BudgetAuthorization,
    BudgetLimits,
    BudgetPolicy,
    BudgetUsage,
    remaining_budget,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _limits(
    *,
    max_research_iterations: int = 2,
    max_search_queries_per_run: int = 8,
    max_search_queries_per_iteration: int = 5,
    max_sources_per_run: int = 12,
    max_source_fetches_per_run: int = 10,
    max_llm_calls_per_run: int = 10,
    finalization_llm_reserve: int = 2,
) -> BudgetLimits:
    return BudgetLimits(
        max_research_iterations=max_research_iterations,
        max_search_queries_per_run=max_search_queries_per_run,
        max_search_queries_per_iteration=(
            max_search_queries_per_iteration
        ),
        max_sources_per_run=max_sources_per_run,
        max_source_fetches_per_run=(
            max_source_fetches_per_run
        ),
        max_llm_calls_per_run=max_llm_calls_per_run,
        finalization_llm_reserve=(
            finalization_llm_reserve
        ),
    )


def _policy(
    **overrides,
) -> BudgetPolicy:
    return BudgetPolicy(
        limits=_limits(
            **overrides
        )
    )


# ---------------------------------------------------------------------------
# remaining_budget
# ---------------------------------------------------------------------------


def test_remaining_budget_returns_difference():
    assert remaining_budget(
        limit=10,
        used=4,
    ) == 6


def test_remaining_budget_returns_zero_when_fully_used():
    assert remaining_budget(
        limit=10,
        used=10,
    ) == 0


def test_remaining_budget_clamps_overuse_to_zero():
    assert remaining_budget(
        limit=10,
        used=15,
    ) == 0


def test_remaining_budget_allows_zero_limit():
    assert remaining_budget(
        limit=0,
        used=0,
    ) == 0


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        None,
        1.5,
        "10",
        [],
        {},
    ],
)
def test_remaining_budget_rejects_invalid_limit_type(
    value,
):
    with pytest.raises(TypeError):
        remaining_budget(
            limit=value,
            used=0,
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
def test_remaining_budget_rejects_invalid_used_type(
    value,
):
    with pytest.raises(TypeError):
        remaining_budget(
            limit=10,
            used=value,
        )


def test_remaining_budget_rejects_negative_limit():
    with pytest.raises(ValueError):
        remaining_budget(
            limit=-1,
            used=0,
        )


def test_remaining_budget_rejects_negative_used():
    with pytest.raises(ValueError):
        remaining_budget(
            limit=10,
            used=-1,
        )


# ---------------------------------------------------------------------------
# BudgetLimits
# ---------------------------------------------------------------------------


def test_budget_limits_accept_valid_configuration():
    limits = _limits()

    assert limits.max_research_iterations == 2
    assert limits.max_search_queries_per_run == 8

    assert (
        limits.max_search_queries_per_iteration
        == 5
    )

    assert limits.max_sources_per_run == 12

    assert (
        limits.max_source_fetches_per_run
        == 10
    )

    assert limits.max_llm_calls_per_run == 10
    assert limits.finalization_llm_reserve == 2


def test_budget_limits_are_frozen():
    limits = _limits()

    with pytest.raises(
        FrozenInstanceError
    ):
        limits.max_research_iterations = 99


@pytest.mark.parametrize(
    "field_name",
    [
        "max_research_iterations",
        "max_search_queries_per_run",
        "max_search_queries_per_iteration",
        "max_sources_per_run",
        "max_source_fetches_per_run",
        "max_llm_calls_per_run",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        None,
        1.5,
        "5",
        [],
        {},
    ],
)
def test_budget_limits_reject_invalid_positive_field_types(
    field_name,
    value,
):
    kwargs = {
        "max_research_iterations": 2,
        "max_search_queries_per_run": 8,
        "max_search_queries_per_iteration": 5,
        "max_sources_per_run": 12,
        "max_source_fetches_per_run": 10,
        "max_llm_calls_per_run": 10,
        "finalization_llm_reserve": 2,
    }

    kwargs[field_name] = value

    with pytest.raises(TypeError):
        BudgetLimits(
            **kwargs
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "max_research_iterations",
        "max_search_queries_per_run",
        "max_search_queries_per_iteration",
        "max_sources_per_run",
        "max_source_fetches_per_run",
        "max_llm_calls_per_run",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        -100,
    ],
)
def test_budget_limits_require_positive_whole_run_limits(
    field_name,
    value,
):
    kwargs = {
        "max_research_iterations": 2,
        "max_search_queries_per_run": 8,
        "max_search_queries_per_iteration": 5,
        "max_sources_per_run": 12,
        "max_source_fetches_per_run": 10,
        "max_llm_calls_per_run": 10,
        "finalization_llm_reserve": 2,
    }

    kwargs[field_name] = value

    with pytest.raises(ValueError):
        BudgetLimits(
            **kwargs
        )


def test_finalization_reserve_may_be_zero():
    limits = _limits(
        finalization_llm_reserve=0,
    )

    assert limits.finalization_llm_reserve == 0


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        None,
        1.5,
        "2",
        [],
        {},
    ],
)
def test_finalization_reserve_requires_integer(
    value,
):
    with pytest.raises(TypeError):
        _limits(
            finalization_llm_reserve=value,
        )


def test_finalization_reserve_rejects_negative_value():
    with pytest.raises(ValueError):
        _limits(
            finalization_llm_reserve=-1,
        )


def test_finalization_reserve_must_not_exceed_llm_budget():
    with pytest.raises(
        ValueError,
        match="must not exceed",
    ):
        _limits(
            max_llm_calls_per_run=2,
            finalization_llm_reserve=3,
        )


def test_finalization_reserve_may_equal_llm_budget():
    limits = _limits(
        max_llm_calls_per_run=2,
        finalization_llm_reserve=2,
    )

    assert limits.finalization_llm_reserve == 2


# ---------------------------------------------------------------------------
# BudgetUsage
# ---------------------------------------------------------------------------


def test_budget_usage_defaults_to_zero():
    usage = BudgetUsage()

    assert usage.iteration_count == 0
    assert usage.search_queries_used == 0
    assert usage.source_fetches_used == 0
    assert usage.llm_calls_used == 0
    assert usage.unique_sources == 0


def test_budget_usage_accepts_non_negative_values():
    usage = BudgetUsage(
        iteration_count=2,
        search_queries_used=7,
        source_fetches_used=5,
        llm_calls_used=6,
        unique_sources=9,
    )

    assert usage.iteration_count == 2
    assert usage.search_queries_used == 7
    assert usage.source_fetches_used == 5
    assert usage.llm_calls_used == 6
    assert usage.unique_sources == 9


def test_budget_usage_is_frozen():
    usage = BudgetUsage()

    with pytest.raises(
        FrozenInstanceError
    ):
        usage.llm_calls_used = 1


@pytest.mark.parametrize(
    "field_name",
    [
        "iteration_count",
        "search_queries_used",
        "source_fetches_used",
        "llm_calls_used",
        "unique_sources",
    ],
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
def test_budget_usage_rejects_invalid_types(
    field_name,
    value,
):
    kwargs = {
        "iteration_count": 0,
        "search_queries_used": 0,
        "source_fetches_used": 0,
        "llm_calls_used": 0,
        "unique_sources": 0,
    }

    kwargs[field_name] = value

    with pytest.raises(TypeError):
        BudgetUsage(
            **kwargs
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "iteration_count",
        "search_queries_used",
        "source_fetches_used",
        "llm_calls_used",
        "unique_sources",
    ],
)
def test_budget_usage_rejects_negative_values(
    field_name,
):
    kwargs = {
        "iteration_count": 0,
        "search_queries_used": 0,
        "source_fetches_used": 0,
        "llm_calls_used": 0,
        "unique_sources": 0,
    }

    kwargs[field_name] = -1

    with pytest.raises(ValueError):
        BudgetUsage(
            **kwargs
        )


# ---------------------------------------------------------------------------
# BudgetAuthorization
# ---------------------------------------------------------------------------


def test_full_authorization_reports_no_skipped_work():
    authorization = BudgetAuthorization(
        resource="search_queries",
        requested=3,
        authorized=3,
    )

    assert authorization.skipped == 0
    assert authorization.fully_authorized is True
    assert authorization.exhausted is False


def test_partial_authorization_reports_skipped_work():
    authorization = BudgetAuthorization(
        resource="search_queries",
        requested=5,
        authorized=2,
        reason="budget reached",
    )

    assert authorization.skipped == 3
    assert authorization.fully_authorized is False
    assert authorization.exhausted is False


def test_zero_authorization_reports_exhaustion():
    authorization = BudgetAuthorization(
        resource="source_fetches",
        requested=4,
        authorized=0,
        reason="budget exhausted",
    )

    assert authorization.skipped == 4
    assert authorization.fully_authorized is False
    assert authorization.exhausted is True


def test_zero_requested_work_is_not_exhaustion():
    authorization = BudgetAuthorization(
        resource="search_queries",
        requested=0,
        authorized=0,
    )

    assert authorization.skipped == 0
    assert authorization.fully_authorized is True
    assert authorization.exhausted is False


def test_budget_authorization_is_frozen():
    authorization = BudgetAuthorization(
        resource="llm_calls",
        requested=1,
        authorized=1,
        llm_purpose="optional_research",
    )

    with pytest.raises(
        FrozenInstanceError
    ):
        authorization.authorized = 0

@pytest.mark.parametrize(
    "purpose",
    [
        "optional_research",
        "finalization",
    ],
)
def test_llm_budget_authorization_accepts_known_purpose(
    purpose,
):
    authorization = BudgetAuthorization(
        resource="llm_calls",
        requested=1,
        authorized=1,
        llm_purpose=purpose,
    )

    assert (
        authorization.llm_purpose
        == purpose
    )


def test_llm_budget_authorization_requires_purpose():
    with pytest.raises(
        ValueError,
        match="llm_purpose is required",
    ):
        BudgetAuthorization(
            resource="llm_calls",
            requested=1,
            authorized=1,
        )


@pytest.mark.parametrize(
    "purpose",
    [
        123,
        [],
        {},
    ],
)
def test_llm_budget_authorization_purpose_requires_string(
    purpose,
):
    with pytest.raises(
        TypeError,
        match="llm_purpose must be a string",
    ):
        BudgetAuthorization(
            resource="llm_calls",
            requested=1,
            authorized=1,
            llm_purpose=purpose,
        )


@pytest.mark.parametrize(
    "purpose",
    [
        "",
        " ",
        "research",
        "synthesis",
        "OPTIONAL_RESEARCH",
    ],
)
def test_llm_budget_authorization_rejects_unknown_purpose(
    purpose,
):
    with pytest.raises(
        ValueError,
        match="llm_purpose must be",
    ):
        BudgetAuthorization(
            resource="llm_calls",
            requested=1,
            authorized=1,
            llm_purpose=purpose,
        )


def test_non_llm_authorization_has_no_llm_purpose():
    authorization = BudgetAuthorization(
        resource="search_queries",
        requested=1,
        authorized=1,
    )

    assert (
        authorization.llm_purpose
        is None
    )


@pytest.mark.parametrize(
    "purpose",
    [
        "optional_research",
        "finalization",
    ],
)
def test_non_llm_authorization_rejects_llm_purpose(
    purpose,
):
    with pytest.raises(
        ValueError,
        match="must be None",
    ):
        BudgetAuthorization(
            resource="search_queries",
            requested=1,
            authorized=1,
            llm_purpose=purpose,
        )        


def test_budget_authorization_rejects_unknown_resource():
    with pytest.raises(ValueError):
        BudgetAuthorization(
            resource="unknown",
            requested=1,
            authorized=1,
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "requested",
        "authorized",
    ],
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
def test_budget_authorization_rejects_invalid_count_types(
    field_name,
    value,
):
    kwargs = {
        "resource": "search_queries",
        "requested": 1,
        "authorized": 1,
    }

    kwargs[field_name] = value

    with pytest.raises(TypeError):
        BudgetAuthorization(
            **kwargs
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "requested",
        "authorized",
    ],
)
def test_budget_authorization_rejects_negative_counts(
    field_name,
):
    kwargs = {
        "resource": "search_queries",
        "requested": 1,
        "authorized": 1,
    }

    kwargs[field_name] = -1

    with pytest.raises(ValueError):
        BudgetAuthorization(
            **kwargs
        )


def test_authorized_count_must_not_exceed_requested():
    with pytest.raises(ValueError):
        BudgetAuthorization(
            resource="search_queries",
            requested=2,
            authorized=3,
        )


@pytest.mark.parametrize(
    "value",
    [
        123,
        [],
        {},
    ],
)
def test_budget_authorization_reason_requires_string_or_none(
    value,
):
    with pytest.raises(TypeError):
        BudgetAuthorization(
            resource="search_queries",
            requested=2,
            authorized=1,
            reason=value,
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
def test_budget_authorization_reason_must_not_be_blank(
    value,
):
    with pytest.raises(ValueError):
        BudgetAuthorization(
            resource="search_queries",
            requested=2,
            authorized=1,
            reason=value,
        )


# ---------------------------------------------------------------------------
# BudgetPolicy construction
# ---------------------------------------------------------------------------


def test_budget_policy_accepts_budget_limits():
    limits = _limits()

    policy = BudgetPolicy(
        limits=limits,
    )

    assert policy.limits is limits


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        [],
        "limits",
        123,
    ],
)
def test_budget_policy_requires_budget_limits(
    value,
):
    with pytest.raises(TypeError):
        BudgetPolicy(
            limits=value,
        )


# ---------------------------------------------------------------------------
# Research-iteration authorization
# ---------------------------------------------------------------------------


def test_first_research_iteration_is_authorized():
    policy = _policy(
        max_research_iterations=2,
    )

    result = policy.authorize_iteration(
        usage=BudgetUsage(
            iteration_count=0,
        )
    )

    assert result.resource == "research_iterations"
    assert result.requested == 1
    assert result.authorized == 1
    assert result.reason is None


def test_second_total_research_iteration_is_authorized():
    policy = _policy(
        max_research_iterations=2,
    )

    result = policy.authorize_iteration(
        usage=BudgetUsage(
            iteration_count=1,
        )
    )

    assert result.authorized == 1
    assert result.fully_authorized is True


def test_third_iteration_is_rejected_when_total_limit_is_two():
    """max_research_iterations=2 means two total runs, not 2 retries."""

    policy = _policy(
        max_research_iterations=2,
    )

    result = policy.authorize_iteration(
        usage=BudgetUsage(
            iteration_count=2,
        )
    )

    assert result.authorized == 0
    assert result.exhausted is True

    assert result.reason == (
        "research iteration budget exhausted"
    )


def test_iteration_budget_stays_exhausted_when_usage_exceeds_limit():
    policy = _policy(
        max_research_iterations=2,
    )

    result = policy.authorize_iteration(
        usage=BudgetUsage(
            iteration_count=10,
        )
    )

    assert result.authorized == 0


# ---------------------------------------------------------------------------
# Search-query authorization
# ---------------------------------------------------------------------------


def test_search_queries_fully_authorized_within_both_limits():
    policy = _policy(
        max_search_queries_per_run=8,
        max_search_queries_per_iteration=5,
    )

    result = policy.authorize_search_queries(
        usage=BudgetUsage(
            search_queries_used=2,
        ),
        requested=4,
    )

    assert result.resource == "search_queries"
    assert result.requested == 4
    assert result.authorized == 4
    assert result.skipped == 0
    assert result.reason is None


def test_search_queries_are_capped_by_per_iteration_limit():
    policy = _policy(
        max_search_queries_per_run=8,
        max_search_queries_per_iteration=3,
    )

    result = policy.authorize_search_queries(
        usage=BudgetUsage(
            search_queries_used=0,
        ),
        requested=5,
    )

    assert result.authorized == 3
    assert result.skipped == 2

    assert result.reason == (
        "per-iteration search query limit reached"
    )


def test_search_queries_are_capped_by_whole_run_remaining_budget():
    policy = _policy(
        max_search_queries_per_run=8,
        max_search_queries_per_iteration=5,
    )

    result = policy.authorize_search_queries(
        usage=BudgetUsage(
            search_queries_used=7,
        ),
        requested=5,
    )

    assert result.authorized == 1
    assert result.skipped == 4

    assert result.reason == (
        "whole-run search query budget reached"
    )


def test_search_queries_use_minimum_of_all_constraints():
    policy = _policy(
        max_search_queries_per_run=8,
        max_search_queries_per_iteration=2,
    )

    result = policy.authorize_search_queries(
        usage=BudgetUsage(
            search_queries_used=5,
        ),
        requested=5,
    )

    # requested = 5
    # run remaining = 3
    # per-iteration = 2
    # authorized = min(5, 3, 2) = 2
    assert result.authorized == 2


def test_search_query_budget_exhaustion_returns_zero_not_exception():
    policy = _policy(
        max_search_queries_per_run=8,
        max_search_queries_per_iteration=5,
    )

    result = policy.authorize_search_queries(
        usage=BudgetUsage(
            search_queries_used=8,
        ),
        requested=3,
    )

    assert result.authorized == 0
    assert result.exhausted is True

    assert result.reason == (
        "whole-run search query budget exhausted"
    )


def test_zero_requested_searches_are_fully_authorized():
    result = _policy().authorize_search_queries(
        usage=BudgetUsage(),
        requested=0,
    )

    assert result.authorized == 0
    assert result.fully_authorized is True
    assert result.exhausted is False
    assert result.reason is None


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        None,
        -1,
        1.5,
        "3",
        [],
        {},
    ],
)
def test_search_authorization_rejects_invalid_requested_count(
    value,
):
    expected_error = (
        ValueError
        if value == -1
        else TypeError
    )

    with pytest.raises(expected_error):
        _policy().authorize_search_queries(
            usage=BudgetUsage(),
            requested=value,
        )


# ---------------------------------------------------------------------------
# Unique-source authorization
# ---------------------------------------------------------------------------


def test_new_sources_are_fully_authorized_when_capacity_exists():
    policy = _policy(
        max_sources_per_run=12,
    )

    result = policy.authorize_new_sources(
        usage=BudgetUsage(
            unique_sources=4,
        ),
        requested=5,
    )

    assert result.resource == "sources"
    assert result.authorized == 5
    assert result.reason is None


def test_new_sources_are_partially_authorized_at_run_limit():
    policy = _policy(
        max_sources_per_run=12,
    )

    result = policy.authorize_new_sources(
        usage=BudgetUsage(
            unique_sources=10,
        ),
        requested=5,
    )

    assert result.authorized == 2
    assert result.skipped == 3

    assert result.reason == (
        "whole-run source budget reached"
    )


def test_new_source_budget_exhaustion_returns_zero():
    policy = _policy(
        max_sources_per_run=12,
    )

    result = policy.authorize_new_sources(
        usage=BudgetUsage(
            unique_sources=12,
        ),
        requested=2,
    )

    assert result.authorized == 0
    assert result.exhausted is True


def test_source_usage_above_limit_still_authorizes_zero():
    policy = _policy(
        max_sources_per_run=12,
    )

    result = policy.authorize_new_sources(
        usage=BudgetUsage(
            unique_sources=20,
        ),
        requested=2,
    )

    assert result.authorized == 0


# ---------------------------------------------------------------------------
# Source-fetch authorization
# ---------------------------------------------------------------------------


def test_source_fetches_are_fully_authorized_with_capacity():
    policy = _policy(
        max_source_fetches_per_run=10,
    )

    result = policy.authorize_source_fetches(
        usage=BudgetUsage(
            source_fetches_used=3,
        ),
        requested=4,
    )

    assert result.resource == "source_fetches"
    assert result.authorized == 4
    assert result.reason is None


def test_source_fetches_are_partially_authorized():
    policy = _policy(
        max_source_fetches_per_run=10,
    )

    result = policy.authorize_source_fetches(
        usage=BudgetUsage(
            source_fetches_used=8,
        ),
        requested=5,
    )

    assert result.authorized == 2
    assert result.skipped == 3

    assert result.reason == (
        "whole-run source fetch budget reached"
    )


def test_source_fetch_budget_exhaustion_returns_zero():
    policy = _policy(
        max_source_fetches_per_run=10,
    )

    result = policy.authorize_source_fetches(
        usage=BudgetUsage(
            source_fetches_used=10,
        ),
        requested=1,
    )

    assert result.authorized == 0
    assert result.exhausted is True


# ---------------------------------------------------------------------------
# LLM-call authorization
# ---------------------------------------------------------------------------


def test_optional_llm_calls_leave_finalization_reserve_untouched():
    policy = _policy(
        max_llm_calls_per_run=10,
        finalization_llm_reserve=2,
    )

    result = policy.authorize_llm_calls(
        usage=BudgetUsage(
            llm_calls_used=7,
        ),
        requested=5,
        purpose="optional_research",
    )

    # remaining = 3
    # protected finalization reserve = 2
    # optional capacity = 1
    assert result.resource == "llm_calls"
    assert (
        result.llm_purpose
        == "optional_research"
    )
    assert result.authorized == 1
    assert result.skipped == 4

    assert result.reason == (
        "optional LLM budget reached"
    )


def test_optional_llm_calls_are_blocked_when_only_reserve_remains():
    policy = _policy(
        max_llm_calls_per_run=10,
        finalization_llm_reserve=2,
    )

    result = policy.authorize_llm_calls(
        usage=BudgetUsage(
            llm_calls_used=8,
        ),
        requested=1,
        purpose="optional_research",
    )

    assert result.authorized == 0
    assert result.exhausted is True

    assert result.reason == (
        "LLM finalization reserve protected"
    )


def test_optional_llm_calls_are_fully_authorized_above_reserve():
    policy = _policy(
        max_llm_calls_per_run=10,
        finalization_llm_reserve=2,
    )

    result = policy.authorize_llm_calls(
        usage=BudgetUsage(
            llm_calls_used=4,
        ),
        requested=3,
        purpose="optional_research",
    )

    # remaining = 6
    # optional capacity = 4
    assert result.authorized == 3
    assert result.reason is None


def test_finalization_can_use_protected_capacity():
    policy = _policy(
        max_llm_calls_per_run=10,
        finalization_llm_reserve=2,
    )

    result = policy.authorize_llm_calls(
        usage=BudgetUsage(
            llm_calls_used=8,
        ),
        requested=2,
        purpose="finalization",
    )
    assert (
        result.llm_purpose
        == "finalization"
    )
    assert result.authorized == 2
    assert result.fully_authorized is True
    assert result.reason is None


def test_finalization_is_partially_authorized_when_only_one_call_remains():
    policy = _policy(
        max_llm_calls_per_run=10,
        finalization_llm_reserve=2,
    )

    result = policy.authorize_llm_calls(
        usage=BudgetUsage(
            llm_calls_used=9,
        ),
        requested=2,
        purpose="finalization",
    )

    assert result.authorized == 1
    assert result.skipped == 1

    assert result.reason == (
        "whole-run LLM budget exhausted"
    )


def test_finalization_is_blocked_when_llm_budget_is_exhausted():
    policy = _policy(
        max_llm_calls_per_run=10,
        finalization_llm_reserve=2,
    )

    result = policy.authorize_llm_calls(
        usage=BudgetUsage(
            llm_calls_used=10,
        ),
        requested=2,
        purpose="finalization",
    )

    assert result.authorized == 0
    assert result.exhausted is True

    assert result.reason == (
        "whole-run LLM budget exhausted"
    )


def test_zero_finalization_reserve_allows_optional_use_of_all_remaining_calls():
    policy = _policy(
        max_llm_calls_per_run=10,
        finalization_llm_reserve=0,
    )

    result = policy.authorize_llm_calls(
        usage=BudgetUsage(
            llm_calls_used=7,
        ),
        requested=3,
        purpose="optional_research",
    )

    assert result.authorized == 3
    assert result.reason is None


def test_reserve_equal_to_entire_llm_budget_blocks_optional_calls():
    policy = _policy(
        max_llm_calls_per_run=2,
        finalization_llm_reserve=2,
    )

    result = policy.authorize_llm_calls(
        usage=BudgetUsage(),
        requested=1,
        purpose="optional_research",
    )

    assert result.authorized == 0

    assert result.reason == (
        "LLM finalization reserve protected"
    )


@pytest.mark.parametrize(
    "purpose",
    [
        None,
        "",
        "research",
        "synthesis",
        "OPTIONAL_RESEARCH",
        123,
    ],
)
def test_llm_authorization_rejects_unknown_purpose(
    purpose,
):
    with pytest.raises(ValueError):
        _policy().authorize_llm_calls(
            usage=BudgetUsage(),
            requested=1,
            purpose=purpose,
        )


# ---------------------------------------------------------------------------
# Usage ownership / purity
# ---------------------------------------------------------------------------


def test_budget_policy_does_not_mutate_usage_snapshot():
    usage = BudgetUsage(
        iteration_count=1,
        search_queries_used=3,
        source_fetches_used=4,
        llm_calls_used=5,
        unique_sources=6,
    )

    original = usage

    policy = _policy()

    policy.authorize_search_queries(
        usage=usage,
        requested=2,
    )

    assert usage is original

    assert usage == BudgetUsage(
        iteration_count=1,
        search_queries_used=3,
        source_fetches_used=4,
        llm_calls_used=5,
        unique_sources=6,
    )


@pytest.mark.parametrize(
    "method_name",
    [
        "authorize_iteration",
        "authorize_search_queries",
        "authorize_new_sources",
        "authorize_source_fetches",
        "authorize_llm_calls",
    ],
)
def test_budget_policy_methods_require_budget_usage(
    method_name,
):
    policy = _policy()

    method = getattr(
        policy,
        method_name,
    )

    kwargs = {
        "usage": None,
    }

    if method_name in {
        "authorize_search_queries",
        "authorize_new_sources",
        "authorize_source_fetches",
    }:
        kwargs["requested"] = 1

    if method_name == "authorize_llm_calls":
        kwargs["requested"] = 1
        kwargs["purpose"] = "optional_research"

    with pytest.raises(TypeError):
        method(
            **kwargs
        )