"""Tests for provider-neutral research search authorization and execution.

These tests cover two separate responsibilities:

1. prepare_search_batch()
   - pure whole-run budget authorization
   - deterministic prefix selection
   - no provider side effects

2. SearchNode
   - executes only an already-authorized SearchBatch
   - validates provider responses
   - does not own or reset whole-run search budgets
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from research_agent.graph.budget import (
    BudgetAuthorization,
    BudgetLimits,
    BudgetPolicy,
    BudgetUsage,
)
from research_agent.graph.nodes.search import (
    SearchBatch,
    SearchNode,
    prepare_search_batch,
)
from research_agent.models.schemas import (
    SearchResult,
    SubQuestion,
)


class FakeSearchClient:
    """Simple fake search client for SearchNode tests."""

    def __init__(
        self,
        responses=None,
    ):
        self.responses = responses or {}
        self.calls = []

    def search(
        self,
        *,
        query: str,
        sub_question_id: str,
        max_results: int,
    ) -> list[SearchResult]:
        self.calls.append(
            {
                "query": query,
                "sub_question_id": (
                    sub_question_id
                ),
                "max_results": (
                    max_results
                ),
            }
        )

        return self.responses.get(
            sub_question_id,
            [],
        )


class RaisingSearchClient:
    """Search client that records a call and then fails."""

    def __init__(self):
        self.calls = []

    def search(
        self,
        *,
        query: str,
        sub_question_id: str,
        max_results: int,
    ):
        self.calls.append(
            {
                "query": query,
                "sub_question_id": (
                    sub_question_id
                ),
                "max_results": (
                    max_results
                ),
            }
        )

        raise RuntimeError(
            "provider failed"
        )


def _sub_question(
    *,
    id: str,
    question: str,
) -> SubQuestion:
    return SubQuestion(
        id=id,
        question=question,
    )


def _search_result(
    *,
    sub_question_id: str,
    query: str,
    title: str = "Result",
    url: str = "https://example.com",
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


def _policy(
    *,
    max_search_queries_per_run: int = 8,
    max_search_queries_per_iteration: int = 5,
) -> BudgetPolicy:
    return BudgetPolicy(
        limits=BudgetLimits(
            max_research_iterations=2,
            max_search_queries_per_run=(
                max_search_queries_per_run
            ),
            max_search_queries_per_iteration=(
                max_search_queries_per_iteration
            ),
            max_sources_per_run=12,
            max_source_fetches_per_run=12,
            max_llm_calls_per_run=64,
            finalization_llm_reserve=2,
        )
    )


def _full_authorization(
    count: int,
) -> BudgetAuthorization:
    return BudgetAuthorization(
        resource="search_queries",
        requested=count,
        authorized=count,
    )


def _batch(
    *sub_questions: SubQuestion,
) -> SearchBatch:
    return SearchBatch(
        sub_questions=tuple(
            sub_questions
        ),
        authorization=_full_authorization(
            len(sub_questions)
        ),
    )


# ---------------------------------------------------------------------------
# SearchNode constructor validation
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
def test_max_results_per_query_must_be_integer(
    value,
):
    with pytest.raises(TypeError):
        SearchNode(
            search_client=FakeSearchClient(),
            max_results_per_query=value,
        )


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        -10,
    ],
)
def test_max_results_per_query_must_be_positive(
    value,
):
    with pytest.raises(ValueError):
        SearchNode(
            search_client=FakeSearchClient(),
            max_results_per_query=value,
        )


# ---------------------------------------------------------------------------
# SearchBatch contract
# ---------------------------------------------------------------------------


def test_search_batch_accepts_valid_authorized_work():
    sub_question = _sub_question(
        id="sq_one",
        question="Question one?",
    )

    batch = SearchBatch(
        sub_questions=(
            sub_question,
        ),
        authorization=BudgetAuthorization(
            resource="search_queries",
            requested=1,
            authorized=1,
        ),
    )

    assert batch.sub_questions == (
        sub_question,
    )

    assert batch.search_queries_used == 1
    assert batch.requested == 1
    assert batch.skipped == 0


def test_search_batch_is_frozen():
    batch = _batch()

    with pytest.raises(
        FrozenInstanceError
    ):
        batch.sub_questions = ()


def test_search_batch_requires_tuple_of_sub_questions():
    with pytest.raises(TypeError):
        SearchBatch(
            sub_questions=[],
            authorization=(
                BudgetAuthorization(
                    resource=(
                        "search_queries"
                    ),
                    requested=0,
                    authorized=0,
                )
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
def test_search_batch_requires_budget_authorization(
    value,
):
    with pytest.raises(TypeError):
        SearchBatch(
            sub_questions=(),
            authorization=value,
        )


def test_search_batch_requires_search_query_authorization():
    with pytest.raises(
        ValueError,
        match="search_queries",
    ):
        SearchBatch(
            sub_questions=(),
            authorization=(
                BudgetAuthorization(
                    resource="source_fetches",
                    requested=0,
                    authorized=0,
                )
            ),
        )


def test_search_batch_requires_only_sub_question_models():
    with pytest.raises(TypeError):
        SearchBatch(
            sub_questions=(
                "not a SubQuestion",
            ),
            authorization=(
                BudgetAuthorization(
                    resource=(
                        "search_queries"
                    ),
                    requested=1,
                    authorized=1,
                )
            ),
        )


def test_search_batch_size_must_match_authorized_count():
    sub_question = _sub_question(
        id="sq_one",
        question="Question one?",
    )

    with pytest.raises(
        ValueError,
        match="authorized",
    ):
        SearchBatch(
            sub_questions=(
                sub_question,
            ),
            authorization=(
                BudgetAuthorization(
                    resource=(
                        "search_queries"
                    ),
                    requested=2,
                    authorized=2,
                )
            ),
        )


def test_partial_search_batch_exposes_skipped_count():
    sub_question = _sub_question(
        id="sq_one",
        question="Question one?",
    )

    batch = SearchBatch(
        sub_questions=(
            sub_question,
        ),
        authorization=BudgetAuthorization(
            resource="search_queries",
            requested=3,
            authorized=1,
            reason="budget reached",
        ),
    )

    assert batch.search_queries_used == 1
    assert batch.requested == 3
    assert batch.skipped == 2


# ---------------------------------------------------------------------------
# Search authorization
# ---------------------------------------------------------------------------


def test_empty_plan_produces_empty_authorized_batch():
    batch = prepare_search_batch(
        sub_questions=[],
        usage=BudgetUsage(),
        budget_policy=_policy(),
    )

    assert batch.sub_questions == ()
    assert batch.search_queries_used == 0
    assert batch.requested == 0
    assert batch.skipped == 0


def test_search_plan_is_fully_authorized_with_capacity():
    sub_questions = [
        _sub_question(
            id="sq_one",
            question="Question one?",
        ),
        _sub_question(
            id="sq_two",
            question="Question two?",
        ),
    ]

    batch = prepare_search_batch(
        sub_questions=sub_questions,
        usage=BudgetUsage(),
        budget_policy=_policy(),
    )

    assert batch.sub_questions == tuple(
        sub_questions
    )

    assert batch.search_queries_used == 2
    assert batch.skipped == 0


def test_search_plan_is_limited_by_per_iteration_budget():
    sub_questions = [
        _sub_question(
            id="sq_one",
            question="Question one?",
        ),
        _sub_question(
            id="sq_two",
            question="Question two?",
        ),
        _sub_question(
            id="sq_three",
            question="Question three?",
        ),
    ]

    batch = prepare_search_batch(
        sub_questions=sub_questions,
        usage=BudgetUsage(),
        budget_policy=_policy(
            max_search_queries_per_run=8,
            max_search_queries_per_iteration=2,
        ),
    )

    assert [
        item.id
        for item in batch.sub_questions
    ] == [
        "sq_one",
        "sq_two",
    ]

    assert batch.search_queries_used == 2
    assert batch.requested == 3
    assert batch.skipped == 1

    assert batch.authorization.reason == (
        "per-iteration search query limit reached"
    )


def test_search_plan_is_limited_by_whole_run_remaining_budget():
    sub_questions = [
        _sub_question(
            id="sq_one",
            question="Question one?",
        ),
        _sub_question(
            id="sq_two",
            question="Question two?",
        ),
        _sub_question(
            id="sq_three",
            question="Question three?",
        ),
    ]

    batch = prepare_search_batch(
        sub_questions=sub_questions,
        usage=BudgetUsage(
            search_queries_used=7,
        ),
        budget_policy=_policy(
            max_search_queries_per_run=8,
            max_search_queries_per_iteration=5,
        ),
    )

    assert [
        item.id
        for item in batch.sub_questions
    ] == [
        "sq_one",
    ]

    assert batch.search_queries_used == 1
    assert batch.skipped == 2

    assert batch.authorization.reason == (
        "whole-run search query budget reached"
    )


def test_exhausted_search_budget_returns_empty_batch_not_exception():
    sub_questions = [
        _sub_question(
            id="sq_one",
            question="Question one?",
        ),
    ]

    batch = prepare_search_batch(
        sub_questions=sub_questions,
        usage=BudgetUsage(
            search_queries_used=8,
        ),
        budget_policy=_policy(
            max_search_queries_per_run=8,
        ),
    )

    assert batch.sub_questions == ()
    assert batch.search_queries_used == 0
    assert batch.skipped == 1

    assert batch.authorization.exhausted is True

    assert batch.authorization.reason == (
        "whole-run search query budget exhausted"
    )


def test_authorization_preserves_planner_order():
    sub_questions = [
        _sub_question(
            id="sq_three",
            question="Question three?",
        ),
        _sub_question(
            id="sq_one",
            question="Question one?",
        ),
        _sub_question(
            id="sq_two",
            question="Question two?",
        ),
    ]

    batch = prepare_search_batch(
        sub_questions=sub_questions,
        usage=BudgetUsage(),
        budget_policy=_policy(
            max_search_queries_per_iteration=2,
        ),
    )

    assert [
        item.id
        for item in batch.sub_questions
    ] == [
        "sq_three",
        "sq_one",
    ]


def test_prepare_search_batch_does_not_mutate_input_list():
    sub_questions = [
        _sub_question(
            id="sq_one",
            question="Question one?",
        ),
        _sub_question(
            id="sq_two",
            question="Question two?",
        ),
    ]

    before = list(
        sub_questions
    )

    prepare_search_batch(
        sub_questions=sub_questions,
        usage=BudgetUsage(),
        budget_policy=_policy(
            max_search_queries_per_iteration=1,
        ),
    )

    assert sub_questions == before


@pytest.mark.parametrize(
    "value",
    [
        None,
        (),
        {},
        "not a list",
    ],
)
def test_prepare_search_batch_requires_list(
    value,
):
    with pytest.raises(TypeError):
        prepare_search_batch(
            sub_questions=value,
            usage=BudgetUsage(),
            budget_policy=_policy(),
        )


def test_prepare_search_batch_validates_every_item_before_authorization():
    valid = _sub_question(
        id="sq_one",
        question="Question one?",
    )

    with pytest.raises(TypeError):
        prepare_search_batch(
            sub_questions=[
                valid,
                "invalid",
            ],
            usage=BudgetUsage(),
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
def test_prepare_search_batch_requires_budget_policy(
    value,
):
    with pytest.raises(TypeError):
        prepare_search_batch(
            sub_questions=[],
            usage=BudgetUsage(),
            budget_policy=value,
        )


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        [],
        "usage",
        123,
    ],
)
def test_prepare_search_batch_requires_budget_usage(
    value,
):
    with pytest.raises(TypeError):
        prepare_search_batch(
            sub_questions=[],
            usage=value,
            budget_policy=_policy(),
        )


# ---------------------------------------------------------------------------
# Search execution
# ---------------------------------------------------------------------------


def test_empty_authorized_batch_returns_empty_results():
    client = FakeSearchClient()

    node = SearchNode(
        search_client=client,
        max_results_per_query=3,
    )

    result = node.search(
        _batch()
    )

    assert result == []
    assert client.calls == []


def test_one_authorized_sub_question_executes_one_search():
    sub_question = _sub_question(
        id="sq_one",
        question="What methods are used?",
    )

    client = FakeSearchClient(
        responses={
            "sq_one": [
                _search_result(
                    sub_question_id="sq_one",
                    query=(
                        "What methods are used?"
                    ),
                )
            ]
        }
    )

    node = SearchNode(
        search_client=client,
        max_results_per_query=3,
    )

    result = node.search(
        _batch(
            sub_question
        )
    )

    assert len(result) == 1

    assert (
        result[0].sub_question_id
        == "sq_one"
    )


def test_search_uses_sub_question_question_as_query():
    sub_question = _sub_question(
        id="sq_one",
        question="What limitations exist?",
    )

    client = FakeSearchClient()

    node = SearchNode(
        search_client=client,
        max_results_per_query=3,
    )

    node.search(
        _batch(
            sub_question
        )
    )

    assert client.calls == [
        {
            "query": (
                "What limitations exist?"
            ),
            "sub_question_id": "sq_one",
            "max_results": 3,
        }
    ]


def test_configured_result_limit_is_forwarded():
    sub_question = _sub_question(
        id="sq_one",
        question="Question one?",
    )

    client = FakeSearchClient()

    node = SearchNode(
        search_client=client,
        max_results_per_query=7,
    )

    node.search(
        _batch(
            sub_question
        )
    )

    assert (
        client.calls[0]["max_results"]
        == 7
    )


def test_multiple_authorized_sub_questions_are_searched_in_order():
    sub_questions = [
        _sub_question(
            id="sq_one",
            question="Question one?",
        ),
        _sub_question(
            id="sq_two",
            question="Question two?",
        ),
        _sub_question(
            id="sq_three",
            question="Question three?",
        ),
    ]

    client = FakeSearchClient()

    node = SearchNode(
        search_client=client,
        max_results_per_query=2,
    )

    node.search(
        _batch(
            *sub_questions
        )
    )

    assert [
        call["sub_question_id"]
        for call in client.calls
    ] == [
        "sq_one",
        "sq_two",
        "sq_three",
    ]


def test_results_from_multiple_queries_are_flattened_in_order():
    sub_questions = [
        _sub_question(
            id="sq_one",
            question="Question one?",
        ),
        _sub_question(
            id="sq_two",
            question="Question two?",
        ),
    ]

    client = FakeSearchClient(
        responses={
            "sq_one": [
                _search_result(
                    sub_question_id="sq_one",
                    query="Question one?",
                    title="A",
                    url="https://example.com/a",
                    rank=1,
                ),
                _search_result(
                    sub_question_id="sq_one",
                    query="Question one?",
                    title="B",
                    url="https://example.com/b",
                    rank=2,
                ),
            ],
            "sq_two": [
                _search_result(
                    sub_question_id="sq_two",
                    query="Question two?",
                    title="C",
                    url="https://example.com/c",
                    rank=1,
                ),
            ],
        }
    )

    node = SearchNode(
        search_client=client,
        max_results_per_query=3,
    )

    result = node.search(
        _batch(
            *sub_questions
        )
    )

    assert [
        item.title
        for item in result
    ] == [
        "A",
        "B",
        "C",
    ]


def test_only_authorized_prefix_is_executed():
    sub_questions = [
        _sub_question(
            id="sq_one",
            question="Question one?",
        ),
        _sub_question(
            id="sq_two",
            question="Question two?",
        ),
        _sub_question(
            id="sq_three",
            question="Question three?",
        ),
    ]

    batch = prepare_search_batch(
        sub_questions=sub_questions,
        usage=BudgetUsage(),
        budget_policy=_policy(
            max_search_queries_per_iteration=2,
        ),
    )

    client = FakeSearchClient()

    node = SearchNode(
        search_client=client,
        max_results_per_query=3,
    )

    node.search(
        batch
    )

    assert [
        call["sub_question_id"]
        for call in client.calls
    ] == [
        "sq_one",
        "sq_two",
    ]


def test_exhausted_budget_batch_executes_no_searches():
    sub_question = _sub_question(
        id="sq_one",
        question="Question one?",
    )

    batch = prepare_search_batch(
        sub_questions=[
            sub_question,
        ],
        usage=BudgetUsage(
            search_queries_used=8,
        ),
        budget_policy=_policy(
            max_search_queries_per_run=8,
        ),
    )

    client = FakeSearchClient()

    node = SearchNode(
        search_client=client,
        max_results_per_query=3,
    )

    result = node.search(
        batch
    )

    assert result == []
    assert client.calls == []


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        (),
        {},
        "not a batch",
        123,
    ],
)
def test_search_requires_authorized_search_batch(
    value,
):
    node = SearchNode(
        search_client=FakeSearchClient(),
        max_results_per_query=3,
    )

    with pytest.raises(TypeError):
        node.search(
            value
        )


def test_authorized_usage_exists_before_provider_failure():
    """The charge is known before the external side effect is attempted."""

    sub_question = _sub_question(
        id="sq_one",
        question="Question one?",
    )

    batch = prepare_search_batch(
        sub_questions=[
            sub_question,
        ],
        usage=BudgetUsage(),
        budget_policy=_policy(),
    )

    assert batch.search_queries_used == 1

    client = RaisingSearchClient()

    node = SearchNode(
        search_client=client,
        max_results_per_query=3,
    )

    with pytest.raises(
        RuntimeError,
        match="provider failed",
    ):
        node.search(
            batch
        )

    assert len(client.calls) == 1

    # The authorization remains one consumed attempt even though the
    # provider operation failed.
    assert batch.search_queries_used == 1


# ---------------------------------------------------------------------------
# Search-client response validation
# ---------------------------------------------------------------------------


def test_search_client_must_return_list():
    sub_question = _sub_question(
        id="sq_one",
        question="Question one?",
    )

    class InvalidClient:
        def search(
            self,
            *,
            query,
            sub_question_id,
            max_results,
        ):
            return None

    node = SearchNode(
        search_client=InvalidClient(),
        max_results_per_query=3,
    )

    with pytest.raises(TypeError):
        node.search(
            _batch(
                sub_question
            )
        )


def test_search_client_results_must_be_search_result_models():
    sub_question = _sub_question(
        id="sq_one",
        question="Question one?",
    )

    client = FakeSearchClient(
        responses={
            "sq_one": [
                "not a SearchResult",
            ]
        }
    )

    node = SearchNode(
        search_client=client,
        max_results_per_query=3,
    )

    with pytest.raises(TypeError):
        node.search(
            _batch(
                sub_question
            )
        )


def test_result_must_match_sub_question_that_produced_it():
    sub_question = _sub_question(
        id="sq_one",
        question="Question one?",
    )

    client = FakeSearchClient(
        responses={
            "sq_one": [
                _search_result(
                    sub_question_id="sq_wrong",
                    query="Question one?",
                )
            ]
        }
    )

    node = SearchNode(
        search_client=client,
        max_results_per_query=3,
    )

    with pytest.raises(ValueError):
        node.search(
            _batch(
                sub_question
            )
        )