"""Tests for the provider-neutral research search node."""

import pytest

from research_agent.graph.nodes.search import (
    SearchBudgetError,
    SearchNode,
)
from research_agent.models.schemas import (
    SearchResult,
    SubQuestion,
)


class FakeSearchClient:
    """Simple fake search client for SearchNode tests."""

    def __init__(self, responses=None):
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
                "sub_question_id": sub_question_id,
                "max_results": max_results,
            }
        )

        return self.responses.get(
            sub_question_id,
            [],
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


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        None,
        1.5,
        "3",
        [],
        {},
        True,
        False,
    ],
)
def test_max_search_queries_must_be_integer(value):
    with pytest.raises(TypeError):
        SearchNode(
            search_client=FakeSearchClient(),
            max_search_queries=value,
            max_results_per_query=5,
        )


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        -10,
    ],
)
def test_max_search_queries_must_be_positive(value):
    with pytest.raises(ValueError):
        SearchNode(
            search_client=FakeSearchClient(),
            max_search_queries=value,
            max_results_per_query=5,
        )


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
def test_max_results_per_query_must_be_integer(value):
    with pytest.raises(TypeError):
        SearchNode(
            search_client=FakeSearchClient(),
            max_search_queries=5,
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
def test_max_results_per_query_must_be_positive(value):
    with pytest.raises(ValueError):
        SearchNode(
            search_client=FakeSearchClient(),
            max_search_queries=5,
            max_results_per_query=value,
        )


# ---------------------------------------------------------------------------
# Search behavior
# ---------------------------------------------------------------------------


def test_empty_sub_question_list_returns_empty_results():
    client = FakeSearchClient()

    node = SearchNode(
        search_client=client,
        max_search_queries=5,
        max_results_per_query=3,
    )

    result = node.search([])

    assert result == []
    assert client.calls == []


def test_one_sub_question_executes_one_search():
    sub_question = _sub_question(
        id="sq_one",
        question="What methods are used?",
    )

    client = FakeSearchClient(
        responses={
            "sq_one": [
                _search_result(
                    sub_question_id="sq_one",
                    query="What methods are used?",
                )
            ]
        }
    )

    node = SearchNode(
        search_client=client,
        max_search_queries=5,
        max_results_per_query=3,
    )

    result = node.search(
        [sub_question]
    )

    assert len(result) == 1
    assert result[0].sub_question_id == "sq_one"


def test_search_uses_sub_question_question_as_query():
    sub_question = _sub_question(
        id="sq_one",
        question="What limitations exist?",
    )

    client = FakeSearchClient()

    node = SearchNode(
        search_client=client,
        max_search_queries=5,
        max_results_per_query=3,
    )

    node.search(
        [sub_question]
    )

    assert client.calls == [
        {
            "query": "What limitations exist?",
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
        max_search_queries=5,
        max_results_per_query=7,
    )

    node.search(
        [sub_question]
    )

    assert client.calls[0]["max_results"] == 7


def test_multiple_sub_questions_are_searched_in_order():
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
        max_search_queries=5,
        max_results_per_query=2,
    )

    node.search(sub_questions)

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
        max_search_queries=5,
        max_results_per_query=3,
    )

    result = node.search(
        sub_questions
    )

    assert [
        item.title
        for item in result
    ] == [
        "A",
        "B",
        "C",
    ]


def test_search_allows_batch_exactly_at_query_budget():
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

    node = SearchNode(
        search_client=FakeSearchClient(),
        max_search_queries=2,
        max_results_per_query=3,
    )

    result = node.search(
        sub_questions
    )

    assert result == []


def test_search_rejects_batch_above_query_budget():
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

    client = FakeSearchClient()

    node = SearchNode(
        search_client=client,
        max_search_queries=1,
        max_results_per_query=3,
    )

    with pytest.raises(SearchBudgetError):
        node.search(
            sub_questions
        )

    assert client.calls == []


@pytest.mark.parametrize(
    "value",
    [
        None,
        (),
        {},
        "not a list",
    ],
)
def test_sub_questions_must_be_list(value):
    node = SearchNode(
        search_client=FakeSearchClient(),
        max_search_queries=5,
        max_results_per_query=3,
    )

    with pytest.raises(TypeError):
        node.search(value)


def test_every_sub_question_must_be_domain_model():
    node = SearchNode(
        search_client=FakeSearchClient(),
        max_search_queries=5,
        max_results_per_query=3,
    )

    with pytest.raises(TypeError):
        node.search(
            [
                "not a SubQuestion",
            ]
        )


def test_invalid_sub_question_stops_before_searching_that_item():
    valid = _sub_question(
        id="sq_one",
        question="Question one?",
    )

    client = FakeSearchClient()

    node = SearchNode(
        search_client=client,
        max_search_queries=5,
        max_results_per_query=3,
    )

    with pytest.raises(TypeError):
        node.search(
            [
                valid,
                "invalid",
            ]
        )

    assert len(client.calls) == 1


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
        max_search_queries=5,
        max_results_per_query=3,
    )

    with pytest.raises(TypeError):
        node.search(
            [sub_question]
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
        max_search_queries=5,
        max_results_per_query=3,
    )

    with pytest.raises(TypeError):
        node.search(
            [sub_question]
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
        max_search_queries=5,
        max_results_per_query=3,
    )

    with pytest.raises(ValueError):
        node.search(
            [sub_question]
        )