"""Provider-neutral search authorization and execution.

Search budgeting uses a two-phase design.

Phase 1 - pure authorization:

    planned SubQuestions
        ↓
    prepare_search_batch(...)
        ↓
    BudgetPolicy
        ↓
    SearchBatch

Phase 2 - side effects:

    SearchBatch
        ↓
    SearchNode.search(...)
        ↓
    external search provider

The SearchBatch exposes the number of authorized search attempts as
``search_queries_used``.

LangGraph orchestration will later write that value to ResearchState
*before* calling SearchNode.search(). This preserves conservative
whole-run accounting even if the provider call subsequently fails.

Budget exhaustion is expected control flow. It therefore produces an
empty or partial SearchBatch rather than raising a budget exception.

The budget layer always preserves planner order. If five planned queries
are supplied and only two are authorized, the first two are executed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from research_agent.graph.budget import (
    BudgetAuthorization,
    BudgetPolicy,
    BudgetUsage,
)
from research_agent.models.schemas import (
    SearchResult,
    SubQuestion,
)


class SearchClient(Protocol):
    """Minimal search interface required by the search node."""

    def search(
        self,
        *,
        query: str,
        sub_question_id: str,
        max_results: int,
    ) -> list[SearchResult]:
        """Execute one search query and return normalized results."""

        ...


@dataclass(frozen=True)
class SearchBatch:
    """One already-authorized batch of search work.

    ``sub_questions`` contains only the deterministic prefix that may
    actually be executed.

    ``authorization`` preserves the complete budget decision, including
    how much work was requested, authorized, and skipped.

    Constructing a batch manually with more or fewer SubQuestions than
    the authorization permits is rejected.
    """

    sub_questions: tuple[SubQuestion, ...]
    authorization: BudgetAuthorization

    def __post_init__(self) -> None:
        if not isinstance(
            self.sub_questions,
            tuple,
        ):
            raise TypeError(
                "sub_questions must be a tuple."
            )

        if not isinstance(
            self.authorization,
            BudgetAuthorization,
        ):
            raise TypeError(
                "authorization must be a "
                "BudgetAuthorization object."
            )

        if (
            self.authorization.resource
            != "search_queries"
        ):
            raise ValueError(
                "SearchBatch authorization must be for "
                "search_queries."
            )

        for sub_question in self.sub_questions:
            if not isinstance(
                sub_question,
                SubQuestion,
            ):
                raise TypeError(
                    "Every SearchBatch item must be a "
                    "SubQuestion."
                )

        if (
            len(self.sub_questions)
            != self.authorization.authorized
        ):
            raise ValueError(
                "SearchBatch size must match the authorized "
                "search-query count."
            )

    @property
    def search_queries_used(self) -> int:
        """Usage delta that must be charged before execution."""

        return self.authorization.authorized

    @property
    def requested(self) -> int:
        """Number of search queries originally requested."""

        return self.authorization.requested

    @property
    def skipped(self) -> int:
        """Number of planned searches omitted by the budget."""

        return self.authorization.skipped


def prepare_search_batch(
    *,
    sub_questions: list[SubQuestion],
    usage: BudgetUsage,
    budget_policy: BudgetPolicy,
) -> SearchBatch:
    """Authorize a deterministic prefix of planned search queries.

    This function is pure. It performs no external searches and does not
    mutate BudgetUsage, BudgetPolicy, or the supplied list.

    All supplied SubQuestions are validated before the budget decision is
    made.

    The returned SearchBatch contains only work that may proceed.

    Example:

        requested = 5
        per-iteration capacity = 3
        whole-run remaining = 2

        authorized = 2

    The resulting batch contains the first two SubQuestions in planner
    order and exposes ``search_queries_used == 2``.
    """

    if not isinstance(
        sub_questions,
        list,
    ):
        raise TypeError(
            "sub_questions must be a list."
        )

    for sub_question in sub_questions:
        if not isinstance(
            sub_question,
            SubQuestion,
        ):
            raise TypeError(
                "Every item in sub_questions must be a "
                "SubQuestion."
            )

    if not isinstance(
        budget_policy,
        BudgetPolicy,
    ):
        raise TypeError(
            "budget_policy must be a BudgetPolicy object."
        )

    # BudgetPolicy performs its own strict BudgetUsage validation.
    authorization = (
        budget_policy.authorize_search_queries(
            usage=usage,
            requested=len(sub_questions),
        )
    )

    authorized_sub_questions = tuple(
        sub_questions[
            : authorization.authorized
        ]
    )

    return SearchBatch(
        sub_questions=authorized_sub_questions,
        authorization=authorization,
    )


class SearchNode:
    """Execute an already-authorized search batch.

    SearchNode intentionally does not own a search-query budget.

    Whole-run authorization belongs to BudgetPolicy and must happen before
    this node performs provider side effects.
    """

    def __init__(
        self,
        *,
        search_client: SearchClient,
        max_results_per_query: int,
    ) -> None:
        if isinstance(
            max_results_per_query,
            bool,
        ) or not isinstance(
            max_results_per_query,
            int,
        ):
            raise TypeError(
                "max_results_per_query must be an integer."
            )

        if max_results_per_query < 1:
            raise ValueError(
                "max_results_per_query must be at least 1."
            )

        self._search_client = search_client
        self._max_results_per_query = (
            max_results_per_query
        )

    def search(
        self,
        batch: SearchBatch,
    ) -> list[SearchResult]:
        """Execute only the work present in an authorized SearchBatch."""

        if not isinstance(
            batch,
            SearchBatch,
        ):
            raise TypeError(
                "batch must be a SearchBatch object."
            )

        results: list[SearchResult] = []

        for sub_question in batch.sub_questions:
            query_results = (
                self._search_client.search(
                    query=sub_question.question,
                    sub_question_id=(
                        sub_question.id
                    ),
                    max_results=(
                        self._max_results_per_query
                    ),
                )
            )

            if not isinstance(
                query_results,
                list,
            ):
                raise TypeError(
                    "Search client must return a list."
                )

            for result in query_results:
                if not isinstance(
                    result,
                    SearchResult,
                ):
                    raise TypeError(
                        "Search client returned an invalid "
                        "result type."
                    )

                if (
                    result.sub_question_id
                    != sub_question.id
                ):
                    raise ValueError(
                        "Search result sub_question_id does "
                        "not match the query that produced it."
                    )

            results.extend(
                query_results
            )

        return results