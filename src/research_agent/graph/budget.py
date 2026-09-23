"""Whole-run budget primitives for the Research Agent.

This module contains deterministic budget policy only.

It does not:

- call external services
- mutate LangGraph state
- construct graph nodes
- know about Tavily, Gemini, or any provider
- perform side effects

The caller supplies the current usage snapshot, asks how much work may be
authorized, records the authorized amount in state, and only then performs
the corresponding side effect.

That ordering is deliberate:

    authorize
        ↓
    record usage
        ↓
    perform external operation

If the external operation later fails, the attempt still consumed budget.

Budget exhaustion is expected control flow, not an exceptional condition.
Authorization therefore returns zero or a deterministic partial amount
instead of raising a "budget exceeded" exception.

The policy also protects LLM capacity for finalization. Optional research
calls are not allowed to consume the configured finalization reserve.

LLM authorizations additionally preserve their purpose so downstream code
can distinguish optional-research capacity from finalization capacity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


BudgetResource = Literal[
    "research_iterations",
    "search_queries",
    "sources",
    "source_fetches",
    "llm_calls",
]

LLMCallPurpose = Literal[
    "optional_research",
    "finalization",
]


def _validate_int(
    *,
    name: str,
    value: int,
    minimum: int,
) -> None:
    """Validate an integer configuration or usage value."""

    if isinstance(value, bool) or not isinstance(
        value,
        int,
    ):
        raise TypeError(
            f"{name} must be an integer."
        )

    if value < minimum:
        raise ValueError(
            f"{name} must be at least {minimum}."
        )


def remaining_budget(
    *,
    limit: int,
    used: int,
) -> int:
    """Return remaining capacity, clamped at zero.

    ``used`` may theoretically be greater than ``limit`` when loading
    historical or externally constructed state. Remaining capacity is still
    never negative.
    """

    _validate_int(
        name="limit",
        value=limit,
        minimum=0,
    )

    _validate_int(
        name="used",
        value=used,
        minimum=0,
    )

    return max(
        0,
        limit - used,
    )


@dataclass(frozen=True)
class BudgetLimits:
    """Static whole-run and local budget configuration.

    Whole-run limits:

    - max_research_iterations
    - max_search_queries_per_run
    - max_sources_per_run
    - max_source_fetches_per_run
    - max_llm_calls_per_run

    Local shaping limit:

    - max_search_queries_per_iteration

    ``finalization_llm_reserve`` protects enough LLM calls for the final
    synthesis stage. With A1.2 enabled, the default reserve is two calls:

    1. synthesis
    2. semantic verification
    """

    max_research_iterations: int
    max_search_queries_per_run: int
    max_search_queries_per_iteration: int
    max_sources_per_run: int
    max_source_fetches_per_run: int
    max_llm_calls_per_run: int

    finalization_llm_reserve: int = 2

    def __post_init__(
        self,
    ) -> None:
        positive_fields = {
            "max_research_iterations": (
                self.max_research_iterations
            ),
            "max_search_queries_per_run": (
                self.max_search_queries_per_run
            ),
            "max_search_queries_per_iteration": (
                self.max_search_queries_per_iteration
            ),
            "max_sources_per_run": (
                self.max_sources_per_run
            ),
            "max_source_fetches_per_run": (
                self.max_source_fetches_per_run
            ),
            "max_llm_calls_per_run": (
                self.max_llm_calls_per_run
            ),
        }

        for name, value in (
            positive_fields.items()
        ):
            _validate_int(
                name=name,
                value=value,
                minimum=1,
            )

        _validate_int(
            name="finalization_llm_reserve",
            value=(
                self.finalization_llm_reserve
            ),
            minimum=0,
        )

        if (
            self.finalization_llm_reserve
            > self.max_llm_calls_per_run
        ):
            raise ValueError(
                "finalization_llm_reserve "
                "must not exceed "
                "max_llm_calls_per_run."
            )


@dataclass(frozen=True)
class BudgetUsage:
    """Dynamic usage snapshot for one research run.

    These values represent work already authorized.

    They are absolute snapshot values here. Later, LangGraph node updates
    will use additive delta semantics when writing the corresponding counters
    into ResearchState.

    ``unique_sources`` is supplied from the deduplicated source collection.
    ResearchState does not need a separate sources-used counter.
    """

    iteration_count: int = 0
    search_queries_used: int = 0
    source_fetches_used: int = 0
    llm_calls_used: int = 0
    unique_sources: int = 0

    def __post_init__(
        self,
    ) -> None:
        values = {
            "iteration_count": (
                self.iteration_count
            ),
            "search_queries_used": (
                self.search_queries_used
            ),
            "source_fetches_used": (
                self.source_fetches_used
            ),
            "llm_calls_used": (
                self.llm_calls_used
            ),
            "unique_sources": (
                self.unique_sources
            ),
        }

        for name, value in (
            values.items()
        ):
            _validate_int(
                name=name,
                value=value,
                minimum=0,
            )


@dataclass(frozen=True)
class BudgetAuthorization:
    """Result of one deterministic budget-allocation decision.

    ``requested`` is the amount of work the caller wanted to perform.

    ``authorized`` is the amount that may actually proceed.

    The caller must preserve its original deterministic ordering and execute
    only the first ``authorized`` items.

    For example, if five planned searches are requested but only two are
    authorized, execute the first two planned searches. The budget layer does
    not rerank work.

    ``llm_purpose`` is required for ``llm_calls`` authorizations.

    It records whether the authorized LLM work belongs to:

    - optional research
    - finalization

    Non-LLM authorizations must not carry an LLM purpose.
    """

    resource: BudgetResource
    requested: int
    authorized: int
    reason: str | None = None
    llm_purpose: LLMCallPurpose | None = None

    def __post_init__(
        self,
    ) -> None:
        allowed_resources = {
            "research_iterations",
            "search_queries",
            "sources",
            "source_fetches",
            "llm_calls",
        }

        if (
            self.resource
            not in allowed_resources
        ):
            raise ValueError(
                "resource must be a known "
                "budget resource."
            )

        _validate_int(
            name="requested",
            value=self.requested,
            minimum=0,
        )

        _validate_int(
            name="authorized",
            value=self.authorized,
            minimum=0,
        )

        if (
            self.authorized
            > self.requested
        ):
            raise ValueError(
                "authorized must not exceed "
                "requested."
            )

        if self.reason is not None:
            if not isinstance(
                self.reason,
                str,
            ):
                raise TypeError(
                    "reason must be a string "
                    "or None."
                )

            if not self.reason.strip():
                raise ValueError(
                    "reason must not be blank."
                )

        allowed_llm_purposes = {
            "optional_research",
            "finalization",
        }

        if self.resource == "llm_calls":
            if self.llm_purpose is None:
                raise ValueError(
                    "llm_purpose is required "
                    "for llm_calls."
                )

            if not isinstance(
                self.llm_purpose,
                str,
            ):
                raise TypeError(
                    "llm_purpose must be a string."
                )

            if (
                self.llm_purpose
                not in allowed_llm_purposes
            ):
                raise ValueError(
                    "llm_purpose must be "
                    "'optional_research' or "
                    "'finalization'."
                )

        elif self.llm_purpose is not None:
            raise ValueError(
                "llm_purpose must be None "
                "for non-LLM budget resources."
            )

    @property
    def skipped(
        self,
    ) -> int:
        """Number of requested work items that were not authorized."""

        return (
            self.requested
            - self.authorized
        )

    @property
    def fully_authorized(
        self,
    ) -> bool:
        """Whether every requested work item was authorized."""

        return (
            self.authorized
            == self.requested
        )

    @property
    def exhausted(
        self,
    ) -> bool:
        """Whether none of the requested work could be authorized."""

        return (
            self.requested > 0
            and self.authorized == 0
        )


class BudgetPolicy:
    """Pure whole-run budget allocator.

    This object is immutable in practice: it stores only static limits.

    It does not own usage counters. The caller must provide a BudgetUsage
    snapshot for every decision.

    This is important for LangGraph because graph state remains the source of
    truth rather than hidden mutable state inside a node or service object.
    """

    def __init__(
        self,
        *,
        limits: BudgetLimits,
    ) -> None:
        if not isinstance(
            limits,
            BudgetLimits,
        ):
            raise TypeError(
                "limits must be a "
                "BudgetLimits object."
            )

        self._limits = limits

    @property
    def limits(
        self,
    ) -> BudgetLimits:
        """Return the static budget limits."""

        return self._limits

    @staticmethod
    def _validate_usage(
        usage: BudgetUsage,
    ) -> None:
        if not isinstance(
            usage,
            BudgetUsage,
        ):
            raise TypeError(
                "usage must be a "
                "BudgetUsage object."
            )

    @staticmethod
    def _validate_requested(
        requested: int,
    ) -> None:
        _validate_int(
            name="requested",
            value=requested,
            minimum=0,
        )

    @staticmethod
    def _authorization(
        *,
        resource: BudgetResource,
        requested: int,
        capacity: int,
        reason: str,
        llm_purpose: LLMCallPurpose | None = None,
    ) -> BudgetAuthorization:
        """Create one partial-or-complete authorization."""

        authorized = min(
            requested,
            max(
                0,
                capacity,
            ),
        )

        return BudgetAuthorization(
            resource=resource,
            requested=requested,
            authorized=authorized,
            reason=(
                None
                if authorized == requested
                else reason
            ),
            llm_purpose=llm_purpose,
        )

    def authorize_iteration(
        self,
        *,
        usage: BudgetUsage,
    ) -> BudgetAuthorization:
        """Authorize the start of one research iteration.

        ``max_research_iterations=2`` therefore means two total research
        iterations, including the initial iteration.
        """

        self._validate_usage(
            usage
        )

        remaining = remaining_budget(
            limit=(
                self._limits.max_research_iterations
            ),
            used=(
                usage.iteration_count
            ),
        )

        return self._authorization(
            resource=(
                "research_iterations"
            ),
            requested=1,
            capacity=remaining,
            reason=(
                "research iteration "
                "budget exhausted"
            ),
        )

    def authorize_search_queries(
        self,
        *,
        usage: BudgetUsage,
        requested: int,
    ) -> BudgetAuthorization:
        """Authorize a deterministic prefix of planned search queries.

        The amount authorized now is:

            min(
                requested,
                max_search_queries_per_iteration,
                remaining whole-run search budget,
            )
        """

        self._validate_usage(
            usage
        )

        self._validate_requested(
            requested
        )

        run_remaining = (
            remaining_budget(
                limit=(
                    self._limits
                    .max_search_queries_per_run
                ),
                used=(
                    usage.search_queries_used
                ),
            )
        )

        capacity = min(
            run_remaining,
            (
                self._limits
                .max_search_queries_per_iteration
            ),
        )

        if requested <= capacity:
            reason = (
                "search query budget limited"
            )

        elif run_remaining <= 0:
            reason = (
                "whole-run search query "
                "budget exhausted"
            )

        elif (
            self._limits
            .max_search_queries_per_iteration
            < requested
            and
            self._limits
            .max_search_queries_per_iteration
            <= run_remaining
        ):
            reason = (
                "per-iteration search query "
                "limit reached"
            )

        else:
            reason = (
                "whole-run search query "
                "budget reached"
            )

        return self._authorization(
            resource="search_queries",
            requested=requested,
            capacity=capacity,
            reason=reason,
        )

    def authorize_new_sources(
        self,
        *,
        usage: BudgetUsage,
        requested: int,
    ) -> BudgetAuthorization:
        """Authorize newly discovered unique Sources.

        ``requested`` must represent only candidate Sources that are new after
        deduplication against existing state.

        The caller preserves discovery order and keeps only the deterministic
        authorized prefix.
        """

        self._validate_usage(
            usage
        )

        self._validate_requested(
            requested
        )

        remaining = remaining_budget(
            limit=(
                self._limits
                .max_sources_per_run
            ),
            used=(
                usage.unique_sources
            ),
        )

        return self._authorization(
            resource="sources",
            requested=requested,
            capacity=remaining,
            reason=(
                "whole-run source "
                "budget reached"
            ),
        )

    def authorize_source_fetches(
        self,
        *,
        usage: BudgetUsage,
        requested: int,
    ) -> BudgetAuthorization:
        """Authorize source-fetch attempts.

        A fetch consumes budget when it is authorized for execution, not only
        when it succeeds.
        """

        self._validate_usage(
            usage
        )

        self._validate_requested(
            requested
        )

        remaining = remaining_budget(
            limit=(
                self._limits
                .max_source_fetches_per_run
            ),
            used=(
                usage.source_fetches_used
            ),
        )

        return self._authorization(
            resource="source_fetches",
            requested=requested,
            capacity=remaining,
            reason=(
                "whole-run source fetch "
                "budget reached"
            ),
        )

    def authorize_llm_calls(
        self,
        *,
        usage: BudgetUsage,
        requested: int,
        purpose: LLMCallPurpose,
    ) -> BudgetAuthorization:
        """Authorize LLM calls while protecting finalization capacity.

        Optional research calls may use only:

            remaining_llm_calls - finalization_llm_reserve

        Finalization calls may use the full remaining LLM capacity.

        The current A1.2 architecture reserves two finalization calls:

        1. synthesis
        2. semantic verification

        Orchestration must stop optional research before finalization begins.

        The returned authorization preserves ``purpose`` so downstream
        consumers can prove whether the capacity was allocated for optional
        research or finalization.
        """

        self._validate_usage(
            usage
        )

        self._validate_requested(
            requested
        )

        if purpose not in {
            "optional_research",
            "finalization",
        }:
            raise ValueError(
                "purpose must be "
                "'optional_research' "
                "or 'finalization'."
            )

        remaining = remaining_budget(
            limit=(
                self._limits
                .max_llm_calls_per_run
            ),
            used=(
                usage.llm_calls_used
            ),
        )

        if purpose == "finalization":
            return self._authorization(
                resource="llm_calls",
                requested=requested,
                capacity=remaining,
                reason=(
                    "whole-run LLM "
                    "budget exhausted"
                ),
                llm_purpose=purpose,
            )

        optional_capacity = max(
            0,
            (
                remaining
                - self._limits
                .finalization_llm_reserve
            ),
        )

        if (
            optional_capacity <= 0
            and requested > 0
        ):
            reason = (
                "LLM finalization "
                "reserve protected"
            )

        else:
            reason = (
                "optional LLM budget reached"
            )

        return self._authorization(
            resource="llm_calls",
            requested=requested,
            capacity=optional_capacity,
            reason=reason,
            llm_purpose=purpose,
        )