"""Run-scoped dependencies and transient workspace for LangGraph.

ResearchState contains only durable/checkpointable data that belongs to one
research run.

This module contains the complementary runtime context:

1. long-lived service objects and static BudgetPolicy configuration
2. a per-run transient workspace used to move prepared work between graph
   nodes without persisting raw provider payloads into ResearchState

The transient workspace is intentionally mutable.

That is necessary because LangGraph runtime context is shared across nodes in
one graph invocation, while some intermediate objects must stay outside
checkpointed state.

Examples of transient-only data include:

- prepared budget-authorized calls/batches
- fetched raw webpage text
- EvidenceCollectionPreparation objects
- provider results waiting for deterministic conversion

Budget counters never live here. ResearchState remains the source of truth for
whole-run usage accounting.
"""

from __future__ import annotations

from _thread import LockType
from dataclasses import dataclass, field
from threading import Lock

from research_agent.graph.budget import (
    BudgetAuthorization,
    BudgetPolicy,
)
from research_agent.graph.nodes.critic import (
    Critic,
    CritiqueCall,
)
from research_agent.graph.nodes.evidence import (
    EvidenceBatch,
)
from research_agent.graph.nodes.evidence_collector import (
    EvidenceCollectionPlan,
    EvidenceCollectionPreparation,
    EvidenceCollector,
)
from research_agent.graph.nodes.planner import (
    Planner,
    PlannerCall,
)
from research_agent.graph.nodes.search import (
    SearchBatch,
    SearchNode,
)
from research_agent.graph.nodes.source_fetcher import (
    SourceFetchBatch,
    SourceFetchResult,
    SourceFetcher,
)
from research_agent.graph.nodes.sources import (
    SourceBatch,
    SourceNode,
)
from research_agent.graph.nodes.synthesis import (
    FinalizationCall,
    Synthesizer,
)
from research_agent.models.schemas import (
    SearchResult,
    SubQuestion,
)


@dataclass
class TransientWorkspace:
    """Per-run non-durable orchestration workspace.

    Nothing stored here should be interpreted as committed ResearchState.

    In particular, objects that may transitively contain full webpage text
    belong here rather than in ResearchState.

    Each graph invocation must receive its own workspace instance. Sharing one
    workspace across concurrent research runs would mix transient data between
    runs and is therefore invalid usage.
    """

    _run_lock: LockType = field(
        default_factory=Lock, init=False, repr=False, compare=False
    )
    _run_started: bool = field(default=False, init=False, repr=False, compare=False)

    def claim_run(self) -> None:
        """Atomically bind this workspace to one production invocation.

        The claim is permanent, including after failure or clear_all(). A new
        request must build a fresh context; clearing data cannot make an old
        authorization safe to reuse. The lock protects simultaneous entrants.
        """
        with self._run_lock:
            if self._run_started:
                raise RuntimeError(
                    "Research workspace has already been used. "
                    "Build a fresh research context for each invocation."
                )
            self._run_started = True

    # ------------------------------------------------------------------
    # Iteration authorization
    # ------------------------------------------------------------------

    iteration_authorization: BudgetAuthorization | None = None

    # ------------------------------------------------------------------
    # Planner
    # ------------------------------------------------------------------

    planner_call: PlannerCall | None = None
    planned_sub_questions: list[SubQuestion] | None = None

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    search_batch: SearchBatch | None = None
    search_results_for_iteration: list[SearchResult] | None = None

    # ------------------------------------------------------------------
    # Source admission
    # ------------------------------------------------------------------

    source_batch: SourceBatch | None = None

    # ------------------------------------------------------------------
    # Fetch / evidence collection
    # ------------------------------------------------------------------

    evidence_collection_plan: EvidenceCollectionPlan | None = None

    # Active executable fetch permit.
    #
    # This is cleared before SourceFetcher performs any external network
    # operation so the same authorization cannot be reused accidentally.
    source_fetch_batch: SourceFetchBatch | None = None

    # Immutable record of the fetch batch that completed execution.
    #
    # This is not an active/reusable permit. It exists only so downstream
    # evidence preparation can verify that transient SourceFetchResults
    # correspond exactly to the authorized deterministic fetch prefix.
    completed_source_fetch_batch: SourceFetchBatch | None = None

    source_fetch_results: list[
        SourceFetchResult
    ] | None = None

    evidence_preparation: EvidenceCollectionPreparation | None = None

    evidence_batch: EvidenceBatch | None = None

    # ------------------------------------------------------------------
    # Critique
    # ------------------------------------------------------------------

    # Active executable critique permit.
    #
    # Critique is optional-research LLM work. The call is prepared first so
    # its additive llm_calls_used delta can be persisted into ResearchState
    # before Critic performs any provider side effect.
    critique_call: CritiqueCall | None = None

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    finalization_call: FinalizationCall | None = None

    def clear_iteration_work(
        self,
    ) -> None:
        """Discard transient work from one completed research iteration.

        Durable domain data and budget counters are unaffected because they
        live in ResearchState rather than this workspace.

        Critique preparation is iteration-local and is therefore cleared here.

        Finalization is deliberately left untouched because it belongs to the
        post-research phase rather than iteration-local work.
        """

        self.iteration_authorization = None
        self.planned_sub_questions = None
        self.planner_call = None
        self.search_batch = None
        self.search_results_for_iteration = None
        self.source_batch = None

        self.evidence_collection_plan = None
        self.source_fetch_batch = None
        self.completed_source_fetch_batch = None
        self.source_fetch_results = None
        self.evidence_preparation = None
        self.evidence_batch = None

        self.critique_call = None

    def clear_finalization_work(
        self,
    ) -> None:
        """Discard transient finalization preparation."""

        self.finalization_call = None

    def clear_all(
        self,
    ) -> None:
        """Discard every transient orchestration object."""

        self.clear_iteration_work()
        self.clear_finalization_work()


@dataclass(frozen=True)
class ResearchGraphContext:
    """Run-scoped dependencies supplied through LangGraph Runtime.

    These dependencies are intentionally excluded from ResearchState.

    ``workspace`` is mutable even though this context object is frozen. The
    dependency wiring itself therefore cannot be reassigned accidentally,
    while transient per-run values can still move between graph nodes.
    """

    budget_policy: BudgetPolicy

    planner: Planner
    search_node: SearchNode
    source_node: SourceNode
    source_fetcher: SourceFetcher

    evidence_collector: EvidenceCollector

    critic: Critic
    synthesizer: Synthesizer

    workspace: TransientWorkspace = field(
        default_factory=TransientWorkspace
    )

    def __post_init__(
        self,
    ) -> None:
        """Fail closed on obviously invalid runtime dependency wiring."""

        if not isinstance(
            self.budget_policy,
            BudgetPolicy,
        ):
            raise TypeError(
                "budget_policy must be a BudgetPolicy object."
            )

        if not isinstance(
            self.planner,
            Planner,
        ):
            raise TypeError(
                "planner must be a Planner object."
            )

        if not isinstance(
            self.search_node,
            SearchNode,
        ):
            raise TypeError(
                "search_node must be a SearchNode object."
            )

        if not isinstance(
            self.source_node,
            SourceNode,
        ):
            raise TypeError(
                "source_node must be a SourceNode object."
            )

        if not isinstance(
            self.source_fetcher,
            SourceFetcher,
        ):
            raise TypeError(
                "source_fetcher must be a SourceFetcher object."
            )

        if not isinstance(
            self.evidence_collector,
            EvidenceCollector,
        ):
            raise TypeError(
                "evidence_collector must be an "
                "EvidenceCollector object."
            )

        if not isinstance(
            self.critic,
            Critic,
        ):
            raise TypeError(
                "critic must be a Critic object."
            )

        if not isinstance(
            self.synthesizer,
            Synthesizer,
        ):
            raise TypeError(
                "synthesizer must be a Synthesizer object."
            )

        if not isinstance(
            self.workspace,
            TransientWorkspace,
        ):
            raise TypeError(
                "workspace must be a TransientWorkspace object."
            )