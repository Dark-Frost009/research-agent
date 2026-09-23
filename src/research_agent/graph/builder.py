"""Production LangGraph builder for the Research Agent.

The graph executes the complete bounded research workflow:

    initial iteration
        -> planner
        -> search
        -> source admission
        -> evidence planning
        -> source fetch
        -> evidence extraction
        -> critique

The critique router then either:

    1. finalizes the accumulated evidence, or
    2. reserves another research iteration and converts critique follow-up
       questions into deterministic SubQuestion objects without another
       Planner LLM call.

Finalization is deliberately last:

    reserve_finalization
        -> execute_finalization
        -> assemble_final_report
        -> END

This preserves the protected atomic synthesis + semantic-verification pair.

The graph is compiled without a checkpointer here. Reservation/execution node
separation guarantees normal LangGraph reducer ordering during execution, but
does not by itself claim crash-durable accounting or resumability.
"""

from __future__ import annotations

from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from research_agent.graph.context import (
    ResearchGraphContext,
)
from research_agent.graph.nodes.critic_orchestration import (
    execute_critique,
    reserve_critique,
)
from research_agent.graph.nodes.evidence_orchestration import (
    execute_evidence_extraction,
    prepare_evidence_collection_plan,
    reserve_evidence_extraction,
)
from research_agent.graph.nodes.finalization_orchestration import (
    execute_finalization,
    reserve_finalization,
)
from research_agent.graph.nodes.follow_up_orchestration import (
    adapt_critique_follow_ups,
)
from research_agent.graph.nodes.iteration import (
    reserve_initial_iteration,
    reserve_iteration,
)
from research_agent.graph.nodes.planner_orchestration import (
    execute_planner,
    reserve_planner,
)
from research_agent.graph.nodes.report_assembly import (
    assemble_final_report,
)
from research_agent.graph.nodes.research_routing import (
    route_after_critique,
)
from research_agent.graph.nodes.search_orchestration import (
    execute_search,
    reserve_search,
)
from research_agent.graph.nodes.source_fetch_orchestration import (
    execute_source_fetch,
    reserve_source_fetch,
)
from research_agent.graph.nodes.source_orchestration import (
    admit_sources,
)
from research_agent.graph.state import (
    ResearchState,
)


def build_research_graph():
    """Build and compile the production Research Agent graph.

    The initial entry atomically claims a single-use workspace before
    reserving the first Planner-driven iteration. Follow-up iterations reuse
    that invocation's workspace without claiming it again. Reusing a context
    (or sharing its workspace) across production invocations fails before I/O.

    Follow-up iterations do not invoke Planner again. The Critic already
    produced concrete follow-up questions, which are converted into
    SubQuestion objects by ``adapt_critique_follow_ups``.
    """

    builder = StateGraph(
        ResearchState,
        context_schema=ResearchGraphContext,
    )

    # ------------------------------------------------------------------
    # Iteration entry points
    # ------------------------------------------------------------------

    builder.add_node(
        "reserve_initial_iteration",
        reserve_initial_iteration,
    )

    builder.add_node(
        "reserve_follow_up_iteration",
        reserve_iteration,
    )

    # ------------------------------------------------------------------
    # Initial planning
    # ------------------------------------------------------------------

    builder.add_node(
        "reserve_planner",
        reserve_planner,
    )

    builder.add_node(
        "execute_planner",
        execute_planner,
    )

    # ------------------------------------------------------------------
    # Critique-driven follow-up adaptation
    # ------------------------------------------------------------------

    builder.add_node(
        "adapt_critique_follow_ups",
        adapt_critique_follow_ups,
    )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    builder.add_node(
        "reserve_search",
        reserve_search,
    )

    builder.add_node(
        "execute_search",
        execute_search,
    )

    # ------------------------------------------------------------------
    # Source admission / fetch
    # ------------------------------------------------------------------

    builder.add_node(
        "admit_sources",
        admit_sources,
    )

    builder.add_node(
        "prepare_evidence_collection_plan",
        prepare_evidence_collection_plan,
    )

    builder.add_node(
        "reserve_source_fetch",
        reserve_source_fetch,
    )

    builder.add_node(
        "execute_source_fetch",
        execute_source_fetch,
    )

    # ------------------------------------------------------------------
    # Evidence extraction
    # ------------------------------------------------------------------

    builder.add_node(
        "reserve_evidence_extraction",
        reserve_evidence_extraction,
    )

    builder.add_node(
        "execute_evidence_extraction",
        execute_evidence_extraction,
    )

    # ------------------------------------------------------------------
    # Evidence sufficiency critique
    # ------------------------------------------------------------------

    builder.add_node(
        "reserve_critique",
        reserve_critique,
    )

    builder.add_node(
        "execute_critique",
        execute_critique,
    )

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    builder.add_node(
        "reserve_finalization",
        reserve_finalization,
    )

    builder.add_node(
        "execute_finalization",
        execute_finalization,
    )

    builder.add_node(
        "assemble_final_report",
        assemble_final_report,
    )

    # ------------------------------------------------------------------
    # Initial path
    # ------------------------------------------------------------------

    builder.add_edge(
        START,
        "reserve_initial_iteration",
    )

    builder.add_edge(
        "reserve_initial_iteration",
        "reserve_planner",
    )

    builder.add_edge(
        "reserve_planner",
        "execute_planner",
    )

    builder.add_edge(
        "execute_planner",
        "reserve_search",
    )

    # ------------------------------------------------------------------
    # Follow-up iteration entry
    # ------------------------------------------------------------------

    builder.add_edge(
        "reserve_follow_up_iteration",
        "adapt_critique_follow_ups",
    )

    builder.add_edge(
        "adapt_critique_follow_ups",
        "reserve_search",
    )

    # ------------------------------------------------------------------
    # Shared research pipeline
    # ------------------------------------------------------------------

    builder.add_edge(
        "reserve_search",
        "execute_search",
    )

    builder.add_edge(
        "execute_search",
        "admit_sources",
    )

    builder.add_edge(
        "admit_sources",
        "prepare_evidence_collection_plan",
    )

    builder.add_edge(
        "prepare_evidence_collection_plan",
        "reserve_source_fetch",
    )

    builder.add_edge(
        "reserve_source_fetch",
        "execute_source_fetch",
    )

    builder.add_edge(
        "execute_source_fetch",
        "reserve_evidence_extraction",
    )

    builder.add_edge(
        "reserve_evidence_extraction",
        "execute_evidence_extraction",
    )

    builder.add_edge(
        "execute_evidence_extraction",
        "reserve_critique",
    )

    builder.add_edge(
        "reserve_critique",
        "execute_critique",
    )

    # ------------------------------------------------------------------
    # Research-loop decision
    # ------------------------------------------------------------------

    builder.add_conditional_edges(
        "execute_critique",
        route_after_critique,
        {
            "continue_research": (
                "reserve_follow_up_iteration"
            ),
            "finalize": (
                "reserve_finalization"
            ),
        },
    )

    # ------------------------------------------------------------------
    # Finalization path
    # ------------------------------------------------------------------

    builder.add_edge(
        "reserve_finalization",
        "execute_finalization",
    )

    builder.add_edge(
        "execute_finalization",
        "assemble_final_report",
    )

    builder.add_edge(
        "assemble_final_report",
        END,
    )

    return builder.compile()