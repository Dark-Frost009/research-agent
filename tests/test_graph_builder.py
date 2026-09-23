"""Tests for the production Research Agent graph topology.

These tests verify graph construction and routing structure only.

They do not execute LLM, search, fetch, or other provider side effects.
They also do not claim checkpoint/crash durability.
"""

from __future__ import annotations

from langgraph.graph import (
    END,
    START,
)

from research_agent.graph.builder import (
    build_research_graph,
)


def _graph():
    """Return the inspectable graph representation."""

    return build_research_graph().get_graph()


def _edge_tuples():
    """Return stable edge metadata for topology assertions."""

    return {
        (
            edge.source,
            edge.target,
            edge.data,
            edge.conditional,
        )
        for edge in _graph().edges
    }


def test_build_research_graph_compiles() -> None:
    graph = build_research_graph()

    assert type(graph).__name__ == (
        "CompiledStateGraph"
    )


def test_production_graph_contains_expected_nodes() -> None:
    graph = _graph()

    expected_nodes = {
        START,
        END,
        "reserve_initial_iteration",
        "reserve_follow_up_iteration",
        "reserve_planner",
        "execute_planner",
        "adapt_critique_follow_ups",
        "reserve_search",
        "execute_search",
        "admit_sources",
        "prepare_evidence_collection_plan",
        "reserve_source_fetch",
        "execute_source_fetch",
        "reserve_evidence_extraction",
        "execute_evidence_extraction",
        "reserve_critique",
        "execute_critique",
        "reserve_finalization",
        "execute_finalization",
        "assemble_final_report",
    }

    assert set(
        graph.nodes.keys()
    ) == expected_nodes


def test_initial_iteration_runs_planner() -> None:
    edges = _edge_tuples()

    assert (
        START,
        "reserve_initial_iteration",
        None,
        False,
    ) in edges

    assert (
        "reserve_initial_iteration",
        "reserve_planner",
        None,
        False,
    ) in edges

    assert (
        "reserve_planner",
        "execute_planner",
        None,
        False,
    ) in edges

    assert (
        "execute_planner",
        "reserve_search",
        None,
        False,
    ) in edges


def test_follow_up_iteration_bypasses_planner() -> None:
    edges = _edge_tuples()

    assert (
        "reserve_follow_up_iteration",
        "adapt_critique_follow_ups",
        None,
        False,
    ) in edges

    assert (
        "adapt_critique_follow_ups",
        "reserve_search",
        None,
        False,
    ) in edges

    assert (
        "reserve_follow_up_iteration",
        "reserve_planner",
        None,
        False,
    ) not in edges

    assert (
        "adapt_critique_follow_ups",
        "reserve_planner",
        None,
        False,
    ) not in edges


def test_shared_research_pipeline_order_is_correct() -> None:
    edges = _edge_tuples()

    expected_edges = {
        (
            "reserve_search",
            "execute_search",
            None,
            False,
        ),
        (
            "execute_search",
            "admit_sources",
            None,
            False,
        ),
        (
            "admit_sources",
            "prepare_evidence_collection_plan",
            None,
            False,
        ),
        (
            "prepare_evidence_collection_plan",
            "reserve_source_fetch",
            None,
            False,
        ),
        (
            "reserve_source_fetch",
            "execute_source_fetch",
            None,
            False,
        ),
        (
            "execute_source_fetch",
            "reserve_evidence_extraction",
            None,
            False,
        ),
        (
            "reserve_evidence_extraction",
            "execute_evidence_extraction",
            None,
            False,
        ),
        (
            "execute_evidence_extraction",
            "reserve_critique",
            None,
            False,
        ),
        (
            "reserve_critique",
            "execute_critique",
            None,
            False,
        ),
    }

    assert expected_edges.issubset(
        edges
    )


def test_critique_has_exactly_two_conditional_routes() -> None:
    critique_edges = {
        (
            edge.target,
            edge.data,
            edge.conditional,
        )
        for edge in _graph().edges
        if edge.source == "execute_critique"
    }

    assert critique_edges == {
        (
            "reserve_follow_up_iteration",
            "continue_research",
            True,
        ),
        (
            "reserve_finalization",
            "finalize",
            True,
        ),
    }


def test_continue_research_route_enters_follow_up_iteration() -> None:
    edges = _edge_tuples()

    assert (
        "execute_critique",
        "reserve_follow_up_iteration",
        "continue_research",
        True,
    ) in edges


def test_finalize_route_enters_finalization() -> None:
    edges = _edge_tuples()

    assert (
        "execute_critique",
        "reserve_finalization",
        "finalize",
        True,
    ) in edges


def test_finalization_occurs_after_critique_not_before() -> None:
    edges = _edge_tuples()

    assert (
        "execute_evidence_extraction",
        "reserve_critique",
        None,
        False,
    ) in edges

    assert (
        "execute_evidence_extraction",
        "reserve_finalization",
        None,
        False,
    ) not in edges

    assert (
        "execute_evidence_extraction",
        "execute_finalization",
        None,
        False,
    ) not in edges


def test_finalization_pipeline_ends_with_report_assembly() -> None:
    edges = _edge_tuples()

    assert (
        "reserve_finalization",
        "execute_finalization",
        None,
        False,
    ) in edges

    assert (
        "execute_finalization",
        "assemble_final_report",
        None,
        False,
    ) in edges

    assert (
        "assemble_final_report",
        END,
        None,
        False,
    ) in edges


def test_no_direct_path_from_critique_to_end() -> None:
    edges = _edge_tuples()

    assert (
        "execute_critique",
        END,
        None,
        False,
    ) not in edges

    assert not any(
        edge.source == "execute_critique"
        and edge.target == END
        for edge in _graph().edges
    )


def test_graph_has_only_expected_conditional_edges() -> None:
    conditional_edges = {
        (
            edge.source,
            edge.target,
            edge.data,
        )
        for edge in _graph().edges
        if edge.conditional
    }

    assert conditional_edges == {
        (
            "execute_critique",
            "reserve_follow_up_iteration",
            "continue_research",
        ),
        (
            "execute_critique",
            "reserve_finalization",
            "finalize",
        ),
    }


def test_graph_has_single_start_entry() -> None:
    start_edges = [
        edge
        for edge in _graph().edges
        if edge.source == START
    ]

    assert len(start_edges) == 1

    assert (
        start_edges[0].target
        == "reserve_initial_iteration"
    )

    assert start_edges[0].conditional is False


def test_graph_has_single_end_entry() -> None:
    end_edges = [
        edge
        for edge in _graph().edges
        if edge.target == END
    ]

    assert len(end_edges) == 1

    assert (
        end_edges[0].source
        == "assemble_final_report"
    )

    assert end_edges[0].conditional is False