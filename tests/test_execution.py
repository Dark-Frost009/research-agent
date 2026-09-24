"""Exercise both entry points through the real graph without external services."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import research_agent.main as cli
import research_agent.ui_service as ui
from research_agent.graph.builder import build_research_graph
from research_agent.graph.execution import research_run_config
from research_agent.models.schemas import CritiqueResult
from test_production_graph import _isolated_context, _policy


@pytest.mark.parametrize("rounds", [2, 8])
@pytest.mark.parametrize("interface", ["cli", "streamlit"])
def test_entry_points_finish_all_rounds_with_explicit_shared_limit(monkeypatch, rounds, interface):
    context = replace(_isolated_context(), budget_policy=_policy(
        max_research_iterations=rounds, max_llm_calls_per_run=100))
    context.critic.outcomes = [
        CritiqueResult(sufficient=False, gaps=[f"Gap {index}"],
                       follow_up_questions=[f"What resolves evidence gap {index}?"])
        for index in range(rounds - 1)
    ] + [CritiqueResult(sufficient=True, gaps=[], follow_up_questions=[])]
    # Real graph, reducers, and node orchestration; deterministic providers.
    graph = Mock(wraps=build_research_graph())
    application = SimpleNamespace(graph=graph, context=context)
    if interface == "cli":
        monkeypatch.setattr(cli, "build_research_application", lambda: application)
        report = cli.run_research("Offline multi-round research")
        config = graph.invoke.call_args.kwargs["config"]
    else:
        monkeypatch.setattr(ui, "build_research_application", lambda settings: application)
        result = ui.run_question("Offline multi-round research", lambda _: None)
        assert result.iterations == rounds
        report = result.report
        config = graph.stream.call_args.kwargs["config"]
    assert context.critic.provider_llm_calls == rounds
    assert report.citations
    assert config == research_run_config(context.budget_policy.limits)
    if rounds == 8:
        assert config["recursion_limit"] > 100
