from dataclasses import replace
import json
import sqlite3
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

import research_agent.ui_service as service
import research_agent.history as history_module
from research_agent.graph.builder import build_research_graph
from research_agent.graph.nodes.research_routing import research_decision, route_after_critique
from research_agent.graph.nodes.synthesis import Synthesizer
from research_agent.history import HistoryStore, HistoryError
from research_agent.incomplete import ResearchInterrupted
from research_agent.llm.client import LLMRateLimitError
from research_agent.run_summary import summarize_run
from test_production_graph import _isolated_context, RejectedSynthesisLLM
import test_research_routing as routing
from test_ui import APP, completed
from test_incomplete import partial


@pytest.mark.parametrize("changes,reason", [
    ({"iteration_count": 3}, "round_limit"),
    ({"search_queries_used": 20}, "search_limit"),
    ({"source_count": 20}, "source_limit"),
    ({"source_fetches_used": 20}, "fetch_limit"),
    ({"llm_calls_used": 18}, "ai_limit"),
    ({"critique": routing._sufficient_critique()}, "sufficient"),
    ({"critique": routing._insufficient_critique(follow_ups=[])}, "no_followups"),
    ({"critique": routing._insufficient_critique(follow_ups=["What is already known?"])}, "no_new_followups"),
    ({"critique": routing._insufficient_critique(follow_ups=[]), "llm_calls_used": 18}, "ai_limit"),
])
def test_stop_reasons_match_routing_without_consuming_budget(changes, reason):
    state = routing._state(**changes)
    before = dict(state)
    context = routing._context()
    runtime = routing._runtime(context)
    assert research_decision(state, runtime) == reason
    assert route_after_critique(state, runtime) == "finalize"
    assert state == before


def run_offline(monkeypatch, context):
    monkeypatch.setattr(service, "build_research_application", lambda settings:
                        SimpleNamespace(graph=build_research_graph(), context=context))
    return service.run_question("Offline summary", lambda _: None)


def test_real_graph_summary_counts_finalization_and_preserves_earlier_stop_reason(monkeypatch):
    context = _isolated_context()
    result = run_offline(monkeypatch, context)
    summary = result.summary
    assert summary.end_reason == "sufficient" and summary.outcome == "verified"
    assert summary.ai_calls.used == 5
    assert summary.rounds.used == summary.searches.used == summary.page_reads.used == summary.sources.used == 1
    assert summary.ai_calls.limit == context.budget_policy.limits.max_llm_calls_per_run
    assert "Budget used counts committed reservations" in result.text_export()
    assert json.loads(result.json_export())["summary"]["ai_calls"]["used"] == 5


@pytest.mark.parametrize("rejection", [False, True])
def test_rejected_synthesis_has_distinct_outcome_without_rejected_text(monkeypatch, rejection):
    context = replace(_isolated_context(), synthesizer=Synthesizer(llm=RejectedSynthesisLLM(reject_grounding=rejection)))
    result = run_offline(monkeypatch, context)
    assert result.summary.end_reason == "sufficient"
    assert result.summary.outcome == "verification_rejected"
    assert result.summary.ai_calls.used == 5
    assert result.report.citations == []
    assert "REJECTED DRAFT" not in result.summary.model_dump_json()


def test_failed_finalization_preserves_reserved_pair_and_provider_reason(monkeypatch):
    context = _isolated_context()
    def fail(*args):
        raise LLMRateLimitError("PRIVATE error details")
    monkeypatch.setattr(context.synthesizer, "synthesize", fail)
    with pytest.raises(ResearchInterrupted) as caught:
        run_offline(monkeypatch, context)
    summary = caught.value.partial.summary
    assert summary.end_reason == "quota" and summary.outcome == "incomplete"
    assert summary.ai_calls.used == 5
    assert summary.page_reads.used == 1
    assert "PRIVATE" not in caught.value.partial.json_export()


@pytest.mark.parametrize("outcome", ["no_evidence", "finalization_budget"])
def test_deterministic_fallback_outcomes(monkeypatch, outcome):
    context = _isolated_context()
    if outcome == "no_evidence":
        # No optional AI capacity yields no extracted evidence, while preserving
        # the finalization reserve. The planner's one call is still budgeted.
        limits = replace(context.budget_policy.limits, max_llm_calls_per_run=3)
    else:
        limits = replace(context.budget_policy.limits, max_llm_calls_per_run=3, finalization_llm_reserve=0)
    context = replace(context, budget_policy=type(context.budget_policy)(limits=limits))
    result = run_offline(monkeypatch, context)
    assert result.summary.outcome == outcome
    assert not result.report.citations


@pytest.mark.parametrize("incomplete", [False, True])
def test_history_summary_roundtrip_and_legacy_defaults(tmp_path, incomplete):
    base = completed()
    state = dict(iteration_count=base.iterations, search_queries_used=base.searches,
                 sources=base.sources, source_fetches_used=1, llm_calls_used=5,
                 research_stop_reason="sufficient", finalization_outcome="verified")
    summary = summarize_run(state, _isolated_context().budget_policy.limits,
                            error="quota" if incomplete else None)
    result = partial().model_copy(update={"summary": summary}) if incomplete else replace(base, summary=summary)
    store = HistoryStore(tmp_path / "history.sqlite")
    key = store.save(result)
    assert HistoryStore(store.path).load(key).summary == summary
    table = "incomplete_runs" if incomplete else "reports"
    with sqlite3.connect(store.path) as connection:
        payload = json.loads(connection.execute(f"SELECT payload FROM {table} WHERE id=?", (key,)).fetchone()[0])
        del payload["summary"]
        connection.execute(f"UPDATE {table} SET payload=? WHERE id=?", (json.dumps(payload), key))
    assert store.load(key).summary is None


def test_corrupt_summary_is_rejected_without_hiding_other_reports(tmp_path):
    store = HistoryStore(tmp_path / "history.sqlite")
    good = store.save(completed())
    summary = summarize_run({}, _isolated_context().budget_policy.limits)
    bad = store.save(replace(completed(), summary=summary))
    with sqlite3.connect(store.path) as connection:
        payload = json.loads(connection.execute("SELECT payload FROM reports WHERE id=?", (bad,)).fetchone()[0])
        payload["summary"]["ai_calls"]["used"] = 999
        connection.execute("UPDATE reports SET payload=? WHERE id=?", (json.dumps(payload), bad))
    with pytest.raises(HistoryError):
        store.load(bad)
    assert store.load(good).summary is None


@pytest.mark.parametrize("incomplete", [False, True])
def test_reopened_summary_uses_saved_limits_not_current_configuration(tmp_path, monkeypatch, incomplete):
    store = HistoryStore(tmp_path / "history.sqlite")
    summary = summarize_run(dict(llm_calls_used=5, research_stop_reason="sufficient", finalization_outcome="verified"),
                            _isolated_context().budget_policy.limits, error="quota" if incomplete else None)
    result = partial().model_copy(update={"summary": summary}) if incomplete else replace(completed(), summary=summary)
    store.save(result)
    monkeypatch.setattr(history_module, "get_history_store", lambda: store)
    monkeypatch.setattr(service, "run_question", lambda *args, **kwargs: pytest.fail("No provider call"))
    app = AppTest.from_file(APP, default_timeout=10).run()
    app.button(key="open_history").click().run()
    assert not app.exception
    assert len(app.table) == 1
    rows = app.table[0].value
    assert rows.loc[rows["Resource"] == "AI calls", "Budget used"].iloc[0] == 5
    assert rows.loc[rows["Resource"] == "AI calls", "Limit"].iloc[0] == 10
    assert any(item.value == summary.stop_message for item in app.text)
