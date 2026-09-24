"""Depth selection and persistence, with all provider execution replaced offline."""
from dataclasses import replace
from types import SimpleNamespace
import json
import sqlite3

import pytest
from streamlit.testing.v1 import AppTest

from research_agent.config import Settings
from research_agent.depth import apply_depth, DEPTH_NAMES, DepthConfigurationError
from research_agent.history import HistoryStore
from research_agent.incomplete import ResearchInterrupted
from research_agent.bootstrap import _build_budget_policy
from research_agent.graph.builder import build_research_graph
from research_agent.llm.client import LLMRateLimitError
from research_agent.models.schemas import CritiqueResult
import research_agent.ui_service as service
import research_agent.history as history_module
from test_ui import APP, completed
from test_incomplete import partial
from test_production_graph import _isolated_context


def settings(**overrides):
    values = dict(
        max_research_iterations=3, max_search_queries_per_run=8,
        max_sources_per_run=12, max_source_fetches_per_run=12,
        max_llm_calls_per_run=64, max_sub_questions=5,
        max_search_queries_per_iteration=5, max_search_results_per_query=5,
        finalization_llm_reserve=2, llm_provider="gemini", llm_model="offline-test",
        llm_api_key="PRIVATE_KEY", tavily_api_key="PRIVATE_SEARCH_KEY",
    )
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.mark.parametrize("name", DEPTH_NAMES)
def test_presets_respect_operator_caps_and_leave_original_settings_unchanged(name):
    original = settings(max_research_iterations=1, max_search_queries_per_run=1,
                        max_sources_per_run=2, max_source_fetches_per_run=1,
                        max_llm_calls_per_run=6, finalization_llm_reserve=3)
    before = original.model_dump()
    resolved, depth = apply_depth(original, name)
    assert original.model_dump() == before
    assert resolved is not original
    assert (depth.rounds, depth.searches, depth.sources, depth.fetches, depth.ai_calls) == (1, 1, 2, 1, 6)
    assert depth.finalization_reserve == 3
    assert resolved.llm_api_key == original.llm_api_key
    assert resolved.llm_model == original.llm_model
    assert resolved.request_timeout_seconds == original.request_timeout_seconds
    assert "PRIVATE" not in depth.model_dump_json()


@pytest.mark.parametrize("reserve,calls", [(0, 12), (1, 12), (2, 2), (12, 64)])
def test_unusable_depth_budget_is_rejected_before_provider_setup(monkeypatch, reserve, calls):
    original = settings(finalization_llm_reserve=reserve, max_llm_calls_per_run=calls)
    monkeypatch.setattr(service, "Settings", lambda **kwargs: original)
    monkeypatch.setattr(service, "build_research_application", lambda *args: pytest.fail("No provider setup"))
    with pytest.raises(DepthConfigurationError):
        service.run_question("Question", lambda _: None, depth="Quick")


@pytest.mark.parametrize("name,rounds,searches,sources,calls", [
    ("Quick", 1, 2, 4, 12), ("Standard", 2, 5, 8, 32), ("Thorough", 3, 8, 12, 64)])
def test_selected_limits_reach_real_graph_and_keep_verification(monkeypatch, name, rounds, searches, sources, calls):
    monkeypatch.setattr(service, "Settings", lambda **kwargs: settings())
    contexts = []
    def build(resolved):
        context = replace(_isolated_context(), budget_policy=_build_budget_policy(resolved))
        context.critic.outcomes = [CritiqueResult(
            sufficient=False, gaps=[f"Gap {index}"],
            follow_up_questions=[f"What resolves gap {index}?"])
            for index in range(rounds)]
        contexts.append(context)
        return SimpleNamespace(graph=build_research_graph(), context=context)
    monkeypatch.setattr(service, "build_research_application", build)
    result = service.run_question("Question", lambda _: None, depth=name)
    limits = contexts[0].budget_policy.limits
    assert (limits.max_research_iterations, limits.max_search_queries_per_run,
            limits.max_sources_per_run, limits.max_llm_calls_per_run) == (rounds, searches, sources, calls)
    assert result.iterations == rounds
    assert result.report.citations
    assert contexts[0].synthesizer.entered_calls[0].fully_authorized
    assert contexts[0].synthesizer.provider_llm_calls == 2
    assert result.depth.name == name
    assert json.loads(result.json_export())["depth"]["ai_calls"] == calls


def test_failure_keeps_selected_depth(monkeypatch):
    monkeypatch.setattr(service, "Settings", lambda **kwargs: settings())
    def build(resolved):
        context = replace(_isolated_context(), budget_policy=_build_budget_policy(resolved))
        def fail(*args, **kwargs):
            raise LLMRateLimitError("PRIVATE")
        monkeypatch.setattr(context.synthesizer, "synthesize", fail)
        return SimpleNamespace(graph=build_research_graph(), context=context)
    monkeypatch.setattr(service, "build_research_application", build)
    with pytest.raises(ResearchInterrupted) as caught:
        service.run_question("Question", lambda _: None, depth="Quick")
    assert caught.value.partial.depth.name == "Quick"
    assert caught.value.partial.depth.ai_calls == 12


@pytest.mark.parametrize("incomplete", [False, True])
def test_mode_survives_history_and_old_records_still_load(tmp_path, incomplete):
    depth = apply_depth(settings(), "Quick")[1]
    result = partial().model_copy(update={"depth": depth}) if incomplete else replace(completed(), depth=depth)
    store = HistoryStore(tmp_path / "history.sqlite")
    key = store.save(result)
    assert HistoryStore(store.path).load(key).depth == depth
    if not incomplete:
        assert "Quick" in store.load(key).text_export()
    table = "incomplete_runs" if incomplete else "reports"
    with sqlite3.connect(store.path) as connection:
        payload = json.loads(connection.execute(f"SELECT payload FROM {table} WHERE id=?", (key,)).fetchone()[0])
        del payload["depth"]
        connection.execute(f"UPDATE {table} SET payload=? WHERE id=?", (json.dumps(payload), key))
    assert store.load(key).depth is None


def test_ui_depth_selection_is_idle_and_saved_result_keeps_original_mode(tmp_path, monkeypatch):
    store = HistoryStore(tmp_path / "history.sqlite")
    monkeypatch.setattr(history_module, "get_history_store", lambda: store)
    monkeypatch.setattr(service, "preview_depth", lambda name: apply_depth(settings(), name)[1])
    calls = []
    def run(question, progress, *, depth=None):
        calls.append(depth)
        return replace(completed(question), depth=apply_depth(settings(), depth)[1])
    monkeypatch.setattr(service, "run_question", run)
    app = AppTest.from_file(APP).run()
    assert app.radio(key="research_depth").value == "Standard"
    app.radio(key="research_depth").set_value("Quick").run()
    assert calls == [] and not app.exception
    assert any("12 AI calls" in item.value for item in app.caption)
    app.text_area[0].set_value("Question")
    app.button[0].click().run()
    assert not app.exception and calls == ["Quick"]
    app.radio(key="research_depth").set_value("Thorough").run()
    assert not app.exception and calls == ["Quick"]
    assert any(item.value.startswith("This run: Quick") for item in app.caption)
    restarted = AppTest.from_file(APP).run()
    restarted.button(key="open_history").click().run()
    assert not restarted.exception and calls == ["Quick"]
    assert any(item.value.startswith("This run: Quick") for item in restarted.caption)


def test_ui_invalid_limits_disable_submission_but_history_remains_accessible(tmp_path, monkeypatch):
    store = HistoryStore(tmp_path / "history.sqlite")
    store.save(completed("Saved before configuration change"))
    monkeypatch.setattr(history_module, "get_history_store", lambda: store)
    def invalid(name):
        raise DepthConfigurationError("Check the research budget settings.")
    monkeypatch.setattr(service, "preview_depth", invalid)
    monkeypatch.setattr(service, "run_question", lambda *args, **kwargs: pytest.fail("No run expected"))
    app = AppTest.from_file(APP).run()
    assert not app.exception and app.button[0].disabled
    app.button(key="open_history").click().run()
    assert not app.exception and len(app.get("download_button")) == 2
