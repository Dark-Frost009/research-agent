import json
import sqlite3
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from streamlit.testing.v1 import AppTest

import research_agent.history as history_module
import research_agent.ui_service as service
from research_agent.graph.builder import build_research_graph
from research_agent.history import HistoryError, HistoryStore
from research_agent.incomplete import IncompleteResearch, ResearchInterrupted
from research_agent.llm.client import LLMRateLimitError, LLMUnavailableError
from test_production_graph import _isolated_context
from test_ui import APP, completed


def partial(question="Interrupted question"):
    result = completed()
    return IncompleteResearch(
        question=question, reason="quota", last_stage="Writing and verifying the report",
        sources=result.sources, evidence=result.evidence, iterations=1, searches=2,
        warning_count=0,
    )


@pytest.mark.parametrize("early", [False, True])
def test_real_graph_failure_keeps_only_committed_research(monkeypatch, early):
    context = _isolated_context()
    def fail(*args, **kwargs):
        raise LLMRateLimitError("PRIVATE provider response with REJECTED DRAFT")
    if early:
        monkeypatch.setattr(context.planner, "plan", fail)
    else:
        monkeypatch.setattr(context.synthesizer, "synthesize", fail)
    monkeypatch.setattr(service, "build_research_application", lambda settings:
                        SimpleNamespace(graph=build_research_graph(), context=context))
    with pytest.raises(ResearchInterrupted) as caught:
        service.run_question("My research", lambda _: None)
    saved = caught.value.partial
    assert saved.reason == "quota"
    assert saved.question == "My research"
    assert bool(saved.evidence) is (not early)
    assert bool(saved.sources) is (not early)
    assert saved.searches == (0 if early else 1)
    assert saved.last_stage == ("Planning your research" if early else "Writing and verifying the report")
    exported = saved.json_export()
    assert "PRIVATE" not in exported and "REJECTED DRAFT" not in exported
    assert "report" not in json.loads(exported)
    assert "citations" not in json.loads(exported)


def test_snapshot_excludes_draft_and_raw_errors(monkeypatch):
    result = completed()
    class FailingGraph:
        def stream(self, *args, **kwargs):
            yield "values", dict(sources=result.sources, evidence=result.evidence,
                errors=["PRIVATE"], draft_content="REJECTED DRAFT", citations=result.report.citations)
            raise LLMUnavailableError("PRIVATE")
    monkeypatch.setattr(service, "build_research_application", lambda settings:
                        SimpleNamespace(graph=FailingGraph(), context=_isolated_context()))
    with pytest.raises(ResearchInterrupted) as caught:
        service.run_question("Question", lambda _: None)
    saved = caught.value.partial
    assert saved.reason == "unavailable" and saved.warning_count == 1
    assert "PRIVATE" not in saved.json_export() and "REJECTED DRAFT" not in saved.json_export()
    result.evidence[0].excerpt = "Changed later"
    assert saved.evidence[0].excerpt != "Changed later"


def test_old_history_database_and_mixed_runs_round_trip(tmp_path):
    path = tmp_path / "history.sqlite"
    result = completed()
    payload = history_module._Snapshot.from_result(result).model_dump_json()
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE reports (id TEXT PRIMARY KEY, question TEXT NOT NULL, created_at TEXT NOT NULL, citation_count INTEGER NOT NULL, payload TEXT NOT NULL)")
        connection.execute("INSERT INTO reports VALUES (?, ?, ?, ?, ?)",
                           ("old", result.report.question, result.report.created_at.isoformat(), 1, payload))
    store = HistoryStore(path)
    assert store.load("old") == result
    interrupted = partial()
    key = store.save(interrupted)
    reopened = HistoryStore(path).load(key)
    assert reopened == interrupted
    entries = store.list_reports()
    assert len(entries) == 2
    assert {item.status for item in entries} == {"completed", "incomplete"}
    assert store.list_reports("interrupted")[0].label.startswith("Incomplete")
    assert store.load("old") == result


def test_invalid_partial_records_fail_safely(tmp_path):
    store = HistoryStore(tmp_path / "history.sqlite")
    good = store.save(completed())
    key = store.save(partial())
    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE incomplete_runs SET payload=? WHERE id=?", ('{"version":999}', key))
    with pytest.raises(HistoryError):
        store.load(key)
    assert isinstance(store.load(good), service.CompletedResearch)
    with pytest.raises(HistoryError):
        store.save(partial().model_copy(update={"sources": []}))
    with pytest.raises(ValidationError):
        IncompleteResearch.model_validate({**partial().model_dump(), "report": "Unverified"})


@pytest.fixture
def local_history(tmp_path, monkeypatch):
    store = HistoryStore(tmp_path / "history.sqlite")
    monkeypatch.setattr(history_module, "get_history_store", lambda: store)
    return store


def test_ui_saves_reopens_and_switches_without_provider_calls(local_history, monkeypatch):
    old = local_history.save(completed("Previous answer"))
    calls = []
    def fail(question, progress):
        calls.append(question)
        raise ResearchInterrupted(partial(question))
    monkeypatch.setattr(service, "run_question", fail)
    app = AppTest.from_file(APP).run()
    app.text_area[0].set_value("Failed question")
    app.button[0].click().run()
    assert not app.exception
    assert any(title.value == "Incomplete research" for title in app.subheader)
    assert all(item.value != "Supported answer." for item in app.text)
    assert len(app.get("download_button")) == 1
    assert len(local_history.list_reports()) == 2
    app.run()
    assert len(local_history.list_reports()) == 2
    restarted = AppTest.from_file(APP).run()
    saved_id = next(item.id for item in local_history.list_reports() if item.status == "incomplete")
    restarted.selectbox(key="history_selection").set_value(saved_id)
    restarted.button(key="open_history").click().run()
    assert not restarted.exception
    assert restarted.session_state["incomplete_research"].question == "Failed question"
    restarted.selectbox(key="history_selection").set_value(old)
    restarted.button(key="open_history").click().run()
    assert not restarted.exception
    assert any(item.value == "Supported answer." for item in restarted.text)
    assert len(restarted.get("download_button")) == 2
    assert calls == ["Failed question"]


def test_ui_save_failure_retry_and_new_success(local_history, monkeypatch):
    def fail(*args):
        raise ResearchInterrupted(partial())
    monkeypatch.setattr(service, "run_question", fail)
    original = local_history.save
    def save_failure(*args):
        raise HistoryError("Disk error")
    monkeypatch.setattr(local_history, "save", save_failure)
    app = AppTest.from_file(APP).run()
    app.text_area[0].set_value("Failed question")
    app.button[0].click().run()
    assert not app.exception and len(app.get("download_button")) == 1
    assert any("has not been saved" in item.value for item in app.warning)
    monkeypatch.setattr(local_history, "save", original)
    app.button(key="retry_save_incomplete").click().run()
    app.run()
    assert not app.exception and len(local_history.list_reports()) == 1
    monkeypatch.setattr(service, "run_question", lambda q, p: completed(q))
    app.text_area[0].set_value("New successful question")
    app.button[0].click().run()
    assert not app.exception
    assert len(app.get("download_button")) == 2
    assert not any(item.value == "Incomplete research" for item in app.subheader)
    assert len(local_history.list_reports()) == 2
