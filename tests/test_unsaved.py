import pytest
from streamlit.testing.v1 import AppTest

import research_agent.history as history_module
import research_agent.ui_service as service
from research_agent.history import HistoryError, HistoryStore
from research_agent.incomplete import ResearchInterrupted
from test_incomplete import partial
from test_ui import APP, completed


@pytest.fixture
def store(tmp_path, monkeypatch):
    result = HistoryStore(tmp_path / "history.sqlite")
    monkeypatch.setattr(history_module, "get_history_store", lambda: result)
    return result


@pytest.mark.parametrize("incomplete", [False, True])
def test_unsaved_blocks_navigation_and_provider_calls_then_save_unlocks(store, monkeypatch, incomplete):
    existing = store.save(completed("Saved question"))
    original_save = store.save
    def failed_save(*args):
        raise HistoryError("Disk unavailable")
    monkeypatch.setattr(store, "save", failed_save)
    calls = []
    def run(question, progress, *, depth=None):
        calls.append(question)
        if incomplete:
            raise ResearchInterrupted(partial(question))
        return completed(question)
    monkeypatch.setattr(service, "run_question", run)
    app = AppTest.from_file(APP).run()
    app.text_area[0].set_value("Unsaved question")
    app.button[0].click().run()
    assert not app.exception
    value_key = "incomplete_research" if incomplete else "completed_research"
    original = app.session_state[value_key]
    app.selectbox(key="history_selection").set_value(existing)
    original_load = store.load
    monkeypatch.setattr(store, "load", lambda *args, **kwargs: pytest.fail("Blocked navigation must not load"))
    app.button(key="open_history").click().run()
    assert not app.exception and app.session_state[value_key] == original
    assert any("current research is unsaved" in item.value for item in app.warning)
    app.text_area[0].set_value("Another question")
    app.button[0].click().run()
    assert not app.exception and app.session_state[value_key] == original
    assert calls == ["Unsaved question"]
    assert len(app.get("download_button")) == (1 if incomplete else 2)
    monkeypatch.setattr(store, "save", original_save)
    monkeypatch.setattr(store, "load", original_load)
    app.button(key="retry_save_incomplete" if incomplete else "retry_save_history").click().run()
    assert not app.exception and len(store.list_reports()) == 2
    app.run()
    assert calls == ["Unsaved question"]  # No deferred submission after saving.
    app.selectbox(key="history_selection").set_value(existing)
    app.button(key="open_history").click().run()
    assert not app.exception
    assert app.session_state["completed_research"].report.question == "Saved question"


@pytest.mark.parametrize("incomplete", [False, True])
def test_explicit_discard_only_removes_unsaved_session_copy(store, monkeypatch, incomplete):
    key = store.save(completed("Kept in history"))
    monkeypatch.setattr(service, "run_question", lambda *args, **kwargs: pytest.fail("No provider call expected"))
    app = AppTest.from_file(APP).run()
    value_key = "incomplete_research" if incomplete else "completed_research"
    app.session_state[value_key] = partial() if incomplete else completed("Legacy unsaved")
    app.run()
    assert not app.exception and app.get("download_button")
    app.button(key="confirm_discard_unsaved").click().run()
    assert not app.exception
    assert value_key not in app.session_state
    assert len(store.list_reports()) == 1
    app.selectbox(key="history_selection").set_value(key)
    app.button(key="open_history").click().run()
    assert not app.exception
    assert app.session_state["completed_research"].report.question == "Kept in history"


def test_hidden_unsaved_report_remains_recoverable(store, monkeypatch):
    saved_partial = partial()
    key = store.save(saved_partial)
    monkeypatch.setattr(service, "run_question", lambda *args, **kwargs: pytest.fail("No provider call expected"))
    app = AppTest.from_file(APP).run()
    app.session_state["completed_research"] = completed("Hidden unsaved")
    app.session_state["incomplete_research"] = saved_partial
    app.session_state["saved_incomplete_id"] = key
    app.run()
    assert not app.exception and len(app.get("download_button")) == 3
    app.button(key="save_hidden_report").click().run()
    assert not app.exception and len(store.list_reports()) == 2
    assert store.load(app.session_state["saved_report_id"]).report.question == "Hidden unsaved"
    assert app.session_state["incomplete_research"] == saved_partial
