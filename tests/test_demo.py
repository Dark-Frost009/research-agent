import json

import pytest
from streamlit.testing.v1 import AppTest

import research_agent.history as history_module
import research_agent.ui_service as service
from research_agent.demo import demo_data, demo_text
from research_agent.history import HistoryStore
from test_ui import APP, completed
from test_incomplete import partial


def test_demo_content_is_labeled_and_all_citations_trace_to_bundled_sources():
    data = demo_data()
    assert data["demo"] is True and data["actual_provider_calls"] == 0
    assert data["example_summary"]["simulated"] is True
    sources = {item["id"]: item for item in data["sources"]}
    evidence = {item["id"]: item for item in data["evidence"]}
    for citation in data["citations"]:
        for evidence_id in citation["evidence_ids"]:
            item = evidence[evidence_id]
            assert item["excerpt"] in sources[item["source_id"]]["text"]
    assert demo_text(data).startswith("OFFLINE DEMO")
    assert "fictional" in json.dumps(data)
    assert "Actual provider calls: 0" in demo_text(data)
    data["sources"].clear()
    assert len(demo_data()["sources"]) == 2


def test_demo_bypasses_configuration_providers_and_history(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Demo must not access configuration, providers, or history")

    monkeypatch.setattr(service, "Settings", forbidden)
    monkeypatch.setattr(service, "preview_depth", forbidden)
    monkeypatch.setattr(service, "build_research_application", forbidden)
    monkeypatch.setattr(service, "run_question", forbidden)
    monkeypatch.setattr(history_module, "get_history_store", forbidden)

    app = AppTest.from_file(APP, default_timeout=10)
    app.session_state["offline_demo"] = True
    app.run()

    assert not app.exception
    assert len(app.tabs) == 3
    assert len(app.table) == 1
    assert len(app.get("download_button")) == 2
    assert all("demo" in item.label for item in app.get("download_button"))
    assert not app.get("link_button")
    assert not app.button
    assert any("OFFLINE DEMO" in item.value for item in app.warning)

    app.run()

    assert not app.exception
    assert "completed_research" not in app.session_state
    assert "incomplete_research" not in app.session_state


def test_public_demo_only_bypasses_live_research_and_shared_history(
    monkeypatch,
):
    def forbidden(*args, **kwargs):
        pytest.fail(
            "Public demo-only mode must not access live research or history"
        )

    monkeypatch.setenv("PUBLIC_DEMO_ONLY", "true")
    monkeypatch.setattr(service, "Settings", forbidden)
    monkeypatch.setattr(service, "preview_depth", forbidden)
    monkeypatch.setattr(service, "build_research_application", forbidden)
    monkeypatch.setattr(service, "run_question", forbidden)
    monkeypatch.setattr(history_module, "get_history_store", forbidden)

    app = AppTest.from_file(APP, default_timeout=10).run()

    assert not app.exception
    assert any("Public portfolio demo" in item.value for item in app.info)
    assert any("OFFLINE DEMO" in item.value for item in app.warning)
    assert len(app.tabs) == 3
    assert len(app.table) == 1
    assert len(app.get("download_button")) == 2
    assert not app.button
    assert not app.toggle
    assert "completed_research" not in app.session_state
    assert "incomplete_research" not in app.session_state


@pytest.mark.parametrize("incomplete", [False, True])
def test_demo_roundtrip_preserves_unsaved_work_and_inputs(
    tmp_path,
    monkeypatch,
    incomplete,
):
    store = HistoryStore(tmp_path / "history.sqlite")
    store.save(completed("Previously saved"))
    monkeypatch.setattr(history_module, "get_history_store", lambda: store)
    monkeypatch.setattr(
        service,
        "run_question",
        lambda *args, **kwargs: pytest.fail("No live research"),
    )

    app = AppTest.from_file(APP, default_timeout=10).run()
    original = (
        partial("Unsaved incomplete")
        if incomplete
        else completed("Unsaved completed")
    )
    key = "incomplete_research" if incomplete else "completed_research"
    saved_key = "saved_incomplete_id" if incomplete else "saved_report_id"

    app.session_state[key] = original
    app.session_state[saved_key] = None
    app.session_state["history_save_failed"] = True

    app.text_area[0].set_value("Question to keep").run()
    app.radio(key="research_depth").set_value("Quick").run()

    before = store.path.read_bytes()
    app.toggle(key="offline_demo").set_value(True).run()

    assert not app.exception
    assert app.session_state[key] == original
    assert store.path.read_bytes() == before
    assert not app.button

    app.run()
    app.toggle(key="offline_demo").set_value(False).run()

    assert not app.exception
    assert app.session_state[key] == original
    assert app.session_state[saved_key] is None
    assert app.text_area[0].value == "Question to keep"
    assert app.radio(key="research_depth").value == "Quick"
    assert len(store.list_reports()) == 1
    assert any(
        "Unsaved research is protected" in item.value
        for item in app.info
    )
    assert app.button(
        key=(
            "retry_save_incomplete"
            if incomplete
            else "retry_save_history"
        )
    )