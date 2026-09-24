from pathlib import Path
from types import SimpleNamespace
import json

import pytest
from streamlit.testing.v1 import AppTest

import research_agent.ui_service as service
from research_agent.models.schemas import Citation, Evidence, ResearchReport, Source
from research_agent.llm.client import LLMProviderError
from test_production_graph import _isolated_context
from research_agent.graph.builder import build_research_graph

APP = Path(__file__).resolve().parents[1] / "app.py"


def completed(question="Test question"):
    return service.CompletedResearch(
        report=ResearchReport(question=question, content="Supported answer.", citations=[
            Citation(id="c1", claim_text="Supported answer.", evidence_ids=["e1"])]),
        evidence=[Evidence(id="e1", source_id="s1", excerpt="Supporting source excerpt.")],
        sources=[Source(id="s1", url="https://example.com/original", domain="example.com",
                        final_url="https://example.com/final", title="Source title")],
        iterations=1, searches=2, warning_count=0,
    )


def test_adapter_runs_real_graph_with_fresh_contexts(monkeypatch):
    contexts = []
    def build(settings):
        context = _isolated_context()
        contexts.append(context)
        return SimpleNamespace(graph=build_research_graph(), context=context)
    monkeypatch.setattr(service, "build_research_application", build)
    progress = []
    first = service.run_question("First question", progress.append)
    second = service.run_question("Second question", progress.append)
    assert first.report.question == "First question"
    assert second.report.question == "Second question"
    assert contexts[0].workspace is not contexts[1].workspace
    assert "Searching the web" in progress
    assert "Writing and verifying the report" in progress
    assert first.report.citations
    assert first.citation_sources(first.report.citations[0])


def test_exports_preserve_citation_to_source_trace():
    result = completed()
    assert "https://example.com/final" in result.text_export()
    assert "[1] Supported answer." in result.text_export()
    data = json.loads(result.json_export())
    assert data["report"]["citations"][0]["evidence_ids"] == [data["evidence"][0]["id"]]
    assert data["evidence"][0]["source_id"] == data["sources"][0]["id"]


@pytest.mark.parametrize("url", ["javascript:alert(1)", "file:///secret", "https://user:pass@example.com", "https://[bad", "https://example.com/\nsecret"])
def test_source_links_reject_unsafe_syntax(url):
    source = completed().sources[0].model_copy(update={"final_url": url})
    assert service.safe_source_url(source) is None


def test_ui_requires_question_without_provider_calls(monkeypatch):
    monkeypatch.setattr(service, "run_question", lambda *a, **kwargs: pytest.fail("Unexpected provider call"))
    app = AppTest.from_file(APP).run()
    app.button[0].click().run()
    assert not app.exception
    assert "Enter a research question" in app.warning[0].value


def test_ui_success_rerun_and_session_isolation(monkeypatch):
    calls = []
    def run(question, progress, *, depth=None):
        calls.append(question)
        progress("Searching the web")
        return completed(question)
    monkeypatch.setattr(service, "run_question", run)
    app = AppTest.from_file(APP).run()
    app.text_area[0].set_value("My question")
    app.button[0].click().run()
    assert not app.exception
    assert calls == ["My question"]
    assert any(t.value == "Supported answer." for t in app.text)
    assert len(app.get("download_button")) == 2
    assert all(link.proto.url == "https://example.com/final" for link in app.get("link_button"))
    app.run()
    assert calls == ["My question"]
    assert app.text_area[0].value == "My question"
    other = AppTest.from_file(APP).run()
    assert not other.get("download_button")


def test_ui_failed_run_preserves_previous_report_without_leaking_error(monkeypatch):
    monkeypatch.setattr(service, "run_question", lambda q, p, **kwargs: completed(q))
    app = AppTest.from_file(APP).run()
    app.text_area[0].set_value("Successful question")
    app.button[0].click().run()
    def fail(*args, **kwargs):
        raise LLMProviderError("PRIVATE_API_KEY_DO_NOT_DISPLAY")
    monkeypatch.setattr(service, "run_question", fail)
    app.text_area[0].set_value("Failed question")
    app.button[0].click().run()
    assert not app.exception
    assert "AI service could not complete" in app.error[0].value
    assert "PRIVATE_API_KEY" not in str(app)
    assert app.session_state["completed_research"].report.question == "Successful question"


def test_ui_fallback_does_not_claim_verified_answer(monkeypatch):
    result = completed()
    result.report.citations.clear()
    result.report.content = "Insufficient evidence."
    monkeypatch.setattr(service, "run_question", lambda *args, **kwargs: result)
    app = AppTest.from_file(APP).run()
    app.text_area[0].set_value("Question")
    app.button[0].click().run()
    assert not app.exception
    assert "did not produce an answer with verified citations" in app.warning[0].value
    assert all(link.label == "Visit page" for link in app.get("link_button"))


def test_missing_search_configuration_has_actionable_safe_message():
    from research_agent.tools.web_search import SearchConfigurationError
    message = service.friendly_error(SearchConfigurationError("PRIVATE_CONFIG"))
    assert "configuration" in message
    assert "PRIVATE_CONFIG" not in message


def test_adapter_rejects_broken_citation_links(monkeypatch):
    result = completed()
    state = dict(final_report=result.report, evidence=[], sources=result.sources,
                 iteration_count=1, search_queries_used=1, errors=[])
    class BrokenGraph:
        def stream(self, *args, **kwargs):
            yield "values", state
    monkeypatch.setattr(service, "build_research_application", lambda settings:
                        SimpleNamespace(graph=BrokenGraph(), context=_isolated_context()))
    with pytest.raises(KeyError):
        service.run_question("Question", lambda message: None)


def test_extraction_diagnostics_are_safe_and_exported(monkeypatch):
    errors = [
        "Evidence extraction failed for source s and sub-question q: LLMUnavailableError: PRIVATE",
        "Evidence extraction failed for source s and sub-question q: EvidenceGroundingError: PRIVATE",
    ]
    issues = service.summarize_issues(errors)
    assert len(issues) == 2
    assert "temporarily unavailable" in issues[0]
    assert "PRIVATE" not in repr(issues)
    from dataclasses import replace
    result = replace(completed(), issues=issues, warning_count=2)
    assert issues[0] in result.text_export()
    assert json.loads(result.json_export())["issues"] == list(issues)
    monkeypatch.setattr(service, "run_question", lambda *args, **kwargs: result)
    app = AppTest.from_file(APP).run()
    app.text_area[0].set_value("Question")
    app.button[0].click().run()
    assert not app.exception
    assert any("temporarily unavailable" in warning.value for warning in app.warning)


def test_provider_outage_is_not_presented_as_insufficient_evidence(monkeypatch):
    from research_agent.llm.client import LLMUnavailableError
    def fail(*args, **kwargs):
        raise LLMUnavailableError("PRIVATE")
    monkeypatch.setattr(service, "run_question", fail)
    app = AppTest.from_file(APP).run()
    app.text_area[0].set_value("Question")
    app.button[0].click().run()
    assert not app.exception
    assert "temporarily unavailable" in app.error[0].value
    assert "PRIVATE" not in app.error[0].value


@pytest.fixture(autouse=True)
def isolated_ui_history(tmp_path, monkeypatch):
    import research_agent.history as history_module
    store = history_module.HistoryStore(tmp_path / "ui-history.sqlite")
    monkeypatch.setattr(history_module, "get_history_store", lambda: store)
