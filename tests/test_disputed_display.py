"""Tests for showing disputed claims to people (phase C2).

A claim that other evidence contradicts is kept, and must be clearly marked
wherever a report is shown or exported: the local app, the public app, the
text download, the evidence download and the command line. Titles and
excerpts come from web pages, so they must stay plain text.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import research_agent.public_service as public
import research_agent.ui_service as service
from research_agent.main import _print_report
from research_agent.models.schemas import Citation, Evidence, ResearchReport, Source
from research_agent.ui_service import DISPUTE_NOTE
from test_public_service import fill, public_env  # noqa: F401 - pytest fixture
from test_ui import APP, assert_only_plain_text, completed

PUBLIC_APP = Path(__file__).resolve().parents[1] / "public_app.py"
BLOG = "Cubic implements a binary search increase."
WIKI = "CUBIC replaced the binary search increase of BIC with a cubic function."
HOSTILE_HOST = "attacker.example/disputed"


def disputed(*, wiki_title: str = "TCP congestion control - Wikipedia"):
    return service.CompletedResearch(
        report=ResearchReport(
            question="How does CUBIC differ from Reno?",
            content=f"{BLOG} Reno grows linearly.",
            citations=[
                Citation(id="c1", claim_text=BLOG, evidence_ids=["e1"], disputed_by=["e2"]),
                Citation(id="c2", claim_text="Reno grows linearly.", evidence_ids=["e3"]),
            ],
        ),
        evidence=[
            Evidence(id="e1", source_id="s1", excerpt=BLOG),
            Evidence(id="e2", source_id="s2", excerpt=WIKI),
            Evidence(id="e3", source_id="s1", excerpt="Reno grows linearly."),
        ],
        sources=[
            Source(id="s1", url="https://blog.example/tcp", domain="blog.example", title="A blog"),
            Source(
                id="s2",
                url="https://en.wikipedia.org/wiki/TCP_congestion_control",
                domain="en.wikipedia.org",
                title=wiki_title,
            ),
        ],
        iterations=1,
        searches=2,
        warning_count=0,
    )


# ---------------------------------------------------------------------------
# Result helpers and downloads
# ---------------------------------------------------------------------------


def test_disputing_evidence_comes_with_its_source():
    result = disputed()
    first, second = result.report.citations

    [(item, source)] = result.disputing_evidence(first)

    assert (item.id, source.id) == ("e2", "s2")
    assert result.disputing_evidence(second) == []
    assert result.disputed_count == 1


def test_a_dispute_pointing_to_missing_evidence_is_refused():
    result = disputed()
    broken = result.report.citations[0].model_copy(update={"disputed_by": ["missing"]})

    with pytest.raises(KeyError):
        result.disputing_evidence(broken)


def test_the_text_download_marks_the_disputed_claim_and_shows_the_other_side():
    text = disputed().text_export()
    claim_one, claim_two = text.split("[2]")

    assert f"DISPUTED: {DISPUTE_NOTE}" in claim_one
    assert "https://en.wikipedia.org/wiki/TCP_congestion_control" in claim_one
    assert WIKI in claim_one
    assert "DISPUTED" not in claim_two


def test_the_evidence_download_keeps_the_dispute():
    data = json.loads(disputed().json_export())

    assert data["report"]["citations"][0]["disputed_by"] == ["e2"]
    assert data["report"]["citations"][1]["disputed_by"] == []


def test_an_undisputed_report_shows_no_dispute_anywhere():
    result = completed()

    assert result.disputed_count == 0
    assert "DISPUTED" not in result.text_export()


def test_the_command_line_marks_disputed_claims(capsys):
    _print_report(disputed().report)
    out = capsys.readouterr().out

    assert "DISPUTED by evidence: e2" in out
    assert out.count("DISPUTED") == 1


# ---------------------------------------------------------------------------
# Local app
# ---------------------------------------------------------------------------


def _run_local(monkeypatch, result):
    monkeypatch.setattr(service, "run_question", lambda *args, **kwargs: result)
    app = AppTest.from_file(APP).run()
    app.text_area[0].set_value("Question")
    app.button[0].click().run()
    assert not app.exception

    return app


def test_the_local_app_warns_about_disputed_claims(monkeypatch):
    app = _run_local(monkeypatch, disputed())
    warnings = [item.value for item in app.warning]

    assert DISPUTE_NOTE in warnings
    assert any("1 claim(s) below are contradicted" in value for value in warnings)
    assert any(item.label == "Contradicting evidence for claim 1" for item in app.expander)
    assert any(item.value == WIKI for item in app.text)


def test_the_local_app_shows_no_dispute_for_an_undisputed_report(monkeypatch):
    app = _run_local(monkeypatch, completed())

    assert DISPUTE_NOTE not in [item.value for item in app.warning]
    assert not any("Contradicting evidence" in item.label for item in app.expander)


def test_disputing_titles_are_plain_text(monkeypatch):
    hostile = disputed(wiki_title=f"Wiki [Verify your account](https://{HOSTILE_HOST})")
    app = _run_local(monkeypatch, hostile)

    assert_only_plain_text(app, HOSTILE_HOST)


# ---------------------------------------------------------------------------
# Public app
# ---------------------------------------------------------------------------


def test_the_public_app_warns_about_disputed_claims(public_env, monkeypatch):  # noqa: F811
    monkeypatch.setattr(public, "run_visitor_question", lambda question, progress, **kwargs: disputed())
    app = fill(AppTest.from_file(PUBLIC_APP).run())

    assert not app.exception
    assert DISPUTE_NOTE in [item.value for item in app.warning]
    assert any(item.value == WIKI for item in app.text)
