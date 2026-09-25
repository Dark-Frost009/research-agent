import sqlite3

import pytest
from streamlit.testing.v1 import AppTest

import research_agent.history as history_module
import research_agent.ui_service as service
from research_agent.history import HistoryStore
from test_ui import APP, completed
from test_incomplete import partial


@pytest.mark.parametrize('query,location,excerpt', [
    ('test question', 'Question', 'Test question'),
    ('supported answer', 'Report text', 'Supported answer.'),
    ('source title', 'Source title', 'Source title'),
    ('supporting source excerpt', 'Evidence excerpt', 'Supporting source excerpt.'),
])
def test_previews_identify_field_and_preserve_original_text(tmp_path, query, location, excerpt):
    store = HistoryStore(tmp_path / 'history.sqlite')
    store.save(completed())
    before = store.path.read_bytes()
    result = store.list_reports(query)[0]
    assert result.match.location == location
    assert result.match.excerpt == excerpt
    assert store.path.read_bytes() == before
    assert store.list_reports()[0].match is None


def test_long_unicode_excerpt_centers_on_match(tmp_path):
    store = HistoryStore(tmp_path / 'history.sqlite')
    result = completed()
    result.report.content = 'ß' * 400 + ' Straße needle\nword ' + 'tail ' * 100
    store.save(result)
    match = store.list_reports('STRASSE NEEDLE\nWORD')[0].match
    assert 'Straße needle word' in match.excerpt
    assert match.excerpt.startswith('…') and match.excerpt.endswith('…')
    assert len(match.excerpt) <= 182
    assert '\n' not in match.excerpt


def test_long_query_and_end_match_are_bounded(tmp_path):
    store = HistoryStore(tmp_path / 'history.sqlite')
    result = completed()
    result.report.content = 'start ' + 'x' * 400 + ' finish'
    store.save(result)
    assert len(store.list_reports('x' * 300)[0].match.excerpt) <= 182
    assert store.list_reports('finish')[0].match.excerpt.endswith('finish')


def test_incomplete_and_damaged_question_previews(tmp_path):
    store = HistoryStore(tmp_path / 'history.sqlite')
    store.save(partial())
    assert store.list_reports('excerpt')[0].match.location == 'Evidence excerpt'
    key = store.save(completed('Damaged question'))
    with sqlite3.connect(store.path) as connection:
        connection.execute('UPDATE reports SET payload=? WHERE id=?', ('broken', key))
    assert store.list_reports('damaged')[0].match.excerpt == 'Damaged question'


def test_multiple_matches_prefer_question_then_report(tmp_path):
    store = HistoryStore(tmp_path / 'history.sqlite')
    result = completed('Shared query')
    result.report.content = 'Shared answer'
    result.sources[0].title = 'Shared title'
    store.save(result)
    assert store.list_reports('shared')[0].match.location == 'Question'
    assert store.list_reports('answer')[0].match.location == 'Report text'


def test_ui_preview_before_open_is_plain_text_and_preserves_unsaved_work(tmp_path, monkeypatch):
    store = HistoryStore(tmp_path / 'history.sqlite')
    result = completed('Saved entry')
    result.report.content = '<script>alert(1)</script> **needle** [link](https://example.com)'
    store.save(result)
    monkeypatch.setattr(history_module, 'get_history_store', lambda: store)
    monkeypatch.setattr(service, 'run_question', lambda *a, **k: pytest.fail('No provider calls'))
    app = AppTest.from_file(APP, default_timeout=10)
    unsaved = completed('Unsaved work')
    app.session_state['completed_research'] = unsaved
    app.session_state['saved_report_id'] = None
    app.run()
    app.text_input(key='history_search').set_value('needle').run()
    assert not app.exception
    assert any(item.value == result.report.content for item in app.text)
    assert any(item.value == 'Match in report text' for item in app.caption)
    assert 'Report text:' in app.selectbox(key='history_selection').options[0]
    assert app.session_state['completed_research'] == unsaved
    app.text_input(key='history_search').set_value('source title').run()
    assert any(item.value == 'Match in source title' for item in app.caption)
    assert not any(item.value == result.report.content for item in app.text)
    app.text_input(key='history_search').set_value('').run()
    assert not any(item.value.startswith('Match in ') for item in app.caption)
    assert app.session_state['completed_research'] == unsaved
