import sqlite3

import pytest
from streamlit.testing.v1 import AppTest

import research_agent.history as history_module
import research_agent.ui_service as service
from research_agent.history import HistoryStore
from test_ui import APP, completed
from test_incomplete import partial


@pytest.mark.parametrize('query', ['test QUESTION', '  SUPPORTED answer  ', 'source TITLE', 'supporting source EXCERPT'])
def test_completed_content_is_searchable_without_changing_history(tmp_path, query):
    store = HistoryStore(tmp_path / 'history.sqlite')
    key = store.save(completed())
    before = store.path.read_bytes()
    assert [item.id for item in store.list_reports(query)] == [key]
    assert store.path.read_bytes() == before


@pytest.mark.parametrize('query', ['Interrupted question', 'source title', 'supporting source excerpt'])
def test_incomplete_content_is_searchable(tmp_path, query):
    store = HistoryStore(tmp_path / 'history.sqlite')
    key = store.save(partial())
    assert [item.id for item in store.list_reports(query)] == [key]
    assert store.list_reports('Supported answer') == []


def test_literal_unicode_search_and_noncontent_fields(tmp_path):
    store = HistoryStore(tmp_path / 'history.sqlite')
    result = completed()
    result.report.content = 'Straße has 100% coverage_of this phrase.'
    key = store.save(result)
    for query in ['STRASSE', '100%', 'coverage_', 'this phrase']:
        assert [item.id for item in store.list_reports(query)] == [key]
    for query in ['%', '_']:
        assert len(store.list_reports(query)) == 1
    for query in ['missing%', 'coverageX', "' OR 1=1 --", 'example.com', 'warning_count', key]:
        assert store.list_reports(query) == []


def test_order_empty_search_and_missing_database(tmp_path):
    store = HistoryStore(tmp_path / 'history.sqlite')
    assert store.list_reports('text') == []
    assert not store.path.exists()
    store.save(completed('First'))
    store.save(completed('Second'))
    store.save(partial('Third'))
    all_entries = store.list_reports()
    assert store.list_reports('   ') == all_entries
    assert store.list_reports('source title') == all_entries


def test_damaged_record_does_not_block_search_and_question_remains_visible(tmp_path):
    store = HistoryStore(tmp_path / 'history.sqlite')
    bad = store.save(completed('Damaged question'))
    good = store.save(completed('Good question'))
    with sqlite3.connect(store.path) as connection:
        connection.execute('UPDATE reports SET payload=? WHERE id=?', ('{broken', bad))
    assert [item.id for item in store.list_reports('supported answer')] == [good]
    assert [item.id for item in store.list_reports('damaged')] == [bad]
    assert len(store.list_reports()) == 2


def test_missing_source_title_and_legacy_metadata(tmp_path):
    import json
    store = HistoryStore(tmp_path / 'history.sqlite')
    result = completed()
    result.sources[0].title = None
    key = store.save(result)
    with sqlite3.connect(store.path) as connection:
        data = json.loads(connection.execute('SELECT payload FROM reports').fetchone()[0])
        data.pop('depth')
        data.pop('summary')
        connection.execute('UPDATE reports SET payload=?', (json.dumps(data),))
    assert [item.id for item in store.list_reports('supported answer')] == [key]
    assert store.list_reports('source title') == []


def test_ui_content_search_open_and_unsaved_protection(tmp_path, monkeypatch):
    store = HistoryStore(tmp_path / 'history.sqlite')
    saved = completed('A question without the search term')
    store.save(saved)
    monkeypatch.setattr(history_module, 'get_history_store', lambda: store)
    monkeypatch.setattr(service, 'run_question', lambda *a, **k: pytest.fail('Search must not call providers'))
    app = AppTest.from_file(APP, default_timeout=10).run()
    app.text_input(key='history_search').set_value('supporting source excerpt').run()
    assert not app.exception
    app.button(key='open_history').click().run()
    assert app.session_state['completed_research'] == saved
    unsaved = completed('Unsaved work')
    app.session_state['completed_research'] = unsaved
    app.session_state['saved_report_id'] = None
    app.text_input(key='history_search').set_value('source title').run()
    app.button(key='open_history').click().run()
    assert not app.exception
    assert app.session_state['completed_research'] == unsaved
    assert any('Your current research is unsaved' in item.value for item in app.warning)
    app.text_input(key='history_search').set_value('no matching content').run()
    assert any('No matching saved reports' in item.value for item in app.caption)
    assert app.session_state['completed_research'] == unsaved
