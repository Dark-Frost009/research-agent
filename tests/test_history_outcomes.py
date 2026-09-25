from dataclasses import replace
from datetime import date
import sqlite3

import pytest
from streamlit.testing.v1 import AppTest

import research_agent.history as history_module
import research_agent.ui_service as service
from research_agent.history import HistoryStore, OUTCOME_FILTERS
from research_agent.run_summary import RunSummary
from test_ui import APP, completed
from test_incomplete import partial
from test_history_filters import depth


def summary(outcome):
    counter = dict(used=0, limit=12)
    return RunSummary(rounds=counter, searches=counter, sources=counter,
                      page_reads=counter, ai_calls=counter,
                      end_reason='unknown', outcome=outcome)


@pytest.fixture
def history(tmp_path):
    store = HistoryStore(tmp_path / 'history.sqlite')
    ids = {}
    for outcome in OUTCOME_FILTERS.values():
        item = (partial().model_copy(update={'summary': summary(outcome)}) if outcome == 'incomplete'
                else replace(completed(), summary=summary(outcome), depth=depth('Quick')))
        ids[outcome] = store.save(item)
    ids['legacy_completed'] = store.save(completed('Older completed report'))
    ids['legacy_incomplete'] = store.save(partial('Older interrupted run'))
    return store, ids


@pytest.mark.parametrize('outcome', list(OUTCOME_FILTERS.values()))
def test_saved_outcomes_filter_without_inferring_from_content(history, outcome):
    store, ids = history
    expected = {ids[outcome]}
    if outcome == 'unknown':
        expected.add(ids['legacy_completed'])
    if outcome == 'incomplete':
        expected.add(ids['legacy_incomplete'])
    before = store.path.read_bytes()
    rows = store.list_reports(outcome=outcome)
    assert {item.id for item in rows} == expected
    assert rows == [item for item in store.list_reports() if item.id in expected]
    assert store.path.read_bytes() == before


def test_outcome_combines_with_search_status_depth_and_dates(history):
    store, ids = history
    rows = store.list_reports('source title', outcome='verified', status='completed',
                              depth='Quick', start_date=date(2000, 1, 1), end_date=date(2100, 1, 1))
    assert [item.id for item in rows] == [ids['verified']]
    assert rows[0].match.location == 'Source title'
    assert store.list_reports(outcome='verified', status='incomplete') == []
    assert store.list_reports(outcome='verified', depth='Thorough') == []
    assert store.list_reports('absent term', outcome='verified') == []


def test_invalid_and_corrupt_outcomes(history):
    store, ids = history
    with pytest.raises(ValueError, match='outcome'):
        store.list_reports(outcome='made_up')
    with sqlite3.connect(store.path) as connection:
        connection.execute('UPDATE reports SET payload=? WHERE id=?', ('broken', ids['verified']))
    assert store.list_reports(outcome='verified') == []
    assert ids['verified'] not in {item.id for item in store.list_reports(outcome='unknown')}
    assert ids['verified'] in {item.id for item in store.list_reports()}


def test_ui_outcome_filter_preserves_work_and_survives_demo(history, monkeypatch):
    store, ids = history
    monkeypatch.setattr(history_module, 'get_history_store', lambda: store)
    monkeypatch.setattr(service, 'run_question', lambda *a, **k: pytest.fail('No provider calls'))
    app = AppTest.from_file(APP, default_timeout=10)
    unsaved = completed('Unsaved')
    app.session_state['completed_research'] = unsaved
    app.session_state['saved_report_id'] = None
    app.run()
    app.text_input(key='history_search').set_value('source title').run()
    app.radio(key='history_outcome').set_value('Verified answer').run()
    assert not app.exception
    assert app.selectbox(key='history_selection').value == ids['verified']
    assert len(app.selectbox(key='history_selection').options) == 1
    app.button(key='open_history').click().run()
    assert app.session_state['completed_research'] == unsaved
    assert any('Your current research is unsaved' in item.value for item in app.warning)
    app.toggle(key='offline_demo').set_value(True).run()
    app.toggle(key='offline_demo').set_value(False).run()
    assert app.radio(key='history_outcome').value == 'Verified answer'
    app.radio(key='history_status').set_value('Incomplete').run()
    assert any('No matching saved reports' in item.value for item in app.caption)
    app.button(key='clear_history_filters').click().run()
    assert not app.exception
    assert app.radio(key='history_outcome').value == 'All'
    assert app.text_input(key='history_search').value == 'source title'
    assert len(app.selectbox(key='history_selection').options) == len(ids)
    assert app.session_state['completed_research'] == unsaved
