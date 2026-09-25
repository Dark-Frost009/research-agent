from dataclasses import replace
from datetime import date, datetime, timezone
import sqlite3

import pytest
from streamlit.testing.v1 import AppTest

import research_agent.history as history_module
import research_agent.ui_service as service
from research_agent.depth import ResearchDepth
from research_agent.history import HistoryStore
from test_ui import APP, completed
from test_incomplete import partial


def depth(name):
    return ResearchDepth(name=name, rounds=1, searches=2, sources=4, fetches=4,
                         ai_calls=12, sub_questions=2, searches_per_round=2,
                         results_per_search=3, finalization_reserve=2)


@pytest.fixture
def history(tmp_path):
    store = HistoryStore(tmp_path / 'history.sqlite')
    first = replace(completed('First'), depth=depth('Quick'))
    first.report.created_at = datetime(2026, 9, 1, tzinfo=timezone.utc)
    second = partial('Second').model_copy(update={
        'depth': depth('Thorough'), 'created_at': datetime(2026, 9, 2, 23, 59, tzinfo=timezone.utc)})
    third = completed('Third')
    third.report.created_at = datetime(2026, 9, 3, tzinfo=timezone.utc)
    ids = [store.save(item) for item in (first, second, third)]
    return store, ids


@pytest.mark.parametrize('filters,indices', [
    ({'status': 'completed'}, [2, 0]),
    ({'status': 'incomplete'}, [1]),
    ({'depth': 'Quick'}, [0]),
    ({'depth': 'Thorough'}, [1]),
    ({'depth': 'Standard'}, []),
    ({'depth': 'Not recorded'}, [2]),
    ({'start_date': date(2026, 9, 2)}, [2, 1]),
    ({'end_date': date(2026, 9, 2)}, [1, 0]),
    ({'start_date': date(2026, 9, 2), 'end_date': date(2026, 9, 2)}, [1]),
    ({'status': 'incomplete', 'depth': 'Quick'}, []),
    ({'status': 'completed', 'depth': 'Quick', 'end_date': date(2026, 9, 1)}, [0]),
])
def test_filters_combine_with_search_and_keep_order(history, filters, indices):
    store, ids = history
    before = store.path.read_bytes()
    assert [item.id for item in store.list_reports(**filters)] == [ids[i] for i in indices]
    results = store.list_reports('source title', **filters)
    assert [item.id for item in results] == [ids[i] for i in indices]
    assert all(item.match.location == 'Source title' for item in results)
    assert store.path.read_bytes() == before


def test_date_filter_normalizes_timezone(history):
    store, ids = history
    with sqlite3.connect(store.path) as connection:
        connection.execute('UPDATE reports SET created_at=? WHERE id=?',
                           ('2026-09-02T01:00:00+05:30', ids[0]))
    assert [item.id for item in store.list_reports(end_date=date(2026, 9, 1))] == [ids[0]]


def test_invalid_filters_and_corrupt_metadata(history):
    store, ids = history
    for filters in ({'status': 'wrong'}, {'depth': 'wrong'},
                    {'start_date': date(2026, 9, 3), 'end_date': date(2026, 9, 1)}):
        with pytest.raises(ValueError):
            store.list_reports(**filters)
    with sqlite3.connect(store.path) as connection:
        connection.execute('UPDATE reports SET payload=?, created_at=? WHERE id=?', ('broken', 'invalid', ids[0]))
    assert ids[0] in [item.id for item in store.list_reports(status='completed')]
    assert ids[0] not in [item.id for item in store.list_reports(depth='Not recorded')]
    assert ids[0] not in [item.id for item in store.list_reports(start_date=date(2026, 9, 1))]


def test_ui_filters_clear_dates_demo_and_unsaved_protection(history, monkeypatch):
    store, ids = history
    monkeypatch.setattr(history_module, 'get_history_store', lambda: store)
    monkeypatch.setattr(service, 'run_question', lambda *a, **k: pytest.fail('No provider calls'))
    app = AppTest.from_file(APP, default_timeout=10)
    unsaved = completed('Unsaved work')
    app.session_state['completed_research'] = unsaved
    app.session_state['saved_report_id'] = None
    app.run()
    app.text_input(key='history_search').set_value('source title').run()
    app.radio(key='history_status').set_value('Incomplete').run()
    app.radio(key='history_depth').set_value('Thorough').run()
    app.date_input(key='history_start').set_value(date(2026, 9, 2)).run()
    app.date_input(key='history_end').set_value(date(2026, 9, 2)).run()
    assert not app.exception
    assert app.selectbox(key='history_selection').value == ids[1]
    assert len(app.selectbox(key='history_selection').options) == 1
    app.button(key='open_history').click().run()
    assert app.session_state['completed_research'] == unsaved
    assert any('Your current research is unsaved' in item.value for item in app.warning)
    app.toggle(key='offline_demo').set_value(True).run()
    app.toggle(key='offline_demo').set_value(False).run()
    assert app.radio(key='history_depth').value == 'Thorough'
    assert app.date_input(key='history_start').value == date(2026, 9, 2)
    app.date_input(key='history_start').set_value(date(2026, 9, 3)).run()
    assert any('From date must be on or before' in item.value for item in app.warning)
    assert not app.selectbox
    app.button(key='clear_history_filters').click().run()
    assert not app.exception
    assert app.radio(key='history_status').value == 'All'
    assert app.radio(key='history_depth').value == 'All'
    assert app.date_input(key='history_start').value is None
    assert app.date_input(key='history_end').value is None
    assert app.text_input(key='history_search').value == 'source title'
    assert len(app.selectbox(key='history_selection').options) == 3
    assert app.session_state['completed_research'] == unsaved
