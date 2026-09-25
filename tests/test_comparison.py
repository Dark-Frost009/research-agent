from dataclasses import replace
import sqlite3

import pytest
from streamlit.testing.v1 import AppTest

import research_agent.history as history_module
import research_agent.ui_service as service
from research_agent.history import HistoryStore
from research_agent.comparison_ui import _source_urls
from test_ui import APP, completed
from test_incomplete import partial
from test_history_outcomes import summary
from test_history_filters import depth


@pytest.fixture
def setup(tmp_path, monkeypatch):
    store = HistoryStore(tmp_path / 'history.sqlite')
    monkeypatch.setattr(history_module, 'get_history_store', lambda: store)
    monkeypatch.setattr(service, 'run_question', lambda *a, **k: pytest.fail('No provider calls'))
    return store


@pytest.mark.parametrize('incomplete', [False, True])
def test_comparison_preserves_unsaved_work_and_displays_saved_content(setup, incomplete):
    store = setup
    first = replace(completed('Same question'), depth=depth('Quick'), summary=summary('verified'))
    first.report.content = '<script>text</script> **Original answer**'
    second = partial('Different question') if incomplete else completed('Different question')
    a, b = store.save(first), store.save(second)
    before = store.path.read_bytes()
    app = AppTest.from_file(APP, default_timeout=10)
    unsaved = completed('Unsaved work')
    app.session_state['completed_research'] = unsaved
    app.session_state['saved_report_id'] = None
    app.run()
    app.toggle(key='compare_runs').set_value(True).run()
    app.selectbox(key='compare_first').set_value(a).run()
    app.selectbox(key='compare_second').set_value(b).run()
    assert not app.exception
    assert app.session_state['completed_research'] == unsaved
    assert app.session_state['saved_report_id'] is None
    assert store.path.read_bytes() == before
    assert any(item.value == first.report.content for item in app.text)
    assert any('different questions' in item.value for item in app.info)
    assert any('1 shared' in item.value for item in app.caption)
    assert any(item.value == 'Research depth: Quick' for item in app.caption)
    assert any('Detailed budget usage and limits were not recorded' in item.value for item in app.text)
    assert len(app.table) == 1
    if incomplete:
        assert any('No completed answer is available' in item.value for item in app.text)
    else:
        assert any('Outcome not recorded' in item.value for item in app.text)
    app.toggle(key='compare_runs').set_value(False).run()
    assert not app.exception
    assert app.session_state['completed_research'] == unsaved


def test_selection_changes_cannot_compare_same_run_and_ignore_sidebar_filters(setup):
    a, b = setup.save(completed('First')), setup.save(partial('Second'))
    app = AppTest.from_file(APP, default_timeout=10).run()
    app.text_input(key='history_search').set_value('No matches').run()
    app.toggle(key='compare_runs').set_value(True).run()
    app.selectbox(key='compare_first').set_value(a).run()
    assert len(app.selectbox(key='compare_second').options) == 1
    app.selectbox(key='compare_second').set_value(b).run()
    app.selectbox(key='compare_first').set_value(b).run()
    assert not app.exception
    assert app.selectbox(key='compare_second').value != b
    assert any('Select both runs' in item.value for item in app.caption)


@pytest.mark.parametrize('count', [0, 1])
def test_needs_two_saved_runs(setup, count):
    if count:
        setup.save(completed())
    app = AppTest.from_file(APP, default_timeout=10).run()
    app.toggle(key='compare_runs').set_value(True).run()
    assert not app.exception
    assert any('Save at least two runs' in item.value for item in app.info)


def test_corrupt_record_fails_safely(setup):
    a, b = setup.save(completed()), setup.save(partial())
    with sqlite3.connect(setup.path) as connection:
        connection.execute('UPDATE reports SET payload=? WHERE id=?', ('broken', a))
    app = AppTest.from_file(APP, default_timeout=10).run()
    app.toggle(key='compare_runs').set_value(True).run()
    app.selectbox(key='compare_first').set_value(a).run()
    app.selectbox(key='compare_second').set_value(b).run()
    assert not app.exception
    assert any('could not be opened' in item.value for item in app.warning)
    assert not app.table


def test_source_overlap_uses_safe_distinct_urls():
    result = completed()
    result.sources.append(result.sources[0].model_copy(update={'id': 'duplicate'}))
    assert _source_urls(result) == {'https://example.com/final'}
    result.sources[0].final_url = 'https://different.example/page'
    assert len(_source_urls(result)) == 2
