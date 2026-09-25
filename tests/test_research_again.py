from dataclasses import replace
import sqlite3

import pytest
from streamlit.testing.v1 import AppTest

import research_agent.history as history_module
import research_agent.ui_service as service
from research_agent.history import HistoryStore
from test_ui import APP, completed
from test_incomplete import partial
from test_history_filters import depth


@pytest.mark.parametrize('incomplete', [False, True])
@pytest.mark.parametrize('recorded_depth', [None, 'Quick', 'Thorough'])
def test_prepare_only_then_explicit_start_preserves_original(tmp_path, monkeypatch, incomplete, recorded_depth):
    store = HistoryStore(tmp_path / 'history.sqlite')
    mode = depth(recorded_depth) if recorded_depth else None
    original = (partial('Original question').model_copy(update={'depth': mode}) if incomplete
                else replace(completed('Original question'), depth=mode))
    key = store.save(original)
    monkeypatch.setattr(history_module, 'get_history_store', lambda: store)
    calls = []
    def run(question, progress, *, depth):
        calls.append((question, depth))
        return completed(question)
    monkeypatch.setattr(service, 'run_question', run)
    app = AppTest.from_file(APP, default_timeout=10).run()
    app.button(key='open_history').click().run()
    result_key = 'incomplete_research' if incomplete else 'completed_research'
    app.text_area(key='question').set_value('Replace this question').run()
    before = store.path.read_bytes()
    app.button(key='research_again').click().run()
    assert not app.exception
    assert app.text_area(key='question').value == 'Original question'
    assert app.radio(key='research_depth').value == (recorded_depth or 'Standard')
    assert app.session_state[result_key] == original
    assert store.path.read_bytes() == before
    assert calls == []
    assert any('No research has started' in item.value for item in app.info)
    if recorded_depth is None:
        assert any('No depth was recorded' in item.value for item in app.info)
    app.run()
    assert calls == []
    app.text_area(key='question').set_value('Edited question')
    app.button[0].click().run()
    assert not app.exception
    assert calls == [('Edited question', recorded_depth or 'Standard')]
    assert len(store.list_reports()) == 2
    assert store.load(key) == original
    assert app.session_state['saved_report_id'] != key


@pytest.mark.parametrize('hidden', [False, True])
def test_unsaved_work_blocks_preparation_without_changing_inputs(tmp_path, monkeypatch, hidden):
    store = HistoryStore(tmp_path / 'history.sqlite')
    store.save(replace(completed('Saved question'), depth=depth('Quick')))
    monkeypatch.setattr(history_module, 'get_history_store', lambda: store)
    monkeypatch.setattr(service, 'run_question', lambda *a, **k: pytest.fail('No research'))
    app = AppTest.from_file(APP, default_timeout=10)
    unsaved = completed('Keep unsaved')
    app.session_state['completed_research'] = unsaved
    app.session_state['saved_report_id'] = None
    if hidden:
        app.session_state['incomplete_research'] = partial()
        app.session_state['saved_incomplete_id'] = store.save(partial())
    app.run()
    app.text_area(key='question').set_value('Keep draft').run()
    app.button(key='research_again').click().run()
    assert not app.exception
    assert app.text_area(key='question').value == 'Keep draft'
    assert app.radio(key='research_depth').value == 'Standard'
    assert app.session_state['completed_research'] == unsaved
    assert any('before preparing another run' in item.value for item in app.warning)


@pytest.mark.parametrize('deleted', [False, True])
def test_unreadable_or_removed_entry_preserves_inputs(tmp_path, monkeypatch, deleted):
    store = HistoryStore(tmp_path / 'history.sqlite')
    key = store.save(completed())
    monkeypatch.setattr(history_module, 'get_history_store', lambda: store)
    app = AppTest.from_file(APP, default_timeout=10).run()
    app.text_area(key='question').set_value('Keep draft').run()
    with sqlite3.connect(store.path) as connection:
        if deleted:
            connection.execute('DELETE FROM reports WHERE id=?', (key,))
        else:
            connection.execute('UPDATE reports SET payload=? WHERE id=?', ('broken', key))
    app.button(key='research_again').click().run()
    assert not app.exception
    assert app.text_area(key='question').value == 'Keep draft'
    assert app.radio(key='research_depth').value == 'Standard'
    assert app.warning
