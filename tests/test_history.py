from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import sqlite3
import pytest
from streamlit.testing.v1 import AppTest
import research_agent.history as history_module
import research_agent.ui_service as service
from research_agent.history import HistoryError, HistoryStore
from test_ui import APP, completed


def test_restart_preserves_complete_report_and_exports(tmp_path):
    path = tmp_path / 'history.sqlite'
    original = replace(completed('RAG research'), issues=('Source unavailable.',), warning_count=1)
    key = HistoryStore(path).save(original)
    restored = HistoryStore(path).load(key)
    assert restored == original
    assert restored.text_export() == original.text_export()
    assert restored.json_export() == original.json_export()
    assert HistoryStore(path).list_reports('rag')[0].id == key
    assert HistoryStore(path).list_reports('unrelated') == []


def test_concurrent_saves_preserve_distinct_runs(tmp_path):
    store = HistoryStore(tmp_path / 'history.sqlite')
    with ThreadPoolExecutor(max_workers=4) as pool:
        keys = list(pool.map(lambda _: store.save(completed()), range(8)))
    assert len(set(keys)) == len(store.list_reports()) == 8
    assert all(store.load(key).report.question == 'Test question' for key in keys)


def test_fallback_preserves_warnings(tmp_path):
    result = completed()
    result.report.citations.clear()
    result.report.content = 'Insufficient evidence.'
    result = replace(result, issues=('Provider unavailable.',), warning_count=2)
    store = HistoryStore(tmp_path / 'history.sqlite')
    assert store.load(store.save(result)) == result


@pytest.mark.parametrize('payload', ['{broken', '{"version":999}'])
def test_invalid_record_does_not_hide_good_records(tmp_path, payload):
    store = HistoryStore(tmp_path / 'history.sqlite')
    good, bad = store.save(completed('Good')), store.save(completed('Bad'))
    with sqlite3.connect(store.path) as connection:
        connection.execute('UPDATE reports SET payload=? WHERE id=?', (payload, bad))
    assert len(store.list_reports()) == 2
    assert store.load(good).report.question == 'Good'
    with pytest.raises(HistoryError):
        store.load(bad)


def test_missing_and_broken_citations_fail_safely(tmp_path):
    store = HistoryStore(tmp_path / 'history.sqlite')
    assert store.list_reports() == []
    with pytest.raises(HistoryError):
        store.load('missing')
    with pytest.raises(HistoryError):
        store.save(replace(completed(), evidence=[]))
    assert store.list_reports() == []


def test_storage_error_preserves_existing_data(tmp_path):
    parent = tmp_path / 'not-directory'
    parent.write_text('existing data')
    with pytest.raises(HistoryError):
        HistoryStore(parent / 'history.sqlite').save(completed())
    assert parent.read_text() == 'existing data'


@pytest.fixture
def local_history(tmp_path, monkeypatch):
    store = HistoryStore(tmp_path / 'history.sqlite')
    monkeypatch.setattr(history_module, 'get_history_store', lambda: store)
    return store


def test_ui_auto_save_and_reopen_without_provider_calls(local_history, monkeypatch):
    calls = []
    def run(question, progress, *, depth=None):
        calls.append(question)
        return completed(question)
    monkeypatch.setattr(service, 'run_question', run)
    first = AppTest.from_file(APP).run()
    first.text_area[0].set_value('Persistent research')
    first.button[0].click().run()
    assert not first.exception
    assert len(local_history.list_reports()) == 1
    first.run()
    assert len(local_history.list_reports()) == 1
    restarted = AppTest.from_file(APP).run()
    restarted.button(key='open_history').click().run()
    assert not restarted.exception
    assert restarted.session_state['completed_research'].report.question == 'Persistent research'
    assert len(restarted.get('download_button')) == 2
    assert calls == ['Persistent research']


def test_ui_failed_request_does_not_save(local_history, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('Failure')
    monkeypatch.setattr(service, 'run_question', fail)
    app = AppTest.from_file(APP).run()
    app.text_area[0].set_value('Failed research')
    app.button[0].click().run()
    assert not app.exception
    assert local_history.list_reports() == []


def test_ui_failed_save_keeps_downloads_and_retry_saves_once(local_history, monkeypatch):
    monkeypatch.setattr(service, 'run_question', lambda q, p, **kwargs: completed(q))
    original = local_history.save
    def fail(result):
        raise HistoryError('Could not save')
    monkeypatch.setattr(local_history, 'save', fail)
    app = AppTest.from_file(APP).run()
    app.text_area[0].set_value('Keep this result')
    app.button[0].click().run()
    assert not app.exception
    assert len(app.get('download_button')) == 2
    assert any('could not be saved' in warning.value for warning in app.warning)
    monkeypatch.setattr(local_history, 'save', original)
    app.button(key='retry_save_history').click().run()
    assert not app.exception
    assert len(local_history.list_reports()) == 1
    app.run()
    assert len(local_history.list_reports()) == 1
