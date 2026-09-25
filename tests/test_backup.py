import json
import sqlite3
from types import SimpleNamespace

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import research_agent.backup as backup_module
import research_agent.history as history_module
import research_agent.ui_service as service
from research_agent.backup import export_backup, read_backup, restore_backup
from research_agent.history import HistoryError, HistoryStore
from test_ui import APP, completed
from test_incomplete import partial


@pytest.fixture
def stores(tmp_path):
    return HistoryStore(tmp_path / 'source.sqlite'), HistoryStore(tmp_path / 'target.sqlite')


def test_roundtrip_merge_and_repeated_restore(stores):
    source, target = stores
    results = [completed('Completed'), partial('Interrupted')]
    ids = [source.save(result) for result in results]
    existing = target.save(completed('Keep me'))
    raw = export_backup(source)
    assert restore_backup(target, raw) == (2, 0)
    assert restore_backup(target, raw) == (0, 2)
    assert len(target.list_reports()) == 3
    assert target.load(existing).report.question == 'Keep me'
    for key, result in zip(ids, results):
        assert target.load(key) == result
        assert target.load(key).json_export() == result.json_export()


@pytest.mark.parametrize('cross_kind', [False, True])
def test_conflict_rolls_back_prior_insert(stores, cross_kind):
    source, target = stores
    conflict = target.save(completed('Keep me'))
    source.save(completed('New'))
    source.save(partial('Conflicting') if cross_kind else completed('Conflicting'))
    data = json.loads(export_backup(source))
    data['records'].sort(key=lambda r: r['payload'].get('question', r['payload'].get('report', {}).get('question')) == 'Conflicting')
    data['records'][1]['id'] = conflict
    before = target.path.read_bytes()
    with pytest.raises(HistoryError, match='conflicts'):
        restore_backup(target, json.dumps(data).encode())
    assert target.path.read_bytes() == before
    assert len(target.list_reports()) == 1


@pytest.mark.parametrize('mutation', ['version', 'format', 'duplicate', 'extra', 'reference', 'id'])
def test_invalid_backup_never_writes(stores, mutation):
    source, target = stores
    source.save(completed())
    data = json.loads(export_backup(source))
    if mutation == 'version':
        data['version'] = 999
    elif mutation == 'format':
        data['format'] = 'unrelated'
    elif mutation == 'duplicate':
        data['records'] *= 2
    elif mutation == 'extra':
        data['records'][0]['payload']['draft_content'] = 'Rejected draft'
    elif mutation == 'reference':
        data['records'][0]['payload']['evidence'] = []
    else:
        data['records'][0]['id'] = '../unsafe'
    with pytest.raises(HistoryError):
        restore_backup(target, json.dumps(data).encode())
    assert not target.path.exists()


def test_empty_backup_does_not_create_database(stores):
    source, target = stores
    assert restore_backup(target, export_backup(source)) == (0, 0)
    assert not source.path.exists() and not target.path.exists()


def test_size_and_json_limits(stores, monkeypatch):
    source, target = stores
    monkeypatch.setattr(backup_module, 'MAX_BACKUP_BYTES', 100)
    with pytest.raises(HistoryError, match='20 MB'):
        restore_backup(target, b' ' * 101)
    with pytest.raises(HistoryError):
        read_backup(b'not json')
    with pytest.raises(HistoryError, match='20 MB'):
        export_backup(source)
    assert not target.path.exists()


def test_corrupt_saved_record_prevents_partial_backup(stores):
    source, _ = stores
    key = source.save(completed())
    source.save(partial())
    with sqlite3.connect(source.path) as connection:
        connection.execute('UPDATE reports SET payload=? WHERE id=?', ('broken', key))
    before = source.path.read_bytes()
    with pytest.raises(HistoryError, match='complete backup'):
        export_backup(source)
    assert source.path.read_bytes() == before


def test_older_optional_metadata_restores(stores):
    source, target = stores
    key = source.save(completed())
    data = json.loads(export_backup(source))
    for field in ('depth', 'summary'):
        data['records'][0]['payload'].pop(field)
    assert restore_backup(target, json.dumps(data).encode()) == (1, 0)
    assert target.load(key) == source.load(key)


def test_write_error_is_safe(stores, monkeypatch):
    source, target = stores
    source.save(completed())
    def fail():
        raise sqlite3.OperationalError('PRIVATE DETAILS')
    monkeypatch.setattr(target, '_connect', fail)
    with pytest.raises(HistoryError, match='No entries') as error:
        restore_backup(target, export_backup(source))
    assert 'PRIVATE' not in str(error.value)


@pytest.mark.parametrize('incomplete', [False, True])
def test_ui_restore_requires_click_and_preserves_unsaved_work(stores, monkeypatch, incomplete):
    source, target = stores
    source.save(completed('To restore'))
    raw = export_backup(source)
    monkeypatch.setattr(history_module, 'get_history_store', lambda: target)
    monkeypatch.setattr(service, 'run_question', lambda *a, **k: pytest.fail('No provider calls'))
    monkeypatch.setattr(st, 'file_uploader', lambda *a, **k: SimpleNamespace(size=len(raw), getvalue=lambda: raw))
    app = AppTest.from_file(APP, default_timeout=10)
    key = 'incomplete_research' if incomplete else 'completed_research'
    original = partial() if incomplete else completed('Unsaved')
    app.session_state[key] = original
    app.run()
    assert not app.exception
    assert target.list_reports() == []
    app.button(key='restore_history_backup').click().run()
    assert not app.exception
    assert len(target.list_reports()) == 1
    assert app.session_state[key] == original
    assert any('Restored 1 entries' in item.value for item in app.success)


def test_ui_prepare_download_only_on_request(stores, monkeypatch):
    source, _ = stores
    source.save(completed())
    monkeypatch.setattr(history_module, 'get_history_store', lambda: source)
    app = AppTest.from_file(APP, default_timeout=10).run()
    assert not app.exception
    assert not app.get('download_button')
    app.button(key='prepare_history_backup').click().run()
    assert not app.exception
    assert len(app.get('download_button')) == 1
    assert len(read_backup(app.session_state['history_backup_bytes']).records) == 1
