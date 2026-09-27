"""No paid/provider calls: exercise real admission storage and process shutdown."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from threading import Barrier
from unittest.mock import Mock

import pytest

import research_agent.public_limits as limits
import research_agent.public_execution as execution
import research_agent.public_service as public
from test_ui import completed


def test_limits_survive_new_connections_and_count_failures(tmp_path, monkeypatch):
    clock = [100_000.0]
    monkeypatch.setattr(limits.time, 'time', lambda: clock[0])
    db = tmp_path / 'limits.db'
    for _ in range(3):
        with pytest.raises(RuntimeError):
            with limits.admit_run(db, ('issuer', 'alice')):
                raise RuntimeError('failure')
    with pytest.raises(limits.AdmissionError, match='Your research limit'):
        with limits.admit_run(db, ('issuer', 'alice')):
            pytest.fail('Limit bypassed')
    with limits.admit_run(db, ('issuer', 'bob')):
        pass
    clock[0] += 3600
    with limits.admit_run(db, ('issuer', 'alice')):
        pass
    with sqlite3.connect(db) as connection:
        rows = connection.execute('SELECT identity, active_until FROM runs').fetchall()
    assert len(rows) == 5
    assert all(len(identity) == 64 and active == 0 for identity, active in rows)
    assert b'alice' not in db.read_bytes()
    clock[0] += 86400
    with limits.admit_run(db, ('issuer', 'alice')):
        pass
    with sqlite3.connect(db) as connection:
        assert connection.execute('SELECT COUNT(*) FROM runs').fetchone()[0] == 1


def test_daily_and_global_limits(tmp_path, monkeypatch):
    now = [100_000.0]
    monkeypatch.setattr(limits.time, 'time', lambda: now[0])
    db = tmp_path / 'usage.db'
    for n in range(10):
        now[0] += 3601
        with limits.admit_run(db, ('issuer', 'alice')):
            pass
    with pytest.raises(limits.AdmissionError, match='Your research limit'):
        with limits.admit_run(db, ('issuer', 'alice')):
            pass
    monkeypatch.setattr(limits, 'GLOBAL_HOURLY', 1)
    with pytest.raises(limits.AdmissionError, match='service research limit'):
        with limits.admit_run(db, ('issuer', 'bob')):
            pass
    now[0] += 3601
    monkeypatch.setattr(limits, 'GLOBAL_DAILY', 10)
    with pytest.raises(limits.AdmissionError, match='service research limit'):
        with limits.admit_run(db, ('issuer', 'bob')):
            pass


def test_atomic_capacity_and_per_account_concurrency(tmp_path):
    db = tmp_path / 'usage.db'
    ready, release = Barrier(3), Barrier(3)
    def hold(user):
        with limits.admit_run(db, ('issuer', user)):
            ready.wait(timeout=5)
            release.wait(timeout=5)
    with ThreadPoolExecutor(2) as pool:
        jobs = [pool.submit(hold, user) for user in ('alice', 'bob')]
        ready.wait(timeout=5)
        try:
            for user, message in [('alice', 'already'), ('charlie', 'busy')]:
                with pytest.raises(limits.AdmissionError, match=message):
                    with limits.admit_run(db, ('issuer', user)):
                        pytest.fail('Concurrent limit bypassed')
        finally:
            release.wait(timeout=5)
        for job in jobs:
            job.result()
    with limits.admit_run(db, ('issuer', 'charlie')):
        pass


def test_expired_crash_lease_releases_capacity(tmp_path):
    db = tmp_path / 'usage.db'
    with limits.admit_run(db, ('issuer', 'alice')):
        pass
    with sqlite3.connect(db) as connection:
        connection.execute('UPDATE runs SET active_until = ?', (limits.time.time() - 1,))
    with limits.admit_run(db, ('issuer', 'alice')):
        pass


@pytest.mark.parametrize('identity', [None, ('', 'alice'), ('issuer', ''), 'alice'])
def test_bad_identity_never_starts_provider(identity, tmp_path, monkeypatch):
    monkeypatch.setenv('PUBLIC_USAGE_DB', str(tmp_path / 'usage.db'))
    monkeypatch.setattr(public, 'execute_question', lambda *a: pytest.fail('Provider called'))
    with pytest.raises(limits.AdmissionError):
        public.run_visitor_question('Question', lambda _: None, gemini_key='key',
                                    tavily_key='key', model='model', identity=identity)


def test_storage_failure_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv('PUBLIC_USAGE_DB', str(tmp_path))  # directory is not a DB
    monkeypatch.setattr(public, 'execute_question', lambda *a: pytest.fail('Provider called'))
    with pytest.raises(limits.AdmissionError, match='unavailable'):
        public.run_visitor_question('Question', lambda _: None, gemini_key='key',
                                    tavily_key='key', model='model', identity=('issuer', 'alice'))


def test_parent_timeout_reaps_a_real_hung_child():
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    with pytest.raises(execution.ResearchTimeoutError):
        execution._communicate(process, '', 0.1)
    assert process.poll() is not None
    assert process.stdin.closed and process.stdout.closed


def test_worker_watchdog_stops_even_without_parent_timeout():
    env = {**os.environ, 'PYTHONPATH': str(Path(public.__file__).resolve().parents[1])}
    process = subprocess.Popen([sys.executable, '-c',
        'import research_agent.public_limits as p; p.RUN_SECONDS=0.1; '
        'from research_agent.public_worker import main; main()'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    try:
        process.wait(timeout=10)
        assert process.returncode == 124
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate()


def test_result_roundtrip_and_credentials_only_in_pipe(monkeypatch):
    original = completed('A question')
    process = Mock()
    process.returncode = 0
    process.poll.return_value = 0
    process.communicate.return_value = (json.dumps(dict(kind='completed',
        value=json.loads(original.json_export()), warning_count=original.warning_count)), None)
    factory = Mock(return_value=process)
    monkeypatch.setattr(execution.subprocess, 'Popen', factory)
    result = execution.execute_question('A question', public.visitor_settings('PRIVATE_KEY', 'SEARCH_KEY', 'model'))
    assert result.json_export() == original.json_export()
    assert 'PRIVATE_KEY' not in str(factory.call_args)
    assert json.loads(process.communicate.call_args.args[0])['gemini_key'] == 'PRIVATE_KEY'


def test_public_provider_timeout_policy(monkeypatch):
    from research_agent.llm import gemini
    from research_agent.tools.web_search import TavilySearchClient
    from research_agent.bootstrap import _build_llm_clients, _build_search_client
    make = Mock()
    monkeypatch.setattr(gemini.genai, 'Client', make)
    settings = public.visitor_settings('key', 'key', 'model')
    _build_llm_clients(settings)
    kwargs = make.call_args.kwargs
    assert kwargs['vertexai'] is False
    assert kwargs['http_options'].timeout == 30_000
    assert kwargs['http_options'].retry_options.attempts == 1
    search = _build_search_client(settings)
    assert search._bounded_requests is True
    stub = Mock()
    stub.search.return_value = {'results': []}
    TavilySearchClient(client=stub, bounded_requests=True).search('query', 'sq1', 3)
    assert stub.search.call_args.kwargs['timeout'] == 15
    assert stub.search.call_args.kwargs['auto_parameters'] is False


@pytest.mark.parametrize('mode', ['completed', 'error'])
def test_real_worker_protocol_suppresses_provider_logs(mode):
    # Stub the research call inside a real worker; never contact providers.
    wire = completed('Question').json_export()
    script = (
        "from types import SimpleNamespace\n"
        "import research_agent.ui_service as service\n"
        "def fake(*args, **kwargs):\n"
        "    print('PRIVATE_PROVIDER_LOG')\n"
        + ("    raise RuntimeError('PRIVATE_PROVIDER_ERROR')\n" if mode == 'error' else
           f"    return SimpleNamespace(json_export=lambda: {wire!r}, warning_count=0)\n")
        + "service.run_question = fake\n"
        "from research_agent.public_worker import main\nmain()\n"
    )
    env = {**os.environ, 'PYTHONPATH': str(Path(public.__file__).resolve().parents[1])}
    response = subprocess.run([sys.executable, '-c', script], input=json.dumps(dict(
        question='Question', gemini_key='fake', tavily_key='fake', model='test')),
        capture_output=True, text=True, encoding='utf-8', timeout=20, env=env)
    assert response.returncode == 0
    assert 'PRIVATE_PROVIDER' not in response.stdout + response.stderr
    assert json.loads(response.stdout)['kind'] == mode


def test_ui_passes_verified_identity_and_shows_timeout(monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    from test_public_service import fill, APP
    monkeypatch.setenv('PUBLIC_ENABLED', 'true')
    monkeypatch.setenv('PUBLIC_GEMINI_MODEL', 'test')
    monkeypatch.setattr(st, 'user', dict(is_logged_in=True, iss='issuer', sub='alice',
                                       exp=limits.time.time() + 3600))
    def timeout(*a, **kwargs):
        assert kwargs['identity'] == ('issuer', 'alice')
        raise execution.ResearchTimeoutError('Research reached the three-minute time limit.')
    monkeypatch.setattr(public, 'run_visitor_question', timeout)
    app = fill(AppTest.from_file(APP).run())
    assert not app.exception
    assert 'three-minute' in app.warning[0].value
    assert not app.get('download_button')
