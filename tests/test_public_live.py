"""Password-gated live research on the operator's keys: offline, providers stubbed."""
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

import research_agent.public_access as access
import research_agent.public_execution as execution
import research_agent.public_limits as limits
import research_agent.public_service as public
from test_ui import completed

APP = Path(__file__).resolve().parents[1] / 'public_app.py'
PASSWORD = 'test-only-passphrase-7'
OPERATOR_GEMINI = 'OPERATOR-GEMINI-SECRET'
OPERATOR_TAVILY = 'OPERATOR-TAVILY-SECRET'


@pytest.fixture(scope='module')
def stored():
    return access.hash_password(PASSWORD)


@pytest.fixture
def live_env(monkeypatch, tmp_path, stored):
    for name in ('PUBLIC_ENABLED', 'PUBLIC_AUTH_PREVIEW'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('PUBLIC_USAGE_DB', str(tmp_path / 'usage.db'))
    monkeypatch.setenv('PUBLIC_LIVE_ENABLED', 'true')
    monkeypatch.setenv('PUBLIC_LIVE_PASSWORD_HASH', stored)
    monkeypatch.setenv('PUBLIC_LIVE_GEMINI_API_KEY', OPERATOR_GEMINI)
    monkeypatch.setenv('PUBLIC_LIVE_TAVILY_API_KEY', OPERATOR_TAVILY)
    monkeypatch.setenv('PUBLIC_LIVE_GEMINI_MODEL', 'test-model')
    monkeypatch.setattr(public, 'run_visitor_question', lambda *a, **k: pytest.fail('BYOK path used'))
    return tmp_path / 'usage.db'


def unlock(app, password=PASSWORD):
    app.text_input(key='live_password_input').set_value(password)
    next(b for b in app.button if b.label == 'Unlock live research').click().run()
    return app


def ask(app, question='My live question'):
    app.text_area(key='live_question').set_value(question)
    next(b for b in app.button if b.label == 'Start live research').click().run()
    return app


def downloads(app):
    """Report downloads only; the fictional demo has its own download buttons."""
    return [b for b in app.get('download_button') if not b.label.startswith('Download demo')]


def shows_demo(app):
    return any(item.value == 'Sample research question' for item in app.subheader)


@pytest.mark.parametrize('missing', ['ENABLED', 'PASSWORD_HASH', 'GEMINI_API_KEY',
                                     'TAVILY_API_KEY', 'GEMINI_MODEL'])
def test_incomplete_live_config_shows_only_the_demo(live_env, monkeypatch, missing):
    monkeypatch.delenv(f'PUBLIC_LIVE_{missing}')
    app = AppTest.from_file(APP).run()
    assert not app.exception
    assert shows_demo(app)
    assert not app.text_input and not app.text_area


def test_malformed_live_config_fails_closed_without_an_error(live_env, monkeypatch):
    monkeypatch.setenv('PUBLIC_LIVE_ENABLED', 'not-a-boolean')
    app = AppTest.from_file(APP).run()
    assert not app.exception and not app.error
    assert shows_demo(app) and not app.text_input


def test_locked_page_shows_demo_and_password_only(live_env):
    app = AppTest.from_file(APP).run()
    assert not app.exception
    assert shows_demo(app)
    assert [item.key for item in app.text_input] == ['live_password_input']
    assert not app.text_area
    page = str(app)
    for secret in (OPERATOR_GEMINI, OPERATOR_TAVILY, PASSWORD):
        assert secret not in page


def test_wrong_password_never_unlocks_and_is_counted(live_env, monkeypatch):
    monkeypatch.setattr(public, 'run_live_question', lambda *a, **k: pytest.fail('Research started'))
    app = unlock(AppTest.from_file(APP).run(), 'not-the-password-123')
    assert not app.exception
    assert any('Incorrect password' in w.value for w in app.warning)
    assert not app.text_area
    import sqlite3
    with sqlite3.connect(live_env) as db:
        assert db.execute('SELECT COUNT(*) FROM login_failures').fetchone()[0] == 1


def test_lockout_applies_to_new_sessions_and_the_correct_password(live_env):
    for _ in range(access.FAILED_ATTEMPTS_PER_HOUR):
        unlock(AppTest.from_file(APP).run(), 'not-the-password-123')
    app = unlock(AppTest.from_file(APP).run())
    assert any('Too many incorrect passwords' in w.value for w in app.warning)
    assert not app.text_area
    assert shows_demo(app)


def test_correct_password_runs_on_operator_keys_and_shows_report(live_env, monkeypatch):
    calls = []

    def run(question, progress, *, config):
        calls.append((question, config))
        return completed(question)

    monkeypatch.setattr(public, 'run_live_question', run)
    app = ask(unlock(AppTest.from_file(APP).run()))
    assert not app.exception
    assert calls[0][0] == 'My live question'
    assert calls[0][1].gemini_api_key.get_secret_value() == OPERATOR_GEMINI
    assert any(item.value == 'Research report' for item in app.subheader)
    assert len(downloads(app)) == 2
    page = str(app)
    for secret in (OPERATOR_GEMINI, OPERATOR_TAVILY, PASSWORD):
        assert secret not in page
    # Another browser session is still locked and sees no results.
    other = AppTest.from_file(APP).run()
    assert not other.text_area and not downloads(other)


def test_lock_button_and_password_change_relock_the_session(live_env, monkeypatch):
    monkeypatch.setattr(public, 'run_live_question', lambda q, p, **k: completed(q))
    app = ask(unlock(AppTest.from_file(APP).run()))
    next(b for b in app.button if b.label == 'Lock live research').click().run()
    assert not app.exception and not app.text_area and not downloads(app)
    app = unlock(app)
    assert app.text_area
    monkeypatch.setenv('PUBLIC_LIVE_PASSWORD_HASH', access.hash_password('a-new-passphrase-99'))
    app.run()
    assert not app.text_area
    assert [item.key for item in app.text_input] == ['live_password_input']


@pytest.mark.parametrize('error', [
    public.ServiceBusyError('The service is busy.'),
    limits.AdmissionError('The live research limit has been reached. Try again later.'),
    RuntimeError(OPERATOR_GEMINI),
])
def test_run_failures_are_reported_without_secrets(live_env, monkeypatch, error):
    def fail(*a, **k):
        raise error
    monkeypatch.setattr(public, 'run_live_question', fail)
    app = ask(unlock(AppTest.from_file(APP).run()))
    assert not app.exception
    assert app.warning or app.error
    assert OPERATOR_GEMINI not in str(app)
    assert not downloads(app)


def _ready_config(monkeypatch, stored):
    for name, value in dict(ENABLED='true', PASSWORD_HASH=stored, GEMINI_API_KEY=OPERATOR_GEMINI,
                            TAVILY_API_KEY=OPERATOR_TAVILY, GEMINI_MODEL='test-model').items():
        monkeypatch.setenv(f'PUBLIC_LIVE_{name}', value)
    return access.LiveAccessConfig()


def test_run_live_question_uses_shared_identity_and_limits(monkeypatch, tmp_path, stored):
    monkeypatch.setenv('PUBLIC_USAGE_DB', str(tmp_path / 'usage.db'))
    clock = [100_000.0]
    monkeypatch.setattr(limits.time, 'time', lambda: clock[0])
    seen = []
    monkeypatch.setattr(public, 'execute_question',
                        lambda question, settings: seen.append(settings) or completed(question))
    config = _ready_config(monkeypatch, stored)
    for _ in range(public.LIVE_HOURLY):
        public.run_live_question('Q', lambda _: None, config=config)
    with pytest.raises(limits.AdmissionError, match='live research limit'):
        public.run_live_question('Q', lambda _: None, config=config)
    assert all(s.app_env == 'public_live' for s in seen)
    assert seen[0].llm_api_key.get_secret_value() == OPERATOR_GEMINI
    assert seen[0].max_llm_calls_per_run == 12 and seen[0].max_research_iterations == 1
    clock[0] += 86400
    for hour in range(4):
        clock[0] += 3601
        for _ in range(public.LIVE_HOURLY):
            public.run_live_question('Q', lambda _: None, config=config)
    clock[0] += 3601
    with pytest.raises(limits.AdmissionError, match='live research limit'):
        public.run_live_question('Q', lambda _: None, config=config)


def test_only_one_live_run_at_a_time(monkeypatch, tmp_path, stored):
    db = tmp_path / 'usage.db'
    with limits.admit_run(db, public.LIVE_IDENTITY, hourly=5, daily=20, shared=True):
        with pytest.raises(limits.AdmissionError, match='Another live research run'):
            with limits.admit_run(db, public.LIVE_IDENTITY, hourly=5, daily=20, shared=True):
                pass


def test_run_live_question_refuses_incomplete_config(monkeypatch, stored):
    monkeypatch.setattr(public, 'execute_question', lambda *a: pytest.fail('Research started'))
    config = _ready_config(monkeypatch, stored)
    monkeypatch.setenv('PUBLIC_LIVE_TAVILY_API_KEY', '')
    with pytest.raises(public.PublicInputError):
        public.run_live_question('Q', lambda _: None, config=access.LiveAccessConfig())
    with pytest.raises(public.PublicInputError):
        public.run_live_question('', lambda _: None, config=config)


def test_live_gemini_retries_only_transient_errors_and_visitor_keys_do_not(monkeypatch):
    from research_agent.llm import gemini
    from research_agent.bootstrap import _build_llm_clients, _build_search_client
    make = Mock()
    monkeypatch.setattr(gemini.genai, 'Client', make)
    _build_llm_clients(public.visitor_settings('key', 'key', 'model', live=True))
    options = make.call_args.kwargs['http_options']
    assert options.timeout == 30_000
    assert options.retry_options.attempts == 3
    assert options.retry_options.max_delay == 4.0
    assert 429 not in options.retry_options.http_status_codes
    assert 503 in options.retry_options.http_status_codes
    assert _build_search_client(public.visitor_settings('key', 'key', 'model', live=True))._bounded_requests
    _build_llm_clients(public.visitor_settings('key', 'key', 'model'))
    assert make.call_args.kwargs['http_options'].retry_options.attempts == 1


def test_worker_gets_live_flag_through_pipe_and_no_hosted_secrets_in_env(monkeypatch):
    monkeypatch.setenv('PUBLIC_LIVE_GEMINI_API_KEY', OPERATOR_GEMINI)
    monkeypatch.setenv('PUBLIC_LIVE_PASSWORD_HASH', 'scrypt$secret-hash')
    original = completed('A question')
    process = Mock(returncode=0)
    process.poll.return_value = 0
    process.communicate.return_value = (json.dumps(dict(kind='completed',
        value=json.loads(original.json_export()), warning_count=0)), None)
    factory = Mock(return_value=process)
    monkeypatch.setattr(execution.subprocess, 'Popen', factory)
    execution.execute_question('A question', public.visitor_settings('K1', 'K2', 'm', live=True))
    request = json.loads(process.communicate.call_args.args[0])
    assert request['live'] is True and request['gemini_key'] == 'K1'
    env = factory.call_args.kwargs['env']
    assert not [name for name in env if name.startswith('PUBLIC_LIVE_')]
    assert OPERATOR_GEMINI not in str(factory.call_args)
    execution.execute_question('A question', public.visitor_settings('K1', 'K2', 'm'))
    assert json.loads(process.communicate.call_args.args[0])['live'] is False
