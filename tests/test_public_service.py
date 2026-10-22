from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
import streamlit as st
import time
from streamlit.testing.v1 import AppTest

import research_agent.public_service as public
import research_agent.ui_service as service
from test_ui import completed

APP = Path(__file__).resolve().parents[1] / 'public_app.py'


@pytest.fixture
def public_env(monkeypatch):
    monkeypatch.setenv('PUBLIC_ENABLED', 'true')
    monkeypatch.setattr(st, 'user', {'is_logged_in': True, 'iss': 'https://issuer.example', 'sub': 'alice', 'exp': time.time() + 3600})
    monkeypatch.setenv('PUBLIC_GEMINI_MODEL', 'test-model')


def test_visitor_settings_never_inherit_host_secrets_or_limits(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / '.env').write_text('LLM_API_KEY=FILE_SECRET\nLLM_WORKER_MODEL=wrong\n')
    monkeypatch.setenv('LLM_API_KEY', 'HOST_SECRET')
    monkeypatch.setenv('TAVILY_API_KEY', 'HOST_SEARCH_SECRET')
    monkeypatch.setenv('MAX_LLM_CALLS_PER_RUN', '999')
    monkeypatch.setenv('LLM_FINAL_MODEL', 'wrong')
    settings = public.visitor_settings('visitor-gemini', 'visitor-tavily', 'test-model')
    assert settings.llm_api_key.get_secret_value() == 'visitor-gemini'
    assert settings.tavily_api_key.get_secret_value() == 'visitor-tavily'
    assert settings.worker_llm_model == settings.final_llm_model == 'test-model'
    assert settings.max_llm_calls_per_run == 12
    assert 'visitor-gemini' not in repr(settings)
    with pytest.raises(public.PublicInputError):
        public.visitor_settings('', '', 'test-model')


@pytest.mark.parametrize('question', ['', ' ', 'x' * 2001])
def test_invalid_question_never_calls_provider(monkeypatch, question):
    monkeypatch.setattr(public, 'run_question', lambda *a, **k: pytest.fail('Provider called'))
    with pytest.raises(public.PublicInputError):
        public.run_visitor_question(question, lambda x: None,
                                   gemini_key='key', tavily_key='key', model='test')


def test_settings_injection_does_not_load_local_settings(monkeypatch):
    settings = public.visitor_settings('one', 'two', 'model')
    seen = []
    class StopBuild(Exception):
        pass
    def build(value):
        seen.append(value)
        raise StopBuild()
    monkeypatch.setattr(service, 'Settings', lambda **kw: pytest.fail('Host settings loaded'))
    monkeypatch.setattr(service, 'build_research_application', build)
    with pytest.raises(StopBuild):
        service.run_question('A question', lambda x: None, settings=settings)
    assert seen == [settings]


def test_capacity_is_shared_across_threads_and_released_on_failure():
    barrier = Barrier(3)
    release = Barrier(3)
    def hold():
        with public.research_slot():
            barrier.wait(timeout=5)
            release.wait(timeout=5)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(hold) for _ in range(2)]
        barrier.wait(timeout=5)
        try:
            with pytest.raises(public.ServiceBusyError):
                with public.research_slot():
                    pytest.fail('Capacity bypassed')
        finally:
            release.wait(timeout=5)
        for job in jobs:
            job.result()
    with pytest.raises(RuntimeError):
        with public.research_slot():
            raise RuntimeError()
    with public.research_slot():
        pass


def test_disabled_service_does_not_render_credentials(monkeypatch):
    monkeypatch.setenv('PUBLIC_ENABLED', 'false')
    app = AppTest.from_file(APP).run()
    assert not app.exception
    assert not app.text_input
    assert not app.text_area


@pytest.mark.parametrize('enabled', ['false', None])
def test_hosted_entry_fails_closed_to_the_demo_when_flags_are_off_or_missing(monkeypatch, enabled):
    # A missing flag (for example a TOML boolean that Streamlit never promoted
    # to an environment variable) must land on the fictional demo only.
    import research_agent.history as history
    monkeypatch.setattr(history, 'get_history_store', lambda: pytest.fail('History accessed'))
    monkeypatch.setattr(public, 'run_visitor_question', lambda *a, **k: pytest.fail('Research started'))
    monkeypatch.delenv('PUBLIC_AUTH_PREVIEW', raising=False)
    if enabled is None:
        monkeypatch.delenv('PUBLIC_ENABLED', raising=False)
    else:
        monkeypatch.setenv('PUBLIC_ENABLED', enabled)
    app = AppTest.from_file(APP).run()
    assert not app.exception
    assert any(item.value == 'Sample research question' for item in app.subheader)
    assert not app.text_input
    assert not app.text_area
    assert not app.checkbox
    assert not [b for b in app.button if b.label in {'Sign in with Google', 'Start research'}]
    assert not app.get('toggle')


@pytest.mark.parametrize('local_flag', [None, '', '0', 'false', 'yes please'])
def test_app_py_without_explicit_local_opt_in_never_serves_the_local_app(monkeypatch, local_flag):
    # A hosted deployment cannot change its main file path, so app.py itself must
    # fail closed: no flag, or a mistyped one, lands on the fictional demo only.
    import research_agent.history as history
    monkeypatch.setattr(history, 'get_history_store', lambda: pytest.fail('History accessed'))
    monkeypatch.setattr(service, 'run_question', lambda *a, **k: pytest.fail('Research started'))
    for name in ('PUBLIC_ENABLED', 'PUBLIC_AUTH_PREVIEW', 'PUBLIC_DEMO_ONLY'):
        monkeypatch.delenv(name, raising=False)
    if local_flag is None:
        monkeypatch.delenv('RESEARCH_AGENT_LOCAL', raising=False)
    else:
        monkeypatch.setenv('RESEARCH_AGENT_LOCAL', local_flag)
    app = AppTest.from_file(APP.parent / 'app.py').run()
    assert not app.exception
    assert any(item.value == 'Sample research question' for item in app.subheader)
    assert not app.text_input
    assert not app.text_area
    assert not app.checkbox
    assert not app.get('toggle')
    assert not any('Saved research' in item.value for item in app.markdown)


@pytest.mark.parametrize('local_flag', ['1', 'true', ' TRUE '])
def test_explicit_local_opt_in_serves_the_local_app(monkeypatch, tmp_path, local_flag):
    import research_agent.history as history
    store = history.HistoryStore(tmp_path / 'history.sqlite')
    monkeypatch.setattr(history, 'get_history_store', lambda: store)
    monkeypatch.setenv('RESEARCH_AGENT_LOCAL', local_flag)
    app = AppTest.from_file(APP.parent / 'app.py').run()
    assert not app.exception
    assert app.text_area
    assert not any(item.value == 'Sample research question' for item in app.subheader)


def test_login_is_required_before_keys_or_results(public_env, monkeypatch):
    monkeypatch.setattr(st, 'user', {'is_logged_in': False})
    app = AppTest.from_file(APP).run()
    assert not app.exception
    assert [b.label for b in app.button] == ['Sign in with Google']
    assert not app.text_input
    assert not app.get('download_button')


def fill(app, question='My question'):
    app.text_input(key='gemini_key').set_value('visitor-gemini')
    app.text_input(key='tavily_key').set_value('visitor-tavily')
    app.text_area(key='question').set_value(question)
    app.checkbox(key='consent').check()
    next(b for b in app.button if b.label == 'Start research').click().run()
    return app


def test_consent_required_and_no_shared_history(public_env, monkeypatch):
    import research_agent.history as history
    monkeypatch.setattr(history, 'get_history_store', lambda: pytest.fail('Shared history accessed'))
    monkeypatch.setattr(public, 'run_visitor_question', lambda *a, **k: pytest.fail('Unapproved call'))
    app = AppTest.from_file(APP).run()
    next(b for b in app.button if b.label == 'Start research').click().run()
    assert not app.exception
    assert 'Confirm authorization' in app.warning[0].value


def test_results_and_credentials_are_isolated_and_clearable(public_env, monkeypatch):
    calls = []
    def run(question, progress, **kwargs):
        calls.append(kwargs)
        return completed(question)
    monkeypatch.setattr(public, 'run_visitor_question', run)
    app = fill(AppTest.from_file(APP).run())
    assert not app.exception
    assert len(app.get('download_button')) == 2
    assert calls[0]['gemini_key'] == 'visitor-gemini'
    app.run()
    assert len(calls) == 1
    other = AppTest.from_file(APP).run()
    assert not other.get('download_button')
    assert other.text_input(key='gemini_key').value == ''
    next(b for b in app.button if b.label == 'Clear keys and results').click().run()
    assert not app.exception
    assert not app.get('download_button')
    assert app.text_input(key='gemini_key').value == ''


def test_provider_errors_do_not_expose_credentials(public_env, monkeypatch):
    def fail(*a, **k):
        raise RuntimeError('VERY_PRIVATE_KEY')
    monkeypatch.setattr(public, 'run_visitor_question', fail)
    app = fill(AppTest.from_file(APP).run())
    assert not app.exception
    assert 'VERY_PRIVATE_KEY' not in str(app)
    assert not app.get('download_button')
    assert app.error


@pytest.mark.parametrize('overrides', [
    {'exp': 1}, {'exp': None}, {'exp': float('nan')}, {'exp': float('inf')}, {'sub': ''}, {'iss': ''},
])
def test_invalid_identity_fails_closed(public_env, monkeypatch, overrides):
    user = dict(st.user)
    user.update(overrides)
    monkeypatch.setattr(st, 'user', user)
    app = AppTest.from_file(APP).run()
    assert not app.exception
    assert not app.text_input
    assert not app.get('download_button')
    assert app.error


def test_identity_change_clears_previous_session(public_env, monkeypatch):
    monkeypatch.setattr(public, 'run_visitor_question', lambda q, p, **kw: completed(q))
    app = fill(AppTest.from_file(APP).run())
    assert app.get('download_button')
    monkeypatch.setattr(st, 'user', {**st.user, 'sub': 'bob'})
    app.run()
    assert not app.exception
    assert not app.get('download_button')
    assert app.text_input(key='gemini_key').value == ''


def test_expired_identity_clears_session_before_display(public_env, monkeypatch):
    monkeypatch.setattr(public, 'run_visitor_question', lambda q, p, **kw: completed(q))
    app = fill(AppTest.from_file(APP).run())
    monkeypatch.setattr(st, 'user', {**st.user, 'exp': 1})
    app.run()
    assert not app.exception
    assert not app.get('download_button')
    assert 'result' not in app.session_state
    assert 'gemini_key' not in app.session_state


@pytest.mark.parametrize('entry', ['public_app.py', 'app.py'])
@pytest.mark.parametrize('logged_in', [False, True])
def test_auth_preview_cannot_run_research(public_env, monkeypatch, entry, logged_in):
    import research_agent.history as history
    monkeypatch.delenv('RESEARCH_AGENT_LOCAL', raising=False)
    monkeypatch.setenv('PUBLIC_ENABLED', 'false')
    monkeypatch.setenv('PUBLIC_AUTH_PREVIEW', 'true')
    monkeypatch.setenv('PUBLIC_DEMO_ONLY', 'true')
    monkeypatch.setattr(st, 'user', {**st.user, 'is_logged_in': logged_in})
    monkeypatch.setattr(public, 'run_visitor_question', lambda *a, **k: pytest.fail('Provider called'))
    monkeypatch.setattr(history, 'get_history_store', lambda: pytest.fail('History accessed'))
    app = AppTest.from_file(APP.parent / entry).run()
    assert not app.exception
    assert not app.text_input
    assert not app.text_area
    assert not app.get('download_button')
    if logged_in:
        assert app.success[0].value == 'Google sign-in is working. You are signed in.'
        assert [b.label for b in app.button] == ['Sign out']
    else:
        assert [b.label for b in app.button] == ['Sign in with Google']
