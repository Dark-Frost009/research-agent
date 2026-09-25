import json
import subprocess

import pytest
from streamlit.testing.v1 import AppTest

import research_agent.evaluation_dashboard as dashboard
import research_agent.history as history_module
import research_agent.ui_service as service
from test_ui import APP, completed


CASES = [dict(id='one', sources=[{}], expected='verified')]
PASS_XML = b'''<testsuites><testsuite><testcase name="test_dataset"/>
<testcase name="test_offline_answer_quality[one]"><properties>
<property name="case_id" value="one"/><property name="actual_outcome" value="verified"/>
</properties></testcase></testsuite></testsuites>'''


def test_pass_and_process_failure_are_distinct():
    report = dashboard.summarize_report(CASES, PASS_XML, 0)
    assert report['success'] and report['passed'] == 1
    assert not dashboard.summarize_report(CASES, PASS_XML, 1)['success']
    assert json.loads(json.dumps(report)) == report


@pytest.mark.parametrize('node,status', [('<failure>assertion failed</failure>', 'Failed'),
                                       ('<error>setup failed</error>', 'Error'),
                                       ('<skipped message="disabled"/>', 'Skipped')])
def test_failed_skipped_and_error_cases_are_not_passes(node, status):
    xml = f'<testsuite><testcase name="test_offline_answer_quality[one]">{node}</testcase></testsuite>'.encode()
    report = dashboard.summarize_report(CASES, xml, 1)
    assert not report['success'] and report['passed'] == 0
    assert report['scenarios'][0]['status'] == status
    assert report['scenarios'][0]['detail']


def test_missing_metadata_missing_case_and_failed_suite_check():
    assert dashboard.summarize_report(CASES, b'<testsuite/>', 0)['scenarios'][0]['status'] == 'Not run'
    xml = PASS_XML.replace(b'value="verified"', b'value="unknown"')
    assert dashboard.summarize_report(CASES, xml, 0)['scenarios'][0]['status'] == 'Error'
    xml = PASS_XML.replace(b'<testcase name="test_dataset"/>', b'<testcase name="test_dataset"><failure>bad fixture</failure></testcase>')
    report = dashboard.summarize_report(CASES, xml, 1)
    assert report['passed'] == 1 and not report['success']
    assert report['suite_checks'][0]['status'] == 'Failed'


def test_report_rejects_entities():
    with pytest.raises(ValueError):
        dashboard.summarize_report(CASES, b'<!DOCTYPE x><testsuite/>', 0)


def test_runner_is_fixed_and_uses_temporary_report(tmp_path, monkeypatch):
    (tmp_path / 'evals').mkdir()
    (tmp_path / 'evals' / 'answer_quality.json').write_text(json.dumps({'cases': CASES}))
    monkeypatch.setattr(dashboard, 'ROOT', tmp_path)
    monkeypatch.setenv('PYTEST_ADDOPTS', '--bad-option')
    monkeypatch.setenv('PYTEST_PLUGINS', 'unwanted')
    report_paths = []
    def run(command, **kwargs):
        from pathlib import Path
        assert 'tests/test_offline_evaluation.py' in command
        assert kwargs['cwd'] == tmp_path and kwargs['timeout'] == 90
        assert kwargs['env']['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] == '1'
        assert 'PYTEST_ADDOPTS' not in kwargs['env'] and 'PYTEST_PLUGINS' not in kwargs['env']
        path = Path(next(arg.split('=', 1)[1] for arg in command if arg.startswith('--junitxml=')))
        report_paths.append(path)
        path.write_bytes(PASS_XML)
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(dashboard.subprocess, 'run', run)
    assert dashboard.run_evaluations()['success']
    assert not report_paths[0].exists()


def test_dashboard_isolated_no_auto_run_and_preserves_work(monkeypatch, tmp_path):
    monkeypatch.setattr(dashboard, 'ROOT', tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail('Dashboard must not access providers, settings, or research history')
    monkeypatch.setattr(history_module, 'get_history_store', forbidden)
    monkeypatch.setattr(service, 'preview_depth', forbidden)
    monkeypatch.setattr(service, 'run_question', forbidden)
    calls = []
    def run():
        calls.append(True)
        return dashboard.summarize_report(CASES, PASS_XML, 0)
    monkeypatch.setattr(dashboard, 'run_evaluations', run)
    app = AppTest.from_file(APP, default_timeout=10)
    original = completed('Unsaved')
    app.session_state['completed_research'] = original
    app.session_state['saved_report_id'] = None
    app.session_state['question'] = 'Keep question'
    app.session_state['evaluation_dashboard'] = True
    app.run()
    assert not app.exception and calls == []
    app.button(key='run_offline_evaluations').click().run()
    assert not app.exception and calls == [True]
    assert len(app.get('download_button')) == 1
    assert app.session_state['completed_research'] == original
    assert app.session_state['question'] == 'Keep question'
    app.run()
    assert calls == [True]
    def timeout():
        raise subprocess.TimeoutExpired('pytest', 90)
    monkeypatch.setattr(dashboard, 'run_evaluations', timeout)
    app.button(key='run_offline_evaluations').click().run()
    assert not app.exception
    assert any('90 seconds' in item.value for item in app.error)
    assert not app.get('download_button')
