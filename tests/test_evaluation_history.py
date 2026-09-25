from copy import deepcopy
import sqlite3

import pytest
from streamlit.testing.v1 import AppTest

import research_agent.evaluation_dashboard as dashboard
from research_agent.evaluation_history import EvaluationHistory, EvaluationHistoryError, compare_evaluations
from test_evaluation_dashboard import CASES, PASS_XML
from test_ui import APP, completed


def report():
    return dashboard.summarize_report(CASES, PASS_XML, 0)


def test_roundtrip_restart_and_invalid_reports(tmp_path):
    path = tmp_path / 'evaluation.sqlite'
    store = EvaluationHistory(path)
    assert store.list_runs() == [] and not path.exists()
    original = report()
    key = store.save(original)
    loaded = EvaluationHistory(path).load(key)
    assert loaded['scenarios'] == original['scenarios']
    assert loaded['success'] == original['success']
    assert store.list_runs()[0][0] == key
    invalid = deepcopy(original)
    invalid['passed'] = 99
    with pytest.raises(EvaluationHistoryError):
        store.save(invalid)
    assert len(store.list_runs()) == 1
    with sqlite3.connect(path) as connection:
        connection.execute('UPDATE evaluations SET payload=? WHERE id=?', ('broken', key))
    with pytest.raises(EvaluationHistoryError):
        store.load(key)


@pytest.mark.parametrize('status,change', [('Failed','Regression'),('Error','Regression'),
                                         ('Skipped','Lost coverage'),('Not run','Lost coverage'),('Passed','Unchanged')])
def test_comparison_status_transitions(status, change):
    before = report()
    after = deepcopy(before)
    after['scenarios'][0]['status'] = status
    assert compare_evaluations(before, after)[0]['change'] == change
    if status != 'Passed':
        assert compare_evaluations(after, before)[0]['change'] == 'Now passing'


def test_changed_added_removed_and_legacy_not_false_regressions():
    before = report()
    after = deepcopy(before)
    after['scenarios'][0].update(status='Failed', fixture_hash='different')
    assert compare_evaluations(before, after)[0]['change'] == 'Scenario changed'
    after['scenarios'][0].pop('fixture_hash')
    assert 'Not comparable' in compare_evaluations(before, after)[0]['change']
    after['scenarios'][0]['scenario'] = 'new'
    assert {row['change'] for row in compare_evaluations(before, after)} == {'Added scenario', 'Removed scenario'}


def test_fixture_hash_changes_with_content():
    changed = deepcopy(CASES)
    changed[0]['question'] = 'New wording'
    assert report()['scenarios'][0]['fixture_hash'] != dashboard.summarize_report(changed, PASS_XML, 0)['scenarios'][0]['fixture_hash']


def test_dashboard_autosaves_once_reopens_and_compares(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, 'ROOT', tmp_path)
    store = EvaluationHistory(tmp_path / '.local' / 'evaluation_history.sqlite')
    baseline = store.save(report())
    failed_xml = PASS_XML.replace(b'</properties>', b'</properties><failure>Example failed check</failure>')
    failed = dashboard.summarize_report(CASES, failed_xml, 1)
    calls = []
    def run():
        calls.append(True)
        return failed
    monkeypatch.setattr(dashboard, 'run_evaluations', run)
    app = AppTest.from_file(APP, default_timeout=10)
    original = completed('Unsaved research')
    app.session_state['completed_research'] = original
    app.session_state['saved_report_id'] = None
    app.session_state['evaluation_dashboard'] = True
    app.run()
    app.button(key='run_offline_evaluations').click().run()
    assert not app.exception
    assert len(store.list_runs()) == 2
    app.run()
    assert len(store.list_runs()) == 2 and len(calls) == 1
    app.selectbox(key='evaluation_baseline').set_value(baseline).run()
    assert not app.exception
    assert any('1 regressions' in item.value for item in app.text)
    assert app.session_state['completed_research'] == original
    assert len(app.get('download_button')) == 2
    restarted = AppTest.from_file(APP, default_timeout=10)
    restarted.session_state['evaluation_dashboard'] = True
    restarted.run()
    restarted.selectbox(key='evaluation_selection').set_value(baseline).run()
    restarted.button(key='open_evaluation').click().run()
    assert not restarted.exception
    assert restarted.session_state['offline_evaluation_report']['success']
    assert len(calls) == 1


def test_failed_save_retains_summary_and_retry_saves_once(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, 'ROOT', tmp_path)
    monkeypatch.setattr(dashboard, 'run_evaluations', report)
    original_save = EvaluationHistory.save
    def fail(*args):
        raise EvaluationHistoryError('Disk unavailable')
    monkeypatch.setattr(EvaluationHistory, 'save', fail)
    app = AppTest.from_file(APP, default_timeout=10)
    app.session_state['evaluation_dashboard'] = True
    app.run()
    app.button(key='run_offline_evaluations').click().run()
    assert not app.exception
    assert app.session_state['offline_evaluation_report']['success']
    assert len(app.get('download_button')) == 1
    app.button(key='run_offline_evaluations').click().run()
    assert any('Save the current evaluation' in item.value for item in app.warning)
    monkeypatch.setattr(EvaluationHistory, 'save', original_save)
    app.button(key='retry_save_evaluation').click().run()
    assert not app.exception
    app.run()
    store = EvaluationHistory(tmp_path / '.local' / 'evaluation_history.sqlite')
    assert len(store.list_runs()) == 1
