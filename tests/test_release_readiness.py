from copy import deepcopy

import pytest

from research_agent.release_readiness import assess, fingerprint, markdown
from research_agent.evaluation_dashboard import summarize_report
from test_evaluation_dashboard import CASES, PASS_XML


def test_pass_is_only_offline_not_live_certification():
    report = assess(CASES, PASS_XML, 0, True)
    assert report['offline_passed']
    assert 'live validation pending' in report['status']
    assert report['changes'] is None
    report.update(source_fingerprint='abc', baseline_note='Not assessed')
    text = markdown(report)
    assert '2/2 passed' in text and 'Not assessed' in text
    assert 'human citation review' in text


@pytest.mark.parametrize('node', ['<failure/>', '<error/>', '<skipped/>'])
def test_any_failure_or_skipped_check_blocks_offline_pass(node):
    xml = PASS_XML.replace(b'<testcase name="test_dataset"/>',
                           f'<testcase name="test_dataset">{node}</testcase>'.encode())
    assert not assess(CASES, xml, 0, True)['offline_passed']


def test_incomplete_process_changed_sources_and_missing_cases_do_not_pass():
    assert not assess(CASES, PASS_XML, 1, True)['offline_passed']
    assert not assess(CASES, PASS_XML, 0, False)['offline_passed']
    assert not assess(CASES, b'<testsuite/>', 0, True)['offline_passed']


def test_baseline_regression_and_changed_fixture_are_distinct():
    baseline = summarize_report(CASES, PASS_XML, 0)
    failed = PASS_XML.replace(b'</properties>', b'</properties><failure/>')
    report = assess(CASES, failed, 1, True, baseline)
    assert report['changes'][0]['change'] == 'Regression'
    changed = deepcopy(CASES)
    changed[0]['question'] = 'Changed fixture'
    assert assess(changed, failed, 1, True, baseline)['changes'][0]['change'] == 'Scenario changed'


def test_fingerprint_tracks_code_and_fixtures_but_not_secrets(tmp_path):
    (tmp_path / 'src').mkdir()
    code = tmp_path / 'src' / 'example.py'
    code.write_text('first')
    before = fingerprint(tmp_path)
    (tmp_path / '.env').write_text('dummy secret')
    assert fingerprint(tmp_path) == before
    code.write_text('second')
    assert fingerprint(tmp_path) != before
