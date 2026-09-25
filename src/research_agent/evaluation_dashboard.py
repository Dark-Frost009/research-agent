"""Run the fixed offline suite in a separate process and summarize its JUnit report."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]


def category(case):
    if case.get('attack_marker'):
        return 'Prompt injection'
    if len(case['sources']) > 1:
        return 'Multiple sources and claims'
    if case['expected'] == 'no_evidence':
        return 'Insufficient evidence'
    if case['expected'] == 'verified':
        return 'Grounded answers'
    return 'Citation and verification rejection'


def summarize_report(cases, xml, returncode):
    if len(xml) > 5 * 1024 * 1024 or b'<!DOCTYPE' in xml or b'<!ENTITY' in xml:
        raise ValueError('Unsupported evaluation report.')
    root = ET.fromstring(xml)
    tests = list(root.iter('testcase'))
    rows = []
    used = set()
    for case in cases:
        expected_name = f"test_offline_answer_quality[{case['id']}]"
        matches = [node for node in tests if node.get('name') == expected_name]
        status, actual, detail = 'Not run', 'Not recorded', ''
        if len(matches) == 1:
            node = matches[0]
            used.add(id(node))
            props = {item.get('name'): item.get('value') for item in node.findall('./properties/property')}
            actual = props.get('actual_outcome', 'Not recorded')
            failure = node.find('failure')
            error = node.find('error')
            skipped = node.find('skipped')
            if failure is not None or error is not None:
                status = 'Failed' if failure is not None else 'Error'
                issue = failure if failure is not None else error
                detail = (issue.text or issue.get('message', 'Test failed'))[:6000]
            elif skipped is not None:
                status, detail = 'Skipped', skipped.get('message', 'Skipped')[:6000]
            elif props.get('case_id') == case['id'] and actual == case['expected']:
                status = 'Passed'
            else:
                status, detail = 'Error', 'Outcome metadata is missing or inconsistent.'
        elif len(matches) > 1:
            status, detail = 'Error', 'Duplicate scenario results in evaluation report.'
        rows.append(dict(scenario=case['id'], category=category(case), status=status,
                         expected=case['expected'], actual=actual, detail=detail))
    checks = []
    for node in tests:
        if id(node) not in used:
            issue = node.find('failure')
            if issue is None:
                issue = node.find('error')
            checks.append(dict(name=node.get('name', 'Suite check'),
                               status='Failed' if issue is not None else 'Skipped' if node.find('skipped') is not None else 'Passed'))
    passed = sum(row['status'] == 'Passed' for row in rows)
    return dict(version=1, generated_at=datetime.now(timezone.utc).isoformat(),
                scope='Offline scripted safeguards; not live model accuracy or prompt-injection resistance.',
                success=returncode == 0 and passed == len(cases) and bool(checks)
                        and all(check['status'] == 'Passed' for check in checks),
                returncode=returncode, passed=passed, total=len(cases), scenarios=rows, suite_checks=checks)


def run_evaluations():
    cases = json.loads((ROOT / 'evals' / 'answer_quality.json').read_text(encoding='utf-8'))['cases']
    with tempfile.TemporaryDirectory(prefix='research-offline-eval-') as directory:
        report_path = Path(directory) / 'results.xml'
        env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTHONDONTWRITEBYTECODE='1',
                   LANGSMITH_TRACING='false', LANGCHAIN_TRACING_V2='false')
        env.pop('PYTEST_ADDOPTS', None)
        env.pop('PYTEST_PLUGINS', None)
        command = [sys.executable, '-B', '-m', 'pytest', 'tests/test_offline_evaluation.py',
                   '-q', '-p', 'no:cacheprovider', '-o', 'junit_family=legacy',
                   '--junitxml=' + str(report_path), '--basetemp=' + str(Path(directory) / 'work')]
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, timeout=90,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if not report_path.exists():
            raise RuntimeError('The offline suite did not produce a report. Check that the project tests and pytest are installed.')
        return summarize_report(cases, report_path.read_bytes(), result.returncode)


def render_evaluation_dashboard():
    st.header('Offline evaluation dashboard')
    st.caption('Runs the bundled scenarios with scripted providers. This checks safeguards, not live Gemini accuracy or resistance to attacks. No research history is read or changed.')
    if st.button('Run offline evaluations', key='run_offline_evaluations'):
        st.session_state.pop('offline_evaluation_report', None)
        try:
            with st.spinner('Running offline scenarios…'):
                st.session_state['offline_evaluation_report'] = run_evaluations()
        except subprocess.TimeoutExpired:
            st.error('Evaluation stopped after 90 seconds. No completed report is available; try again after checking the local test setup.')
        except (OSError, ValueError, ET.ParseError, RuntimeError):
            st.error('Could not complete the evaluation report. Check that the project fixtures, tests, and pytest are installed, then try again.')
    report = st.session_state.get('offline_evaluation_report')
    if report is None:
        st.info('Run the offline evaluations to see current results. Nothing runs automatically.')
        return
    st.caption('Snapshot from ' + report['generated_at'] + '. Run again after code or fixture changes.')
    (st.success if report['success'] else st.warning)(f"{report['passed']} of {report['total']} scenarios passed."
        + (' All suite checks passed.' if report['success'] else 'Review failures, missing results, and suite checks below.'))
    st.table([{key: row[key] for key in ('scenario', 'category', 'status', 'expected', 'actual')}
              for row in report['scenarios']])
    failures = [row for row in report['scenarios'] if row['status'] != 'Passed']
    if failures:
        st.subheader('Failures and incomplete checks by category')
        for group in sorted({row['category'] for row in failures}):
            with st.expander(group, expanded=True):
                for row in failures:
                    if row['category'] == group:
                        st.text(row['scenario'] + ': ' + row['status'])
                        st.text(row['detail'] or 'This scenario did not finish.')
    if report['suite_checks']:
        st.subheader('Suite checks')
        st.table(report['suite_checks'])
    st.download_button('Download evaluation summary (.json)', json.dumps(report, indent=2),
                       file_name='offline-evaluation-summary.json', mime='application/json', key='download_evaluation_summary')
