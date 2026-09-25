"""Produce an offline release-readiness snapshot; never certify live research."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

from research_agent.evaluation_dashboard import summarize_report
from research_agent.evaluation_history import EvaluationHistory, EvaluationHistoryError, compare_evaluations

ROOT = Path(__file__).resolve().parents[2]


def fingerprint(root):
    files = [*root.glob('src/**/*.py'), *root.glob('tests/**/*.py'), *root.glob('evals/*.json'),
             root / 'app.py', root / 'pyproject.toml']
    digest = hashlib.sha256()
    for path in sorted(files):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(b'\0' + path.read_bytes() + b'\0')
    return digest.hexdigest()


def assess(cases, xml, returncode, unchanged, baseline=None):
    tree = ET.fromstring(xml)
    nodes = list(tree.iter('testcase'))
    counts = dict(total=len(nodes), failed=sum(node.find('failure') is not None for node in nodes),
                  errors=sum(node.find('error') is not None for node in nodes),
                  skipped=sum(node.find('skipped') is not None for node in nodes))
    counts['passed'] = counts['total'] - counts['failed'] - counts['errors'] - counts['skipped']
    evaluation = summarize_report(cases, xml, returncode)
    changes = compare_evaluations(baseline, evaluation) if baseline else None
    passed = (returncode == 0 and bool(nodes) and not any(counts[key] for key in ('failed', 'errors', 'skipped'))
              and evaluation['success'] and unchanged)
    return dict(generated_at=datetime.now(timezone.utc).isoformat(),
                status='Offline checks passed; live validation pending' if passed else 'Offline checks need attention',
                offline_passed=passed, source_unchanged_during_check=unchanged, tests=counts,
                scenarios=[{key: row[key] for key in ('scenario', 'category', 'status', 'expected', 'actual')} for row in evaluation['scenarios']],
                baseline_at=baseline['generated_at'] if baseline else None, changes=changes,
                live_validation='Not assessed by this offline check. A successful live run and human citation review are still required.',
                remaining=['Complete a live research run with Gemini and inspect every citation against its source.',
                           'Confirm insufficient-evidence and provider-error behavior with live services.',
                           'Check setup from a clean local environment and complete a final interface review.'])


def markdown(report):
    counts = report['tests']
    lines = ['# Local MVP release readiness', report['status'], 'Checked: ' + report['generated_at'],
             'Source snapshot: `' + report['source_fingerprint'] + '`',
             '## Offline checks', f"Tests: {counts['passed']}/{counts['total']} passed; {counts['failed']} failed; "
             f"{counts['errors']} errors; {counts['skipped']} skipped.",
             'Source files unchanged during check: ' + str(report['source_unchanged_during_check']),
             'Scenarios use scripted providers and verdicts. Passing does not establish live answer quality or prompt-injection resistance.',
             '| Scenario | Result |\n| --- | --- |']
    lines[-1] += '\n' + '\n'.join('| ' + row['scenario'].replace('|', '\\|') + ' | ' + row['status'] + ' |' for row in report['scenarios'])
    lines += ['## Evaluation baseline', report['baseline_note']]
    if report['changes'] is not None:
        for change in sorted({row['change'] for row in report['changes']}):
            lines.append(f"- {change}: {sum(row['change'] == change for row in report['changes'])}")
    lines += ['## Live validation', report['live_validation'], '## Remaining work',
              '\n'.join('- ' + item for item in report['remaining']),
              'This is a timestamped local-MVP checklist, not a public deployment or security certification. Rerun after code changes.']
    return '\n\n'.join(lines) + '\n'


def run_check(root=ROOT):
    before = fingerprint(root)
    cases = json.loads((root / 'evals' / 'answer_quality.json').read_text(encoding='utf-8'))['cases']
    baseline = None
    note = 'No saved evaluation baseline was available; regression comparison was not assessed.'
    try:
        history = EvaluationHistory(root / '.local' / 'evaluation_history.sqlite')
        saved = history.list_runs()
        if saved:
            baseline = history.load(saved[0][0])
            note = 'Compared with the most recent saved evaluation: ' + baseline['generated_at']
    except EvaluationHistoryError:
        note = 'Saved evaluation history was unreadable; regression comparison was not assessed.'
    with tempfile.TemporaryDirectory(prefix='research-readiness-') as directory:
        xml_path = Path(directory) / 'suite.xml'
        env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTHONDONTWRITEBYTECODE='1',
                   LANGSMITH_TRACING='false', LANGCHAIN_TRACING_V2='false')
        env.pop('PYTEST_ADDOPTS', None)
        env.pop('PYTEST_PLUGINS', None)
        result = subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
            '-o', 'junit_family=legacy', '--junitxml=' + str(xml_path), '--basetemp=' + str(Path(directory) / 'work')],
            cwd=root, env=env, capture_output=True, timeout=180,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if not xml_path.exists():
            raise RuntimeError('Test run produced no report. No readiness conclusion is available.')
        report = assess(cases, xml_path.read_bytes(), result.returncode, before == fingerprint(root), baseline)
    report.update(source_fingerprint=before, baseline_note=note)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / '.local' / 'readiness')
    args = parser.parse_args()
    try:
        report = run_check()
    except (OSError, ValueError, ET.ParseError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print('Readiness check did not complete: ' + type(exc).__name__, file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'release-readiness.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (args.output_dir / 'release-readiness.md').write_text(markdown(report), encoding='utf-8')
    print(report['status'])
    print(json.dumps(report['tests']))
    print(args.output_dir / 'release-readiness.md')
    return 0 if report['offline_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
