"""Independent local storage and comparison of evaluation snapshots."""
from contextlib import closing
from pathlib import Path
import sqlite3
from typing import Literal
from uuid import uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator


class EvaluationHistoryError(RuntimeError):
    pass


class Scenario(BaseModel):
    model_config = ConfigDict(extra='forbid')
    scenario: str
    category: str
    status: Literal['Passed', 'Failed', 'Error', 'Skipped', 'Not run']
    expected: str
    actual: str
    detail: str
    fixture_hash: str | None = None


class SuiteCheck(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str
    status: Literal['Passed', 'Failed', 'Skipped']


class EvaluationReport(BaseModel):
    model_config = ConfigDict(extra='forbid')
    version: Literal[1]
    generated_at: AwareDatetime
    scope: str
    success: bool
    returncode: int
    passed: int = Field(ge=0)
    total: int = Field(ge=0)
    scenarios: list[Scenario]
    suite_checks: list[SuiteCheck]

    @model_validator(mode='after')
    def consistent(self):
        if len({row.scenario for row in self.scenarios}) != len(self.scenarios):
            raise ValueError('Duplicate scenarios')
        if self.total != len(self.scenarios) or self.passed != sum(row.status == 'Passed' for row in self.scenarios):
            raise ValueError('Inconsistent counts')
        success = (self.returncode == 0 and self.passed == self.total and bool(self.suite_checks)
                   and all(row.status == 'Passed' for row in self.suite_checks))
        if self.success != success:
            raise ValueError('Inconsistent success status')
        return self


class EvaluationHistory:
    def __init__(self, path):
        self.path = Path(path)

    def save(self, report):
        try:
            validated = EvaluationReport.model_validate(report)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            key = uuid4().hex
            with closing(sqlite3.connect(self.path, timeout=10)) as connection, connection:
                connection.execute('CREATE TABLE IF NOT EXISTS evaluations (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, payload TEXT NOT NULL)')
                connection.execute('INSERT INTO evaluations VALUES (?, ?, ?)',
                                   (key, validated.generated_at.isoformat(), validated.model_dump_json()))
            return key
        except (OSError, sqlite3.Error, ValueError) as exc:
            raise EvaluationHistoryError('Could not save the evaluation locally. The current summary remains available to download.') from exc

    def list_runs(self):
        if not self.path.exists():
            return []
        try:
            with closing(sqlite3.connect(self.path, timeout=10)) as connection:
                return connection.execute('SELECT id, created_at FROM evaluations ORDER BY created_at DESC, id DESC').fetchall()
        except (OSError, sqlite3.Error) as exc:
            raise EvaluationHistoryError('Could not read evaluation history.') from exc

    def load(self, key):
        if not self.path.exists():
            raise EvaluationHistoryError('This saved evaluation is no longer available.')
        try:
            with closing(sqlite3.connect(self.path, timeout=10)) as connection:
                row = connection.execute('SELECT payload FROM evaluations WHERE id=?', (key,)).fetchone()
            if row is None:
                raise EvaluationHistoryError('This saved evaluation is no longer available.')
            return EvaluationReport.model_validate_json(row[0]).model_dump(mode='json')
        except (OSError, sqlite3.Error, ValueError) as exc:
            raise EvaluationHistoryError('This saved evaluation could not be opened.') from exc


def compare_evaluations(baseline, current):
    before = {row['scenario']: row for row in baseline['scenarios']}
    after = {row['scenario']: row for row in current['scenarios']}
    changes = []
    for name in sorted(before.keys() | after.keys()):
        old, new = before.get(name), after.get(name)
        if old is None:
            change = 'Added scenario'
        elif new is None:
            change = 'Removed scenario'
        elif not old.get('fixture_hash') or not new.get('fixture_hash'):
            change = 'Not comparable: fixture identity missing'
        elif old['fixture_hash'] != new['fixture_hash'] or old['expected'] != new['expected']:
            change = 'Scenario changed'
        elif old['status'] == 'Passed' and new['status'] in ('Failed', 'Error'):
            change = 'Regression'
        elif old['status'] == 'Passed' and new['status'] in ('Skipped', 'Not run'):
            change = 'Lost coverage'
        elif old['status'] != 'Passed' and new['status'] == 'Passed':
            change = 'Now passing'
        elif old['status'] != new['status'] or old['actual'] != new['actual']:
            change = 'Result changed'
        else:
            change = 'Unchanged'
        changes.append(dict(scenario=name, category=(new or old)['category'], change=change,
                            before=old['status'] if old else 'Absent', after=new['status'] if new else 'Absent'))
    return changes
