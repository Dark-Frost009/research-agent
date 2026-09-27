"""Transactional admission limits for one host and shared SQLite file.

Only a hash of the verified issuer/subject, timestamps and random run IDs are
stored. This is not a distributed limiter or a durable store on ephemeral hosts.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import uuid

RUN_SECONDS = 180
USER_HOURLY = 3
USER_DAILY = 10
GLOBAL_HOURLY = 60
GLOBAL_DAILY = 200


class AdmissionError(RuntimeError):
    pass


def _connect(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=3, isolation_level=None)
    try:
        db.execute('CREATE TABLE IF NOT EXISTS runs '
                   '(id TEXT PRIMARY KEY, identity TEXT NOT NULL, '
                   'started REAL NOT NULL, active_until REAL NOT NULL)')
        db.execute('CREATE INDEX IF NOT EXISTS runs_started ON runs(started)')
        return db
    except BaseException:
        db.close()
        raise


@contextmanager
def admit_run(path, identity):
    if (not isinstance(identity, tuple) or len(identity) != 2 or
            any(not isinstance(v, str) or not v or len(v) > 2048 for v in identity)):
        raise AdmissionError('Sign in again before starting research.')
    fingerprint = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
    run_id = uuid.uuid4().hex
    try:
        db = _connect(path)
        try:
            db.execute('BEGIN IMMEDIATE')
            now = time.time()
            db.execute('DELETE FROM runs WHERE started <= ? AND active_until <= ?',
                       (now - 86400, now))
            active, own_active = db.execute(
                'SELECT COUNT(*), COALESCE(SUM(identity = ?), 0) '
                'FROM runs WHERE active_until > ?', (fingerprint, now)).fetchone()
            if own_active:
                raise AdmissionError('You already have research running. Wait for it to finish.')
            if active >= 2:
                raise AdmissionError('The service is busy. Please try again shortly.')
            for seconds, personal, total in [(3600, USER_HOURLY, GLOBAL_HOURLY),
                                              (86400, USER_DAILY, GLOBAL_DAILY)]:
                count, own = db.execute(
                    'SELECT COUNT(*), COALESCE(SUM(identity = ?), 0) '
                    'FROM runs WHERE started > ?', (fingerprint, now - seconds)).fetchone()
                if own >= personal:
                    raise AdmissionError('Your research limit has been reached. Try again later.')
                if count >= total:
                    raise AdmissionError('The service research limit has been reached. Try again later.')
            # Crash leases outlast the process watchdog and parent cleanup window.
            db.execute('INSERT INTO runs VALUES (?, ?, ?, ?)',
                       (run_id, fingerprint, now, now + RUN_SECONDS + 60))
            db.execute('COMMIT')
        finally:
            db.close()  # rolls back any rejected admission
    except (OSError, sqlite3.Error):
        raise AdmissionError('Usage checks are unavailable. Research did not start.') from None
    try:
        yield
    finally:
        # Failures still count. A cleanup failure leaves a conservative crash lease.
        try:
            db = _connect(path)
            try:
                db.execute('UPDATE runs SET active_until = 0 WHERE id = ?', (run_id,))
            finally:
                db.close()
        except (OSError, sqlite3.Error):
            pass
