"""Password gate for live research: offline, no provider calls."""
from concurrent.futures import ThreadPoolExecutor
import sqlite3
from threading import Barrier

import pytest

import research_agent.public_access as access

PASSWORD = 'test-only-passphrase-7'


@pytest.fixture(scope='module')
def stored():
    return access.hash_password(PASSWORD)


def test_hash_round_trip_and_wrong_passwords(stored):
    assert access.verify_password(PASSWORD, stored)
    for wrong in ['', 'test-only-passphrase-8', PASSWORD + ' ', PASSWORD.upper(), PASSWORD[:-1]]:
        assert not access.verify_password(wrong, stored)
    assert PASSWORD not in stored


def test_same_password_gets_a_new_salt_each_time(stored):
    again = access.hash_password(PASSWORD)
    assert again != stored
    assert access.verify_password(PASSWORD, again)


@pytest.mark.parametrize('password', ['short-pass', ' ' + PASSWORD, PASSWORD + '\n', 'x' * 257])
def test_hash_rejects_weak_or_ambiguous_passwords(password):
    with pytest.raises(access.PasswordPolicyError):
        access.hash_password(password)


@pytest.mark.parametrize('mutate', [
    lambda s: '',
    lambda s: 'not-a-hash',
    lambda s: s.replace('scrypt', 'sha256', 1),
    lambda s: s.replace('$16384$', '$2$', 1),          # cheap cost refused
    lambda s: s.replace('$16384$', '$1048576$', 1),    # memory-exhausting cost refused
    lambda s: s.rsplit('$', 1)[0] + '$AAAA',           # truncated digest
    lambda s: s + '$extra',
    lambda s: s.replace('$', '$!', 1),
])
def test_malformed_or_tampered_hashes_never_verify(stored, mutate):
    bad = mutate(stored)
    assert not access.is_valid_hash(bad)
    assert not access.verify_password(PASSWORD, bad)


def _config(monkeypatch, **values):
    for name in ['ENABLED', 'PASSWORD_HASH', 'GEMINI_API_KEY', 'TAVILY_API_KEY', 'GEMINI_MODEL']:
        monkeypatch.delenv(f'PUBLIC_LIVE_{name}', raising=False)
    for name, value in values.items():
        monkeypatch.setenv(f'PUBLIC_LIVE_{name.upper()}', value)
    return access.LiveAccessConfig()


def test_config_is_ready_only_when_everything_is_present(monkeypatch, stored):
    complete = dict(enabled='true', password_hash=stored, gemini_api_key='g-key',
                    tavily_api_key='t-key', gemini_model='gemini-test')
    assert _config(monkeypatch, **complete).ready
    for missing in complete:
        partial = {k: v for k, v in complete.items() if k != missing}
        assert not _config(monkeypatch, **partial).ready, missing
    assert not _config(monkeypatch, **{**complete, 'enabled': 'false'}).ready
    assert not _config(monkeypatch, **{**complete, 'password_hash': PASSWORD}).ready
    assert not _config(monkeypatch, **{**complete, 'gemini_api_key': '   '}).ready


def test_config_never_reveals_keys_or_hash(monkeypatch, stored):
    config = _config(monkeypatch, enabled='true', password_hash=stored,
                     gemini_api_key='g-secret-key', tavily_api_key='t-secret-key',
                     gemini_model='gemini-test')
    shown = repr(config) + str(config.model_dump())
    for secret in ['g-secret-key', 't-secret-key', stored]:
        assert secret not in shown


def test_failed_attempts_lock_site_wide_and_expire(tmp_path, monkeypatch, stored):
    clock = [100_000.0]
    monkeypatch.setattr(access.time, 'time', lambda: clock[0])
    db = tmp_path / 'usage.db'
    assert access.check_password(db, PASSWORD, stored)
    for _ in range(access.FAILED_ATTEMPTS_PER_HOUR):
        assert not access.check_password(db, 'wrong-password!', stored)
    # Locked even for the correct password; a new connection (page reload) changes nothing.
    with pytest.raises(access.LoginLockedError):
        access.check_password(db, PASSWORD, stored)
    with pytest.raises(access.LoginLockedError):
        access.check_password(db, 'wrong-password!', stored)
    clock[0] += 3601
    assert access.check_password(db, PASSWORD, stored)
    with sqlite3.connect(db) as connection:
        assert connection.execute('SELECT COUNT(*) FROM login_failures').fetchone()[0] == 0


def test_successful_logins_do_not_count_toward_the_lock(tmp_path, stored):
    db = tmp_path / 'usage.db'
    for _ in range(access.FAILED_ATTEMPTS_PER_HOUR + 5):
        assert access.check_password(db, PASSWORD, stored)


def test_store_holds_only_timestamps(tmp_path, stored):
    db = tmp_path / 'usage.db'
    access.check_password(db, 'a-wrong-guess-123', stored)
    data = db.read_bytes()
    assert b'a-wrong-guess-123' not in data and stored.encode() not in data


def test_concurrent_guesses_cannot_exceed_the_limit(tmp_path, stored):
    db = tmp_path / 'usage.db'
    attempts = access.FAILED_ATTEMPTS_PER_HOUR + 10
    barrier = Barrier(attempts)

    def guess(_):
        barrier.wait()
        try:
            return access.check_password(db, 'wrong-password!', stored)
        except access.LoginLockedError:
            return 'locked'
        except access.LoginUnavailableError:
            return 'unavailable'

    with ThreadPoolExecutor(attempts) as pool:
        outcomes = list(pool.map(guess, range(attempts)))
    with sqlite3.connect(db) as connection:
        recorded = connection.execute('SELECT COUNT(*) FROM login_failures').fetchone()[0]
    assert recorded == outcomes.count(False) <= access.FAILED_ATTEMPTS_PER_HOUR
    assert True not in outcomes


def test_unusable_store_fails_closed(tmp_path, stored):
    blocker = tmp_path / 'not-a-directory'
    blocker.write_text('x')
    with pytest.raises(access.LoginUnavailableError):
        access.check_password(blocker / 'usage.db', PASSWORD, stored)
    corrupt = tmp_path / 'corrupt.db'
    corrupt.write_bytes(b'this is not sqlite' * 100)
    with pytest.raises(access.LoginUnavailableError):
        access.check_password(corrupt, PASSWORD, stored)
