"""Password gate for live research on the operator's own provider keys.

The operator chooses one password and stores only its salted scrypt hash in
the hosted secrets. A human-chosen password can be guessed far more easily
than a random token, so two protections apply:

- scrypt makes every offline guess slow if the stored hash ever leaks;
- failed attempts are counted site-wide in the shared SQLite usage store, so
  reloading the page or opening new sessions does not reset the limit.

The lock check runs before the password is verified: once locked, even a
correct guess is refused until the hour has passed.

Configuration fails closed. Live access is offered only when it is explicitly
enabled AND a well-formed hash, both provider keys and a model are present.
"""
import base64
import hashlib
import hmac
import secrets
import sqlite3
import time
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

MIN_PASSWORD_CHARS = 12
MAX_PASSWORD_CHARS = 256
FAILED_ATTEMPTS_PER_HOUR = 10

# scrypt cost: about 16 MiB of memory and tens of milliseconds per check.
SCRYPT_N, SCRYPT_R, SCRYPT_P = 2 ** 14, 8, 1
SCRYPT_MAXMEM = 64 * 1024 * 1024
SALT_BYTES = 16
HASH_BYTES = 32
_PREFIX = 'scrypt'


class PasswordPolicyError(ValueError):
    pass


class LoginLockedError(RuntimeError):
    pass


class LoginUnavailableError(RuntimeError):
    pass


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode('ascii')


def _derive(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    return hashlib.scrypt(password.encode('utf-8'), salt=salt, n=n, r=r, p=p,
                          maxmem=SCRYPT_MAXMEM, dklen=HASH_BYTES)


def hash_password(password: str) -> str:
    """Return 'scrypt$n$r$p$salt$hash' for the operator's chosen password.

    Length is enforced here, when the operator creates the hash, because the
    application can never learn the password's length from the hash itself.
    """
    if not isinstance(password, str) or password != password.strip():
        raise PasswordPolicyError('The password must not start or end with spaces.')
    if not MIN_PASSWORD_CHARS <= len(password) <= MAX_PASSWORD_CHARS:
        raise PasswordPolicyError(
            f'Use between {MIN_PASSWORD_CHARS} and {MAX_PASSWORD_CHARS} characters.')
    salt = secrets.token_bytes(SALT_BYTES)
    digest = _derive(password, salt, SCRYPT_N, SCRYPT_R, SCRYPT_P)
    return '$'.join([_PREFIX, str(SCRYPT_N), str(SCRYPT_R), str(SCRYPT_P), _b64(salt), _b64(digest)])


def _parse(stored: str):
    """Return (salt, digest, n, r, p) or None for anything not produced by hash_password."""
    parts = stored.split('$') if isinstance(stored, str) else []
    if len(parts) != 6 or parts[0] != _PREFIX:
        return None
    try:
        n, r, p = (int(value) for value in parts[1:4])
        salt = base64.urlsafe_b64decode(parts[4].encode('ascii'))
        digest = base64.urlsafe_b64decode(parts[5].encode('ascii'))
    except (ValueError, UnicodeEncodeError):
        return None
    # Accept only the cost this module writes, so a tampered secret cannot make
    # every login attempt allocate excessive memory or become trivially cheap.
    if (n, r, p) != (SCRYPT_N, SCRYPT_R, SCRYPT_P):
        return None
    if len(salt) != SALT_BYTES or len(digest) != HASH_BYTES:
        return None
    return salt, digest, n, r, p


def is_valid_hash(stored: str) -> bool:
    return _parse(stored) is not None


def verify_password(password: str, stored: str) -> bool:
    parsed = _parse(stored)
    if parsed is None or not isinstance(password, str):
        return False
    if not 1 <= len(password) <= MAX_PASSWORD_CHARS:
        return False
    salt, digest, n, r, p = parsed
    # Constant-time comparison: timing reveals nothing about a near miss.
    return hmac.compare_digest(_derive(password, salt, n, r, p), digest)


class LiveAccessConfig(BaseSettings):
    """Hosted secrets for password-gated live research (PUBLIC_LIVE_*)."""
    model_config = SettingsConfigDict(env_prefix='PUBLIC_LIVE_', env_file=None, extra='ignore')
    enabled: bool = False
    password_hash: SecretStr = SecretStr('')
    gemini_api_key: SecretStr = SecretStr('')
    tavily_api_key: SecretStr = SecretStr('')
    gemini_model: str = Field(default='', max_length=150)

    @property
    def ready(self) -> bool:
        return (self.enabled
                and is_valid_hash(self.password_hash.get_secret_value())
                and bool(self.gemini_api_key.get_secret_value().strip())
                and bool(self.tavily_api_key.get_secret_value().strip())
                and bool(self.gemini_model.strip()))


def _connect(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=3, isolation_level=None)
    try:
        db.execute('CREATE TABLE IF NOT EXISTS login_failures (at REAL NOT NULL)')
        return db
    except BaseException:
        db.close()
        raise


def check_password(path, password: str, stored: str) -> bool:
    """Verify one login attempt under the site-wide failed-attempt limit.

    Returns True or False for the password. Raises LoginLockedError when the
    limit is reached and LoginUnavailableError when the store cannot be used:
    without a working counter, guessing would be unlimited, so access is denied.
    """
    try:
        db = _connect(path)
        try:
            # One writer at a time, so concurrent guesses cannot all slip under the limit.
            db.execute('BEGIN IMMEDIATE')
            now = time.time()
            db.execute('DELETE FROM login_failures WHERE at <= ?', (now - 3600,))
            (failures,) = db.execute('SELECT COUNT(*) FROM login_failures').fetchone()
            if failures >= FAILED_ATTEMPTS_PER_HOUR:
                raise LoginLockedError(
                    'Too many incorrect passwords. Live access is paused for up to an hour. '
                    'The offline demo is still available.')
            accepted = verify_password(password, stored)
            if not accepted:
                db.execute('INSERT INTO login_failures VALUES (?)', (now,))
            db.execute('COMMIT')
            return accepted
        finally:
            db.close()  # rolls back if anything above failed
    except (OSError, sqlite3.Error):
        raise LoginUnavailableError('Live access is unavailable right now.') from None
