"""Create the hash for the live-research password (PUBLIC_LIVE_PASSWORD_HASH).

Run it in VS Code with "Run Python File", or from the repo folder:

    python scripts/make_password_hash.py

The password is typed hidden and never saved anywhere. Only the hash is
printed; paste that line into the Streamlit Cloud app's Secrets.
"""
import getpass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from research_agent.public_access import PasswordPolicyError, hash_password  # noqa: E402


def main() -> int:
    password = getpass.getpass('New password (typing is hidden): ')
    if password != getpass.getpass('Type it again: '):
        print('The two entries did not match. Nothing was created.')
        return 1
    try:
        stored = hash_password(password)
    except PasswordPolicyError as exc:
        print(f'{exc} Nothing was created.')
        return 1
    print('\nPaste this line into Streamlit Cloud > your app > Settings > Secrets:\n')
    print(f'PUBLIC_LIVE_PASSWORD_HASH = "{stored}"')
    return 0


if __name__ == '__main__':
    sys.exit(main())
