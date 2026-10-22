"""Offline checks that the deployed pins and pyproject.toml cannot drift apart.

requirements.txt is what Streamlit Community Cloud installs; pyproject.toml holds
the allowed ranges. No network access or package installation is involved.
"""
from pathlib import Path
import tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _pins():
    pins = {}
    for number, raw in enumerate((ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, version = line.partition("==")
        assert separator and name and version, f"line {number} is not an exact pin: {line!r}"
        assert not any(mark in line for mark in "<>~!;[@ "), f"line {number} is not a plain name==version pin: {line!r}"
        Version(version)  # raises if the version is not valid
        key = canonicalize_name(name)
        assert key not in pins, f"{name} is pinned twice"
        pins[key] = version
    return pins


def _direct_requirements():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    texts = list(project["dependencies"]) + list(project["optional-dependencies"]["public"])
    return [Requirement(text) for text in texts]


def test_requirements_are_exact_pins_without_duplicates():
    assert len(_pins()) > 0


@pytest.mark.parametrize("requirement", _direct_requirements(), ids=lambda r: r.name)
def test_every_direct_dependency_is_pinned_within_its_allowed_range(requirement):
    pins = _pins()
    key = canonicalize_name(requirement.name)
    assert key in pins, f"{requirement.name} is in pyproject.toml but not pinned in requirements.txt"
    assert requirement.specifier.contains(Version(pins[key]), prereleases=True), (
        f"{requirement.name}=={pins[key]} is outside the allowed range {requirement.specifier}"
    )


def test_google_sign_in_support_is_pinned():
    # public_app.py uses st.login(), which needs Streamlit's auth extra (Authlib).
    # The extra is declared in pyproject.toml but not visible in a flat pin list.
    assert "authlib" in _pins()


def test_development_only_packages_are_not_deployed():
    pins = _pins()
    for name in ("pytest", "pytest-cov", "setuptools"):
        assert name not in pins
