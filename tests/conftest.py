"""Shared test configuration."""

import pytest


@pytest.fixture(autouse=True)
def local_app_opt_in(monkeypatch):
    # app.py serves the local history application only with an explicit local
    # opt-in. Most UI tests drive that application; tests of the fail-closed
    # behavior delete this variable themselves.
    monkeypatch.setenv("RESEARCH_AGENT_LOCAL", "1")
