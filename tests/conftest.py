"""Pytest fixtures shared by all test modules.

Two principles:

1. Tests never call Google. Anything that would hit the network is mocked.
2. Each test gets a fresh SessionStore so they cannot bleed state.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Make `from backend import ...` work when running pytest from the app root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Force a known config before any backend module is imported.
os.environ.setdefault("ENVIRONMENT", "dev")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("REQUIRE_API_KEY", "FALSE")
os.environ.setdefault("SESSION_SECRET", "test-secret-only")


@pytest.fixture(autouse=True)
def fresh_session_store():
    """Reset the singleton store between tests."""
    from backend import session_store

    session_store._STORE = session_store.SessionStore()
    yield
