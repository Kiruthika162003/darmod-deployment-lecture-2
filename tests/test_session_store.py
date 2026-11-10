"""Session store basics. No google-genai involved."""

from __future__ import annotations

import time

from backend.session_store import SessionStore


def test_create_and_get_roundtrip():
    store = SessionStore()
    sess = store.create(api_key="AIzaXXX", backend="ai_studio")
    fetched = store.get(sess.session_id)
    assert fetched is not None
    assert fetched.api_key == "AIzaXXX"
    assert fetched.backend == "ai_studio"


def test_get_unknown_returns_none():
    store = SessionStore()
    assert store.get("nope") is None


def test_pending_tool_call_lifecycle():
    store = SessionStore()
    sess = store.create(api_key="AIzaXXX", backend="ai_studio")
    store.add_pending_tool_call(sess.session_id, "call-1", "book_demo", {"name": "Sam"})
    popped = store.pop_pending_tool_call(sess.session_id, "call-1")
    assert popped is not None
    assert popped["tool_name"] == "book_demo"
    assert popped["arguments"]["name"] == "Sam"
    # second pop returns None
    assert store.pop_pending_tool_call(sess.session_id, "call-1") is None


def test_remove():
    store = SessionStore()
    sess = store.create("k", "ai_studio")
    store.remove(sess.session_id)
    assert store.get(sess.session_id) is None


def test_ttl_eviction(monkeypatch):
    store = SessionStore()
    store._ttl_seconds = 1  # short for the test
    sess = store.create("k", "ai_studio")
    # Roll the session's last_seen_at into the past
    sess.last_seen_at = time.time() - 5
    assert store.get(sess.session_id) is None
