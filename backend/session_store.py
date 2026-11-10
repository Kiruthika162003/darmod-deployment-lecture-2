"""Per-student session state, in-memory.

When a student pastes their key into Maple's heart, we validate it and stash
it against a signed session cookie. The cookie itself never carries the key,
only an opaque session id. The key lives in this process's memory and never
hits disk or logs.

For a real production deployment with multiple replicas, swap this for Redis
or Memorystore. The interface is small enough that the swap is local. The
in-memory version is correct for a single-replica lecture demo.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from threading import RLock

from .config import get_settings


@dataclass
class Session:
    """One student's awakened session."""

    session_id: str
    api_key: str
    backend: str  # "ai_studio" or "vertex"
    awakened_at: float
    last_seen_at: float
    pending_tool_calls: dict[str, dict] = field(default_factory=dict)


class SessionStore:
    """Thread-safe in-memory session store with TTL eviction."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = RLock()
        self._ttl_seconds = get_settings().session_ttl_minutes * 60

    def create(self, api_key: str, backend: str) -> Session:
        sid = secrets.token_urlsafe(24)
        now = time.time()
        sess = Session(
            session_id=sid,
            api_key=api_key,
            backend=backend,
            awakened_at=now,
            last_seen_at=now,
        )
        with self._lock:
            self._sessions[sid] = sess
            self._evict_expired_locked()
        return sess

    def get(self, sid: str) -> Session | None:
        with self._lock:
            sess = self._sessions.get(sid)
            if sess is None:
                return None
            if time.time() - sess.last_seen_at > self._ttl_seconds:
                self._sessions.pop(sid, None)
                return None
            sess.last_seen_at = time.time()
            return sess

    def remove(self, sid: str) -> None:
        with self._lock:
            self._sessions.pop(sid, None)

    def add_pending_tool_call(
        self, sid: str, call_id: str, tool_name: str, arguments: dict
    ) -> None:
        with self._lock:
            sess = self._sessions.get(sid)
            if sess is None:
                return
            sess.pending_tool_calls[call_id] = {
                "tool_name": tool_name,
                "arguments": arguments,
                "queued_at": time.time(),
            }

    def pop_pending_tool_call(self, sid: str, call_id: str) -> dict | None:
        with self._lock:
            sess = self._sessions.get(sid)
            if sess is None:
                return None
            return sess.pending_tool_calls.pop(call_id, None)

    def _evict_expired_locked(self) -> None:
        cutoff = time.time() - self._ttl_seconds
        expired = [sid for sid, s in self._sessions.items() if s.last_seen_at < cutoff]
        for sid in expired:
            self._sessions.pop(sid, None)


# Module-level singleton. FastAPI dependency injection wraps it.
_STORE = SessionStore()


def get_session_store() -> SessionStore:
    return _STORE
