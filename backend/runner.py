"""Thin wrapper around ADK's Runner that knows about Maple's sessions.

Each browser session (cookie-based) gets its own ADK session under the hood.
This keeps conversation history per-student, so two students hitting the same
Cloud Run instance don't see each other's chats.

For streaming, we yield text deltas as the model emits them. For non-streaming,
we accumulate and return.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from .agent import build_maple_agent
from .config import ai_studio_auth
from .logging_setup import get_logger
from .session_store import get_session_store

log = get_logger(__name__)

_APP_NAME = "maple"
_AGENT = None  # built lazily on first request
_SESSION_SERVICE = InMemorySessionService()
_RUNNER: Runner | None = None
_BOOT = time.monotonic()


def is_cold() -> bool:
    """True until the first agent turn has built the runner.

    On Cloud Run that first turn is the one that pays for a cold start: a new
    container has just booted and the agent and model client are built while
    the user waits.
    """
    return _RUNNER is None


def instance_age_s() -> int:
    """Seconds since this process started."""
    return int(time.monotonic() - _BOOT)


def _get_runner() -> Runner:
    global _AGENT, _RUNNER
    if _RUNNER is None:
        _AGENT = build_maple_agent()
        _RUNNER = Runner(
            agent=_AGENT,
            app_name=_APP_NAME,
            session_service=_SESSION_SERVICE,
        )
    return _RUNNER


async def _ensure_session(session_id: str) -> None:
    """Create the ADK session if it doesn't exist yet."""
    existing = await _SESSION_SERVICE.get_session(
        app_name=_APP_NAME, user_id=session_id, session_id=session_id
    )
    if existing is None:
        await _SESSION_SERVICE.create_session(
            app_name=_APP_NAME,
            user_id=session_id,
            session_id=session_id,
            state={"session_id": session_id},
        )


_NO_SEARCH_HINT = (
    "(For this question, answer from your own knowledge without calling "
    "search_agent.)\n\n"
)


def _prompt_for(prompt: str, use_search: bool) -> str:
    """Maple searches the web before answering a factual question. Turning
    search off saves her the extra model calls, at the cost of fresh facts."""
    return prompt if use_search else _NO_SEARCH_HINT + prompt


class _Trace:
    """What one agent turn did, step by step, so the page can show it.

    A step's `ms` is the time since the previous step: how long the model took
    to decide on a tool, how long the tool took, how long the answer took.
    """

    def __init__(self, started: float) -> None:
        self._mark = started
        self.steps: list[dict[str, Any]] = []
        self.model_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.thinking_tokens = 0

    def observe(self, event: Any) -> list[dict[str, Any]]:
        """Fold one ADK event in and return the steps it added."""
        if getattr(event, "partial", False):
            return []
        usage = getattr(event, "usage_metadata", None)
        if usage is not None:
            self.model_calls += 1
            self.input_tokens += usage.prompt_token_count or 0
            self.output_tokens += usage.candidates_token_count or 0
            self.thinking_tokens += getattr(usage, "thoughts_token_count", None) or 0
        now = time.monotonic()
        ms = int((now - self._mark) * 1000)
        added: list[dict[str, Any]] = []
        for call in event.get_function_calls():
            detail = str(dict(call.args or {}))[:160]
            added.append({"kind": "decide", "name": call.name, "ms": ms, "detail": detail})
        for result in event.get_function_responses():
            added.append({"kind": "tool", "name": result.name, "ms": ms, "detail": ""})
        if event.is_final_response() and event.content and event.content.parts:
            added.append({"kind": "answer", "name": "", "ms": ms, "detail": ""})
        if added:
            self._mark = now
            self.steps.extend(added)
        return added

    def usage(self) -> dict[str, int]:
        return {
            "model_calls": self.model_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "thinking_tokens": self.thinking_tokens,
        }


async def run_turn(session_id: str, prompt: str, use_search: bool = True) -> dict[str, Any]:
    """Drive one agent turn. Returns reply text, citations, used tool names.

    A 429 or 503 from Gemini is retried inside the model client (see agent._brain).
    """
    cold = is_cold()
    runner = _get_runner()
    await _ensure_session(session_id)

    session = get_session_store().get(session_id)
    started = time.monotonic()
    with ai_studio_auth(session.api_key if session else None):
        result = await _drive_once(runner, session_id, _prompt_for(prompt, use_search), started)
    result["cold_start"] = cold
    result["instance_age_s"] = instance_age_s()
    return result


async def _drive_once(
    runner: Runner, session_id: str, prompt: str, started: float
) -> dict[str, Any]:
    content = genai_types.Content(role="user", parts=[genai_types.Part(text=prompt)])
    reply_parts: list[str] = []
    used_tools: list[str] = []
    citations: list[dict[str, str]] = []
    trace = _Trace(started)

    async for event in runner.run_async(
        user_id=session_id, session_id=session_id, new_message=content
    ):
        # Tool-call observation
        used_tools.extend(call.name for call in event.get_function_calls())
        trace.observe(event)

        # Grounding citations on the final event
        grounding = getattr(event, "grounding_metadata", None)
        if grounding is not None:
            for attribution in getattr(grounding, "grounding_attributions", []) or []:
                web = getattr(attribution, "web", None)
                if web is not None and getattr(web, "uri", None):
                    citations.append({
                        "url": web.uri,
                        "title": getattr(web, "title", None) or "",
                    })

        if event.is_final_response() and event.content and event.content.parts:
            reply_parts.append(
                "".join(p.text or "" for p in event.content.parts)
            )

    latency_ms = int((time.monotonic() - started) * 1000)
    log.info(
        "agent.turn.done",
        session_id=session_id,
        latency_ms=latency_ms,
        used_tools=used_tools,
        n_citations=len(citations),
    )
    return {
        "reply": "".join(reply_parts) or "I did not get that. Try again?",
        "citations": citations,
        "used_tools": used_tools,
        "latency_ms": latency_ms,
        "steps": trace.steps,
        "usage": trace.usage(),
    }


async def run_turn_streaming(
    session_id: str, prompt: str, use_search: bool = True
) -> AsyncIterator[dict[str, Any]]:
    """Stream agent events. Each yield is a dict the SSE layer serializes.

    Event shapes:
        {"type": "token", "text": "..."}             incremental text
        {"type": "tool_call", "name": "..."}         tool started
        {"type": "step", "kind": "decide|tool|answer", "name": "...",
         "ms": N, "detail": "..."}                   one step of the turn
        {"type": "awaiting_approval", "call_id":...,
         "tool_name": "...", "arguments": {...}}     HITL pause
        {"type": "citation", "url": "...", "title": "..."}
        {"type": "done", "latency_ms": N, "usage": {...},
         "cold_start": bool, "instance_age_s": N}    final marker
    """
    cold = is_cold()
    runner = _get_runner()
    await _ensure_session(session_id)
    started = time.monotonic()

    content = genai_types.Content(
        role="user", parts=[genai_types.Part(text=_prompt_for(prompt, use_search))]
    )
    trace = _Trace(started)

    # Enable SSE streaming so partial events arrive token-by-token from the
    # model. Without this, ADK yields one event per complete part, which the
    # browser sees as the full reply in a single chunk.
    stream_cfg = RunConfig(streaming_mode=StreamingMode.SSE)

    async for event in runner.run_async(
        user_id=session_id, session_id=session_id, new_message=content,
        run_config=stream_cfg,
    ):
        # Stream tool-call notices so the UI can animate Maple's mouth and
        # show "Maple is searching the Vector site..." style hints.
        for call in event.get_function_calls():
            yield {"type": "tool_call", "name": call.name}
            # If a HITL-gated tool returned a PENDING marker, surface it
            # to the frontend as an awaiting-approval event.
            if call.name in ("submit_order",):
                yield {
                    "type": "awaiting_approval",
                    "call_id": _extract_call_id_from_event(event),
                    "tool_name": call.name,
                    "arguments": dict(getattr(call, "args", {}) or {}),
                }
        for step in trace.observe(event):
            yield {"type": "step", **step}

        # SSE mode emits a stream of `partial=True` events with text deltas
        # and one final non-partial event containing the cumulative text.
        # Only yield text from partials, so the browser doesn't get the full
        # reply duplicated at the end.
        if event.content and event.content.parts and getattr(event, "partial", False):
            for p in event.content.parts:
                if p.text:
                    yield {"type": "token", "text": p.text}

        grounding = getattr(event, "grounding_metadata", None)
        if grounding is not None:
            for attribution in getattr(grounding, "grounding_attributions", []) or []:
                web = getattr(attribution, "web", None)
                if web is not None and getattr(web, "uri", None):
                    yield {
                        "type": "citation",
                        "url": web.uri,
                        "title": getattr(web, "title", None) or "",
                    }

    yield {
        "type": "done",
        "latency_ms": int((time.monotonic() - started) * 1000),
        "usage": trace.usage(),
        "cold_start": cold,
        "instance_age_s": instance_age_s(),
    }


def _extract_call_id_from_event(event: Any) -> str:
    """Pull a call_id out of an event's function_response if our callback
    embedded one in its PENDING_HUMAN_APPROVAL string. Best-effort.
    """
    try:
        responses = getattr(event, "function_responses", []) or []
        for r in responses:
            text = str(getattr(r, "response", "") or "")
            if "call_id=" in text:
                return text.split("call_id=", 1)[1].split()[0].strip()
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


async def resume_after_approval(
    session_id: str, call_id: str, approved: bool
) -> dict[str, Any]:
    """Resume the conversation after a HITL approval/denial.

    We send a synthetic user-side message back to the agent that tells it
    what the human decided. The agent then either runs the tool (approved)
    or apologizes and asks what to do instead (denied).
    """
    runner = _get_runner()
    await _ensure_session(session_id)

    if approved:
        # On approval, we re-call the real tool function ourselves (bypassing
        # the HITL gate this time) and feed the result back to the agent.
        from .session_store import get_session_store
        from .tools import submit_order

        pending = get_session_store().pop_pending_tool_call(session_id, call_id)
        if pending is None:
            return {"ok": False, "reason": "no_such_pending_call"}
        result = submit_order(**pending["arguments"])
        synthetic = (
            f"HUMAN_APPROVED_AND_RAN_TOOL: {pending['tool_name']} "
            f"returned: {result}"
        )
    else:
        synthetic = (
            f"HUMAN_DENIED_TOOL_CALL: call_id={call_id}. "
            f"The admin declined this order. Apologize, ask what to "
            f"change, and offer to resubmit."
        )

    return await run_turn(session_id, synthetic)
