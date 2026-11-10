"""Construct the Maple agent.

One ADK LlmAgent with three custom tools and a HITL gate.

Note on grounding. Gemini does not allow mixing the built-in
google_search_retrieval grounding with custom function tools in the same
agent call. So Maple cannot have both. We chose custom tools: she can
fetch_vector_url for the authoritative source on Vector pages, draw images,
and book demos. For freshness on people/news, the runbook's optional
"researcher" pattern (a second agent with only google_search and no
functions) can be plugged in later. For the lecture demo, the LLM's own
knowledge of Vector and Canadian AI is enough.
"""

from __future__ import annotations

import uuid
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.models import Gemini
from google.adk.tools import FunctionTool, google_search
from google.adk.tools.agent_tool import AgentTool
from google.genai import types as genai_types

from .config import get_settings
from .logging_setup import get_logger
from .session_store import get_session_store
from .tools import (
    CATALOG,
    submit_order,
)

log = get_logger(__name__)

# Maple is an LLM with a web-search tool. Vector Institute knowledge is
# layered on top by this prompt; she does not pretend to be a website.
SYSTEM_PROMPT = f"""
You are Maple, the assistant for Vector Institute, Canada's leading
institute for advanced AI research, headquartered at MaRS Discovery
District in Toronto. You speak in warm, concrete sentences. Specific
names, real dates, real numbers when you have them. No emojis. No em
dashes.

How you answer:
1. For any factual question, run search_agent first, then answer in one
   concise paragraph that uses the results.
2. If search_agent is unavailable or empty, answer from your own knowledge
   and say openly that the answer may not be current.
3. Vector Institute is your priority topic. Programs (FastLane Applied
   Projects, DARMOD, NextAI, Vector Scholarships in AI), people who work
   there, history since 2017, events, the MaRS location.
4. For general AI or Canadian AI questions, still answer; relate it back
   to Vector if there is a sensible angle.
5. Never invent specific people, dates, or program details. If you don't
   know, search; if search doesn't know, say so.

You also operate the small Vector store. The catalog has exactly four
items; use these ids when calling submit_order:
  vector-hoodie    Vector Institute hoodie ($65 CAD)
  vector-mug       Vector Institute mug ($18 CAD)
  darmod-packet    DARMOD program info packet (free)
  fastlane-packet  FastLane Applied Projects info packet (free)

When the user wants to order: confirm items, get their name and email
(only ask if not already in the conversation), then call submit_order.
The order goes to an admin queue and only ships after manual approval.

Tools available:
    search_agent(query)                 Google Search specialist tool.
  submit_order(items, name, email)  place a Vector store order, gated.

End each response with one short follow-up question.
""".strip()


def _maple_before_tool_callback(*_args: Any, **kwargs: Any) -> Any | None:
    """Handle multiple ADK callback signatures for tool pre-hooks.

    ADK versions call the before_tool_callback with different parameter
    names/orders (positional callback_context + args, or keyword args like
    `tool`, `args`, `tool_context`). Normalize these into `tool_name` and
    `session_id` for the existing HITL logic.
    """
    callback_context = kwargs.get("callback_context")
    args = kwargs.get("args")
    tool = kwargs.get("tool")
    tool_context = kwargs.get("tool_context")

    # Normalize positional forms
    if callback_context is None and _args:
        first = _args[0]
        # If first has .tool and .state it's the callback_context object
        if hasattr(first, "tool") and hasattr(first, "state"):
            callback_context = first
            if args is None and len(_args) > 1:
                args = _args[1]
        else:
            # Could be (tool, args, tool_context)
            if tool is None and hasattr(first, "name"):
                tool = first
            if args is None and len(_args) > 1:
                args = _args[1]
            if tool_context is None and len(_args) > 2:
                tool_context = _args[2]

    # Determine tool name
    tool_name = None
    if callback_context is not None and hasattr(callback_context, "tool"):
        try:
            tool_name = callback_context.tool.name
        except Exception:
            tool_name = None
    if tool_name is None and tool is not None:
        tool_name = getattr(tool, "name", None) or (tool.get("name") if isinstance(tool, dict) else None)

    # Determine session id
    session_id = None
    if callback_context is not None and hasattr(callback_context, "state"):
        session_id = callback_context.state.get("session_id")
    if session_id is None and tool_context is not None and hasattr(tool_context, "state"):
        session_id = tool_context.state.get("session_id")

    # Fallback for args if still none
    if args is None:
        args = {}
    """Intercept HITL-gated tool calls and route them to the approval queue.

    ADK calls this before each tool actually runs. We check whether the tool
    name is in the configured hitl_tools list. If it is, we stash the call in
    the session's pending queue and return a string that tells the model the
    request is awaiting approval. The model then says something like "I have
    queued the booking request for human approval" instead of pretending the
    call succeeded.

    The HTTP layer notices the pending call and surfaces it to the frontend
    as a streaming pause event, where it shows the approve/deny modal.
    """
    settings = get_settings()
    if not tool_name or tool_name not in settings.hitl_tools:
        return None  # let the tool run normally

    if not session_id:
        # Defensive: if we somehow lost the session id, just refuse the call.
        log.warning("hitl.no_session_id", tool=tool_name)
        return (
            "I cannot run this action because I lost track of who I am "
            "talking to. Please refresh and try again."
        )

    call_id = uuid.uuid4().hex
    get_session_store().add_pending_tool_call(
        sid=session_id,
        call_id=call_id,
        tool_name=tool_name,
        arguments=dict(args),
    )
    # If this is an order, also surface it on the admin page.
    if tool_name == "submit_order":
        from .order_store import get_order_store
        items = list(args.get("items") or [])
        total = sum(CATALOG[i]["price_cad"] for i in items if i in CATALOG)
        get_order_store().queue(
            session_id=session_id,
            call_id=call_id,
            items=items,
            name=str(args.get("name") or ""),
            email=str(args.get("email") or ""),
            total_cad=total,
        )
    log.info(
        "hitl.queued",
        session_id=session_id,
        call_id=call_id,
        tool=tool_name,
        args=args,
    )
    # Returning a non-None value short-circuits the tool. ADK passes our
    # return value back to the model as the tool's "result".
    return (
        f"PENDING_HUMAN_APPROVAL: call_id={call_id}. The order has been "
        f"queued for an admin to review. Tell the customer their order is "
        f"awaiting human approval and will not ship until a Vector admin "
        f"clicks approve."
    )


def _brain() -> dict[str, Any]:
    """The model settings every Maple agent shares.

    Three things keep a reply fast and a stuck call from hanging the page:
    thinking is off by default, a 429 from Vertex is retried a few times with
    a growing wait, and a call that hangs is cut off after 30 seconds instead
    of being waited on for minutes.
    """
    settings = get_settings()
    if not settings.chat_model.startswith("gemini"):
        return {"model": settings.chat_model}
    return {
        "model": Gemini(
            model=settings.chat_model,
            retry_options=genai_types.HttpRetryOptions(attempts=4, initial_delay=2),
        ),
        "generate_content_config": genai_types.GenerateContentConfig(
            thinking_config=genai_types.ThinkingConfig(thinking_budget=settings.thinking_budget),
            http_options=genai_types.HttpOptions(timeout=30_000),
        ),
    }


def build_maple_agent() -> LlmAgent:
    """Build the agent. One per process is enough; ADK is thread-safe enough
    for the lecture demo scale. For higher fanout, build per-session.
    """
    settings = get_settings()

    search_agent = LlmAgent(
        **_brain(),
        name="search_agent",
        instruction="You're a specialist in Google Search.",
        tools=[google_search],
    )
    search_tool = AgentTool(search_agent)

    tools = [
        search_tool,
        FunctionTool(func=submit_order),
    ]
    # Google search now runs through a dedicated sub-agent via AgentTool,
    # while submit_order remains a HITL-gated function tool.

    return LlmAgent(
        **_brain(),
        name="maple",
        instruction=SYSTEM_PROMPT,
        tools=tools,
        before_tool_callback=_maple_before_tool_callback,
    )


def _run_async_tool(async_fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Run an async tool from sync ADK callbacks safely.

    `asyncio.run(...)` fails if called while an event loop is already running.
    When that happens, run the coroutine in a short-lived worker thread with
    its own event loop.
    """
    import asyncio
    import threading

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(async_fn(*args, **kwargs))

    out: dict[str, Any] = {}
    err: dict[str, Exception] = {}

    def _worker() -> None:
        try:
            out["value"] = asyncio.run(async_fn(*args, **kwargs))
        except Exception as exc:  # noqa: BLE001
            err["exc"] = exc

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join()

    if "exc" in err:
        raise err["exc"]
    return out.get("value")

