"""FastAPI surface for Maple.

Endpoints:

  POST /awaken         Validate a student's API key, create a session.
  GET  /me             Who am I, what backend, what model?
  POST /chat           Online inference. One request, one reply.
  POST /chat/batch     Batch inference. List of prompts, list of replies.
  POST /chat/stream    Streaming inference. SSE tokens.
  POST /chat/compare   Run prompt against Flash and Pro in parallel.
  POST /tool/approve   Human approval response for a HITL-gated tool call.
  GET  /catalog        The four-item Vector store catalog.
  GET  /admin/orders   Admin: list orders (token-gated).
  POST /admin/orders/{id}/approve  Admin: approve a pending order.
  POST /admin/orders/{id}/deny     Admin: deny a pending order.
  GET  /healthz        Liveness probe.

Static frontend (HTML/CSS/JS for Maple's UI) is served from /static under the
mount point /. So a GET / returns the index page.

Sessions are signed cookies. The API key never leaves the server; only an
opaque session id is exchanged with the browser.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, AsyncIterator

import structlog
from fastapi import Cookie, Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from itsdangerous import BadSignature, URLSafeSerializer
from structlog.contextvars import bind_contextvars, clear_contextvars

from . import __version__
from .config import ai_studio_auth, detect_backend, get_settings
from .logging_setup import configure_logging, get_logger
from .models import (
    ApprovalRequest,
    ApprovalResponse,
    AwakenRequest,
    AwakenResponse,
    BatchItem,
    BatchRequest,
    BatchResponse,
    ChatRequest,
    ChatResponse,
    Citation,
    CompareReply,
    CompareRequest,
    CompareResponse,
    HealthResponse,
)
from .runner import (
    instance_age_s,
    is_cold,
    resume_after_approval,
    run_turn,
    run_turn_streaming,
)
from .lab import router as lab_router
from .lab import warm as warm_lab
from .session_store import SessionStore, get_session_store

log = get_logger(__name__)

# Hold a reference to fire-and-forget tasks, or the event loop may drop them
# before they finish.
_BACKGROUND: set[asyncio.Task] = set()


def _spawn(coro) -> None:
    task = asyncio.create_task(coro)
    _BACKGROUND.add(task)
    task.add_done_callback(_BACKGROUND.discard)


async def _resume_and_remember(order_id: str, session_id: str, call_id: str, approved: bool) -> None:
    """Resume the customer's chat after an admin decision and keep Maple's
    reply on the order, so the page can show it in the chat."""
    from .order_store import get_order_store

    try:
        result = await resume_after_approval(session_id, call_id, approved)
    except Exception as exc:  # noqa: BLE001
        log.warning("order.resume_failed", order_id=order_id, error=str(exc))
        return
    reply = result.get("reply") if isinstance(result, dict) else None
    if reply:
        get_order_store().set_confirmation(order_id, reply)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    log.info(
        "maple.start",
        version=__version__,
        backend=detect_backend(),
        model=get_settings().chat_model,
        region=get_settings().deployment_region,
    )
    if os.environ.get("LAB_WARM", "1") != "0":
        # Load the heavy libraries and train the lab's small models now, so the
        # first person to open a demo is not the one who waits for them.
        threading.Thread(target=warm_lab, daemon=True).start()
    yield
    log.info("maple.stop")


app = FastAPI(
    title="Maple",
    version=__version__,
    description="DARMOD program demo. A Gemini-3-grounded agent for Vector Institute and Canadian AI.",
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session cookies are signed (not encrypted). The cookie carries an opaque id
# only; the api_key lives in the SessionStore in memory.
_signer = URLSafeSerializer(get_settings().session_secret, salt="maple-cookie")


# --- Request-scoped helpers ------------------------------------------------

@app.middleware("http")
async def request_context(request: Request, call_next):
    """Bind request_id and (when present) session_id into structlog context."""
    request_id = uuid.uuid4().hex[:12]
    bind_contextvars(request_id=request_id)
    sid = _read_session_id(request.cookies.get(get_settings().session_cookie_name))
    if sid:
        bind_contextvars(session_id=sid)
    try:
        response = await call_next(request)
    finally:
        clear_contextvars()
    response.headers["X-Request-ID"] = request_id
    return response


def _read_session_id(raw: str | None) -> str | None:
    if not raw:
        return None
    try:
        return _signer.loads(raw)
    except BadSignature:
        return None


def _write_session_cookie(response: Response, session_id: str) -> None:
    response.set_cookie(
        key=get_settings().session_cookie_name,
        value=_signer.dumps(session_id),
        httponly=True,
        samesite="lax",
        secure=get_settings().environment != "dev",
        max_age=get_settings().session_ttl_minutes * 60,
    )


def _require_session(
    request: Request,
    response: Response,
    sid_cookie: Annotated[str | None, Cookie(alias=get_settings().session_cookie_name)] = None,
    store: SessionStore = Depends(get_session_store),
):
    """FastAPI dependency: resolve the caller's session.

    In API-key mode this 401s unauthenticated callers. In lecturer/recording
    mode (require_api_key=False), it reads the session cookie if present;
    otherwise it creates ONE persistent session per browser and writes the
    cookie back so subsequent turns reuse it. This is what makes the ADK
    conversation history work - same session id every turn.
    """
    if not get_settings().require_api_key:
        sid = _read_session_id(sid_cookie)
        sess = store.get(sid) if sid else None
        if sess:
            return sess
        sess = store.create(api_key="", backend="vertex")
        _write_session_cookie(response, sess.session_id)
        return sess
    sid = _read_session_id(sid_cookie)
    if not sid:
        raise HTTPException(401, "Maple is asleep. Insert your API key.")
    sess = store.get(sid)
    if not sess:
        raise HTTPException(401, "Session expired. Re-insert your API key.")
    return sess


# --- Endpoints -------------------------------------------------------------

@app.get("/healthz", response_model=HealthResponse)
async def healthz():
    return HealthResponse(
        backend=detect_backend(),
        model=get_settings().chat_model,
        region=get_settings().deployment_region,
        version=__version__,
    )


@app.post("/awaken", response_model=AwakenResponse)
async def awaken(
    body: AwakenRequest,
    response: Response,
    store: SessionStore = Depends(get_session_store),
):
    """Validate the student's AI Studio key with one cheap call.

    On success, store the key against a fresh session id, set a signed
    cookie, return OK so the frontend can wake Maple up. On failure, we
    return a short reason and keep Maple asleep.
    """
    key = body.api_key.strip()
    if not re.fullmatch(r"[A-Za-z0-9._-]{20,}", key):
        return AwakenResponse(ok=False, message="That does not look like a Google AI Studio key.")
    try:
        with ai_studio_auth(key):
            await _validate_key(key)
    except Exception as exc:  # noqa: BLE001
        log.warning("awaken.invalid_key", error=str(exc))
        return AwakenResponse(ok=False, message=f"Google rejected the key: {exc}")

    sess = store.create(api_key=key, backend="ai_studio")
    _write_session_cookie(response, sess.session_id)
    log.info("awaken.ok", session_id=sess.session_id, backend="ai_studio")
    return AwakenResponse(
        ok=True,
        message="Maple is awake.",
        backend="ai_studio",
        model=get_settings().chat_model,
    )


async def _validate_key(api_key: str) -> None:
    """Make one five-token call to verify the key works."""
    from google import genai

    def _go():
        with ai_studio_auth(api_key):
            client = genai.Client(api_key=api_key)
            client.models.generate_content(
                model=get_settings().chat_model,
                contents="Say OK.",
            )

    await asyncio.to_thread(_go)


@app.get("/me")
async def me(sess=Depends(_require_session)):
    return {
        "session_id": sess.session_id,
        "backend": sess.backend,
        "model": get_settings().chat_model,
        "region": get_settings().deployment_region,
        "pro_model": get_settings().pro_model,
        "awakened_at": sess.awakened_at,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, sess=Depends(_require_session)):
    """Online inference. One request, one reply."""
    result = await run_turn(sess.session_id, body.prompt, body.use_search)
    return ChatResponse(
        reply=result["reply"],
        citations=[Citation(**c) for c in result["citations"]],
        used_tools=result["used_tools"],
        latency_ms=result["latency_ms"],
        cold_start=result.get("cold_start", False),
        instance_age_s=result.get("instance_age_s", 0),
        steps=result.get("steps", []),
        usage=result.get("usage", {}),
    )


@app.post("/chat/batch", response_model=BatchResponse)
async def chat_batch(body: BatchRequest, sess=Depends(_require_session)):
    """Batch inference. List of prompts, list of replies.

    Runs the prompts sequentially against one session so the agent has a
    coherent context. For the lecture demo this is the right behavior: the
    audience sees Maple chew through ten things in a row, not ten Maples in
    parallel. For real batch jobs you would fan out across sessions.
    """
    items: list[BatchItem] = []
    cold = is_cold()
    started = time.monotonic()
    for p in body.prompts:
        t0 = time.monotonic()
        try:
            r = await run_turn(sess.session_id, p, body.use_search)
            items.append(BatchItem(
                prompt=p, reply=r["reply"], latency_ms=r["latency_ms"]
            ))
        except Exception as exc:  # noqa: BLE001
            items.append(BatchItem(
                prompt=p, reply="", error=str(exc),
                latency_ms=int((time.monotonic() - t0) * 1000),
            ))
    return BatchResponse(
        items=items,
        total_latency_ms=int((time.monotonic() - started) * 1000),
        cold_start=cold,
        instance_age_s=instance_age_s(),
    )


@app.post("/chat/stream")
async def chat_stream(body: ChatRequest, sess=Depends(_require_session)):
    """Streaming inference via Server-Sent Events.

    Each event is one JSON object on a `data:` line. Event types:
      token, tool_call, awaiting_approval, citation, done.
    """
    async def event_gen() -> AsyncIterator[bytes]:
        try:
            async for ev in run_turn_streaming(sess.session_id, body.prompt, body.use_search):
                line = f"data: {json.dumps(ev)}\n\n"
                yield line.encode("utf-8")
        except Exception as exc:  # noqa: BLE001
            err = {"type": "error", "message": str(exc)}
            yield f"data: {json.dumps(err)}\n\n".encode("utf-8")

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/chat/compare", response_model=CompareResponse)
async def chat_compare(body: CompareRequest, sess=Depends(_require_session)):
    """Fire the same prompt at the chat model AND the pro model in
    parallel. Return both replies with latencies, token counts and a cost.

    Bypasses the ADK runner because this endpoint is about model comparison,
    not agent loops. We want a clean side-by-side: one prompt, two models,
    no tool calls in between.
    """
    from google import genai
    from google.genai import types as genai_types

    # USD per 1M tokens: (input, output). Output includes thinking tokens.
    # The 2.5 prices are for prompts up to 200K tokens, from Google's Vertex AI
    # pricing page (docs.cloud.google.com/vertex-ai/generative-ai/pricing).
    # The 3.1 prices come from third-party price trackers, not Google's page:
    # check them before you present. Update when Google reprices. A model
    # missing here shows tokens but no cost.
    PRICE_PER_1M_USD = {
        "gemini-3.1-flash-lite":  (0.25, 1.50),
        "gemini-3.1-pro-preview": (2.00, 12.00),
        "gemini-2.5-flash":       (0.30, 2.50),
        "gemini-2.5-pro":         (1.25, 10.00),
    }
    settings = get_settings()

    # Compare is a model-vs-model comparison; the side calls have NO tools.
    # Don't reuse the agent's tools-aware system prompt - mentioning tools
    # the model can't actually call causes gemini-2.5-pro on Vertex to
    # finish with UNEXPECTED_TOOL_CALL and return empty text. Use a stripped
    # prompt that scopes the answer to Vector Institute but says nothing
    # about tools.
    MAPLE_SYSTEM_PROMPT = (
        "You are Maple, the assistant for Vector Institute, Canada's leading "
        "institute for advanced AI research, headquartered at MaRS Discovery "
        "District in Toronto. Answer in concrete sentences with real names, "
        "dates, and numbers when you have them. No emojis, no em dashes. "
        "If a question is ambiguous between Vector Institute and another "
        "Vector (e.g. fastlane.tools), assume Vector Institute. End with "
        "one short follow-up question."
    )

    async def _ask(model_id: str) -> CompareReply:
        t0 = time.monotonic()
        # Only the chat model can switch thinking off; Pro cannot, which is a
        # large part of why it is slower and dearer.
        thinking = (
            genai_types.ThinkingConfig(thinking_budget=settings.thinking_budget)
            if model_id == settings.chat_model else None
        )
        def _go():
            client = genai.Client(api_key=sess.api_key) if sess.api_key else genai.Client()
            return client.models.generate_content(
                model=model_id,
                contents=body.prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=MAPLE_SYSTEM_PROMPT,
                    temperature=0.5,
                    # Thinking tokens count against this cap and Pro spends
                    # thousands of them, so leave room for the answer itself.
                    max_output_tokens=8192,
                    thinking_config=thinking,
                    http_options=genai_types.HttpOptions(timeout=60_000),
                ),
            )
        try:
            resp = await asyncio.to_thread(_go)
            text = getattr(resp, "text", "") or ""
            # When Gemini returns an empty text on a successful call, the
            # reason is usually a safety block, a missing region, or the
            # model is gated. Dig into the response for an actionable hint.
            error: str | None = None
            if not text:
                hint = _diagnose_empty_response(resp, model_id)
                if hint:
                    error = hint
            usage = getattr(resp, "usage_metadata", None)
            tokens_in = getattr(usage, "prompt_token_count", None)
            tokens_out = getattr(usage, "candidates_token_count", None)
            tokens_thinking = getattr(usage, "thoughts_token_count", None)
            cents = None
            price = PRICE_PER_1M_USD.get(model_id)
            if price and tokens_in is not None:
                billed_out = (tokens_out or 0) + (tokens_thinking or 0)
                cents = round((tokens_in * price[0] + billed_out * price[1]) / 1_000_000 * 100, 4)
            return CompareReply(
                model=model_id,
                reply=text or "(no text returned)",
                latency_ms=int((time.monotonic() - t0) * 1000),
                output_chars=len(text),
                input_tokens=tokens_in,
                output_tokens=tokens_out,
                thinking_tokens=tokens_thinking,
                estimated_cents=cents,
                error=error,
            )
        except Exception as exc:  # noqa: BLE001
            return CompareReply(
                model=model_id, reply="", latency_ms=int((time.monotonic() - t0) * 1000),
                output_chars=0, estimated_cents=None, error=str(exc),
            )

    flash, pro = await asyncio.gather(
        _ask(settings.chat_model),
        _ask(settings.pro_model),
    )
    return CompareResponse(prompt=body.prompt, flash=flash, pro=pro)


def _diagnose_empty_response(resp, model_id: str) -> str | None:
    """Best-effort diagnostic when generate_content returns no text.

    The most common causes when the call itself succeeds:
      - the request was filtered (finish_reason SAFETY / MAX_TOKENS)
      - the response came back with only function_call parts
      - the model isn't enabled for this Vertex project + region
    """
    try:
        cands = getattr(resp, "candidates", None) or []
        if not cands:
            return f"{model_id} returned no candidates. Likely model not enabled on this Vertex project."
        first = cands[0]
        reason = getattr(first, "finish_reason", None)
        reason_str = getattr(reason, "name", None) or str(reason or "")
        if reason_str and reason_str.upper() not in ("STOP", "FINISH_REASON_STOP"):
            return f"{model_id} finished with {reason_str}."
        return f"{model_id} returned a response with no text."
    except Exception:  # noqa: BLE001
        return f"{model_id} returned an empty reply (no further detail)."


@app.post("/tool/approve", response_model=ApprovalResponse)
async def tool_approve(body: ApprovalRequest, sess=Depends(_require_session)):
    """Human approval response for a HITL-gated tool call (per-session)."""
    result = await resume_after_approval(
        session_id=sess.session_id,
        call_id=body.call_id,
        approved=body.approved,
    )
    return ApprovalResponse(
        ok=True,
        resumed=True,
        final_reply=result.get("reply") if isinstance(result, dict) else None,
    )


# --- Catalog and admin endpoints ------------------------------------------

@app.get("/catalog")
async def catalog():
    """Public catalog so the frontend Order tab can render item cards."""
    from .tools import CATALOG
    return {
        "items": [
            {"id": cid, **meta} for cid, meta in CATALOG.items()
        ]
    }


@app.get("/orders/my")
async def my_orders(sess=Depends(_require_session)):
    """Customer view of their own orders for approval status polling."""
    from .order_store import get_order_store
    from .tools import CATALOG

    all_orders = get_order_store().list(status=None)
    mine = [o for o in all_orders if o.session_id == sess.session_id]
    return {
        "orders": [
            {
                "order_id": o.order_id,
                "call_id": o.call_id,
                "items": [
                    {"id": i, **CATALOG.get(i, {"name": i})} for i in o.items
                ],
                "name": o.name,
                "email": o.email,
                "total_cad": o.total_cad,
                "queued_at": o.queued_at,
                "status": o.status,
                "resolved_at": o.resolved_at,
                "reason": o.reason,
                "confirmation": o.confirmation,
            }
            for o in mine
        ]
    }


def _check_admin_token(token: str | None) -> None:
    """Mild guard so the admin endpoints aren't world-readable.

    Accepts either the password from settings (default: "1234") OR the legacy
    env var ADMIN_TOKEN for backwards compatibility with existing scripts.
    """
    import os
    expected_password = get_settings().admin_password
    legacy_token = os.environ.get("ADMIN_TOKEN")
    if token == expected_password:
        return
    if legacy_token and token == legacy_token:
        return
    raise HTTPException(401, "admin password required")


@app.get("/admin/orders")
async def admin_orders(status: str = "pending", token: str | None = None):
    """List orders. status in {pending, approved, denied, all}."""
    _check_admin_token(token)
    from .order_store import get_order_store
    s = None if status == "all" else status
    orders = get_order_store().list(status=s)  # type: ignore[arg-type]
    from .tools import CATALOG
    return {
        "orders": [
            {
                "order_id": o.order_id,
                "session_id": o.session_id,
                "call_id": o.call_id,
                "items": [
                    {"id": i, **CATALOG.get(i, {"name": i})} for i in o.items
                ],
                "name": o.name,
                "email": o.email,
                "total_cad": o.total_cad,
                "queued_at": o.queued_at,
                "status": o.status,
                "resolved_at": o.resolved_at,
                "reason": o.reason,
                "confirmation": o.confirmation,
            }
            for o in orders
        ]
    }


@app.post("/admin/orders/{order_id}/approve")
async def admin_approve(order_id: str, token: str | None = None, reason: str | None = None):
    """Admin approves a pending order. Resumes the customer's chat in the
    background so the customer sees Maple confirm out loud."""
    _check_admin_token(token)
    from .order_store import get_order_store
    order = get_order_store().get(order_id)
    if order is None:
        raise HTTPException(404, "no such order")
    if order.status != "pending":
        return {"ok": True, "status": order.status, "noop": True}
    get_order_store().resolve(order_id, "approved", reason=reason)
    # Fire and forget the chat resume for this customer. Maple's reply is
    # kept on the order so the customer's page can show it in the chat.
    _spawn(_resume_and_remember(order_id, order.session_id, order.call_id, approved=True))
    return {"ok": True, "status": "approved"}


@app.post("/admin/orders/{order_id}/deny")
async def admin_deny(order_id: str, token: str | None = None, reason: str | None = None):
    """Admin denies a pending order. Resumes the customer's chat so Maple
    can apologize and ask what to change."""
    _check_admin_token(token)
    from .order_store import get_order_store
    order = get_order_store().get(order_id)
    if order is None:
        raise HTTPException(404, "no such order")
    if order.status != "pending":
        return {"ok": True, "status": order.status, "noop": True}
    get_order_store().resolve(order_id, "denied", reason=reason)
    _spawn(_resume_and_remember(order_id, order.session_id, order.call_id, approved=False))
    return {"ok": True, "status": "denied"}


# --- Static frontend ------------------------------------------------------

app.include_router(lab_router)

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

if _FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")

    @app.get("/")
    async def index():
        index_path = _FRONTEND_DIR / "index.html"
        if not index_path.exists():
            return {"message": "Frontend not built yet."}
        return FileResponse(str(index_path))

    @app.get("/lab")
    async def lab_page():
        """Lecture 2 lab: live demos of what keeps an AI app alive in production."""
        return FileResponse(str(_FRONTEND_DIR / "lab.html"))

    @app.get("/admin")
    async def admin_page():
        """Separate admin page at /admin (password gated client-side)."""
        admin_path = _FRONTEND_DIR / "admin.html"
        if not admin_path.exists():
            return {"message": "Admin page not built yet."}
        return FileResponse(str(admin_path))
