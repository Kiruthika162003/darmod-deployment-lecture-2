"""Lecture 2 lab: ten small live demos of what keeps an AI app alive in production.

Each one does real work (a model trained on the spot, real statistics, real
timing, real Gemini calls for the cache) and none of them needs Maple. State
lives in memory inside this process, which is itself one of the lessons: a
second copy of the app on Cloud Run knows nothing about the first.
"""

from __future__ import annotations

import asyncio
import functools
import json
import math
import os
import re
import time
import uuid
from datetime import date

import numpy as np
from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse

from .config import get_settings

router = APIRouter(prefix="/lab/api", tags=["lab"])

# Which copy of the app this is. On Cloud Run every new instance gets its own.
INSTANCE_ID = uuid.uuid4().hex[:6]
REVISION = os.environ.get("K_REVISION", "local")


def _fake() -> bool:
    """True when there is no Gemini to call (tests, offline rehearsal)."""
    return os.environ.get("LAB_FAKE") == "1" or get_settings().chat_model.startswith("fake")


@router.get("/whoami")
def whoami() -> dict:
    return {"copy": INSTANCE_ID, "revision": REVISION, "fake": _fake()}


# --- 1. Feature store: one definition, two stores ---------------------------

_DOB = date(2003, 3, 15)


def _age(rule: str) -> int:
    today = date.today()
    completed = today.year - _DOB.year - ((today.month, today.day) < (_DOB.month, _DOB.day))
    return completed if rule == "completed" else completed + 1


class _Features:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.serving_rule = "completed"
        self.online: dict | None = None
        self.offline: list[dict] = []


_F = _Features()


def _features_view() -> dict:
    return {
        "serving_rule": _F.serving_rule,
        "online": _F.online,
        "offline": _F.offline[-1] if _F.offline else None,
    }


@router.get("/features")
def features() -> dict:
    return _features_view()


@router.post("/features/write")
def features_write() -> dict:
    """One pipeline, one rule, written to both stores: they agree by design."""
    row = {"age": _age("completed"), "rule": "completed"}
    _F.online = dict(row)
    _F.offline.append(dict(row))
    return _features_view()


@router.post("/features/change-rule")
def features_change_rule() -> dict:
    """A developer edits the rule in the serving pipeline only."""
    _F.serving_rule = "running"
    return _features_view()


@router.post("/features/serve")
def features_serve() -> dict:
    """The serving pipeline runs with whatever rule it has now."""
    _F.online = {"age": _age(_F.serving_rule), "rule": _F.serving_rule}
    return _features_view()


@router.get("/features/check")
def features_check() -> dict:
    view = _features_view()
    agree = bool(view["online"] and view["offline"] and view["online"]["age"] == view["offline"]["age"])
    return {**view, "agree": agree, "ready": bool(view["online"] and view["offline"])}


@router.post("/features/reset")
def features_reset() -> dict:
    _F.reset()
    return _features_view()


# --- 2. Data drift: the people changed --------------------------------------

_AGE_BINS = np.linspace(18, 78, 11)


def _ages(mean: float, n: int = 4000) -> np.ndarray:
    # Same seed every time, so moving the slider slides the same crowd along.
    rng = np.random.default_rng(1)
    return np.clip(rng.normal(mean, 9, n), 18, 77.99)


def _psi(reference: np.ndarray, current: np.ndarray, bins: np.ndarray) -> float:
    r, _ = np.histogram(reference, bins)
    c, _ = np.histogram(current, bins)
    rp = np.clip(r / r.sum(), 1e-4, None)
    cp = np.clip(c / c.sum(), 1e-4, None)
    return float(np.sum((cp - rp) * np.log(cp / rp)))


def _status(psi: float) -> str:
    return "stable" if psi < 0.1 else "watch" if psi < 0.25 else "drifted"


@router.get("/drift")
def drift(shift: float = Query(0, ge=-25, le=25)) -> dict:
    ref, cur = _ages(42), _ages(42 + shift)
    r, _ = np.histogram(ref, _AGE_BINS)
    c, _ = np.histogram(cur, _AGE_BINS)
    return {
        "bins": [f"{int(a)}" for a in _AGE_BINS[:-1]],
        "yesterday": (r / r.sum()).round(4).tolist(),
        "today": (c / c.sum()).round(4).tolist(),
        "status": _status(_psi(ref, cur, _AGE_BINS)),
    }


# --- 3. Model decay and retraining ------------------------------------------

def _cohort(week: int, seed: int, n: int = 3000) -> tuple[np.ndarray, np.ndarray]:
    """Loan applicants in a given week. The people shift a little each week, and
    so does what really predicts repayment: the world moves under the model."""
    rng = np.random.default_rng(seed)
    age = rng.normal(42 - 0.6 * week, 9, n)
    income = rng.normal(60 + 0.8 * week, 18, n)
    debt = rng.beta(2, 5, n)
    cards = rng.poisson(2.2 + 0.12 * week, n)
    z = (
        0.05 * (age - 40)
        + (0.06 - 0.005 * week) * (income - 60)
        - 5.0 * (debt - 0.28)
        + (-0.5 + 0.06 * week) * (cards - 2.2)
        + 0.3
    )
    y = rng.random(n) < 1 / (1 + np.exp(-2.5 * z))
    return np.column_stack([age, income, debt, cards]), y


def _fit(week: int):
    from sklearn.linear_model import LogisticRegression

    X, y = _cohort(week, seed=1000 + week)
    return LogisticRegression(max_iter=500).fit(X, y)


class _Decay:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.week = 0
        self.model = _fit(0)
        self.base = self._score(0)
        self.history = [{"week": 0, "accuracy": self.base, "verdict": "healthy"}]

    def _score(self, week: int) -> float:
        X, y = _cohort(week, seed=5000 + week, n=8000)  # fresh people, never seen in training
        return float(self.model.score(X, y))

    def verdict(self, accuracy: float) -> str:
        drop = self.base - accuracy
        return "healthy" if drop < 0.03 else "watch" if drop < 0.07 else "retrain"


_D: _Decay | None = None


def _decay() -> _Decay:
    global _D
    if _D is None:
        _D = _Decay()
    return _D


def _decay_view() -> dict:
    d = _decay()
    return {"week": d.week, "history": d.history, "verdict": d.history[-1]["verdict"]}


@router.get("/decay")
def decay() -> dict:
    return _decay_view()


@router.post("/decay/next")
def decay_next() -> dict:
    d = _decay()
    d.week += 1
    acc = d._score(d.week)
    d.history.append({"week": d.week, "accuracy": acc, "verdict": d.verdict(acc)})
    return _decay_view()


@router.post("/decay/retrain")
def decay_retrain() -> dict:
    d = _decay()
    d.model = _fit(d.week)
    acc = d._score(d.week)
    d.history.append({"week": d.week, "accuracy": acc, "verdict": "healthy", "retrained": True})
    return _decay_view()


@router.post("/decay/reset")
def decay_reset() -> dict:
    global _D
    _D = _Decay()
    return _decay_view()


# --- 4. Pipeline with a data gate -------------------------------------------

def _gate(day: str) -> tuple[bool, str]:
    """Compare the data arriving today with the data the model learned from."""
    X, _ = _cohort(0, seed=1000)
    today, _ = _cohort(0, seed=7777)
    if day == "bad":
        today[:, 1] = 0.0  # a bank changed its file format: income is all zeros
    worst, worst_col = 0.0, ""
    for i, name in enumerate(["age", "income", "debt", "cards"]):
        lo, hi = float(X[:, i].min()), float(X[:, i].max())
        bins = np.linspace(lo, hi + 1e-9, 11)
        p = _psi(X[:, i], np.clip(today[:, i], lo, hi), bins)
        if p > worst:
            worst, worst_col = p, name
    return worst < 0.25, worst_col


async def _pipeline(day: str):
    def ev(stage: str, status: str, note: str = "") -> str:
        return f"data: {json.dumps({'stage': stage, 'status': status, 'note': note})}\n\n"

    stages = ["Code check", "Unit checks", "Train", "Evaluate", "Data gate", "Register"]
    for s in stages:
        yield ev(s, "waiting")
    for stage in stages:
        yield ev(stage, "running")
        await asyncio.sleep(0.7)
        ok, note = True, ""
        if stage == "Code check":
            import ast
            from pathlib import Path

            try:  # read-only check: the container user cannot write compiled files
                for source in Path(__file__).parent.glob("*.py"):
                    ast.parse(source.read_text(encoding="utf-8"))
            except SyntaxError:
                ok = False
            note = "every file parses" if ok else "a file has a syntax error"
        elif stage == "Unit checks":
            from .tools import CATALOG

            ok = CATALOG["vector-hoodie"]["price_cad"] > CATALOG["vector-mug"]["price_cad"] > 0
            note = "the store catalog is sane"
        elif stage == "Train":
            _fit(0)
            note = "a fresh model learned from last month"
        elif stage == "Evaluate":
            ok = _decay().base > 0.7
            note = "good enough on people it has not seen"
        elif stage == "Data gate":
            ok, col = _gate(day)
            note = (
                "today's data looks like the data the model learned from"
                if ok
                else f"the '{col}' column looks nothing like before: refusing to ship"
            )
        elif stage == "Register":
            note = "saved as the next model version"
        yield ev(stage, "pass" if ok else "fail", note)
        if not ok:
            for later in stages[stages.index(stage) + 1 :]:
                yield ev(later, "skipped")
            break
    yield ev("done", "done")


@router.get("/pipeline/run")
async def pipeline_run(day: str = "normal") -> StreamingResponse:
    return StreamingResponse(_pipeline(day), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


# --- 5. Scaling shapes: one, many, or a queue -------------------------------

async def _scale(mode: str, jobs: int, wait: float):
    start = time.perf_counter()

    def ms() -> int:
        return int((time.perf_counter() - start) * 1000)

    def ev(**kw) -> str:
        return f"data: {json.dumps({**kw, 'ms': ms()})}\n\n"

    if mode == "queue":
        yield ev(kind="ticket", ticket="T-" + uuid.uuid4().hex[:4].upper())
    done: asyncio.Queue = asyncio.Queue()
    sem = asyncio.Semaphore(1 if mode == "one" else 4)

    async def job(i: int) -> None:
        async with sem:
            await asyncio.sleep(wait)  # waits like a call to a model
        await done.put(i)

    tasks = [asyncio.create_task(job(i)) for i in range(jobs)]
    for _ in range(jobs):
        yield ev(kind="job", job=await done.get())
    await asyncio.gather(*tasks)
    yield ev(kind="done")


@router.get("/scale/run")
async def scale_run(mode: str = "one", jobs: int = Query(12, ge=1, le=40)) -> StreamingResponse:
    return StreamingResponse(_scale(mode, jobs, 0.4), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


# --- 6. Autoscaling: which copies are serving -------------------------------

_SEEN: dict[str, float] = {}
_INFLIGHT = 0


@router.get("/scale/work")
async def scale_work() -> dict:
    """A request that takes a while, so many at once keep copies busy."""
    global _INFLIGHT
    _INFLIGHT += 1
    try:
        await asyncio.sleep(1.5)
    finally:
        _INFLIGHT -= 1
    _SEEN[INSTANCE_ID] = time.time()
    return {"copy": INSTANCE_ID, "revision": REVISION}


async def _crowd(base: str, visitors: int):
    """Send a crowd at this service's own public address. On Cloud Run that goes
    back in through the front door, so Cloud Run really does spread the visitors
    over as many copies as it needs. A browser cannot do this: it only keeps a
    few connections open at a time."""
    import httpx

    async def one(client: httpx.AsyncClient) -> dict:
        try:
            r = await client.get(f"{base}/lab/api/scale/work")
            r.raise_for_status()
            return r.json()
        except Exception:  # noqa: BLE001  (cannot reach ourselves from outside: answer here)
            return await scale_work()

    limits = httpx.Limits(max_connections=200)
    async with httpx.AsyncClient(timeout=30, limits=limits) as client:
        tasks = [asyncio.create_task(one(client)) for _ in range(visitors)]
        for finished in asyncio.as_completed(tasks):
            visitor = await finished
            yield f"data: {json.dumps({'kind': 'visitor', **visitor})}\n\n"
    yield f"data: {json.dumps({'kind': 'done'})}\n\n"


@router.get("/scale/crowd")
async def scale_crowd(request: Request, n: int = Query(30, ge=1, le=100)) -> StreamingResponse:
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    base = f"{proto}://{request.headers.get('host', request.url.netloc)}"
    return StreamingResponse(_crowd(base, n), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


# --- 7. Serving many at once: answer people together ------------------------

@router.get("/batch/run")
def batch_run() -> dict:
    """The idea behind engines like vLLM, with real maths: reading the model's
    weights once for everyone is far cheaper than reading them once per person."""
    d, people = 4096, 64
    rng = np.random.default_rng(0)
    w = rng.standard_normal((d, d)).astype(np.float32)
    x = rng.standard_normal((people, d)).astype(np.float32)
    _ = x @ w  # warm up

    def best_of(runs: int, work) -> float:
        # Best of a few runs: a busy moment on the machine slows one run, not all of them.
        times = []
        for _ in range(runs):
            t = time.perf_counter()
            work()
            times.append(time.perf_counter() - t)
        return min(times)

    # Cap the maths library's threads. Left alone it spreads a small job over every
    # core the machine reports (24 here), and the coordination costs more than the work.
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=2):
        _ = x @ w
        one_by_one = best_of(3, lambda: [row @ w for row in x])
        together = best_of(3, lambda: x @ w)
    return {"people": people, "one_by_one_ms": round(one_by_one * 1000, 1), "together_ms": round(together * 1000, 1)}


# --- 8. Shrinking the model (quantization) ----------------------------------

_Q: dict | None = None


def _quant_state() -> dict:
    global _Q
    if _Q is None:
        from sklearn.datasets import load_digits
        from sklearn.linear_model import LogisticRegression

        digits = load_digits()
        x, y = digits.data / 16.0, digits.target
        model = LogisticRegression(max_iter=2000).fit(x[:1200], y[:1200])
        _Q = {"model": model, "x": x[1200:], "y": y[1200:]}
    return _Q


def _shrink(w: np.ndarray, bits: int) -> np.ndarray:
    if bits >= 32:
        return w
    if bits == 16:
        return w.astype(np.float16).astype(np.float64)
    levels = 2 ** (bits - 1) - 1
    scale = np.abs(w).max() / levels
    return np.round(w / scale).clip(-levels, levels) * scale


@router.get("/quant")
def quant(sample: int = Query(0, ge=0, le=500)) -> dict:
    q = _quant_state()
    model, x, y = q["model"], q["x"], q["y"]
    i = sample % len(x)
    rows = []
    for bits in (32, 16, 8, 4, 3, 2):
        w = _shrink(model.coef_, bits)
        scores = x @ w.T + model.intercept_
        pred = scores.argmax(axis=1)
        rows.append({
            "bits": bits,
            "size": bits / 32,
            "accuracy": float((pred == y).mean()),
            "guess": int(pred[i]),
        })
    return {"truth": int(y[i]), "pixels": (x[i] * 16).round().astype(int).tolist(), "rows": rows}


# --- 9. Cache: remember answers, even for reworded questions ----------------

_CACHE: list[dict] = []
# Measured with the real embedding model: reworded questions score 0.91 or more,
# different questions about the same topic score 0.87 or less. A missed match only
# costs a fresh answer; a wrong match hands someone the wrong answer, so lean high.
_SIMILAR = 0.90
_SIMILAR_FAKE = 0.6  # the stand-in embedding is cruder, so it is allowed to be looser


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


@functools.lru_cache(maxsize=1)
def _genai():
    """One Gemini client for the whole app. It has to be kept referenced: a client
    that is created and dropped in one expression is closed before it is used."""
    from google import genai

    return genai.Client()


def _fake_embed(text: str) -> list[float]:
    """Letter triples hashed into a vector: crude, but enough to match rewording."""
    v = np.zeros(256)
    t = f"  {_norm(text)}  "
    for i in range(len(t) - 2):
        v[hash(t[i : i + 3]) % 256] += 1
    n = np.linalg.norm(v)
    return (v / n).tolist() if n else v.tolist()


async def _embed(text: str) -> list[float] | None:
    if _fake():
        return _fake_embed(text)

    def go() -> list[float]:
        res = _genai().models.embed_content(model=os.environ.get("EMBEDDING_MODEL", "gemini-embedding-2"), contents=text)
        return list(res.embeddings[0].values)

    try:
        return await asyncio.to_thread(go)
    except Exception:  # noqa: BLE001
        return None


async def _llm(question: str) -> str:
    if _fake():
        await asyncio.sleep(1.2)
        return f"(scripted answer) {question.strip()}"

    def go() -> str:
        from google.genai import types

        s = get_settings()
        res = _genai().models.generate_content(
            model=s.chat_model,
            contents=question,
            config=types.GenerateContentConfig(
                system_instruction="You are Maple, the assistant for Vector Institute in Toronto. Answer in two short sentences.",
                max_output_tokens=300,
                thinking_config=types.ThinkingConfig(thinking_budget=s.thinking_budget),
                http_options=types.HttpOptions(timeout=30_000),
            ),
        )
        return res.text or ""

    return await asyncio.to_thread(go)


def _cosine(a: list[float], b: list[float]) -> float:
    x, y = np.asarray(a), np.asarray(b)
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


@router.post("/cache/ask")
async def cache_ask(body: dict) -> dict:
    question = str(body.get("question", "")).strip()[:300]
    t0 = time.perf_counter()
    key = _norm(question)
    for e in _CACHE:
        if e["key"] == key:
            return _cache_reply(e["answer"], "exact", e["question"], t0)
    emb = await _embed(question)
    if emb is not None:
        best = max(((_cosine(emb, e["emb"]), e) for e in _CACHE if e["emb"] is not None), key=lambda p: p[0], default=None)
        if best and best[0] >= (_SIMILAR_FAKE if _fake() else _SIMILAR):
            return _cache_reply(best[1]["answer"], "similar", best[1]["question"], t0)
    answer = await _llm(question)
    _CACHE.append({"key": key, "question": question, "emb": emb, "answer": answer})
    return _cache_reply(answer, "fresh", None, t0)


def _cache_reply(answer: str, source: str, matched: str | None, t0: float) -> dict:
    return {
        "answer": answer,
        "source": source,
        "matched": matched,
        "ms": int((time.perf_counter() - t0) * 1000),
        "copy": INSTANCE_ID,
        "remembered": len(_CACHE),
    }


@router.post("/cache/clear")
def cache_clear() -> dict:
    _CACHE.clear()
    return {"remembered": 0}


# --- 10. Many teams, one service: fair use ----------------------------------

_BUCKETS: dict[str, tuple[float, float]] = {}
_CAPACITY, _REFILL = 5.0, 1.0  # five quick requests, then one more every second


@router.post("/tenant/request")
def tenant_request(team: str = "A", fair: bool = True) -> dict:
    """One shared allowance for everyone, or a separate allowance per team."""
    key = team if fair else "everyone"
    now = time.monotonic()
    tokens, last = _BUCKETS.get(key, (_CAPACITY, now))
    tokens = min(_CAPACITY, tokens + (now - last) * _REFILL)
    ok = tokens >= 1
    _BUCKETS[key] = (tokens - 1 if ok else tokens, now)
    return {"team": team, "accepted": ok, "left": math.floor(_BUCKETS[key][0])}


@router.post("/tenant/reset")
def tenant_reset() -> dict:
    _BUCKETS.clear()
    return {"ok": True}
