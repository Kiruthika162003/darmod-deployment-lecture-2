"""What the lecture slides promise the demo shows, checked against a scripted model.

No Google calls: tests/fake_model.py stands in for Gemini.
"""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def fake_agent(monkeypatch):
    """Point the agent at the scripted model and start from a cold runner."""
    from backend import config, runner

    import tests.fake_model  # noqa: F401  (registers the model)

    monkeypatch.setattr(config.get_settings(), "chat_model", "fake-maple")
    monkeypatch.setattr(runner, "_RUNNER", None)
    monkeypatch.setattr(runner, "_AGENT", None)


@pytest.fixture
def client(fake_agent):
    from backend.main import app

    with TestClient(app) as c:
        c.get("/me")  # opens the browser session, like the page does on load
        yield c


def _sse(response) -> list[dict]:
    return [
        json.loads(line[5:])
        for line in response.text.splitlines()
        if line.startswith("data:")
    ]


def test_first_request_is_a_cold_start_then_warm(client):
    first = client.post("/chat", json={"prompt": "What is DARMOD?"}).json()
    second = client.post("/chat", json={"prompt": "What is FastLane?"}).json()
    assert first["cold_start"] is True
    assert second["cold_start"] is False
    assert second["instance_age_s"] >= 0


def test_batch_answers_every_prompt_in_one_request(client):
    prompts = ["What is DARMOD?", "What is FastLane?", "Where is Vector?"]
    body = client.post("/chat/batch", json={"prompts": prompts}).json()
    assert [i["prompt"] for i in body["items"]] == prompts
    assert all(i["reply"] and i["error"] is None for i in body["items"])
    assert body["total_latency_ms"] >= 0
    assert body["cold_start"] is True


def test_stream_reports_cold_start_on_the_done_event(client):
    events = _sse(client.post("/chat/stream", json={"prompt": "What is DARMOD?"}))
    assert any(e["type"] == "token" for e in events)
    done = events[-1]
    assert done["type"] == "done"
    assert done["cold_start"] is True
    assert "instance_age_s" in done


def _place_order(client) -> dict:
    from backend.tools import CATALOG

    name = CATALOG["vector-hoodie"]["name"]
    prompt = (
        f"Place an order for me. Items: {name}. Name: Ada. "
        "Email: ada@example.com. Use submit_order; don't ask me to confirm."
    )
    client.post("/chat", json={"prompt": prompt})
    orders = client.get("/orders/my").json()["orders"]
    assert len(orders) == 1 and orders[0]["status"] == "pending"
    assert orders[0]["confirmation"] is None
    return orders[0]


def _wait_for_confirmation(client, seconds: float = 5.0) -> dict:
    deadline = time.time() + seconds
    while time.time() < deadline:
        order = client.get("/orders/my").json()["orders"][0]
        if order["confirmation"]:
            return order
        time.sleep(0.1)
    raise AssertionError("Maple's confirmation never reached the order")


def test_approval_puts_maples_confirmation_on_the_order(client):
    from backend.config import get_settings

    order = _place_order(client)
    token = get_settings().admin_password
    res = client.post(f"/admin/orders/{order['order_id']}/approve", params={"token": token})
    assert res.json()["status"] == "approved"

    done = _wait_for_confirmation(client)
    assert done["status"] == "approved"
    assert "approved" in done["confirmation"].lower()


def test_denial_puts_maples_reply_on_the_order(client):
    from backend.config import get_settings

    order = _place_order(client)
    token = get_settings().admin_password
    res = client.post(f"/admin/orders/{order['order_id']}/deny", params={"token": token})
    assert res.json()["status"] == "denied"

    done = _wait_for_confirmation(client)
    assert done["status"] == "denied"
    assert "declined" in done["confirmation"].lower()


# --- What the agent did, step by step ---------------------------------------

def _order_prompt() -> str:
    from backend.tools import CATALOG

    name = CATALOG["vector-hoodie"]["name"]
    return (
        f"Place an order for me. Items: {name}. Name: Ada. "
        "Email: ada@example.com. Use submit_order; don't ask me to confirm."
    )


def test_a_plain_answer_is_a_single_step(client):
    body = client.post("/chat", json={"prompt": "What is DARMOD?"}).json()
    assert [s["kind"] for s in body["steps"]] == ["answer"]
    assert body["used_tools"] == []


def test_an_order_shows_the_tool_maple_decided_to_call(client):
    body = client.post("/chat", json={"prompt": _order_prompt()}).json()
    assert body["used_tools"] == ["submit_order"]
    decide = body["steps"][0]
    assert decide["kind"] == "decide" and decide["name"] == "submit_order"
    assert "ada@example.com" in decide["detail"]
    assert all(s["ms"] >= 0 for s in body["steps"])


def test_the_stream_reports_tool_calls_and_steps(client):
    events = _sse(client.post("/chat/stream", json={"prompt": _order_prompt()}))
    kinds = [e["type"] for e in events]
    assert "tool_call" in kinds and "awaiting_approval" in kinds
    assert [e["kind"] for e in events if e["type"] == "step"][0] == "decide"
    assert events[-1]["type"] == "done" and "usage" in events[-1]


def test_search_off_asks_maple_to_skip_the_search():
    from backend.runner import _prompt_for

    assert _prompt_for("What is FastLane?", True) == "What is FastLane?"
    off = _prompt_for("What is FastLane?", False)
    assert off.endswith("What is FastLane?")
    assert "without calling search_agent" in off


# --- Compare: real token counts and the cost they imply ---------------------

def test_compare_prices_the_tokens_the_model_reports(monkeypatch):
    from types import SimpleNamespace

    from backend.main import app

    class FakeModels:
        def generate_content(self, model, contents, config):
            thinking = 1000 if "pro" in model else 0
            return SimpleNamespace(
                text="An answer.",
                usage_metadata=SimpleNamespace(
                    prompt_token_count=100, candidates_token_count=50, thoughts_token_count=thinking
                ),
            )

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.models = FakeModels()

    monkeypatch.setattr("google.genai.Client", FakeClient)
    with TestClient(app) as c:
        c.get("/me")
        body = c.post("/chat/compare", json={"prompt": "What is DARMOD?"}).json()

    flash, pro = body["flash"], body["pro"]
    assert flash["thinking_tokens"] == 0 and pro["thinking_tokens"] == 1000
    # (100 in x $0.25 + 50 out x $1.50) per 1M tokens, in cents
    assert flash["estimated_cents"] == pytest.approx(0.01)
    # thinking tokens are billed as output: (100 x $2.00 + 1050 x $12.00) per 1M
    assert pro["estimated_cents"] == pytest.approx(1.28)
