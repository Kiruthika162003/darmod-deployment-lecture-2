"""The lecture 2 lab: each demo does what its slide says it does.

LAB_FAKE=1 swaps Gemini for a scripted answer, so these tests make no Google calls.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("LAB_FAKE", "1")
    from backend import lab
    from backend.main import app

    lab._CACHE.clear()
    lab._BUCKETS.clear()
    lab._F.reset()
    lab._D = None
    with TestClient(app) as c:
        yield c


def _events(response) -> list[dict]:
    return [json.loads(line[5:]) for line in response.text.splitlines() if line.startswith("data:")]


def test_the_lab_page_and_its_files_are_served(client):
    page = client.get("/lab")
    assert page.status_code == 200 and "Lecture 2 lab" in page.text
    assert client.get("/static/lab.js").status_code == 200
    assert client.get("/static/lab.css").status_code == 200
    assert 'href="/lab"' in client.get("/").text  # Maple links to the lab


def test_feature_store_stays_in_sync_until_one_rule_changes(client):
    wrote = client.post("/lab/api/features/write").json()
    assert wrote["online"]["age"] == wrote["offline"]["age"]
    assert client.get("/lab/api/features/check").json()["agree"] is True

    client.post("/lab/api/features/change-rule")
    client.post("/lab/api/features/serve")
    check = client.get("/lab/api/features/check").json()
    assert check["agree"] is False
    assert check["online"]["age"] == check["offline"]["age"] + 1

    assert client.post("/lab/api/features/reset").json()["online"] is None


def test_drift_is_stable_for_the_same_crowd_and_flagged_for_a_different_one(client):
    assert client.get("/lab/api/drift?shift=0").json()["status"] == "stable"
    order = {"stable": 0, "watch": 1, "drifted": 2}
    states = [order[client.get(f"/lab/api/drift?shift={s}").json()["status"]] for s in (0, 3, 6, 12)]
    assert states == sorted(states) and states[-1] == 2
    shifted = client.get("/lab/api/drift?shift=10").json()
    assert shifted["yesterday"] != shifted["today"]


def test_a_model_decays_week_by_week_and_retraining_fixes_it(client):
    first = client.post("/lab/api/decay/reset").json()
    assert first["verdict"] == "healthy"
    for _ in range(8):
        v = client.post("/lab/api/decay/next").json()
    start, end = v["history"][0]["accuracy"], v["history"][-1]["accuracy"]
    assert end < start - 0.06 and v["verdict"] == "retrain"
    fixed = client.post("/lab/api/decay/retrain").json()
    assert fixed["verdict"] == "healthy" and fixed["history"][-1]["accuracy"] > end + 0.05


def test_the_pipeline_ships_on_a_normal_day_and_stops_at_the_gate_on_a_bad_one(client):
    good = {e["stage"]: e["status"] for e in _events(client.get("/lab/api/pipeline/run?day=normal"))}
    assert [good[s] for s in ("Code check", "Unit checks", "Train", "Evaluate", "Data gate", "Register")] == ["pass"] * 6

    bad = {e["stage"]: e["status"] for e in _events(client.get("/lab/api/pipeline/run?day=bad"))}
    assert bad["Evaluate"] == "pass" and bad["Data gate"] == "fail" and bad["Register"] == "skipped"


def test_more_workers_finish_sooner_and_a_queue_hands_back_a_ticket_at_once(client):
    def run(mode):
        ev = _events(client.get(f"/lab/api/scale/run?mode={mode}&jobs=4"))
        return ev, next(e["ms"] for e in ev if e["kind"] == "done")

    one_events, one = run("one")
    many_events, many = run("many")
    queue_events, _ = run("queue")
    assert sum(e["kind"] == "job" for e in one_events) == 4
    assert one > many * 2
    assert queue_events[0]["kind"] == "ticket" and queue_events[0]["ms"] < 150


def test_a_busy_request_says_which_copy_answered(client):
    reply = client.get("/lab/api/scale/work").json()
    assert reply["copy"] == client.get("/lab/api/whoami").json()["copy"]


def test_a_crowd_is_sent_and_every_visitor_is_answered(client):
    events = _events(client.get("/lab/api/scale/crowd?n=6"))
    visitors = [e for e in events if e["kind"] == "visitor"]
    assert len(visitors) == 6 and events[-1]["kind"] == "done"
    assert {v["copy"] for v in visitors} == {client.get("/lab/api/whoami").json()["copy"]}


def test_the_default_digit_survives_three_bits_and_fails_at_two(client):
    data = client.get("/lab/api/quant?sample=1").json()
    guess = {r["bits"]: r["guess"] for r in data["rows"]}
    assert all(guess[b] == data["truth"] for b in (32, 16, 8, 4, 3)) and guess[2] != data["truth"]


def test_answering_people_together_beats_one_at_a_time(client):
    race = client.get("/lab/api/batch/run").json()
    assert race["together_ms"] * 1.5 < race["one_by_one_ms"]


def test_a_model_survives_shrinking_until_it_does_not(client):
    rows = {r["bits"]: r["accuracy"] for r in client.get("/lab/api/quant?sample=4").json()["rows"]}
    assert rows[8] > rows[32] - 0.03 and rows[2] < rows[32] - 0.15


def test_the_cache_remembers_exact_and_reworded_questions(client):
    first = client.post("/lab/api/cache/ask", json={"question": "What is DARMOD?"}).json()
    assert first["source"] == "fresh" and first["ms"] > 500

    again = client.post("/lab/api/cache/ask", json={"question": "what is darmod"}).json()
    assert again["source"] == "exact" and again["ms"] < 50 and again["answer"] == first["answer"]

    reworded = client.post("/lab/api/cache/ask", json={"question": "whats DARMOD"}).json()
    assert reworded["source"] == "similar" and reworded["matched"] == "What is DARMOD?"

    other = client.post("/lab/api/cache/ask", json={"question": "Where is the Toronto office?"}).json()
    assert other["source"] == "fresh"
    assert client.post("/lab/api/cache/clear").json()["remembered"] == 0


def test_one_team_flooding_does_not_block_another_unless_they_share(client):
    flood = [client.post("/lab/api/tenant/request?team=A&fair=true").json()["accepted"] for _ in range(20)]
    assert flood.count(True) == 5
    assert client.post("/lab/api/tenant/request?team=B&fair=true").json()["accepted"] is True

    client.post("/lab/api/tenant/reset")
    for _ in range(20):
        client.post("/lab/api/tenant/request?team=A&fair=false")
    assert client.post("/lab/api/tenant/request?team=B&fair=false").json()["accepted"] is False
