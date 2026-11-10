"""Tests for the tools Maple actually uses now: submit_order and web_search.

Tests don't touch the network: external HTTP calls are mocked via patches.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.tools import (
    CATALOG,
    clear_order_log,
    get_order_log,
    submit_order,
    web_search,
)


@pytest.fixture(autouse=True)
def _clear_log():
    clear_order_log()
    yield


# --- submit_order ---------------------------------------------------------

def test_submit_order_records_and_returns_confirmation():
    out = submit_order(
        items=["vector-hoodie", "darmod-packet"],
        name="Sam",
        email="sam@example.com",
    )
    assert "sam@example.com" in out
    log = get_order_log()
    assert len(log) == 1
    assert log[0]["name"] == "Sam"
    assert log[0]["items"] == ["vector-hoodie", "darmod-packet"]


def test_submit_order_rejects_unknown_items_without_recording():
    out = submit_order(items=["mystery-thing"], name="Sam", email="sam@example.com")
    assert "don't have" in out.lower()
    assert get_order_log() == []


def test_catalog_has_all_four_items():
    assert set(CATALOG.keys()) == {
        "vector-hoodie", "vector-mug", "darmod-packet", "fastlane-packet",
    }
    for cid, item in CATALOG.items():
        assert "name" in item and "category" in item and "price_cad" in item


# --- web_search (Tavily + DuckDuckGo fallback) ----------------------------

@pytest.mark.asyncio
async def test_web_search_uses_tavily_when_key_present(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
    mock_resp = MagicMock()
    mock_resp.json = MagicMock(return_value={
        "answer": "Vector Institute is in Toronto.",
        "results": [
            {"title": "About", "url": "https://vectorinstitute.ai/about",
             "content": "Founded in 2017."}
        ],
    })
    mock_resp.raise_for_status = MagicMock()
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("backend.tools.httpx.AsyncClient", return_value=mock_client):
        out = await web_search("Vector Institute history")

    assert "Vector Institute is in Toronto" in out
    assert "vectorinstitute.ai/about" in out


@pytest.mark.asyncio
async def test_web_search_falls_back_to_ddg_without_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    mock_resp = MagicMock()
    mock_resp.json = MagicMock(return_value={
        "Heading": "Vector Institute",
        "AbstractText": "Toronto-based AI research institute.",
        "AbstractURL": "https://en.wikipedia.org/wiki/Vector_Institute",
        "RelatedTopics": [],
    })
    mock_resp.raise_for_status = MagicMock()
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_resp)

    with patch("backend.tools.httpx.AsyncClient", return_value=mock_client):
        out = await web_search("Vector Institute")

    assert "Toronto-based AI research institute" in out
