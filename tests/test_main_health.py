"""End-to-end check that the FastAPI app boots and /healthz responds.

We do not exercise /chat here because that would require a real Gemini key.
The runbook walks the live route. CI runs this much, which is enough to
catch import-time bugs and route-wiring mistakes.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    # Import inside the fixture so env vars set by conftest.py take effect first.
    from backend.main import app

    with TestClient(app) as c:
        yield c


def test_healthz_responds(client):
    res = client.get("/healthz")
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert "model" in body
    assert "region" in body


def test_chat_requires_session_when_required(client, monkeypatch):
    # Flip the flag at the settings level for this one test.
    from backend import config

    monkeypatch.setattr(config.get_settings(), "require_api_key", True)
    res = client.post("/chat", json={"prompt": "hi"})
    assert res.status_code == 401


def test_default_models_use_supported_gemini_names():
    from backend import config

    settings = config.get_settings()
    assert settings.chat_model == "gemini-3.1-flash-lite"
    assert settings.image_model == "gemini-2.5-flash-image"


def test_awaken_allows_alternate_google_api_key_prefixes(client, monkeypatch):
    from backend import main

    async def fake_validate_key(api_key: str) -> None:
        return None

    monkeypatch.setattr(main, "_validate_key", fake_validate_key)
    res = client.post(
        "/awaken",
        json={"api_key": "not-a-real-key-just-twenty-plus-chars-for-the-format-check"},
    )
    assert res.status_code == 200
    assert res.json()["ok"] is True


def test_session_api_key_forces_ai_studio_environment(monkeypatch):
    from backend import config

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")

    with config.ai_studio_auth("AIza-test-key"):
        assert os.environ["GOOGLE_API_KEY"] == "AIza-test-key"
        assert os.environ["GOOGLE_GENAI_USE_VERTEXAI"] == "FALSE"

    assert os.environ.get("GOOGLE_API_KEY") is None
    assert os.environ.get("GOOGLE_GENAI_USE_VERTEXAI") == "TRUE"
