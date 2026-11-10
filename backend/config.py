"""Runtime configuration for Maple.

Mirrors the dual-auth pattern from flockcv. If GOOGLE_API_KEY is set, the
agent runs against Google AI Studio (free-tier friendly). Otherwise it falls
back to Vertex AI via gcloud Application Default Credentials, which is what
Cloud Run uses in production with a service account.

This separation lets one codebase serve two very different deployment modes:
a student running locally with a free key, and the lecturer's Cloud Run
service with no key at all.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

BackendName = Literal["ai_studio", "vertex"]


class Settings(BaseSettings):
    """Application settings, loaded once at process start."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Server
    environment: Literal["dev", "staging", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    deployment_region: str = "local"

    # Agent
    chat_model: str = "gemini-3.1-flash-lite"
    # The bigger model the Compare tab sets against chat_model.
    pro_model: str = "gemini-3.1-pro-preview"
    # How many tokens the model may spend thinking before it answers. 0 turns
    # thinking off: Maple's questions do not need it, and it added about four
    # seconds a call. Raise it (try 1024) to watch the answers get slower.
    thinking_budget: int = 0
    image_model: str = "gemini-2.5-flash-image"
    image_fallback_model: str = "imagen-4.0-generate-001"
    max_output_tokens: int = 1024
    temperature: float = 0.5

    # Auth
    # Default: open access. The deployed Maple uses Vertex AI via the
    # lecturer's gcloud Application Default Credentials, so no per-user key
    # is needed. Flip REQUIRE_API_KEY=TRUE in .env to bring back the
    # heart-style key prompt (the /awaken endpoint is still wired in).
    require_api_key: bool = False
    session_secret: str = "change-me-in-prod"
    session_cookie_name: str = "maple_sid"
    session_ttl_minutes: int = 60

    # HITL: tool names listed here pause for human approval before execution.
    # The runbook flips this list to demo what happens without HITL.
    hitl_tools: list[str] = Field(
        default_factory=lambda: ["submit_order"]
    )

    # Admin page password. Overridable via ADMIN_PASSWORD env var. Default
    # is the lecturer's recording password so the demo runs out of the box.
    admin_password: str = "1234"

    # CORS: relax for local dev; lock down to the frontend origin in prod.
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def detect_backend() -> BackendName:
    """Choose AI Studio vs Vertex based on env, exactly as flockcv does.

    If neither GOOGLE_CLOUD_PROJECT nor GOOGLE_CLOUD_LOCATION is set in the
    environment, fall back to whatever `gcloud config` reports. That way the
    lecturer's `gcloud auth application-default login` + `gcloud config set
    project ...` is enough to drive Maple. No .env editing required.
    """
    if os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
        return "ai_studio"
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")
    if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
        try:
            import subprocess
            proj = subprocess.run(
                ["gcloud", "config", "get-value", "project"],
                capture_output=True, text=True, timeout=5, check=False,
            ).stdout.strip()
            if proj and proj != "(unset)":
                os.environ["GOOGLE_CLOUD_PROJECT"] = proj
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")
    return "vertex"


@contextmanager
def ai_studio_auth(api_key: str | None = None):
    """Temporarily force AI Studio mode for a session-scoped API key."""
    previous_api_key = os.environ.get("GOOGLE_API_KEY")
    previous_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI")

    try:
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
        else:
            if previous_api_key is None:
                os.environ.pop("GOOGLE_API_KEY", None)
            else:
                os.environ["GOOGLE_API_KEY"] = previous_api_key
            if previous_vertex is None:
                os.environ.pop("GOOGLE_GENAI_USE_VERTEXAI", None)
            else:
                os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = previous_vertex
        yield
    finally:
        if previous_api_key is None:
            os.environ.pop("GOOGLE_API_KEY", None)
        else:
            os.environ["GOOGLE_API_KEY"] = previous_api_key
        if previous_vertex is None:
            os.environ.pop("GOOGLE_GENAI_USE_VERTEXAI", None)
        else:
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = previous_vertex


BACKEND: BackendName = detect_backend()
