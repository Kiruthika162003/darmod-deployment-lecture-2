"""Pydantic request and response models for the Maple API.

Every endpoint declares its shapes here. FastAPI uses them for validation,
OpenAPI docs, and example payloads. Keeping them centralized means the
frontend can be code-generated from /openapi.json if we want to later.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AwakenRequest(BaseModel):
    """Student pastes their Google AI Studio API key into Maple's heart."""

    api_key: str = Field(..., min_length=20, description="Google AI Studio key")


class AwakenResponse(BaseModel):
    ok: bool
    message: str
    backend: Literal["ai_studio", "vertex", "unknown"] = "unknown"
    model: str | None = None


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    # False asks Maple to answer from her own knowledge instead of searching
    # the web first: faster, but she may get facts wrong.
    use_search: bool = True


class Citation(BaseModel):
    """One source the grounded LLM used to construct its answer."""

    url: str
    title: str | None = None


class Step(BaseModel):
    """One step of an agent turn. `ms` is the time since the previous step."""
    kind: Literal["decide", "tool", "answer"]
    name: str = ""
    ms: int
    detail: str = ""


class Usage(BaseModel):
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0


class ChatResponse(BaseModel):
    reply: str
    citations: list[Citation] = []
    used_tools: list[str] = []
    latency_ms: int
    # True when this was the first request this process served, so the model
    # client and agent had to be built first. On Cloud Run that is a cold start.
    cold_start: bool = False
    instance_age_s: int = 0
    steps: list[Step] = []
    usage: Usage = Usage()


class BatchRequest(BaseModel):
    prompts: list[str] = Field(..., min_length=1, max_length=25)
    use_search: bool = True


class BatchItem(BaseModel):
    prompt: str
    reply: str
    error: str | None = None
    latency_ms: int


class BatchResponse(BaseModel):
    items: list[BatchItem]
    total_latency_ms: int
    cold_start: bool = False
    instance_age_s: int = 0


class PendingToolCall(BaseModel):
    """A HITL-gated tool call awaiting human approval."""

    call_id: str
    tool_name: str
    arguments: dict[str, str]


class StreamingPauseEvent(BaseModel):
    """SSE event when a HITL-gated tool wants to fire."""

    event: Literal["awaiting_approval"] = "awaiting_approval"
    pending: PendingToolCall


class ApprovalRequest(BaseModel):
    call_id: str
    approved: bool
    reason: str | None = None


class ApprovalResponse(BaseModel):
    ok: bool
    resumed: bool
    final_reply: str | None = None


class HealthResponse(BaseModel):
    ok: bool = True
    backend: str
    model: str
    region: str
    version: str


class CompareRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)


class CompareReply(BaseModel):
    model: str
    reply: str
    latency_ms: int
    output_chars: int
    # Token counts as the model reports them. Thinking tokens are billed
    # as output, which is why Pro costs more than its answer suggests.
    input_tokens: int | None = None
    output_tokens: int | None = None
    thinking_tokens: int | None = None
    # None when we have no price for the model.
    estimated_cents: float | None = None
    error: str | None = None


class CompareResponse(BaseModel):
    prompt: str
    flash: CompareReply
    pro: CompareReply
