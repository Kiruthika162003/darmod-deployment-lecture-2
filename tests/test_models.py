"""Pydantic shape checks."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.models import (
    AwakenRequest,
    BatchRequest,
    ChatRequest,
    CompareRequest,
)


def test_chat_request_rejects_empty():
    with pytest.raises(ValidationError):
        ChatRequest(prompt="")


def test_chat_request_caps_length():
    with pytest.raises(ValidationError):
        ChatRequest(prompt="x" * 5000)


def test_awaken_requires_min_length():
    with pytest.raises(ValidationError):
        AwakenRequest(api_key="short")


def test_batch_caps_at_25():
    BatchRequest(prompts=["one"] * 25)
    with pytest.raises(ValidationError):
        BatchRequest(prompts=["one"] * 26)


def test_compare_request_rejects_empty():
    with pytest.raises(ValidationError):
        CompareRequest(prompt="")
