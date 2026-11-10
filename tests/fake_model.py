"""A scripted stand-in for Gemini, so tests make no network calls.

Register it by importing this module, then point the agent at it with
CHAT_MODEL=fake-maple. It plays three parts:

  - a plain question gets a short multi-sentence answer, streamed word by word
  - a cart order ("Place an order for me ...") calls submit_order, the same
    tool the real agent calls, so the approval flow runs for real
  - the messages the runner sends after an admin decision get a confirmation

Set FAKE_MODEL_DELAY (seconds per streamed word) to slow it down enough to
watch the stopwatch and the sentence-by-sentence speech.
"""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import AsyncGenerator

from google.adk.models import BaseLlm, LLMRegistry
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from backend.tools import CATALOG

_ORDER = re.compile(r"Items: (.+?)\. Name: (.+?)\. Email: (.+?)\. Use submit_order")


def _text_response(text: str, *, partial: bool = False) -> LlmResponse:
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=text)]),
        partial=partial,
    )


class FakeGemini(BaseLlm):
    model: str = "fake-maple"

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r"fake-maple"]

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        parts = llm_request.contents[-1].parts if llm_request.contents else []
        user_text = " ".join(p.text or "" for p in parts)
        got_tool_result = any(p.function_response for p in parts)

        if got_tool_result:
            reply = (
                "I have queued your order for an admin to review. It will not "
                "ship until a Vector admin approves it. Anything else I can help with?"
            )
        elif user_text.startswith("HUMAN_APPROVED_AND_RAN_TOOL"):
            reply = (
                "Good news. An admin approved your order and it is on its way. "
                "Thanks for shopping with Vector. Anything else I can help with?"
            )
        elif user_text.startswith("HUMAN_DENIED_TOOL_CALL"):
            reply = (
                "Sorry, an admin declined that order. Tell me what to change and "
                "I will submit it again. What would you like to do?"
            )
        elif (match := _ORDER.search(user_text)) is not None:
            by_name = {meta["name"]: cid for cid, meta in CATALOG.items()}
            wanted = [n.strip() for n in match.group(1).split(",")]
            call = types.FunctionCall(
                name="submit_order",
                args={
                    "items": [by_name.get(n, n) for n in wanted],
                    "name": match.group(2),
                    "email": match.group(3),
                },
            )
            yield LlmResponse(content=types.Content(role="model", parts=[types.Part(function_call=call)]))
            return
        else:
            reply = (
                "Vector Institute is Canada's AI institute, based at MaRS in Toronto. "
                "DARMOD is its applied program for working professionals. "
                "You asked: " + user_text.strip()[:80] + ". "
                "Want to know more about the program?"
            )

        delay = float(os.environ.get("FAKE_MODEL_DELAY", "0"))
        if stream:
            words = reply.split(" ")
            for i, word in enumerate(words):
                if delay:
                    await asyncio.sleep(delay)
                yield _text_response(word + (" " if i < len(words) - 1 else ""), partial=True)
        elif delay:
            await asyncio.sleep(delay * len(reply.split(" ")))
        yield _text_response(reply)


LLMRegistry.register(FakeGemini)
