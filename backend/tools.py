"""Tools available to Maple.

Two tools, that is the whole surface:

    web_search        Safe. Google Custom Search first (if configured),
                                        Tavily second, DuckDuckGo Instant Answer as last fallback.
                                        Maple calls this for any factual question; Vector Institute
                                        knowledge is layered on top by the system prompt.

  submit_order      HITL-gated. Submits a Vector store order. The order is
                    queued in OrderStore; nothing ships until an admin
                    approves it from the Admin tab.
"""

from __future__ import annotations

import os

import httpx

from .logging_setup import get_logger
from . import config

log = get_logger(__name__)

# The 'ship' equivalent for the ordering tool. In a real app this would call
# Shopify, Salesforce, an internal warehouse system. For the demo we just log
# what would have happened and return a confirmation string.
_DEMO_ORDER_LOG: list[dict] = []

# --- Safe tool: web_search (Google -> Tavily -> DDG fallback) -------------

async def google_search_with_images(query: str, max_results: int = 5) -> tuple[str, str | None]:
    """Search the web with Google, and optionally find an image.

    Args:
        query: The search query.
        max_results: The maximum number of results to return.

    Returns:
        A tuple containing the search results as a string, and an image URL
        if one is found.
    """
    # For simplicity, we'll use the web_search tool for this.
    # In a real application, you might use a dedicated image search API.
    text_results = await web_search(query, max_results)

    # Simple image search: search for the query + "image" and take the first
    # image result.
    image_search_query = f"{query} image"
    image_results = await web_search(image_search_query, 1)

    image_url = None
    if image_results:
        # This is a very simple way to find an image URL. A real implementation
        # would be more robust.
        import re
        match = re.search(r'https?://\S+\.(?:jpg|jpeg|gif|png)', image_results)
        if match:
            image_url = match.group(0)

    return text_results, image_url


async def web_search(query: str, max_results: int = 5) -> str:
    """Search the web and return a concise grounding text block.

    Args:
        query: The search query string.
        max_results: How many results to pull. Cap at 5 for the lecture demo
            so the model isn't drowned in context.

    Provider order:
      1) Google Custom Search (set GOOGLE_API_KEY + GOOGLE_CSE_ID)
      2) Tavily (set TAVILY_API_KEY)
      3) DuckDuckGo Instant Answer (no key)
    """
    max_results = min(max_results, 5)

    # 1) Google Programmable Search (Custom Search JSON API)
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    google_cse_id = os.environ.get("GOOGLE_CSE_ID")
    if google_api_key and google_cse_id:
        google = await _web_search_google_cse(
            query=query,
            max_results=max_results,
            api_key=google_api_key,
            cse_id=google_cse_id,
        )
        if google and not google.startswith("web_search failed"):
            return google

    # 2) Tavily
    api_key = os.environ.get("TAVILY_API_KEY")
    if api_key:
        tavily = await _web_search_tavily(
            query=query,
            max_results=max_results,
            api_key=api_key,
        )
        if tavily and not tavily.startswith("web_search failed"):
            return tavily

    # 3) No-key fallback
    return await _web_search_duckduckgo(query=query, max_results=max_results)


async def _web_search_google_cse(
    query: str,
    max_results: int,
    api_key: str,
    cse_id: str,
) -> str:
    """Search with Google Programmable Search (Custom Search JSON API)."""
    log.info("tool.web_search.google.start", query=query[:80])
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            resp = await client.get(
                "https://www.googleapis.com/customsearch/v1",
                params={
                    "key": api_key,
                    "cx": cse_id,
                    "q": query,
                    "num": max_results,
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as exc:
        log.warning("tool.web_search.google.http_error", error=str(exc))
        return f"web_search failed: {exc}. Answer from what you already know."

    items = data.get("items", []) or []
    parts: list[str] = []
    for r in items[:max_results]:
        title = r.get("title") or "(no title)"
        url = r.get("link") or ""
        snippet = (r.get("snippet") or "").strip().replace("\n", " ")
        if len(snippet) > 300:
            snippet = snippet[:300] + "..."
        parts.append(f"- {title} ({url}): {snippet}")

    if not parts:
        return "No results."
    log.info("tool.web_search.google.ok", n_results=len(items))
    return "\n".join(parts)[:3500]


async def _web_search_tavily(query: str, max_results: int, api_key: str) -> str:
    """Search via Tavily API."""
    log.info("tool.web_search.start", query=query[:80])
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "basic",
                    "include_answer": True,
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as exc:
        log.warning("tool.web_search.http_error", error=str(exc))
        return f"web_search failed: {exc}. Answer from what you already know."

    parts: list[str] = []
    answer = data.get("answer")
    if answer:
        parts.append(f"Summary: {answer}")
    for r in data.get("results", [])[:max_results]:
        title = r.get("title") or "(no title)"
        url = r.get("url") or ""
        snippet = (r.get("content") or "").strip().replace("\n", " ")
        if len(snippet) > 300:
            snippet = snippet[:300] + "..."
        parts.append(f"- {title} ({url}): {snippet}")
    if not parts:
        return "No results."
    log.info("tool.web_search.ok", n_results=len(data.get("results", [])))
    return "\n".join(parts)[:3500]


async def _web_search_duckduckgo(query: str, max_results: int = 5) -> str:
    """Fallback web search with no API key via DuckDuckGo Instant Answer."""
    log.info("tool.web_search.ddg.start", query=query[:80])
    try:
        async with httpx.AsyncClient(timeout=12.0, follow_redirects=True) as client:
            resp = await client.get(
                "https://api.duckduckgo.com/",
                params={
                    "q": query,
                    "format": "json",
                    "no_html": "1",
                    "no_redirect": "1",
                },
                headers={"User-Agent": "MapleBot/0.1"},
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as exc:
        log.warning("tool.web_search.ddg.http_error", error=str(exc))
        return (
            f"web_search failed: {exc}. Answer from what you already know, "
            "and mention this may not be current."
        )

    parts: list[str] = []

    abstract = (data.get("AbstractText") or "").strip()
    abstract_url = data.get("AbstractURL") or ""
    heading = data.get("Heading") or ""
    if abstract:
        parts.append(f"Summary: {heading}: {abstract} ({abstract_url})")

    def _collect_related(rows: list[dict], out: list[tuple[str, str, str]]) -> None:
        for row in rows:
            if "Topics" in row:
                _collect_related(row.get("Topics") or [], out)
                continue
            text = (row.get("Text") or "").strip()
            url = (row.get("FirstURL") or "").strip()
            if text:
                title = text.split(" - ", 1)[0]
                out.append((title, url, text))

    related: list[tuple[str, str, str]] = []
    _collect_related(data.get("RelatedTopics") or [], related)

    for title, url, text in related[: min(max_results, 5)]:
        snippet = text if len(text) <= 280 else text[:280] + "..."
        parts.append(f"- {title} ({url}): {snippet}")

    if not parts:
        return "No results."
    log.info("tool.web_search.ddg.ok", n_results=len(parts))
    return "\n".join(parts)[:3500]
			


# --- HITL-gated tool: submit_order -----------------------------------------

# Catalog of items Maple can order. Two pieces of branded swag and two
# Vector program info packets. The names here are what the agent passes to
# the tool; the metadata is what the admin and the order receipt show.
CATALOG: dict[str, dict] = {
    "vector-hoodie": {
        "name": "Vector Institute hoodie",
        "category": "swag",
        "price_cad": 65,
        "summary": "Pink-on-charcoal logo. Sizes XS to XXL.",
    },
    "vector-mug": {
        "name": "Vector Institute mug",
        "category": "swag",
        "price_cad": 18,
        "summary": "Ceramic, 350ml. Toronto-skyline imprint.",
    },
    "darmod-packet": {
        "name": "DARMOD program info packet",
        "category": "info",
        "price_cad": 0,
        "summary": "Curriculum, eligibility, deadlines, mailed to you.",
    },
    "fastlane-packet": {
        "name": "FastLane Applied Projects info packet",
        "category": "info",
        "price_cad": 0,
        "summary": "Cohort structure, partner orgs, application timeline.",
    },
}


def submit_order(items: list[str], name: str, email: str) -> str:
    """Submit a Vector store order for human review.

    HITL-GATED. The agent must request approval before this fires. The
    actual order is queued in the session and surfaces on the admin page;
    nothing ships until a human clicks Approve.

    Args:
        items: Catalog ids of the items to order. Each must be a key of
            CATALOG. Example: ["vector-hoodie", "darmod-packet"].
        name: Customer name.
        email: Where to send shipping/confirmation.

    Returns:
        A short confirmation string for the user.
    """
    unknown = [i for i in items if i not in CATALOG]
    if unknown:
        return (
            f"I don't have these in the catalog: {', '.join(unknown)}. "
            f"Available items: {', '.join(CATALOG.keys())}."
        )
    line_items = [CATALOG[i] for i in items]
    total = sum(it["price_cad"] for it in line_items)
    entry = {
        "items": items,
        "name": name,
        "email": email,
        "total_cad": total,
    }
    _DEMO_ORDER_LOG.append(entry)
    log.info("tool.submit_order.executed", **entry)
    if total > 0:
        return (
            f"Done. {len(items)} item(s), total ${total} CAD. "
            f"We'll email {email} when it ships."
        )
    return (
        f"Done. {len(items)} info packet(s) on the way to {email}."
    )


def get_order_log() -> list[dict]:
    """For tests and the runbook's 'show what got ordered' moment."""
    return list(_DEMO_ORDER_LOG)


def clear_order_log() -> None:
    _DEMO_ORDER_LOG.clear()


