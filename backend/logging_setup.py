"""Structured logging via structlog.

JSON in production (so Cloud Run logs are queryable), pretty console in dev.
Every log line gets the session_id, request_id, and (when applicable) the
tool name automatically through context vars set by the request middleware.
"""

from __future__ import annotations

import logging
import sys

import structlog
from structlog.contextvars import merge_contextvars

from .config import get_settings


def configure_logging() -> None:
    settings = get_settings()
    level = getattr(logging, settings.log_level)

    # Quiet some noisy upstream libraries; they leak INFO at request volume.
    for noisy in ("httpx", "httpcore", "google_genai", "google_adk"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))

    shared_processors: list[structlog.types.Processor] = [
        merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.environment == "dev":
        # Human-readable colored output for local work.
        renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer()
    else:
        # JSON lines for Cloud Logging.
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
