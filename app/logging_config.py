"""
Structured JSON logging for the API, one line per request.

I wanted every request to leave a trace I could grep later - which endpoint,
how long it took, how many detections came back, whether the guardrail
tripped - without having to reproduce the request to find out. Plain text
logs are fine to read live in a terminal, but they are miserable to query
later ("how many /ask requests hit the guardrail today?"), and JSON lines
answer that with a one-line jq/grep instead of re-running anything.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextlib import contextmanager


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        # Anything passed via logger.info(..., extra={...}) rides along here,
        # rather than needing its own bespoke formatter per call site.
        for key, value in getattr(record, "extra_fields", {}).items():
            payload[key] = value
        return json.dumps(payload)


def configure_logging() -> logging.Logger:
    logger = logging.getLogger("rap_doclayout")
    if logger.handlers:
        # configure_logging() can be called more than once (app reload,
        # tests importing main multiple times) - without this guard every
        # call adds another handler and every request gets logged N times.
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


@contextmanager
def log_request(logger: logging.Logger, endpoint: str, **fields):
    """
    Wraps one request: logs a single structured line when it finishes, with
    latency and whatever extra fields the caller wants attached (detection
    count, guardrail rule, etc.) - regardless of whether the request
    succeeded or raised.

    Usage:
        with log_request(logger, "/detect") as ctx:
            ... do the work ...
            ctx["detection_count"] = len(detections)
    """
    request_id = uuid.uuid4().hex[:12]
    started = time.perf_counter()
    ctx = {"request_id": request_id, "endpoint": endpoint, **fields}
    error: Exception | None = None
    try:
        yield ctx
    except Exception as exc:
        error = exc
        raise
    finally:
        ctx["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        if error is not None:
            ctx["error"] = str(error)
            logger.error("request failed", extra={"extra_fields": ctx})
        else:
            logger.info("request completed", extra={"extra_fields": ctx})
