# Structured JSON logs, one line per request - lets me grep/jq for stuff
# later ("how many /ask requests hit the guardrail today") instead of
# re-running things to find out.
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
        # extra={...} fields ride along here instead of needing a custom
        # formatter per call site
        for key, value in getattr(record, "extra_fields", {}).items():
            payload[key] = value
        return json.dumps(payload)


def configure_logging() -> logging.Logger:
    logger = logging.getLogger("rap_doclayout")
    if logger.handlers:
        # avoid double handlers on repeated calls (app reload, test imports)
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


@contextmanager
def log_request(logger: logging.Logger, endpoint: str, **fields):
    """Logs one line per request with latency, success or failure either
    way.

    Usage:
        with log_request(logger, "/detect") as ctx:
            ...
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
