"""
Structured logging utilities using Python's stdlib logging.
Outputs JSON-formatted logs in production, human-readable in development.
"""
import logging
import json
import sys
from datetime import datetime, timezone
from typing import Any


class JSONFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object."""

    RESERVED = {"message", "levelname", "name", "asctime"}

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Attach extras (e.g. request_id, user, latency_ms)
        for key, value in record.__dict__.items():
            if (
                not key.startswith("_")
                and key not in self.RESERVED
                and key
                not in {
                    "args", "created", "exc_info", "exc_text", "filename",
                    "funcName", "levelno", "lineno", "module", "msecs",
                    "msg", "pathname", "process", "processName", "relativeCreated",
                    "stack_info", "thread", "threadName",
                }
            ):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", json_output: bool = False) -> None:
    """
    Configure root logger.

    Args:
        level: Log level string ("DEBUG", "INFO", "WARNING", etc.)
        json_output: Use JSON formatter for structured logging (set True in production).
    """
    handler = logging.StreamHandler(sys.stdout)

    if json_output:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Quieten noisy third-party loggers
    for noisy in ("httpx", "httpcore", "pinecone", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger (call this in every module)."""
    return logging.getLogger(name)
