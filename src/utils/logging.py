"""Structured logging for Cross-Cloud AgentOps."""
from __future__ import annotations
import logging, sys
from typing import Any
import structlog

def configure_logging(level: str = "INFO", json_output: bool = False) -> None:
    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]
    renderer = structlog.processors.JSONRenderer() if json_output else structlog.dev.ConsoleRenderer(colors=True)
    structlog.configure(
        processors=shared + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, level.upper()))
    for noisy in ("boto3", "botocore", "azure", "google", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

def get_logger(name: str, **ctx: Any) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name).bind(**ctx)

configure_logging()
