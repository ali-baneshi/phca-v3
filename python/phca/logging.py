"""
PHCA v3.0 — Shared Logging Helper.

Provides a `_log()` function that works with both structlog (structured logging)
and stdlib logging (fallback when structlog is not installed).

Usage:
    from phca.logging import logger, _log
    _log(logger, "info", "event.name", key1=value1, key2=value2)
"""

from __future__ import annotations

try:
    import structlog
    _STRUCTLOG_AVAILABLE = True
    logger = structlog.get_logger()
except ImportError:
    import logging as _logging
    _STRUCTLOG_AVAILABLE = False
    logger = _logging.getLogger("phca")

# ── File Logging ─────────────────────────────────────────────

_LOG_DIR = "logs"
_LOG_FILE = "phca.log"
_MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
_BACKUP_COUNT = 3


_FILE_LOGGING_CONFIGURED: bool = False


def setup_file_logging(log_dir: str = _LOG_DIR) -> None:
    """Configure a file handler for the PHCA logger.

    Creates a log file in log_dir that is written by all _log() calls.
    Rotates at _MAX_LOG_SIZE, keeps _BACKUP_COUNT backups.
    Idempotent: calling multiple times does not duplicate handlers.

    Args:
        log_dir: Directory for log files (default: 'logs').
    """
    global _FILE_LOGGING_CONFIGURED
    if _FILE_LOGGING_CONFIGURED:
        return

    import os
    from pathlib import Path
    from logging.handlers import RotatingFileHandler
    import logging

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    log_file_path = log_path / _LOG_FILE

    if _STRUCTLOG_AVAILABLE:
        # structlog: write JSON lines to file via PrintLogger
        import structlog as _structlog
        from structlog.processors import JSONRenderer

        _structlog.configure(
            processors=[
                _structlog.stdlib.filter_by_level,
                _structlog.stdlib.add_log_level,
                _structlog.stdlib.PositionalArgumentsFormatter(),
                _structlog.processors.TimeStamper(fmt="iso"),
                _structlog.processors.StackInfoRenderer(),
                _structlog.processors.format_exc_info,
                JSONRenderer(),
            ],
            wrapper_class=_structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=_structlog.PrintLoggerFactory(
                file=open(log_file_path, "a", buffering=1),
            ),
            cache_logger_on_first_use=True,
        )
    else:
        # stdlib: write formatted logs with key=value pairs
        # Check if handler already exists to avoid duplicates
        existing = [
            h for h in logger.handlers
            if isinstance(h, RotatingFileHandler)
            and h.baseFilename == str(log_file_path)
        ]
        if existing:
            return

        handler = RotatingFileHandler(
            str(log_file_path),
            maxBytes=_MAX_LOG_SIZE,
            backupCount=_BACKUP_COUNT,
        )
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

    _FILE_LOGGING_CONFIGURED = True


def ensure_logging() -> None:
    """Ensure file logging is configured (idempotent).

    Safe to call at module import time — only configures once.
    """
    if not getattr(ensure_logging, "_configured", False):
        setup_file_logging()
        ensure_logging._configured = True


def _log(logger_obj, level: str, msg: str, **kwargs):
    """
    Log a message, compatible with both structlog and stdlib logging.

    Args:
        logger_obj: Logger instance (structlog or stdlib).
        level: Log level string ('info', 'warning', 'error', 'critical', 'debug').
        msg: Event name or log message.
        **kwargs: Key-value pairs for structured logging (structlog)
                  or appended as formatted string (stdlib).
    """
    log_func = getattr(logger_obj, level, None)
    if log_func is None:
        return

    if _STRUCTLOG_AVAILABLE:
        log_func(msg, **kwargs)
    else:
        extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
        if extra:
            log_func("%s %s", msg, extra)
        else:
            log_func("%s", msg)
