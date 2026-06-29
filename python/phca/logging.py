"""
PHCA v3.0 — Shared Logging Helper.

Provides a `_log()` function that works with both structlog (structured logging)
and stdlib logging (fallback when structlog is not installed).

Usage:
    from phca.logging import logger, _log
    _log(logger, "info", "event.name", key1=value1, key2=value2)
"""

try:
    import structlog
    _STRUCTLOG_AVAILABLE = True
    logger = structlog.get_logger()
except ImportError:
    import logging
    _STRUCTLOG_AVAILABLE = False
    logger = logging.getLogger("phca")


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
