from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Iterator

from .logging_utils import get_logger


@contextmanager
def log_timed_step(
    action: str,
    *,
    logger_name: str | None = None,
    level: str = "info",
    **context,
) -> Iterator[None]:
    logger = get_logger(logger_name)
    log_fn = getattr(logger, level, logger.info)
    if context:
        details = ", ".join(f"{key}={value}" for key, value in context.items())
        log_fn("start %s | %s", action, details)
    else:
        log_fn("start %s", action)
    start = perf_counter()
    try:
        yield
    except Exception:
        logger.exception("failed %s after %.3fs", action, perf_counter() - start)
        raise
    else:
        log_fn("done %s in %.3fs", action, perf_counter() - start)
