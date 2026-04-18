from __future__ import annotations

import logging
import os
from pathlib import Path
from datetime import datetime


LOGGER_NAME = "mnemis_build"


def get_logger(name: str | None = None) -> logging.Logger:
    root = logging.getLogger(LOGGER_NAME)
    if name:
        return root.getChild(name)
    return root


def _default_log_path() -> Path:
    base_dir = Path(os.getenv("MNEMIS_LOG_DIR", "results/logs"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir / timestamp / "mnemis_build.log"


def _env_flag(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def configure_logging() -> Path:
    configured_log_path = os.getenv("MNEMIS_LOG_PATH")
    log_path = Path(configured_log_path) if configured_log_path else _default_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        if getattr(handler, "_mnemis_handler", False):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    file_handler._mnemis_handler = True  # type: ignore[attr-defined]
    logger.addHandler(file_handler)

    if _env_flag("MNEMIS_LOG_TO_CONSOLE"):
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        stream_handler._mnemis_handler = True  # type: ignore[attr-defined]
        logger.addHandler(stream_handler)

    logger.info("logging configured at %s", log_path)
    return log_path
