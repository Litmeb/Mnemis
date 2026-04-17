from __future__ import annotations

import logging
import os
from pathlib import Path


LOGGER_NAME = "mnemis_build"


def get_logger(name: str | None = None) -> logging.Logger:
    root = logging.getLogger(LOGGER_NAME)
    if name:
        return root.getChild(name)
    return root


def configure_logging() -> Path:
    log_path = Path(os.getenv("MNEMIS_LOG_PATH", "results/logs/mnemis_build.log"))
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

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    stream_handler._mnemis_handler = True  # type: ignore[attr-defined]
    logger.addHandler(stream_handler)

    logger.info("logging configured at %s", log_path)
    return log_path
