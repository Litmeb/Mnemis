from pathlib import Path

import mnemis_build.logging_utils as logging_utils


def _mnemis_handlers():
    logger = logging_utils.get_logger()
    return [handler for handler in logger.handlers if getattr(handler, "_mnemis_handler", False)]


def test_configure_logging_uses_timestamped_run_directory_by_default(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MNEMIS_LOG_PATH", raising=False)
    monkeypatch.setenv("MNEMIS_LOG_DIR", str(tmp_path))
    monkeypatch.delenv("MNEMIS_LOG_TO_CONSOLE", raising=False)

    log_path = logging_utils.configure_logging()

    assert log_path.parent.parent == tmp_path
    assert log_path.name == "mnemis_build.log"
    assert log_path.parent.name
    assert log_path.parent.exists()
    assert len(_mnemis_handlers()) == 1


def test_configure_logging_respects_explicit_log_path(monkeypatch, tmp_path: Path) -> None:
    explicit_path = tmp_path / "custom" / "run.log"
    monkeypatch.setenv("MNEMIS_LOG_PATH", str(explicit_path))
    monkeypatch.delenv("MNEMIS_LOG_DIR", raising=False)
    monkeypatch.delenv("MNEMIS_LOG_TO_CONSOLE", raising=False)

    log_path = logging_utils.configure_logging()

    assert log_path == explicit_path
    assert log_path.parent.exists()
    assert len(_mnemis_handlers()) == 1


def test_configure_logging_can_enable_console_logging(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MNEMIS_LOG_PATH", raising=False)
    monkeypatch.setenv("MNEMIS_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("MNEMIS_LOG_TO_CONSOLE", "1")

    logging_utils.configure_logging()

    assert len(_mnemis_handlers()) == 2
