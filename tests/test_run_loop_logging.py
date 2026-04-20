"""Tests for configure_run_logging level-restore and file-creation behaviour."""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from milknado.domains.execution.run_loop._logging import configure_run_logging, ts


class TestConfigureRunLoggingLevelRestore:
    def test_restores_warning_level(self, tmp_path: Path) -> None:
        milknado_logger = logging.getLogger("milknado")
        milknado_logger.setLevel(logging.WARNING)

        with configure_run_logging(tmp_path):
            assert milknado_logger.level == logging.INFO

        assert milknado_logger.level == logging.WARNING

    def test_restores_debug_level(self, tmp_path: Path) -> None:
        milknado_logger = logging.getLogger("milknado")
        milknado_logger.setLevel(logging.DEBUG)

        with configure_run_logging(tmp_path):
            pass

        assert milknado_logger.level == logging.DEBUG

    def test_restores_level_on_exception(self, tmp_path: Path) -> None:
        milknado_logger = logging.getLogger("milknado")
        milknado_logger.setLevel(logging.WARNING)

        with pytest.raises(ValueError):
            with configure_run_logging(tmp_path):
                raise ValueError("boom")

        assert milknado_logger.level == logging.WARNING


class TestConfigureRunLoggingFileCreation:
    def test_creates_log_file_under_milknado_dir(self, tmp_path: Path) -> None:
        with configure_run_logging(tmp_path) as log_path:
            assert log_path.exists()
            assert log_path.parent == tmp_path / ".milknado"

    def test_log_filename_starts_with_run(self, tmp_path: Path) -> None:
        with configure_run_logging(tmp_path) as log_path:
            assert log_path.name.startswith("run-")

    def test_log_filename_ends_with_z(self, tmp_path: Path) -> None:
        with configure_run_logging(tmp_path) as log_path:
            assert log_path.name.endswith(".log")

    def test_creates_milknado_dir_if_missing(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        project.mkdir()

        with configure_run_logging(project) as log_path:
            assert (project / ".milknado").is_dir()
            assert log_path.exists()


class TestConfigureRunLoggingHandlerCleanup:
    def test_handler_removed_after_context(self, tmp_path: Path) -> None:
        milknado_logger = logging.getLogger("milknado")
        before_count = len(milknado_logger.handlers)

        with configure_run_logging(tmp_path):
            assert len(milknado_logger.handlers) == before_count + 1

        assert len(milknado_logger.handlers) == before_count

    def test_handler_removed_on_exception(self, tmp_path: Path) -> None:
        milknado_logger = logging.getLogger("milknado")
        before_count = len(milknado_logger.handlers)

        with pytest.raises(RuntimeError):
            with configure_run_logging(tmp_path):
                raise RuntimeError("error during run")

        assert len(milknado_logger.handlers) == before_count

    def test_messages_written_to_log_file(self, tmp_path: Path) -> None:
        with configure_run_logging(tmp_path) as log_path:
            logging.getLogger("milknado").info("test message hello")

        content = log_path.read_text(encoding="utf-8")
        assert "test message hello" in content


class TestTs:
    def test_ts_returns_time_string(self) -> None:
        result = ts()
        parts = result.split(":")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)
