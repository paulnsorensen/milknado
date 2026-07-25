"""Tests for configure_run_logging level-restore and file-creation behaviour."""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import pytest

import milknado.domains.execution.run_loop._logging as run_logging
from milknado.domains.execution.run_loop._logging import (
    _logger,
    configure_run_logging,
    ts,
)
from milknado.process_logging import configure_stderr_logging


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

    def test_log_filename_matches_iso8601_pattern(self, tmp_path: Path) -> None:
        with configure_run_logging(tmp_path) as log_path:
            assert re.fullmatch(r"run-\d{8}T\d{12}Z-[0-9a-f]{32}\.log", log_path.name)

    def test_same_timestamp_log_creation_is_unique(self, tmp_path: Path) -> None:
        with configure_run_logging(tmp_path) as first, configure_run_logging(tmp_path) as second:
            assert first != second

    def test_pruning_protects_active_logs_and_enforces_byte_bound(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        log_dir = tmp_path / ".milknado"
        log_dir.mkdir()
        active = log_dir / "run-active.log"
        active.write_bytes(b"x" * 20)
        for index in range(3):
            (log_dir / f"run-finished-{index}.log").write_bytes(b"x" * 10)
        monkeypatch.setattr(run_logging, "_MAX_LOG_BYTES", 20)
        run_logging._prune_old_logs(log_dir, "run-*.log", keep=2, active_run_ids=("active",))

        assert active.exists()
        finished_bytes = sum(
            path.stat().st_size for path in log_dir.glob("run-*.log") if path != active
        )
        assert finished_bytes <= 20

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


class TestConfigureStderrLogging:
    @pytest.fixture(autouse=True)
    def _restore_handlers(self) -> object:
        original = list(_logger.handlers)
        yield
        _logger.handlers = original

    def test_second_call_reuses_handler_and_adds_no_duplicate(self) -> None:
        """Idempotency matters: the MCP server may reconfigure logging more than
        once in-process (tests, embedding), and a stacked stderr handler would
        double every log line. The second call must return the same handler and
        leave the stderr-handler count at one."""
        first = configure_stderr_logging()
        second = configure_stderr_logging()

        assert second is first
        stderr_handlers = [
            h
            for h in _logger.handlers
            if isinstance(h, logging.StreamHandler) and h.stream is sys.stderr
        ]
        assert len(stderr_handlers) == 1
