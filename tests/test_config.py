from __future__ import annotations

from pathlib import Path

import pytest

from milknado.domains.common.config import (
    MilknadoConfig,
    _escape_toml_string,
    default_config,
    load_config,
    save_config,
)


class TestDefaultConfig:
    def test_returns_milknado_config(self, tmp_path: Path) -> None:
        cfg = default_config(tmp_path)
        assert isinstance(cfg, MilknadoConfig)
        assert cfg.project_root == tmp_path
        assert cfg.agent_family == "claude"

    def test_db_path_under_project_root(self, tmp_path: Path) -> None:
        cfg = default_config(tmp_path)
        assert cfg.db_path == tmp_path / ".milknado" / "milknado.db"

    def test_planning_agent_not_empty(self, tmp_path: Path) -> None:
        cfg = default_config(tmp_path)
        assert len(cfg.planning_agent) > 0

    def test_execution_agent_not_empty(self, tmp_path: Path) -> None:
        cfg = default_config(tmp_path)
        assert len(cfg.execution_agent) > 0


class TestLoadConfig:
    def _write_toml(self, tmp_path: Path, content: str) -> Path:
        p = tmp_path / "milknado.toml"
        p.write_text(content, encoding="utf-8")
        return p

    def test_loads_minimal_config(self, tmp_path: Path) -> None:
        toml = '[milknado]\nagent_family = "claude"\n'
        path = self._write_toml(tmp_path, toml)
        cfg = load_config(path)
        assert cfg.agent_family == "claude"
        assert cfg.project_root == tmp_path

    def test_loads_custom_quality_gates(self, tmp_path: Path) -> None:
        toml = (
            '[milknado]\n'
            'agent_family = "claude"\n'
            'quality_gates = ["uv run pytest", "uv run ruff check"]\n'
        )
        path = self._write_toml(tmp_path, toml)
        cfg = load_config(path)
        assert "uv run pytest" in cfg.quality_gates

    def test_loads_concurrency_limit(self, tmp_path: Path) -> None:
        toml = '[milknado]\nagent_family = "claude"\nconcurrency_limit = 8\n'
        path = self._write_toml(tmp_path, toml)
        cfg = load_config(path)
        assert cfg.concurrency_limit == 8

    def test_loads_custom_planning_agent(self, tmp_path: Path) -> None:
        toml = (
            '[milknado]\n'
            'agent_family = "claude"\n'
            'planning_agent = "claude --model opus"\n'
        )
        path = self._write_toml(tmp_path, toml)
        cfg = load_config(path)
        assert "opus" in cfg.planning_agent

    def test_loads_custom_execution_agent(self, tmp_path: Path) -> None:
        toml = (
            '[milknado]\n'
            'agent_family = "claude"\n'
            'execution_agent = "claude --model haiku"\n'
        )
        path = self._write_toml(tmp_path, toml)
        cfg = load_config(path)
        assert "haiku" in cfg.execution_agent

    def test_invalid_family_raises(self, tmp_path: Path) -> None:
        toml = '[milknado]\nagent_family = "openai"\n'
        path = self._write_toml(tmp_path, toml)
        with pytest.raises(ValueError, match="Invalid agent_family"):
            load_config(path)

    def test_db_path_relative_to_project(self, tmp_path: Path) -> None:
        toml = '[milknado]\nagent_family = "claude"\ndb_path = ".milknado/custom.db"\n'
        path = self._write_toml(tmp_path, toml)
        cfg = load_config(path)
        assert cfg.db_path == tmp_path / ".milknado" / "custom.db"

    def test_loads_all_numeric_fields(self, tmp_path: Path) -> None:
        toml = (
            '[milknado]\n'
            'agent_family = "claude"\n'
            'stall_threshold_seconds = 600\n'
            'dispatch_max_retries = 5\n'
            'dispatch_backoff_seconds = 10.0\n'
            'completion_timeout_seconds = 3600.0\n'
            'eta_sample_size = 20\n'
        )
        path = self._write_toml(tmp_path, toml)
        cfg = load_config(path)
        assert cfg.stall_threshold_seconds == 600
        assert cfg.dispatch_max_retries == 5
        assert cfg.dispatch_backoff_seconds == 10.0
        assert cfg.completion_timeout_seconds == 3600.0
        assert cfg.eta_sample_size == 20

    def test_loads_plugins_list(self, tmp_path: Path) -> None:
        toml = '[milknado]\nagent_family = "claude"\nplugins = ["plugin-a", "plugin-b"]\n'
        path = self._write_toml(tmp_path, toml)
        cfg = load_config(path)
        assert "plugin-a" in cfg.plugins

    def test_loads_protected_branches(self, tmp_path: Path) -> None:
        toml = '[milknado]\nagent_family = "claude"\nprotected_branches = ["main", "develop"]\n'
        path = self._write_toml(tmp_path, toml)
        cfg = load_config(path)
        assert "develop" in cfg.protected_branches

    def test_top_level_without_milknado_section(self, tmp_path: Path) -> None:
        toml = 'agent_family = "claude"\n'
        path = self._write_toml(tmp_path, toml)
        cfg = load_config(path)
        assert cfg.agent_family == "claude"


class TestSaveConfig:
    def test_writes_toml_file(self, tmp_path: Path) -> None:
        cfg = default_config(tmp_path)
        path = tmp_path / "milknado.toml"
        save_config(cfg, path)
        assert path.exists()
        content = path.read_text()
        assert "[milknado]" in content
        assert "agent_family" in content

    def test_roundtrip(self, tmp_path: Path) -> None:
        cfg = default_config(tmp_path)
        path = tmp_path / "milknado.toml"
        save_config(cfg, path)
        loaded = load_config(path)
        assert loaded.agent_family == cfg.agent_family
        assert loaded.concurrency_limit == cfg.concurrency_limit

    def test_escapes_backslashes(self, tmp_path: Path) -> None:
        cfg = MilknadoConfig(
            planning_agent='cmd "quoted"',
            execution_agent="cmd",
            project_root=tmp_path,
            db_path=tmp_path / ".milknado" / "milknado.db",
        )
        path = tmp_path / "milknado.toml"
        save_config(cfg, path)
        content = path.read_text()
        assert '\\"quoted\\"' in content


class TestEscapeTomlString:
    def test_escapes_backslash(self) -> None:
        assert _escape_toml_string("a\\b") == "a\\\\b"

    def test_escapes_double_quote(self) -> None:
        assert _escape_toml_string('a"b') == 'a\\"b'

    def test_no_change_for_plain_string(self) -> None:
        assert _escape_toml_string("hello world") == "hello world"
