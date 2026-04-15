from __future__ import annotations

from pathlib import Path

from milknado.domains.common import MilknadoConfig, default_config
from milknado.domains.common.agent_argv import (
    build_planning_subprocess,
    normalize_preset,
    resolve_execution_agent_command,
)
from milknado.domains.common.config import load_config, save_config


def test_normalize_preset_unknown() -> None:
    assert normalize_preset("unknown") == "custom"


def test_resolve_custom_uses_file_command() -> None:
    assert (
        resolve_execution_agent_command(
            "custom",
            agent_command="my-agent --flag",
            agent_command_key_present=True,
        )
        == "my-agent --flag"
    )


def test_resolve_claude_builtin_without_override() -> None:
    assert (
        resolve_execution_agent_command(
            "claude",
            agent_command="ignored",
            agent_command_key_present=False,
        )
        == "claude -p --dangerously-skip-permissions"
    )


def test_resolve_claude_override_when_key_present() -> None:
    assert (
        resolve_execution_agent_command(
            "claude",
            agent_command="claude -p --custom-flag",
            agent_command_key_present=True,
        )
        == "claude -p --custom-flag"
    )


def test_build_planning_custom_appends_body(tmp_path: Path) -> None:
    p = tmp_path / "ctx.md"
    p.write_text("hello world", encoding="utf-8")
    argv, extra = build_planning_subprocess(p, "custom", "echo")
    assert argv[:-1] == ["echo"]
    assert argv[-1] == "hello world"
    assert extra == {}


def test_build_planning_claude_stdin(tmp_path: Path) -> None:
    p = tmp_path / "ctx.md"
    p.write_text("stdin body", encoding="utf-8")
    argv, extra = build_planning_subprocess(
        p, "claude", "claude -p --dangerously-skip-permissions",
    )
    assert argv[-1] == "-"
    assert extra.get("text") is True
    assert extra.get("input") == "stdin body"


def test_load_config_roundtrip_preset(tmp_path: Path) -> None:
    cfg_path = tmp_path / "milknado.toml"
    cfg = MilknadoConfig(
        agent_preset="gemini",
        agent_command="gemini -p --yolo",
        project_root=tmp_path,
        db_path=tmp_path / ".milknado" / "milknado.db",
    )
    save_config(cfg, cfg_path)
    loaded = load_config(cfg_path)
    assert loaded.agent_preset == "gemini"
    assert "gemini -p" in loaded.agent_command


def test_default_config_uses_claude_preset(tmp_path: Path) -> None:
    cfg = default_config(tmp_path)
    assert cfg.agent_preset == "claude"
    assert "dangerously-skip-permissions" in cfg.agent_command
