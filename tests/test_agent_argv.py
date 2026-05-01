from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from milknado.domains.common import MilknadoConfig, default_config
from milknado.domains.common.agent_argv import (
    build_minimal_mcp_env,
    build_planning_subprocess,
    resolve_execution_agent_command,
    resolve_planning_agent_command,
)
from milknado.domains.common.config import load_config, save_config


def test_resolve_planning_uses_override() -> None:
    assert (
        resolve_planning_agent_command(
            "claude",
            planning_agent="my-planner --flag",
        )
        == "my-planner --flag"
    )


def test_resolve_execution_uses_override() -> None:
    assert (
        resolve_execution_agent_command(
            "claude",
            execution_agent="my-exec --flag",
        )
        == "my-exec --flag"
    )


def test_non_default_family_with_override_keeps_planning_and_execution_consistent(
    tmp_path: Path,
) -> None:
    p = tmp_path / "ctx.md"
    p.write_text("hello world", encoding="utf-8")
    override = "claude --model custom -p --dangerously-skip-permissions"

    planning_command = resolve_planning_agent_command(
        "claude",
        planning_agent=override,
    )
    execution_command = resolve_execution_agent_command(
        "claude",
        execution_agent=override,
    )
    argv, _extra = build_planning_subprocess(p, planning_command)

    assert planning_command == override
    assert execution_command == override
    assert argv[0] == "claude"
    assert "--model" in argv
    assert "custom" in argv
    assert argv[-1] == "-"


def test_build_planning_subprocess_uses_stdin(tmp_path: Path) -> None:
    p = tmp_path / "ctx.md"
    p.write_text("hello world", encoding="utf-8")
    argv, extra = build_planning_subprocess(p, "echo")
    assert argv[0] == "echo"
    assert argv[-1] == "-"
    assert extra.get("text") is True
    assert extra.get("input") == "hello world"
    assert isinstance(extra.get("env"), dict)


def test_build_planning_subprocess_allows_external_mcp(tmp_path: Path) -> None:
    p = tmp_path / "ctx.md"
    p.write_text("hello world", encoding="utf-8")
    _argv, extra = build_planning_subprocess(
        p,
        "echo",
        allow_external_mcp=True,
    )
    assert "env" not in extra


def test_build_planning_subprocess_adds_repo_mcp_config(tmp_path: Path) -> None:
    p = tmp_path / "ctx.md"
    p.write_text("hello world", encoding="utf-8")
    (tmp_path / ".mcp.json").write_text('{"mcpServers": {}}', encoding="utf-8")
    argv, extra = build_planning_subprocess(p, "echo", project_root=tmp_path)
    assert "--mcp-config" in argv
    assert str(tmp_path / ".mcp.json") in argv
    assert isinstance(extra.get("env"), dict)


def test_build_minimal_mcp_env_strips_external_mcp() -> None:
    mocked_env = {
        "MCP_SERVER_URL": "https://example.com/mcp",
        "MILKNADO_MCP_MODE": "local",
        "CRG_EMBEDDING_MODEL": "all-MiniLM-L6-v2",
        "PATH": "/bin",
    }
    with patch("milknado.domains.common.agent_argv.os.environ", mocked_env):
        env = build_minimal_mcp_env()
    assert "MCP_SERVER_URL" not in env
    assert env["MILKNADO_MCP_MODE"] == "local"
    assert env["CRG_EMBEDDING_MODEL"] == "all-MiniLM-L6-v2"


def test_load_config_roundtrip_split_agents(tmp_path: Path) -> None:
    cfg_path = tmp_path / "milknado.toml"
    cfg = MilknadoConfig(
        agent_family="gemini",
        planning_agent="gemini --model gemini-3.1-pro-preview -p --yolo",
        execution_agent="gemini --model gemini-2.5-flash -p --yolo",
        project_root=tmp_path,
        db_path=tmp_path / ".milknado" / "milknado.db",
    )
    save_config(cfg, cfg_path)
    loaded = load_config(cfg_path)
    assert loaded.agent_family == "gemini"
    assert "gemini" in loaded.planning_agent
    assert "gemini" in loaded.execution_agent


def test_default_config_uses_claude_preset(tmp_path: Path) -> None:
    cfg = default_config(tmp_path)
    assert cfg.agent_family == "claude"
    assert "--model opus" in cfg.planning_agent
    assert "--model sonnet" in cfg.execution_agent


def test_load_config_rejects_unknown_family(tmp_path: Path) -> None:
    cfg_path = tmp_path / "milknado.toml"
    cfg_path.write_text(
        '[milknado]\nagent_family = "unknown"\n',
        encoding="utf-8",
    )
    try:
        load_config(cfg_path)
        assert False, "Expected ValueError for invalid agent_family"
    except ValueError as exc:
        assert "Invalid agent_family" in str(exc)
