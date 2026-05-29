from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from milknado.domains.common import MilknadoConfig, default_config
from milknado.domains.common.agent_argv import (
    WORKER_ALLOWED_TOOLS,
    build_minimal_mcp_env,
    build_planning_subprocess,
    resolve_execution_agent_command,
    resolve_planning_agent_command,
    resolve_worker_tools,
)
from milknado.domains.common.config import load_config, save_config


def test_worker_allowlist_grants_track_follow_up_not_delete_or_edit() -> None:
    claude = WORKER_ALLOWED_TOOLS["claude"]
    assert "mcp__milknado__milknado_track_follow_up" in claude
    joined = ",".join(claude)
    assert "delete_node" not in joined
    assert "edit_node" not in joined


def test_gemini_worker_allowlist_grants_track_follow_up() -> None:
    assert "milknado_track_follow_up" in WORKER_ALLOWED_TOOLS["gemini"]


def test_worker_allowlist_grants_serena_symbol_tools() -> None:
    claude = WORKER_ALLOWED_TOOLS["claude"]
    assert "mcp__serena__replace_symbol_body" in claude
    assert "mcp__serena__find_symbol" in claude
    # Shell-exec stays out: no granted serena tool may execute a shell/command.
    serena_shell = [
        t for t in claude if t.startswith("mcp__serena__") and ("shell" in t or "execute" in t)
    ]
    assert not serena_shell
    gemini = WORKER_ALLOWED_TOOLS["gemini"]
    assert "replace_symbol_body" in gemini
    assert "find_symbol" in gemini
    # Same Serena-scoped check for Gemini's raw (unprefixed) names: derive the
    # granted serena tool set from the Claude entries and assert none can exec.
    # Scoping to serena tools — rather than scanning the whole allowlist —
    # keeps the intentionally-granted ShellTool(rtk *) out of the check.
    serena_names = {
        t.removeprefix("mcp__serena__") for t in claude if t.startswith("mcp__serena__")
    }
    gemini_serena = [t for t in gemini if t in serena_names]
    assert not [t for t in gemini_serena if "shell" in t or "execute" in t]


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


# ── resolve_worker_tools / structured allowlist ───────────────────────────────


def test_resolve_worker_tools_default_returns_family_baseline() -> None:
    tools = resolve_worker_tools("claude")
    assert tools == WORKER_ALLOWED_TOOLS["claude"]


def test_resolve_worker_tools_allow_replaces_default() -> None:
    tools = resolve_worker_tools("claude", allow=["Read", "Edit"])
    assert tools == ("Read", "Edit")
    # And the family default really is dropped.
    assert "mcp__tilth__*" not in tools


def test_resolve_worker_tools_extend_appends_dedups() -> None:
    tools = resolve_worker_tools(
        "claude",
        extend=["mcp__github__*", "Read"],  # Read already in default
    )
    assert tools[: len(WORKER_ALLOWED_TOOLS["claude"])] == WORKER_ALLOWED_TOOLS["claude"]
    assert "mcp__github__*" in tools
    assert tools.count("Read") == 1


def test_resolve_worker_tools_deny_removes_entries() -> None:
    tools = resolve_worker_tools("claude", deny=["Write"])
    assert "Write" not in tools
    assert "Read" in tools  # untouched


def test_resolve_worker_tools_allow_then_deny() -> None:
    tools = resolve_worker_tools(
        "claude",
        allow=["Read", "Write", "Edit"],
        deny=["Write"],
    )
    assert tools == ("Read", "Edit")


def test_resolve_execution_agent_uses_tools_kwarg() -> None:
    cmd = resolve_execution_agent_command("claude", tools=["Read", "Edit"])
    assert "--allowedTools 'Read,Edit'" in cmd


def test_resolve_execution_agent_override_wins_over_tools() -> None:
    # Explicit execution_agent string opts out of the allowlist machinery.
    cmd = resolve_execution_agent_command(
        "claude",
        execution_agent="my-exec --custom",
        tools=["Read"],
    )
    assert cmd == "my-exec --custom"


def test_resolve_execution_agent_unknown_family_raises() -> None:
    # No default command template exists for an unrecognised family; building
    # one must fail loudly rather than silently emit a broken CLI string.
    try:
        resolve_execution_agent_command("powershell", tools=["Read"])
        assert False, "Expected KeyError for unknown execution family"
    except KeyError as exc:
        assert "powershell" in str(exc)


def test_resolve_execution_agent_gemini_builds_allowed_tools() -> None:
    cmd = resolve_execution_agent_command("gemini", tools=["tilth_search"])
    assert cmd.startswith("gemini")
    assert "--allowed-tools 'tilth_search'" in cmd


def test_resolve_execution_agent_cursor_ignores_tools() -> None:
    # cursor-agent has no headless tool allowlist; the command is fixed.
    cmd = resolve_execution_agent_command("cursor", tools=["Read"])
    assert cmd == "cursor-agent --model sonnet -p"


def test_resolve_execution_agent_codex_ignores_tools() -> None:
    # codex scopes via --sandbox, not a tool allowlist; the command is fixed.
    cmd = resolve_execution_agent_command("codex", tools=["Read"])
    assert cmd.startswith("codex exec")
    assert "--sandbox workspace-write" in cmd


def test_load_config_structured_worker_tools_extend(tmp_path: Path) -> None:
    cfg_path = tmp_path / "milknado.toml"
    cfg_path.write_text(
        (
            "[milknado]\n"
            'agent_family = "claude"\n\n'
            "[milknado.worker.tools.claude]\n"
            'extend = ["mcp__github__*"]\n'
            'deny = ["Write"]\n'
        ),
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert "mcp__github__*" in cfg.execution_agent
    assert "Write" not in cfg.execution_agent.split("--allowedTools")[1]


def test_load_config_structured_worker_tools_allow_replaces(tmp_path: Path) -> None:
    cfg_path = tmp_path / "milknado.toml"
    cfg_path.write_text(
        (
            "[milknado]\n"
            'agent_family = "claude"\n\n'
            "[milknado.worker.tools.claude]\n"
            'allow = ["Read", "Edit"]\n'
        ),
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert "Read,Edit" in cfg.execution_agent
    # Default tool dropped because allow=replace semantics.
    assert "mcp__tilth__*" not in cfg.execution_agent


def test_load_config_structured_worker_tools_rejects_unknown_key(tmp_path: Path) -> None:
    cfg_path = tmp_path / "milknado.toml"
    cfg_path.write_text(
        (
            "[milknado]\n"
            'agent_family = "claude"\n\n'
            "[milknado.worker.tools.claude]\n"
            'banana = ["x"]\n'
        ),
        encoding="utf-8",
    )
    try:
        load_config(cfg_path)
        assert False, "Expected ValueError for unknown worker.tools key"
    except ValueError as exc:
        assert "unknown keys" in str(exc)


def test_load_config_explicit_execution_agent_bypasses_worker_tools(tmp_path: Path) -> None:
    cfg_path = tmp_path / "milknado.toml"
    cfg_path.write_text(
        (
            "[milknado]\n"
            'agent_family = "claude"\n'
            'execution_agent = "claude --custom"\n\n'
            "[milknado.worker.tools.claude]\n"
            'extend = ["mcp__github__*"]\n'
        ),
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert cfg.execution_agent == "claude --custom"
