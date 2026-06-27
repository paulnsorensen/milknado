"""Tests for milknado config sync — per-flavor agent-def generation.

WHY: config sync must produce harness-correct agent-def files from resolved flavor
config, so the native Workflow path and cross-harness workers honor per-flavor tool
and model identity rather than the single static worker def.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from typer.testing import CliRunner

from milknado.cli import app
from milknado.domains.common.agent_argv import WORKER_ALLOWED_TOOLS
from milknado.domains.common.config import (
    DEFAULT_MAX_TURNS,
    FlavorOverride,
    default_config,
)
from milknado.domains.common.types import TaskFlavor
from milknado.domains.config_sync import (
    FlavorIdentity,
    render_claude,
    render_codex,
    render_opencode,
    resolve_flavor_identity,
    sync,
)
from milknado.domains.config_sync._render import _SyncFilter

runner = CliRunner()


# ── helpers ───────────────────────────────────────────────────────────────


def _cfg(tmp_path: Path, **flavor_kwargs):
    """Default config, optionally with a single IMPLEMENT override."""
    cfg = default_config(tmp_path)
    if flavor_kwargs:
        override = FlavorOverride(**flavor_kwargs)
        cfg = dataclasses.replace(cfg, flavors={TaskFlavor.IMPLEMENT: override})
    return cfg


def _cfg_flavor(tmp_path: Path, flavor: TaskFlavor, **kwargs):
    cfg = default_config(tmp_path)
    override = FlavorOverride(**kwargs)
    return dataclasses.replace(cfg, flavors={flavor: override})


# ── Capability classifier ──────────────────────────────────────────────────


def test_write_capable_bash_tool(tmp_path: Path) -> None:
    # Bash(rtk:*) is a shell-exec tool → write-capable.
    cfg = _cfg(tmp_path, tools=("Bash(rtk:*)", "mcp__tilth__tilth_read"))
    identity = resolve_flavor_identity(cfg, TaskFlavor.IMPLEMENT)
    assert identity.is_write_capable is True


def test_write_capable_mcp_serena_prefix(tmp_path: Path) -> None:
    # mcp__serena__replace_symbol_body → write-capable.
    cfg = _cfg_flavor(
        tmp_path,
        TaskFlavor.RESEARCH,
        tools=("mcp__serena__replace_symbol_body", "mcp__tilth__tilth_read"),
    )
    identity = resolve_flavor_identity(cfg, TaskFlavor.RESEARCH)
    assert identity.is_write_capable is True


def test_write_capable_bare_serena_op(tmp_path: Path) -> None:
    # Bare op name (no mcp__serena__ prefix) also matches.
    cfg = _cfg_flavor(
        tmp_path, TaskFlavor.SPEC, tools=("replace_symbol_body", "mcp__tilth__tilth_read")
    )
    identity = resolve_flavor_identity(cfg, TaskFlavor.SPEC)
    assert identity.is_write_capable is True


def test_write_capable_edit_tool(tmp_path: Path) -> None:
    # Direct edit tool name → write-capable.
    cfg = _cfg_flavor(tmp_path, TaskFlavor.SPIKE, tools=("Edit", "mcp__tilth__tilth_read"))
    identity = resolve_flavor_identity(cfg, TaskFlavor.SPIKE)
    assert identity.is_write_capable is True


def test_read_only_tools_not_write_capable(tmp_path: Path) -> None:
    # Only tilth/serena-read tools → read-only.
    cfg = _cfg_flavor(
        tmp_path,
        TaskFlavor.RESEARCH,
        tools=(
            "mcp__tilth__tilth_read",
            "mcp__tilth__tilth_search",
            "mcp__serena__find_symbol",
            "mcp__serena__get_symbols_overview",
        ),
    )
    identity = resolve_flavor_identity(cfg, TaskFlavor.RESEARCH)
    assert identity.is_write_capable is False


# ── Model parsing ──────────────────────────────────────────────────────────


def test_model_default_is_sonnet(tmp_path: Path) -> None:
    # No flavor override → default execution_agent has --model sonnet.
    identity = resolve_flavor_identity(default_config(tmp_path), TaskFlavor.IMPLEMENT)
    assert identity.model == "sonnet"


def test_model_opus_from_execution_agent(tmp_path: Path) -> None:
    # Spec preset: execution_agent = "claude --model opus -p".
    cfg = _cfg_flavor(tmp_path, TaskFlavor.SPEC, execution_agent="claude --model opus -p")
    identity = resolve_flavor_identity(cfg, TaskFlavor.SPEC)
    assert identity.model == "opus"


def test_model_extracted_from_full_custom_command(tmp_path: Path) -> None:
    cfg = _cfg_flavor(
        tmp_path,
        TaskFlavor.SPIKE,
        execution_agent="claude --model haiku -p --allowedTools 'Read'",
    )
    identity = resolve_flavor_identity(cfg, TaskFlavor.SPIKE)
    assert identity.model == "haiku"


# ── resolve_flavor_identity ────────────────────────────────────────────────


def test_resolve_no_override_returns_family_defaults(tmp_path: Path) -> None:
    # No entry in cfg.flavors → family default tools, sonnet, DEFAULT_MAX_TURNS.
    identity = resolve_flavor_identity(default_config(tmp_path), TaskFlavor.IMPLEMENT)
    assert identity.tools == WORKER_ALLOWED_TOOLS["claude"]
    assert identity.model == "sonnet"
    assert identity.max_turns == DEFAULT_MAX_TURNS
    assert identity.flavor == TaskFlavor.IMPLEMENT


def test_resolve_tools_override_replaces_default(tmp_path: Path) -> None:
    cfg = _cfg_flavor(tmp_path, TaskFlavor.RESEARCH, tools=("mcp__tilth__tilth_read",))
    identity = resolve_flavor_identity(cfg, TaskFlavor.RESEARCH)
    assert identity.tools == ("mcp__tilth__tilth_read",)


def test_resolve_max_turns_override(tmp_path: Path) -> None:
    cfg = _cfg_flavor(tmp_path, TaskFlavor.SPIKE, max_turns=30)
    identity = resolve_flavor_identity(cfg, TaskFlavor.SPIKE)
    assert identity.max_turns == 30


# ── render_claude ──────────────────────────────────────────────────────────


def test_render_claude_write_flavor(tmp_path: Path) -> None:
    agents_dir = tmp_path / ".claude" / "agents"
    identity = FlavorIdentity(
        flavor=TaskFlavor.IMPLEMENT,
        tools=("mcp__tilth__tilth_read", "Bash(rtk:*)"),
        model="sonnet",
        max_turns=60,
        is_write_capable=True,
        user_controlled_tools=False,
    )
    rd = render_claude(identity, agents_dir)
    assert rd.harness == "claude"
    assert rd.path == agents_dir / "milknado-worker-implement.md"
    assert "tools: mcp__tilth__tilth_read, Bash(rtk:*)" in rd.content
    assert "model: sonnet" in rd.content
    assert "maxTurns: 60" in rd.content
    assert "GENERATED" in rd.content
    assert rd.content.startswith("---\n")


def test_render_claude_read_only_flavor(tmp_path: Path) -> None:
    agents_dir = tmp_path / ".claude" / "agents"
    identity = FlavorIdentity(
        flavor=TaskFlavor.RESEARCH,
        tools=("mcp__tilth__tilth_read",),
        model="sonnet",
        max_turns=60,
        is_write_capable=False,
        user_controlled_tools=False,
    )
    rd = render_claude(identity, agents_dir)
    assert "tools: mcp__tilth__tilth_read" in rd.content
    assert rd.path == agents_dir / "milknado-worker-research.md"


# ── render_opencode ────────────────────────────────────────────────────────


def test_render_opencode_write_flavor_with_bash(tmp_path: Path) -> None:
    # Write-capable + bash tool → all permissions allow.
    agents_dir = tmp_path / ".opencode" / "agents"
    identity = FlavorIdentity(
        flavor=TaskFlavor.IMPLEMENT,
        tools=("Bash(rtk:*)", "mcp__tilth__tilth_read"),
        model="sonnet",
        max_turns=60,
        is_write_capable=True,
        user_controlled_tools=False,
    )
    rd = render_opencode(identity, agents_dir)
    assert rd.harness == "opencode"
    assert rd.path == agents_dir / "milknado-worker-implement.md"
    assert "edit: allow" in rd.content
    assert "write: allow" in rd.content
    assert "bash: allow" in rd.content
    assert "GENERATED" in rd.content


def test_render_opencode_read_only_no_bash(tmp_path: Path) -> None:
    # Read-only tools, no bash → all permissions deny.
    agents_dir = tmp_path / ".opencode" / "agents"
    identity = FlavorIdentity(
        flavor=TaskFlavor.RESEARCH,
        tools=("mcp__tilth__tilth_read",),
        model="sonnet",
        max_turns=60,
        is_write_capable=False,
        user_controlled_tools=False,
    )
    rd = render_opencode(identity, agents_dir)
    assert "edit: deny" in rd.content
    assert "write: deny" in rd.content
    assert "bash: deny" in rd.content


# ── render_codex ───────────────────────────────────────────────────────────


def test_render_codex_write_flavor(tmp_path: Path) -> None:
    agents_dir = tmp_path / ".codex" / "agents"
    identity = FlavorIdentity(
        flavor=TaskFlavor.IMPLEMENT,
        tools=("Bash(rtk:*)",),
        model="sonnet",
        max_turns=60,
        is_write_capable=True,
        user_controlled_tools=False,
    )
    rd = render_codex(identity, agents_dir)
    assert rd.harness == "codex"
    assert rd.path == agents_dir / "milknado-worker-implement.toml"
    assert 'sandbox_mode = "workspace-write"' in rd.content
    assert 'name = "milknado-worker-implement"' in rd.content
    assert "developer_instructions" in rd.content
    assert "# GENERATED" in rd.content


def test_render_codex_read_only_flavor(tmp_path: Path) -> None:
    agents_dir = tmp_path / ".codex" / "agents"
    identity = FlavorIdentity(
        flavor=TaskFlavor.RESEARCH,
        tools=("mcp__tilth__tilth_read",),
        model="sonnet",
        max_turns=60,
        is_write_capable=False,
        user_controlled_tools=False,
    )
    rd = render_codex(identity, agents_dir)
    assert 'sandbox_mode = "read-only"' in rd.content
    # Must not emit danger-full-access (spec non-goal).
    assert "danger-full-access" not in rd.content


# ── sync ───────────────────────────────────────────────────────────────────


def test_sync_project_scope_writes_all_harnesses(tmp_path: Path) -> None:
    cfg = default_config(tmp_path)
    results = sync(
        cfg, "project", _SyncFilter(["claude", "opencode", "codex"], [TaskFlavor.IMPLEMENT]), False
    )
    assert len(results) == 3
    assert {r.harness for r in results} == {"claude", "opencode", "codex"}
    assert (tmp_path / ".claude" / "agents" / "milknado-worker-implement.md").exists()
    assert (tmp_path / ".opencode" / "agents" / "milknado-worker-implement.md").exists()
    assert (tmp_path / ".codex" / "agents" / "milknado-worker-implement.toml").exists()


def test_sync_dry_run_writes_nothing(tmp_path: Path) -> None:
    cfg = default_config(tmp_path)
    results = sync(cfg, "project", _SyncFilter(["claude"], [TaskFlavor.IMPLEMENT]), dry_run=True)
    assert len(results) == 1
    assert not (tmp_path / ".claude" / "agents" / "milknado-worker-implement.md").exists()


def test_sync_harness_filter_emits_only_requested(tmp_path: Path) -> None:
    cfg = default_config(tmp_path)
    results = sync(
        cfg,
        "project",
        _SyncFilter(["claude"], [TaskFlavor.IMPLEMENT, TaskFlavor.RESEARCH]),
        dry_run=True,
    )
    assert all(r.harness == "claude" for r in results)
    assert len(results) == 2


def test_sync_flavor_filter_emits_only_requested(tmp_path: Path) -> None:
    cfg = default_config(tmp_path)
    results = sync(
        cfg,
        "project",
        _SyncFilter(["claude", "opencode", "codex"], [TaskFlavor.RESEARCH]),
        dry_run=True,
    )
    assert all("research" in r.path.name for r in results)
    assert len(results) == 3


def test_sync_global_scope_uses_home_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Redirect HOME and XDG so we can inspect paths without touching the real home.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    cfg = default_config(tmp_path)
    results = sync(
        cfg, "global", _SyncFilter(["claude", "codex"], [TaskFlavor.IMPLEMENT]), dry_run=True
    )
    cc = next(r for r in results if r.harness == "claude")
    codex = next(r for r in results if r.harness == "codex")
    assert str(cc.path).startswith(str(tmp_path / "home" / ".claude"))
    assert str(codex.path).startswith(str(tmp_path / "home" / ".codex"))


def test_generated_header_present_in_all_harnesses(tmp_path: Path) -> None:
    cfg = default_config(tmp_path)
    results = sync(
        cfg,
        "project",
        _SyncFilter(["claude", "opencode", "codex"], [TaskFlavor.IMPLEMENT]),
        dry_run=True,
    )
    for rd in results:
        assert "GENERATED" in rd.content, f"Missing GENERATED header in {rd.harness} def"


def test_file_names_match_milknado_worker_prefix(tmp_path: Path) -> None:
    # Every generated file must match milknado-worker-<flavor>.* — never user files.
    cfg = default_config(tmp_path)
    results = sync(
        cfg,
        "project",
        _SyncFilter(["claude", "opencode", "codex"], list(TaskFlavor)),
        dry_run=True,
    )
    for rd in results:
        assert rd.path.name.startswith("milknado-worker-"), rd.path.name


# ── CLI smoke ─────────────────────────────────────────────────────────────


def test_cli_config_sync_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Dry-run via CLI must report paths without writing files.
    monkeypatch.setenv("COLUMNS", "10000")
    result = runner.invoke(app, ["config", "sync", "--project-root", str(tmp_path), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "milknado-worker-implement" in result.output
    assert not (tmp_path / ".claude" / "agents" / "milknado-worker-implement.md").exists()


def test_cli_config_sync_harness_filter(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "config",
            "sync",
            "--project-root",
            str(tmp_path),
            "--harness",
            "claude",
            "--flavor",
            "implement",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / ".claude" / "agents" / "milknado-worker-implement.md").exists()
    assert not (tmp_path / ".opencode" / "agents" / "milknado-worker-implement.md").exists()


def test_cli_config_sync_unknown_harness_exits_1(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["config", "sync", "--project-root", str(tmp_path), "--harness", "unknown"]
    )
    assert result.exit_code == 1


def test_cli_config_sync_unknown_flavor_exits_1(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["config", "sync", "--project-root", str(tmp_path), "--flavor", "bogus"]
    )
    assert result.exit_code == 1


def test_project_scope_sync_reflects_global_flavor_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # WHY: --scope project must render the EFFECTIVE runtime config, which merges
    # the user's global milknado.toml flavor overrides on top of the built-in
    # defaults before the project config is applied (load order: global-base →
    # project-override). A flavor override present only in the global config
    # must appear in the rendered def so workers honour the user's turn budget
    # without every project re-declaring it.  The old include_global=False
    # behaviour would silently drop the global override, causing the rendered def
    # to use DEFAULT_MAX_TURNS (60) even when the user set a different value.
    xdg = tmp_path / "xdg"
    xdg.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))

    # Global config: max_turns=77 for IMPLEMENT, no other flavor overrides.
    global_cfg = xdg / "milknado" / "milknado.toml"
    global_cfg.parent.mkdir(parents=True)
    global_cfg.write_text(
        "[milknado.flavor.implement]\nmax_turns = 77\n",
        encoding="utf-8",
    )
    # Project config: no flavor override — the global value must still appear.
    (tmp_path / "milknado.toml").write_text(
        '[milknado]\nagent_family = "claude"\n',
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "config",
            "sync",
            "--scope",
            "project",
            "--project-root",
            str(tmp_path),
            "--harness",
            "claude",
            "--flavor",
            "implement",
        ],
    )
    assert result.exit_code == 0, result.output

    rendered = (tmp_path / ".claude" / "agents" / "milknado-worker-implement.md").read_text(
        encoding="utf-8"
    )
    assert "maxTurns: 77" in rendered, (
        "project-scope sync must reflect global flavor max_turns override; "
        f"rendered def:\n{rendered}"
    )


# ── global-scope reads global config ──────────────────────────────────────


def test_cli_config_sync_global_scope_reads_global_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # WHY: _load_cfg(scope="global") must read the global milknado.toml
    # (via global_config_path(), respecting $XDG_CONFIG_HOME) and reflect
    # flavor overrides in the rendered defs.  If the scope check breaks—
    # reading the project config instead, or include_global accidentally
    # flips to True (circular re-read)—global-config values are silently
    # dropped and workers use the wrong identity.
    xdg = tmp_path / "xdg"
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.setenv("HOME", str(home))

    global_cfg = xdg / "milknado" / "milknado.toml"
    global_cfg.parent.mkdir(parents=True)
    global_cfg.write_text(
        "[milknado.flavor.implement]\nmax_turns = 99\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "config",
            "sync",
            "--scope",
            "global",
            "--harness",
            "claude",
            "--flavor",
            "implement",
        ],
    )
    assert result.exit_code == 0, result.output

    rendered = (home / ".claude" / "agents" / "milknado-worker-implement.md").read_text(
        encoding="utf-8"
    )
    assert "maxTurns: 99" in rendered, (
        "global-scope sync must read global milknado.toml and apply flavor overrides; "
        f"rendered:\n{rendered}"
    )


# ── spec: full-custom execution_agent emits no tool restriction ────────────


def test_full_custom_execution_agent_emits_no_tool_restriction(tmp_path: Path) -> None:
    # WHY: when execution_agent is a full-custom command (e.g. aider) with no tools
    # override, the user has taken full control of tool access.  The CC def must
    # omit the tools: line entirely rather than emitting the family default list.
    # Spec Design: "If execution_agent is a full custom command, extract the model
    # and emit no tool restriction (the user took full control)."
    cfg = _cfg_flavor(
        tmp_path,
        TaskFlavor.SPIKE,
        execution_agent="aider --model gpt-4o --no-auto-commits",
    )
    identity = resolve_flavor_identity(cfg, TaskFlavor.SPIKE)
    assert identity.user_controlled_tools is True
    rd = render_claude(identity, tmp_path / ".claude" / "agents")
    assert "tools:" not in rd.content


def test_parse_model_equals_form(tmp_path: Path) -> None:
    # WHY: _parse_model must handle --model=X (equals form) in addition to
    # --model X (space form); the space-only regex silently fell back to the
    # family default, yielding a def with the wrong model.
    cfg = _cfg_flavor(
        tmp_path,
        TaskFlavor.RESEARCH,
        execution_agent="claude --model=haiku -p",
    )
    identity = resolve_flavor_identity(cfg, TaskFlavor.RESEARCH)
    assert identity.model == "haiku"


def test_tool_name_with_newline_rejected(tmp_path: Path) -> None:
    # WHY: a newline in a tool name is a YAML-injection vector — it can insert
    # arbitrary frontmatter keys into the generated CC agent-def, enabling
    # capability escalation for any worker spawned from an untrusted repo.
    # The boundary must reject such strings at parse time (fail fast).
    from milknado.domains.common.config import _coerce_single_tool_list

    with pytest.raises(ValueError, match="newline"):
        _coerce_single_tool_list(["Bash\ntools: [Evil]"], "test")
