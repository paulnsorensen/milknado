"""Layered milknado.toml (global + local) and prompt-prepend loading."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from milknado.domains.common import (
    WorkerToolsOverride,
    default_config,
    global_config_path,
    load_config,
    save_config,
)
from milknado.domains.common.config import _parse_worker_tools


def _write(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


@pytest.fixture()
def xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Per-test XDG_CONFIG_HOME so we can write a fake global config."""
    home = tmp_path / "xdg"
    home.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
    return home


def test_global_config_path_honors_xdg(xdg: Path) -> None:
    assert global_config_path() == xdg / "milknado" / "milknado.toml"


def test_global_config_path_falls_back_to_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    expected = tmp_path / ".config" / "milknado" / "milknado.toml"
    # Path.home() reads $HOME on POSIX; assert structural prefix instead of
    # full equality to remain platform-tolerant.
    assert str(global_config_path()).endswith(".config/milknado/milknado.toml")
    assert expected.name == "milknado.toml"


def test_local_overrides_global(xdg: Path, tmp_path: Path) -> None:
    _write(
        xdg / "milknado" / "milknado.toml",
        '[milknado]\nagent_family = "claude"\nconcurrency_limit = 7\n',
    )
    local = _write(
        tmp_path / "milknado.toml",
        '[milknado]\nagent_family = "claude"\nconcurrency_limit = 12\n',
    )
    cfg = load_config(local)
    assert cfg.concurrency_limit == 12


def test_global_fills_unset_local_keys(xdg: Path, tmp_path: Path) -> None:
    _write(
        xdg / "milknado" / "milknado.toml",
        ('[milknado]\nagent_family = "claude"\nstall_threshold_seconds = 999\n'),
    )
    local = _write(
        tmp_path / "milknado.toml",
        '[milknado]\nagent_family = "claude"\n',
    )
    cfg = load_config(local)
    assert cfg.stall_threshold_seconds == 999


def test_global_local_only_keys_ignored(xdg: Path, tmp_path: Path) -> None:
    # plugins and db_path are per-project — global cannot dictate them.
    _write(
        xdg / "milknado" / "milknado.toml",
        (
            "[milknado]\n"
            'agent_family = "claude"\n'
            'db_path = "global/path.db"\n'
            'plugins = ["global-plugin"]\n'
        ),
    )
    local = _write(
        tmp_path / "milknado.toml",
        '[milknado]\nagent_family = "claude"\n',
    )
    cfg = load_config(local)
    # Falls back to default db path, not global's "global/path.db".
    assert cfg.db_path == tmp_path / ".milknado" / "milknado.db"
    assert cfg.plugins == ()


def test_include_global_false_skips_global(xdg: Path, tmp_path: Path) -> None:
    _write(
        xdg / "milknado" / "milknado.toml",
        '[milknado]\nagent_family = "claude"\nconcurrency_limit = 99\n',
    )
    local = _write(
        tmp_path / "milknado.toml",
        '[milknado]\nagent_family = "claude"\n',
    )
    cfg = load_config(local, include_global=False)
    assert cfg.concurrency_limit == 4  # MilknadoConfig default


def test_global_worker_tools_merges_with_local_extend(xdg: Path, tmp_path: Path) -> None:
    _write(
        xdg / "milknado" / "milknado.toml",
        (
            "[milknado]\n"
            'agent_family = "claude"\n\n'
            "[milknado.worker.tools.claude]\n"
            'extend = ["mcp__github__*"]\n'
        ),
    )
    local = _write(
        tmp_path / "milknado.toml",
        '[milknado]\nagent_family = "claude"\n',
    )
    cfg = load_config(local)
    assert "mcp__github__*" in cfg.execution_agent


def test_global_extend_and_local_deny_compose(xdg: Path, tmp_path: Path) -> None:
    """Global and local worker.tools.<family> tables deep-merge — global's
    extend keys and local's deny keys end up in the same effective override.
    Same-key conflicts still go to local (last write wins)."""
    _write(
        xdg / "milknado" / "milknado.toml",
        (
            "[milknado]\n"
            'agent_family = "claude"\n\n'
            "[milknado.worker.tools.claude]\n"
            'extend = ["mcp__github__*"]\n'
        ),
    )
    local = _write(
        tmp_path / "milknado.toml",
        (
            "[milknado]\n"
            'agent_family = "claude"\n\n'
            "[milknado.worker.tools.claude]\n"
            'deny = ["Write"]\n'
        ),
    )
    cfg = load_config(local)
    # Global's extend survives because deep-merge keeps it on the same table.
    assert "mcp__github__*" in cfg.execution_agent
    # Local's deny strips Write from the allowlist.
    assert "Write" not in cfg.execution_agent.split("--allowedTools")[1]


# ── prompt prepends ───────────────────────────────────────────────────────────


def test_planning_prompt_prepend_inline(tmp_path: Path) -> None:
    local = _write(
        tmp_path / "milknado.toml",
        (
            "[milknado]\n"
            'agent_family = "claude"\n\n'
            "[milknado.prompts]\n"
            'planning_prepend = "team rule X"\n'
        ),
    )
    cfg = load_config(local)
    assert cfg.planning_prompt_prepend == "team rule X"


def test_planning_prompt_prepend_path(tmp_path: Path) -> None:
    extras = tmp_path / "planner-extras.md"
    extras.write_text("global team conventions\n", encoding="utf-8")
    local = _write(
        tmp_path / "milknado.toml",
        (
            "[milknado]\n"
            'agent_family = "claude"\n\n'
            "[milknado.prompts]\n"
            'planning_prepend_path = "planner-extras.md"\n'
        ),
    )
    cfg = load_config(local)
    assert cfg.planning_prompt_prepend == "global team conventions"


def test_worker_brief_prepend_inline(tmp_path: Path) -> None:
    local = _write(
        tmp_path / "milknado.toml",
        (
            "[milknado]\n"
            'agent_family = "claude"\n\n'
            "[milknado.prompts]\n"
            'worker_brief_prepend = "always just build"\n'
        ),
    )
    cfg = load_config(local)
    assert cfg.worker_brief_prepend == "always just build"


def test_prompt_prepend_inline_and_path_mutually_exclusive(tmp_path: Path) -> None:
    local = _write(
        tmp_path / "milknado.toml",
        (
            "[milknado]\n"
            'agent_family = "claude"\n\n'
            "[milknado.prompts]\n"
            'planning_prepend = "inline"\n'
            'planning_prepend_path = "p.md"\n'
        ),
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        load_config(local)


def test_prompt_prepend_path_missing_raises(tmp_path: Path) -> None:
    local = _write(
        tmp_path / "milknado.toml",
        (
            "[milknado]\n"
            'agent_family = "claude"\n\n'
            "[milknado.prompts]\n"
            'worker_brief_prepend_path = "no-such-file.md"\n'
        ),
    )
    with pytest.raises(FileNotFoundError):
        load_config(local)


def test_prompt_prepend_empty_string_plus_path_still_rejected(tmp_path: Path) -> None:
    """Empty-string inline + path is still a misconfigured pair: surface it
    rather than silently picking the path branch."""
    extras = tmp_path / "p.md"
    extras.write_text("contents", encoding="utf-8")
    local = _write(
        tmp_path / "milknado.toml",
        (
            "[milknado]\n"
            'agent_family = "claude"\n\n'
            "[milknado.prompts]\n"
            'planning_prepend = ""\n'
            'planning_prepend_path = "p.md"\n'
        ),
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        load_config(local)


def test_local_path_overrides_global_inline(xdg: Path, tmp_path: Path) -> None:
    """A global ``planning_prepend`` plus a local ``planning_prepend_path``
    should resolve to the local path's contents, not raise mutual-exclusion."""
    _write(
        xdg / "milknado" / "milknado.toml",
        (
            "[milknado]\n"
            'agent_family = "claude"\n\n'
            "[milknado.prompts]\n"
            'planning_prepend = "global text"\n'
        ),
    )
    extras = tmp_path / "local.md"
    extras.write_text("local from path", encoding="utf-8")
    local = _write(
        tmp_path / "milknado.toml",
        (
            "[milknado]\n"
            'agent_family = "claude"\n\n'
            "[milknado.prompts]\n"
            'planning_prepend_path = "local.md"\n'
        ),
    )
    cfg = load_config(local)
    assert cfg.planning_prompt_prepend == "local from path"


def test_local_inline_overrides_global_path(xdg: Path, tmp_path: Path) -> None:
    """Reverse direction: global uses ``_path``, local replaces with inline."""
    global_extras = xdg / "global.md"
    global_extras.write_text("global from path", encoding="utf-8")
    _write(
        xdg / "milknado" / "milknado.toml",
        (
            "[milknado]\n"
            'agent_family = "claude"\n\n'
            "[milknado.prompts]\n"
            f'planning_prepend_path = "{global_extras}"\n'
        ),
    )
    local = _write(
        tmp_path / "milknado.toml",
        (
            "[milknado]\n"
            'agent_family = "claude"\n\n'
            "[milknado.prompts]\n"
            'planning_prepend = "local inline wins"\n'
        ),
    )
    cfg = load_config(local)
    assert cfg.planning_prompt_prepend == "local inline wins"


def test_worker_tools_allow_string_rejected(tmp_path: Path) -> None:
    """A scalar string is not a list; reject it instead of splatting it into chars."""
    local = _write(
        tmp_path / "milknado.toml",
        (
            "[milknado]\n"
            'agent_family = "claude"\n\n'
            "[milknado.worker.tools.claude]\n"
            'allow = "Read"\n'
        ),
    )
    with pytest.raises(ValueError, match="must be a list of strings"):
        load_config(local)


def test_worker_tools_extend_with_non_string_rejected(tmp_path: Path) -> None:
    local = _write(
        tmp_path / "milknado.toml",
        ('[milknado]\nagent_family = "claude"\n\n[milknado.worker.tools.claude]\nextend = [42]\n'),
    )
    with pytest.raises(ValueError, match=r"extend\[0\] must be a non-empty string"):
        load_config(local)


def test_worker_tools_deny_with_empty_string_rejected(tmp_path: Path) -> None:
    local = _write(
        tmp_path / "milknado.toml",
        ('[milknado]\nagent_family = "claude"\n\n[milknado.worker.tools.claude]\ndeny = [""]\n'),
    )
    with pytest.raises(ValueError, match="must be a non-empty string"):
        load_config(local)


def test_worker_not_table_rejected(tmp_path: Path) -> None:
    """A scalar `worker` key is not a [milknado.worker] table; reject clearly."""
    local = _write(
        tmp_path / "milknado.toml",
        '[milknado]\nagent_family = "claude"\nworker = "oops"\n',
    )
    with pytest.raises(ValueError, match=r"\[milknado\.worker\] must be a table"):
        load_config(local)


def test_worker_tools_not_table_rejected(tmp_path: Path) -> None:
    """A scalar `tools` key under [milknado.worker] is not a table; reject clearly."""
    local = _write(
        tmp_path / "milknado.toml",
        '[milknado]\nagent_family = "claude"\n\n[milknado.worker]\ntools = "oops"\n',
    )
    with pytest.raises(ValueError, match=r"\[milknado\.worker\.tools\] must be a table"):
        load_config(local)


def test_prompts_not_table_rejected(tmp_path: Path) -> None:
    """A scalar `prompts` key is not a [milknado.prompts] table; reject clearly."""
    local = _write(
        tmp_path / "milknado.toml",
        '[milknado]\nagent_family = "claude"\nprompts = "oops"\n',
    )
    with pytest.raises(ValueError, match=r"\[milknado\.prompts\] must be a table"):
        load_config(local)


def test_worker_table_without_tools_yields_no_overrides(tmp_path: Path) -> None:
    """A [milknado.worker] table without a tools subtable is valid and empty."""
    local = _write(
        tmp_path / "milknado.toml",
        '[milknado]\nagent_family = "claude"\n\n[milknado.worker]\n',
    )
    cfg = load_config(local)
    assert cfg.worker_tools == {}


def test_worker_tools_family_not_table_rejected(tmp_path: Path) -> None:
    """A scalar under [milknado.worker.tools] is not a per-family table; reject."""
    local = _write(
        tmp_path / "milknado.toml",
        '[milknado]\nagent_family = "claude"\n\n[milknado.worker.tools]\nclaude = "oops"\n',
    )
    with pytest.raises(ValueError, match=r"\[milknado\.worker\.tools\.claude\] must be a table"):
        load_config(local)


def test_parse_worker_tools_non_string_family_rejected() -> None:
    """Non-string family keys (only reachable via a raw dict, not TOML) are rejected."""
    with pytest.raises(ValueError, match="family keys must be strings"):
        _parse_worker_tools({"tools": {1: {"allow": ["Read"]}}})


def test_top_level_milknado_not_table_rejected(tmp_path: Path) -> None:
    """A top-level scalar `milknado` value is not the [milknado] table; reject."""
    local = _write(tmp_path / "milknado.toml", 'milknado = "oops"\n')
    with pytest.raises(ValueError, match=r"\[milknado\] is not a table"):
        load_config(local)


def test_save_config_emits_allow_table_and_omits_derived_execution_agent(
    tmp_path: Path,
) -> None:
    """A structured allow override is serialized; the derived execution_agent is
    suppressed so it cannot shadow the override on reload."""
    # Build the config the way load_config does: no explicit execution_agent, so
    # the in-memory execution_agent IS the command derived from the override —
    # a derived artifact that save_config should drop.
    src = _write(
        tmp_path / "in.toml",
        '[milknado]\nagent_family = "claude"\n\n'
        '[milknado.worker.tools.claude]\nallow = ["Read", "Edit"]\n',
    )
    cfg = load_config(src, include_global=False)
    out = tmp_path / "milknado.toml"
    save_config(cfg, out)
    text = out.read_text(encoding="utf-8")
    assert "[milknado.worker.tools.claude]" in text
    assert '"Read"' in text
    assert '"Edit"' in text
    assert "execution_agent" not in text
    # Round-trips back to the same structured override.
    assert load_config(out, include_global=False).worker_tools["claude"].allow == ("Read", "Edit")


def test_save_config_skips_empty_worker_override(tmp_path: Path) -> None:
    """An override with no allow/extend/deny is derived noise; don't serialize it."""
    cfg = replace(
        default_config(tmp_path),
        worker_tools={"claude": WorkerToolsOverride()},
    )
    out = tmp_path / "milknado.toml"
    save_config(cfg, out)
    text = out.read_text(encoding="utf-8")
    assert "[milknado.worker.tools.claude]" not in text


def test_save_config_preserves_explicit_execution_agent_with_override(tmp_path: Path) -> None:
    """An explicit execution_agent that differs from the derived command is user
    intent and must survive a save->load round trip, even alongside a structured
    worker override (which would otherwise re-derive a different command)."""
    cfg = replace(
        default_config(tmp_path),
        execution_agent="claude --my-custom-exec",
        worker_tools={"claude": WorkerToolsOverride(extend=("mcp__github__*",))},
    )
    out = tmp_path / "milknado.toml"
    save_config(cfg, out)
    text = out.read_text(encoding="utf-8")
    assert 'execution_agent = "claude --my-custom-exec"' in text
    # The explicit command wins over the derived allowlist on reload.
    assert load_config(out, include_global=False).execution_agent == "claude --my-custom-exec"


def test_global_relative_prompt_path_resolves_against_global_dir(
    xdg: Path, tmp_path: Path
) -> None:
    """A relative prompt `_path` in the GLOBAL config resolves next to the global
    config file, not the local project root."""
    global_dir = xdg / "milknado"
    _write(
        global_dir / "milknado.toml",
        '[milknado.prompts]\nplanning_prepend_path = "team.md"\n',
    )
    (global_dir / "team.md").write_text("GLOBAL TEAM NOTE", encoding="utf-8")
    # A decoy with the same relative name in the project root: the buggy
    # behaviour resolved against here and would pick this up.
    (tmp_path / "team.md").write_text("LOCAL DECOY", encoding="utf-8")
    local = _write(tmp_path / "milknado.toml", '[milknado]\nagent_family = "claude"\n')
    cfg = load_config(local)
    assert cfg.planning_prompt_prepend == "GLOBAL TEAM NOTE"


def test_prompt_inline_non_string_rejected(tmp_path: Path) -> None:
    """Prompt prepends are strings; a non-string inline value is rejected, not
    stringified into garbage prompt text."""
    local = _write(
        tmp_path / "milknado.toml",
        '[milknado]\nagent_family = "claude"\n\n[milknado.prompts]\nplanning_prepend = ["x"]\n',
    )
    with pytest.raises(ValueError, match="planning_prepend must be a string"):
        load_config(local)


def test_prompt_path_non_string_rejected(tmp_path: Path) -> None:
    """A non-string `_path` is rejected rather than coerced into a filename."""
    local = _write(
        tmp_path / "milknado.toml",
        '[milknado]\nagent_family = "claude"\n\n'
        "[milknado.prompts]\nworker_brief_prepend_path = 42\n",
    )
    with pytest.raises(ValueError, match=r"prepend_path must be a string"):
        load_config(local)
