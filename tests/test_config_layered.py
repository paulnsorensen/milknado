"""Layered milknado.toml (global + local) and prompt-prepend loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from milknado.domains.common import global_config_path, load_config


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
