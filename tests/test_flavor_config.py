"""Tests for per-flavor agent config: AC1-7 from spec per-flavor-agent-config.md."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from milknado.domains.common.agent_argv import (
    WORKER_ALLOWED_TOOLS,
    resolve_worker_tools,
)
from milknado.domains.common.config import (
    FlavorOverride,
    Gate,
    MilknadoConfig,
    load_config,
    save_config,
)
from milknado.domains.common.flavor_profile import (
    resolve_flavor_profile,
)
from milknado.domains.common.types import BUILTIN_FLAVORS

# ── AC2: single-list grammar + sentinel ─────────────────────────────────────


def test_resolve_worker_tools_none_returns_family_default() -> None:
    assert resolve_worker_tools("claude", None) == WORKER_ALLOWED_TOOLS["claude"]


def test_resolve_worker_tools_none_unknown_family_returns_empty() -> None:
    assert resolve_worker_tools("unknownfam", None) == ()


def test_resolve_worker_tools_sentinel_expands_to_default() -> None:
    tools = resolve_worker_tools("claude", ["...", "WebSearch"])
    default = WORKER_ALLOWED_TOOLS["claude"]
    # "..." expands in place; WebSearch appended after
    assert tools[: len(default)] == default
    assert "WebSearch" in tools
    assert "..." not in tools


def test_resolve_worker_tools_no_sentinel_replaces_default() -> None:
    tools = resolve_worker_tools("claude", ["Read", "Edit"])
    assert tools == ("Read", "Edit")
    assert "mcp__tilth__*" not in tools


def test_resolve_worker_tools_sentinel_dedupes_first_wins() -> None:
    # "Read" appears both in default and in explicit list; must appear only once
    tools = resolve_worker_tools("claude", ["...", "Read"])
    assert tools.count("Read") == 1


def test_resolve_worker_tools_at_most_one_sentinel_validated_at_load(tmp_path: Path) -> None:
    cfg_path = tmp_path / "milknado.toml"
    cfg_path.write_text(
        '[milknado]\nagent_family = "claude"\n\n'
        "[milknado.worker.tools]\n"
        'claude = ["...", "Read", "..."]\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="at most one"):
        load_config(cfg_path)


def test_load_config_single_list_worker_tools(tmp_path: Path) -> None:
    cfg_path = tmp_path / "milknado.toml"
    cfg_path.write_text(
        '[milknado]\nagent_family = "claude"\n\n'
        "[milknado.worker.tools]\n"
        'claude = ["...", "Bash(just:*)"]\n',
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    # Raw list is stored — sentinel preserved, expansion deferred to resolution time
    wt = cfg.worker_tools.get("claude", ())
    assert wt == ("...", "Bash(just:*)")
    assert "..." in wt
    assert "Bash(just:*)" in wt
    # execution_agent embeds the resolved (expanded) tool list at build time
    default = WORKER_ALLOWED_TOOLS["claude"]
    assert "Bash(just:*)" in cfg.execution_agent
    assert all(t in cfg.execution_agent for t in default[:3])


def test_load_config_single_list_bare_string_rejected(tmp_path: Path) -> None:
    cfg_path = tmp_path / "milknado.toml"
    cfg_path.write_text(
        '[milknado]\nagent_family = "claude"\n\n[milknado.worker.tools]\nclaude = "Read"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must be a list"):
        load_config(cfg_path)


# ── AC1: FlavorOverride config parsing + validation ──────────────────────────


def test_load_config_flavor_table_parses(tmp_path: Path) -> None:
    cfg_path = tmp_path / "milknado.toml"
    cfg_path.write_text(
        '[milknado]\nagent_family = "claude"\n\n'
        "[milknado.flavor.research]\n"
        "quality_gates = []\n"
        'brief_prepend = "Research mode."\n',
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert "research" in cfg.flavors
    fo = cfg.flavors["research"]
    assert fo.quality_gates == ()
    assert fo.brief_prepend == "Research mode."


def test_load_config_flavor_custom_key_registers(tmp_path: Path) -> None:
    """A TOML-declared flavor name outside BUILTIN_FLAVORS is its own registration (ADR-004)."""
    cfg_path = tmp_path / "milknado.toml"
    cfg_path.write_text(
        '[milknado]\nagent_family = "claude"\n\n'
        "[milknado.flavor.customflavorkey]\n"
        "quality_gates = []\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert "customflavorkey" in cfg.flavors


def test_flavor_registry_includes_builtins_and_declared(tmp_path: Path) -> None:
    """MilknadoConfig.flavor_registry is BUILTIN_FLAVORS unioned with declared TOML names."""
    cfg_path = tmp_path / "milknado.toml"
    cfg_path.write_text(
        '[milknado]\nagent_family = "claude"\n\n'
        "[milknado.flavor.customflavorkey]\n"
        "quality_gates = []\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert cfg.flavor_registry == BUILTIN_FLAVORS | {"customflavorkey"}


def test_flavor_registry_defaults_to_builtins_only(tmp_path: Path) -> None:
    cfg_path = tmp_path / "milknado.toml"
    cfg_path.write_text('[milknado]\nagent_family = "claude"\n', encoding="utf-8")
    cfg = load_config(cfg_path)
    assert cfg.flavor_registry == BUILTIN_FLAVORS


def test_load_config_flavor_worktree_parses(tmp_path: Path) -> None:
    cfg_path = tmp_path / "milknado.toml"
    cfg_path.write_text(
        '[milknado]\nagent_family = "claude"\n\n[milknado.flavor.spec]\nworktree = false\n',
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert cfg.flavors["spec"].worktree is False


def test_load_config_flavor_worktree_not_bool_raises(tmp_path: Path) -> None:
    cfg_path = tmp_path / "milknado.toml"
    cfg_path.write_text(
        '[milknado]\nagent_family = "claude"\n\n[milknado.flavor.spec]\nworktree = "nope"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="worktree"):
        load_config(cfg_path)


def test_load_config_flavor_invalid_execution_agent_raises(tmp_path: Path) -> None:
    cfg_path = tmp_path / "milknado.toml"
    cfg_path.write_text(
        '[milknado]\nagent_family = "claude"\n\n'
        "[milknado.flavor.spike]\n"
        'execution_agent = "evil-bin --flag"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValidationError) as exc_info:
        load_config(cfg_path)

    message = str(exc_info.value)
    assert "execution_agent must start with one of" in message
    assert "evil-bin" not in message


def test_load_config_flavor_tools_malformed_bare_string_raises(tmp_path: Path) -> None:
    cfg_path = tmp_path / "milknado.toml"
    cfg_path.write_text(
        '[milknado]\nagent_family = "claude"\n\n[milknado.flavor.spike]\ntools = "Read"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must be a list"):
        load_config(cfg_path)


def test_load_config_flavor_tools_multiple_sentinels_raises(tmp_path: Path) -> None:
    cfg_path = tmp_path / "milknado.toml"
    cfg_path.write_text(
        '[milknado]\nagent_family = "claude"\n\n'
        "[milknado.flavor.spike]\n"
        'tools = ["...", "Read", "..."]\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="at most one"):
        load_config(cfg_path)


def test_load_config_flavor_brief_prepend_path_conflict_raises(tmp_path: Path) -> None:
    brief_file = tmp_path / "house.md"
    brief_file.write_text("rules", encoding="utf-8")
    cfg_path = tmp_path / "milknado.toml"
    cfg_path.write_text(
        '[milknado]\nagent_family = "claude"\n\n'
        "[milknado.flavor.spike]\n"
        'brief_prepend = "inline"\n'
        'brief_prepend_path = "house.md"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        load_config(cfg_path)


def test_load_config_flavor_brief_prepend_path_missing_file_raises(tmp_path: Path) -> None:
    cfg_path = tmp_path / "milknado.toml"
    cfg_path.write_text(
        '[milknado]\nagent_family = "claude"\n\n'
        "[milknado.flavor.spike]\n"
        'brief_prepend_path = "nonexistent.md"\n',
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError):
        load_config(cfg_path)


def test_load_config_flavor_quality_gates_must_be_list(tmp_path: Path) -> None:
    cfg_path = tmp_path / "milknado.toml"
    cfg_path.write_text(
        '[milknado]\nagent_family = "claude"\n\n'
        "[milknado.flavor.spike]\n"
        'quality_gates = "uv run pytest"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must be a list"):
        load_config(cfg_path)


def test_load_config_flavor_brief_prepend_path_list(tmp_path: Path) -> None:
    f1 = tmp_path / "house.md"
    f2 = tmp_path / "research.md"
    f1.write_text("house rules", encoding="utf-8")
    f2.write_text("research rules", encoding="utf-8")
    cfg_path = tmp_path / "milknado.toml"
    cfg_path.write_text(
        '[milknado]\nagent_family = "claude"\n\n'
        "[milknado.flavor.research]\n"
        'brief_prepend_path = ["house.md", "research.md"]\n',
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    fo = cfg.flavors["research"]
    assert fo.brief_prepend is not None
    assert "house rules" in fo.brief_prepend
    assert "research rules" in fo.brief_prepend


# ── AC3: save_config round-trip ─────────────────────────────────────────────


def test_save_load_roundtrip_with_flavors(tmp_path: Path) -> None:
    brief_file = tmp_path / "house.md"
    brief_file.write_text("house rules here", encoding="utf-8")
    cfg_path = tmp_path / "milknado.toml"
    cfg = MilknadoConfig(
        agent_family="claude",
        project_root=tmp_path,
        db_path=tmp_path / ".milknado" / "milknado.db",
        worker_tools={"claude": ("...", "Bash(just:*)")},
        flavors={
            "research": FlavorOverride(
                quality_gates=(),
                brief_prepend="Research mode",
            ),
            "spike": FlavorOverride(
                tools=("...", "WebSearch"),
            ),
        },
    )
    save_config(cfg, cfg_path)
    loaded = load_config(cfg_path)

    # field-by-field: flavors
    assert "research" in loaded.flavors
    assert "spike" in loaded.flavors
    r = loaded.flavors["research"]
    assert r.quality_gates == ()
    assert r.brief_prepend == "Research mode"
    s = loaded.flavors["spike"]
    assert s.tools == ("...", "WebSearch")

    # field-by-field: worker_tools (single-list — sentinel round-trips raw)
    wt = loaded.worker_tools.get("claude")
    assert wt is not None
    assert wt == ("...", "Bash(just:*)")
    assert "..." in wt  # sentinel preserved through save→load
    assert "Bash(just:*)" in wt


def test_save_load_roundtrip_single_list_worker_tools(tmp_path: Path) -> None:
    cfg_path = tmp_path / "milknado.toml"
    cfg = MilknadoConfig(
        agent_family="claude",
        project_root=tmp_path,
        db_path=tmp_path / ".milknado" / "milknado.db",
        worker_tools={"claude": ("Read", "Edit")},
    )
    save_config(cfg, cfg_path)
    loaded = load_config(cfg_path)
    wt = loaded.worker_tools.get("claude")
    assert wt is not None
    assert wt == ("Read", "Edit")


# ── AC4: resolve_flavor_profile unit tests ──────────────────────────────────


def _base_cfg(tmp_path: Path) -> MilknadoConfig:
    return MilknadoConfig(
        agent_family="claude",
        project_root=tmp_path,
        db_path=tmp_path / ".milknado" / "milknado.db",
    )


def test_resolve_flavor_profile_no_flavor_returns_defaults(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)
    profile = resolve_flavor_profile(cfg, None)
    assert profile.execution_agent == cfg.execution_agent
    assert profile.quality_gates == cfg.quality_gates
    assert profile.brief_prepend == cfg.worker_brief_prepend


def test_resolve_flavor_profile_flavor_no_entry_returns_defaults(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)
    profile = resolve_flavor_profile(cfg, "spike")
    assert profile.execution_agent == cfg.execution_agent
    assert profile.quality_gates == cfg.quality_gates


def test_resolve_flavor_profile_explicit_execution_agent_wins(tmp_path: Path) -> None:
    cfg = MilknadoConfig(
        agent_family="claude",
        project_root=tmp_path,
        db_path=tmp_path / ".milknado" / "milknado.db",
        flavors={
            "research": FlavorOverride(
                execution_agent="claude -p --model opus",
            ),
        },
    )
    profile = resolve_flavor_profile(cfg, "research")
    assert profile.execution_agent == "claude -p --model opus"


def test_resolve_flavor_profile_tools_derive_command(tmp_path: Path) -> None:
    cfg = MilknadoConfig(
        agent_family="claude",
        project_root=tmp_path,
        db_path=tmp_path / ".milknado" / "milknado.db",
        flavors={
            "spike": FlavorOverride(
                tools=("Read", "Edit"),
            ),
        },
    )
    profile = resolve_flavor_profile(cfg, "spike")
    assert "Read,Edit" in profile.execution_agent


def test_resolve_flavor_profile_quality_gates_empty_tuple(tmp_path: Path) -> None:
    cfg = MilknadoConfig(
        agent_family="claude",
        project_root=tmp_path,
        db_path=tmp_path / ".milknado" / "milknado.db",
        flavors={
            "research": FlavorOverride(
                quality_gates=(),
            ),
        },
    )
    profile = resolve_flavor_profile(cfg, "research")
    assert profile.quality_gates == ()


def test_resolve_flavor_profile_quality_gates_none_inherits(tmp_path: Path) -> None:
    cfg = MilknadoConfig(
        agent_family="claude",
        project_root=tmp_path,
        db_path=tmp_path / ".milknado" / "milknado.db",
        quality_gates=(Gate(command="uv run pytest"),),
        flavors={
            "spike": FlavorOverride(),
        },
    )
    profile = resolve_flavor_profile(cfg, "spike")
    assert profile.quality_gates == (Gate(command="uv run pytest"),)


def test_resolve_flavor_profile_no_flavor_worktree_defaults_true(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)
    profile = resolve_flavor_profile(cfg, None)
    assert profile.worktree is True


def test_resolve_flavor_profile_entry_without_worktree_defaults_true(tmp_path: Path) -> None:
    cfg = MilknadoConfig(
        agent_family="claude",
        project_root=tmp_path,
        db_path=tmp_path / ".milknado" / "milknado.db",
        flavors={
            "spike": FlavorOverride(),
        },
    )
    profile = resolve_flavor_profile(cfg, "spike")
    assert profile.worktree is True


def test_resolve_flavor_profile_worktree_false_override(tmp_path: Path) -> None:
    cfg = MilknadoConfig(
        agent_family="claude",
        project_root=tmp_path,
        db_path=tmp_path / ".milknado" / "milknado.db",
        flavors={
            "spec": FlavorOverride(worktree=False),
        },
    )
    profile = resolve_flavor_profile(cfg, "spec")
    assert profile.worktree is False


def test_resolve_flavor_profile_uses_config_worktree_default(tmp_path: Path) -> None:
    cfg = MilknadoConfig(
        agent_family="claude",
        project_root=tmp_path,
        db_path=tmp_path / ".milknado" / "milknado.db",
        worktree=False,
        flavors={"research": FlavorOverride()},
    )
    assert resolve_flavor_profile(cfg, "research").worktree is False


def test_resolve_flavor_profile_worktree_true_override(tmp_path: Path) -> None:
    cfg = MilknadoConfig(
        agent_family="claude",
        project_root=tmp_path,
        db_path=tmp_path / ".milknado" / "milknado.db",
        flavors={
            "spec": FlavorOverride(worktree=True),
        },
    )
    profile = resolve_flavor_profile(cfg, "spec")
    assert profile.worktree is True


def test_resolve_flavor_profile_brief_replaces_global(tmp_path: Path) -> None:
    cfg = MilknadoConfig(
        agent_family="claude",
        project_root=tmp_path,
        db_path=tmp_path / ".milknado" / "milknado.db",
        worker_brief_prepend="global prepend",
        flavors={
            "spike": FlavorOverride(
                brief_prepend="spike prepend",
            ),
        },
    )
    profile = resolve_flavor_profile(cfg, "spike")
    assert profile.brief_prepend == "spike prepend"


def test_resolve_flavor_profile_single_winner_tools_precedence(tmp_path: Path) -> None:
    """Flavor tools win; family tools not merged in (single-winner rule)."""
    cfg = MilknadoConfig(
        agent_family="claude",
        project_root=tmp_path,
        db_path=tmp_path / ".milknado" / "milknado.db",
        worker_tools={"claude": ("mcp__tilth__*", "Bash(rtk:*)")},
        flavors={
            "spike": FlavorOverride(
                tools=("Read", "Edit"),
            ),
        },
    )
    profile = resolve_flavor_profile(cfg, "spike")
    # Flavor tools replace; family tools not merged
    assert "Read,Edit" in profile.execution_agent
    assert "Bash(rtk:*)" not in profile.execution_agent


# ── AC5: todo-run dispatch unification ──────────────────────────────────────


def test_runner_no_default_worker_cmd_constant() -> None:
    """_DEFAULT_WORKER_CMD must be deleted from runner.py."""
    import milknado.domains.dispatch.runner as runner

    assert not hasattr(runner, "_DEFAULT_WORKER_CMD"), (
        "_DEFAULT_WORKER_CMD still present; must be deleted per spec"
    )


def test_runner_no_milknado_worker_cmd_env_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """$MILKNADO_WORKER_CMD must no longer influence dispatch."""
    import inspect

    import milknado.domains.dispatch.runner as runner

    src = inspect.getsource(runner._resolve_worker_cmd)
    assert "MILKNADO_WORKER_CMD" not in src, (
        "_resolve_worker_cmd still reads MILKNADO_WORKER_CMD env var"
    )


def test_validate_worker_argv_still_rejects_unknown_executable() -> None:
    import milknado.domains.dispatch.runner as runner

    with pytest.raises(ValueError, match="worker_cmd must start with"):
        runner.validate_worker_argv(["evil-bin", "--flag"])


# ── AC7: brief semantics ─────────────────────────────────────────────────────


def test_flavor_brief_replaces_global() -> None:
    """resolve_flavor_profile returns flavor brief_prepend, not global."""
    import tempfile
    from pathlib import Path

    from milknado.domains.common.flavor_profile import resolve_flavor_profile

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        cfg = MilknadoConfig(
            agent_family="claude",
            project_root=tmp,
            db_path=tmp / ".milknado" / "milknado.db",
            worker_brief_prepend="global prepend",
            flavors={
                "research": FlavorOverride(
                    brief_prepend="research prepend",
                ),
            },
        )
        profile = resolve_flavor_profile(cfg, "research")
        assert profile.brief_prepend == "research prepend"
        assert profile.brief_prepend != cfg.worker_brief_prepend


# ── Additional coverage for uncovered config paths ───────────────────────────


def test_load_config_flavor_valid_execution_agent(tmp_path: Path) -> None:
    """A valid execution_agent in a flavor entry parses and stores correctly."""
    cfg_path = tmp_path / "milknado.toml"
    cfg_path.write_text(
        '[milknado]\nagent_family = "claude"\n\n'
        "[milknado.flavor.spike]\n"
        'execution_agent = "claude -p --model opus"\n',
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)

    fo = cfg.flavors["spike"]
    assert fo.execution_agent == "claude -p --model opus"


def test_load_config_flavor_omp_execution_agent(tmp_path: Path) -> None:
    cfg_path = tmp_path / "milknado.toml"
    cfg_path.write_text(
        '[milknado]\nagent_family = "claude"\n\n'
        "[milknado.flavor.spike]\n"
        'execution_agent = "omp -p --auto-approve --no-session"\n',
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)

    assert cfg.flavors["spike"].execution_agent == "omp -p --auto-approve --no-session"


def test_save_config_preserves_flavor_execution_agent(tmp_path: Path) -> None:
    """save_config round-trips a flavor with execution_agent set."""
    cfg = MilknadoConfig(
        agent_family="claude",
        project_root=tmp_path,
        db_path=tmp_path / ".milknado" / "milknado.db",
        flavors={
            "spike": FlavorOverride(
                execution_agent="claude -p --model opus",
            ),
        },
    )
    cfg_path = tmp_path / "milknado.toml"
    save_config(cfg, cfg_path)
    loaded = load_config(cfg_path)
    assert loaded.flavors["spike"].execution_agent == "claude -p --model opus"


def test_save_load_roundtrip_flavor_worktree(tmp_path: Path) -> None:
    """save_config round-trips a flavor with worktree set (9th knob, ADR-005)."""
    cfg = MilknadoConfig(
        agent_family="claude",
        project_root=tmp_path,
        db_path=tmp_path / ".milknado" / "milknado.db",
        flavors={
            "spec": FlavorOverride(worktree=False),
        },
    )
    cfg_path = tmp_path / "milknado.toml"
    save_config(cfg, cfg_path)
    loaded = load_config(cfg_path)
    assert loaded.flavors["spec"].worktree is False


def test_load_config_flavor_not_a_table_raises(tmp_path: Path) -> None:
    """[milknado.flavor] being a scalar raises ValueError."""
    cfg_path = tmp_path / "milknado.toml"
    cfg_path.write_text(
        '[milknado]\nagent_family = "claude"\nflavor = "oops"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="\\[milknado.flavor\\] must be a table"):
        load_config(cfg_path)


def test_load_config_flavor_entry_not_a_table_raises(tmp_path: Path) -> None:
    """A scalar flavor entry raises ValueError."""
    # We simulate this via a raw dict through the [milknado] schema directly.
    from milknado.domains.common.config import MilknadoSection

    with pytest.raises(ValueError, match="\\[milknado.flavor.spike\\] must be a table"):
        MilknadoSection.model_validate({"flavor": {"spike": "oops"}})


def test_load_config_flavor_execution_agent_not_string_raises(tmp_path: Path) -> None:
    """Non-string execution_agent in a flavor entry raises ValueError."""
    from milknado.domains.common.flavor_codec import FlavorTable

    with pytest.raises(ValueError, match="execution_agent must be a string"):
        FlavorTable.model_validate({"execution_agent": 42})


@pytest.mark.parametrize(
    "payload",
    [
        {"execution_agent": "TOPSECRET --api-key hidden"},
        {"tools": [{"api_key": "TOPSECRET"}]},
        {"loop_mode": "TOPSECRET"},
        {"session_mode": "TOPSECRET"},
        {"on_reject": "TOPSECRET"},
        {
            "quality_gates": [
                {"command": "test", "fail_on_stdout": "[TOPSECRET"},
            ]
        },
    ],
)
def test_flavor_validation_hides_invalid_input(payload: dict[str, object]) -> None:
    from milknado.domains.common.flavor_codec import FlavorTable

    with pytest.raises(ValidationError) as exc_info:
        FlavorTable.model_validate(payload)

    assert "TOPSECRET" not in str(exc_info.value)


def test_load_config_flavor_quality_gates_non_string_item_raises(tmp_path: Path) -> None:
    """Non-string quality_gates item in a flavor entry raises ValueError."""
    cfg_path = tmp_path / "milknado.toml"
    cfg_path.write_text(
        '[milknado]\nagent_family = "claude"\n\n[milknado.flavor.spike]\nquality_gates = [42]\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="string or a table"):
        load_config(cfg_path)


def test_load_config_flavor_quality_gates_error_names_quality_gates_key(tmp_path: Path) -> None:
    """Parse errors in a flavor's quality_gates mention 'quality_gates' in the message."""
    from milknado.domains.common.flavor_codec import FlavorTable

    with pytest.raises(ValueError, match="quality_gates"):
        FlavorTable.model_validate({"quality_gates": [42]})


def test_load_config_flavor_brief_prepend_not_string_raises(tmp_path: Path) -> None:
    """Non-string brief_prepend in a flavor entry raises ValueError."""
    from milknado.domains.common.flavor_codec import FlavorTable

    with pytest.raises(ValueError, match="brief_prepend must be a string"):
        FlavorTable.model_validate({"brief_prepend": 42})


def test_load_config_flavor_brief_prepend_path_not_string_or_list_raises(tmp_path: Path) -> None:
    """brief_prepend_path being a number raises ValueError."""
    from milknado.domains.common.flavor_codec import FlavorTable

    with pytest.raises(ValueError, match="brief_prepend_path must be a string or list"):
        FlavorTable.model_validate({"brief_prepend_path": 42})


def test_load_config_flavor_brief_prepend_path_list_non_string_raises(tmp_path: Path) -> None:
    """A list brief_prepend_path with a non-string entry raises ValueError."""
    from milknado.domains.common.flavor_codec import FlavorTable

    with pytest.raises(ValueError, match="brief_prepend_path entries must be strings"):
        FlavorTable.model_validate({"brief_prepend_path": [42, "ok.md"]})


def test_absolutize_global_flavor_paths(tmp_path: Path) -> None:
    """_absolutize_global_flavor_paths resolves relative path lists in flavor entries."""
    from milknado.domains.common.flavor_codec import (
        absolutize_global_flavor_paths as _absolutize_global_flavor_paths,
    )

    base_dir = tmp_path / "global"
    base_dir.mkdir()
    raw: dict = {
        "flavor": {
            "research": {
                "brief_prepend_path": ["house.md", "/abs/path.md"],
            },
            "spike": {
                "brief_prepend_path": "relative.md",
            },
        }
    }
    _absolutize_global_flavor_paths(raw, base_dir)
    assert raw["flavor"]["research"]["brief_prepend_path"] == [
        str((base_dir / "house.md").resolve()),
        "/abs/path.md",
    ]
    assert raw["flavor"]["spike"]["brief_prepend_path"] == str(
        (base_dir / "relative.md").resolve()
    )


# ── Press hardening: boundaries + integration seams ─────────────────────────


# AC2 boundary: empty tool list replaces default (no sentinel, no items)
def test_resolve_worker_tools_empty_list_replaces_default() -> None:
    result = resolve_worker_tools("claude", [])
    assert result == ()


# AC2 assertion strength: sentinel expansion produces exact expected tuple
def test_resolve_worker_tools_sentinel_produces_exact_result() -> None:
    base = WORKER_ALLOWED_TOOLS["claude"]
    extra = "ExtraCustomTool"
    result = resolve_worker_tools("claude", ["...", extra])
    assert result == (*base, extra)


# AC3 round-trip: flavor with all four fields populated survives save→load unchanged
def test_save_load_roundtrip_flavor_all_fields(tmp_path: Path) -> None:
    cfg_path = tmp_path / "milknado.toml"
    cfg = MilknadoConfig(
        agent_family="claude",
        project_root=tmp_path,
        db_path=tmp_path / ".milknado" / "milknado.db",
        flavors={
            "prototype": FlavorOverride(
                execution_agent="claude -p --model haiku",
                tools=("Read", "Edit"),
                brief_prepend="Prototype: ship rough cut.",
                quality_gates=(Gate(command="uv run pytest -x"),),
            ),
        },
    )
    save_config(cfg, cfg_path)
    loaded = load_config(cfg_path)

    fo = loaded.flavors["prototype"]
    assert fo.execution_agent == "claude -p --model haiku"
    assert fo.tools == ("Read", "Edit")
    assert fo.brief_prepend == "Prototype: ship rough cut."
    assert fo.quality_gates == (Gate(command="uv run pytest -x"),)


# AC5 dispatch: explicit worker_cmd overrides flavor execution_agent in resolved profile
def test_resolve_worker_cmd_explicit_beats_profile_default() -> None:
    """_resolve_worker_cmd: explicit arg wins over the profile default."""
    import milknado.domains.dispatch.runner as runner

    # Explicit wins; default is ignored.
    result = runner._resolve_worker_cmd("claude -p", "claude --model opus")
    assert result == ["claude", "-p"]


def test_resolve_worker_cmd_empty_explicit_falls_back_to_profile() -> None:
    """_resolve_worker_cmd: empty/None explicit falls back to the profile default."""
    import milknado.domains.dispatch.runner as runner

    result = runner._resolve_worker_cmd(None, "claude -p")
    assert result == ["claude", "-p"]

    result = runner._resolve_worker_cmd("", "claude -p")
    assert result == ["claude", "-p"]


# AC7: milknado_todo_brief returns flavor brief_prepend when config has flavor entry
def test_todo_brief_returns_flavor_prepend_from_config(tmp_path: Path) -> None:
    """milknado_todo_brief uses flavor brief_prepend when the task has a matching flavor."""
    from milknado.mcp.todo import milknado_todo_brief
    from milknado.mcp.todo_mutate import milknado_todo_add

    def _call(tool, **kwargs):
        fn = getattr(tool, "fn", tool)
        return fn(**kwargs)

    root = str(tmp_path)
    cfg_path = tmp_path / "milknado.toml"
    cfg_path.write_text(
        '[milknado]\nagent_family = "claude"\n\n'
        "[milknado.flavor.research]\n"
        'brief_prepend = "RESEARCH_MARKER: go deep."\n',
        encoding="utf-8",
    )

    task = _call(
        milknado_todo_add, description="investigate X", flavor="research", project_root=root
    )
    result = _call(milknado_todo_brief, node_id=task["id"], project_root=root)
    assert "RESEARCH_MARKER: go deep." in result["brief"]


def test_todo_brief_resolves_custom_flavor_brief_path(tmp_path: Path) -> None:
    """A registry flavor's file-backed prepend reaches the worker brief."""
    from milknado.mcp.todo import milknado_todo_brief
    from milknado.mcp.todo_mutate import milknado_todo_add

    def _call(tool, **kwargs):
        fn = getattr(tool, "fn", tool)
        return fn(**kwargs)

    marker = "TRIAGE_MARKER: evidence first."
    brief_file = tmp_path / "triage.md"
    brief_file.write_text(marker, encoding="utf-8")
    cfg_path = tmp_path / "milknado.toml"
    cfg_path.write_text(
        '[milknado]\nagent_family = "claude"\n\n'
        "[milknado.flavor.triage]\n"
        "quality_gates = []\n"
        'brief_prepend_path = "triage.md"\n',
        encoding="utf-8",
    )

    task = _call(
        milknado_todo_add,
        description="triage issue 42",
        flavor="triage",
        project_root=str(tmp_path),
    )
    result = _call(milknado_todo_brief, node_id=task["id"], project_root=str(tmp_path))

    assert result["brief"].startswith(marker)


# ── adversarial-review-loops: session_mode/review config contract ───────────


def test_validate_session_mode_accepts_fresh_and_resume() -> None:
    from milknado.domains.common.flavor_codec import validate_session_mode

    assert validate_session_mode("fresh") == "fresh"
    assert validate_session_mode("resume") == "resume"


def test_validate_session_mode_rejects_invalid() -> None:
    from milknado.domains.common.flavor_codec import validate_session_mode

    with pytest.raises(ValueError, match="session_mode"):
        validate_session_mode("eventual")


def test_validate_on_reject_accepts_block_and_warn() -> None:
    from milknado.domains.common.flavor_codec import validate_on_reject

    assert validate_on_reject("block") == "block"
    assert validate_on_reject("warn") == "warn"


def test_validate_on_reject_rejects_invalid() -> None:
    from milknado.domains.common.flavor_codec import validate_on_reject

    with pytest.raises(ValueError, match="on_reject"):
        validate_on_reject("ignore")


def test_load_config_flavor_review_fields_parse(tmp_path: Path) -> None:
    cfg_path = tmp_path / "milknado.toml"
    cfg_path.write_text(
        '[milknado]\nagent_family = "claude"\n\n'
        "[milknado.flavor.implement]\n"
        'session_mode = "resume"\n'
        "review = true\n"
        'review_agent = "claude -p --model opus"\n'
        "review_max_rounds = 5\n"
        'on_reject = "warn"\n',
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    fo = cfg.flavors["implement"]
    assert fo.session_mode == "resume"
    assert fo.review is True
    assert fo.review_agent == "claude -p --model opus"
    assert fo.review_max_rounds == 5
    assert fo.on_reject == "warn"


def test_save_load_roundtrip_flavor_review_fields(tmp_path: Path) -> None:
    cfg_path = tmp_path / "milknado.toml"
    cfg = MilknadoConfig(
        agent_family="claude",
        project_root=tmp_path,
        db_path=tmp_path / ".milknado" / "milknado.db",
        flavors={
            "spec": FlavorOverride(
                session_mode="resume",
                review=True,
                review_agent="claude -p --model opus",
                review_max_rounds=3,
                on_reject="warn",
            ),
        },
    )
    save_config(cfg, cfg_path)
    loaded = load_config(cfg_path)
    fo = loaded.flavors["spec"]
    assert fo.session_mode == "resume"
    assert fo.review is True
    assert fo.review_agent == "claude -p --model opus"
    assert fo.review_max_rounds == 3
    assert fo.on_reject == "warn"


def test_load_config_flavor_review_not_bool_raises(tmp_path: Path) -> None:
    cfg_path = tmp_path / "milknado.toml"
    cfg_path.write_text(
        '[milknado]\nagent_family = "claude"\n\n[milknado.flavor.spike]\nreview = "yes"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="review must be a boolean"):
        load_config(cfg_path)


def test_load_config_flavor_review_agent_not_string_raises(tmp_path: Path) -> None:
    cfg_path = tmp_path / "milknado.toml"
    cfg_path.write_text(
        '[milknado]\nagent_family = "claude"\n\n[milknado.flavor.spike]\nreview_agent = 42\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="review_agent must be a string"):
        load_config(cfg_path)


# AC — cursor-agent resume fail-fast (adversarial-review-loops-F001), enforced
# at both FlavorTable (config load) and resolve_flavor_profile.


def test_load_config_flavor_resume_cursor_agent_execution_agent_raises(tmp_path: Path) -> None:
    cfg_path = tmp_path / "milknado.toml"
    cfg_path.write_text(
        '[milknado]\nagent_family = "claude"\n\n'
        "[milknado.flavor.implement]\n"
        'session_mode = "resume"\n'
        'execution_agent = "cursor-agent -p"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="adversarial-review-loops-F001"):
        load_config(cfg_path)


def test_load_config_flavor_resume_cursor_agent_review_agent_raises(tmp_path: Path) -> None:
    cfg_path = tmp_path / "milknado.toml"
    cfg_path.write_text(
        '[milknado]\nagent_family = "claude"\n\n'
        "[milknado.flavor.implement]\n"
        'session_mode = "resume"\n'
        'review_agent = "cursor-agent -p"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="adversarial-review-loops-F001"):
        load_config(cfg_path)


def test_load_config_flavor_resume_checks_both_configured_agents(tmp_path: Path) -> None:
    cfg_path = tmp_path / "milknado.toml"
    cfg_path.write_text(
        '[milknado]\nagent_family = "claude"\n\n'
        "[milknado.flavor.implement]\n"
        'session_mode = "resume"\n'
        'execution_agent = "cursor-agent -p"\n'
        'review_agent = "claude -p"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="adversarial-review-loops-F001"):
        load_config(cfg_path)


def test_resolve_flavor_profile_resume_cursor_agent_family_raises(tmp_path: Path) -> None:
    """agent_family alone (no execution_agent override) drives the effective family."""
    cfg = MilknadoConfig(
        agent_family="cursor-agent",
        project_root=tmp_path,
        db_path=tmp_path / ".milknado" / "milknado.db",
        flavors={
            "implement": FlavorOverride(session_mode="resume", execution_agent=""),
        },
    )
    with pytest.raises(ValueError, match="adversarial-review-loops-F001"):
        resolve_flavor_profile(cfg, "implement")


# AC — resolve_flavor_profile: default review, session_mode/on_reject/
# review_agent/review_max_rounds resolution.


def test_resolve_flavor_profile_review_default_true_for_implement(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)
    profile = resolve_flavor_profile(cfg, "implement")
    assert profile.review is True


def test_resolve_flavor_profile_review_default_true_for_spec(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)
    profile = resolve_flavor_profile(cfg, "spec")
    assert profile.review is True


def test_resolve_flavor_profile_review_default_false_for_other_flavors(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)
    profile = resolve_flavor_profile(cfg, "spike")
    assert profile.review is False


def test_resolve_flavor_profile_review_false_override_wins_on_implement(tmp_path: Path) -> None:
    cfg = MilknadoConfig(
        agent_family="claude",
        project_root=tmp_path,
        db_path=tmp_path / ".milknado" / "milknado.db",
        flavors={
            "implement": FlavorOverride(review=False),
        },
    )
    profile = resolve_flavor_profile(cfg, "implement")
    assert profile.review is False


def test_resolve_flavor_profile_review_config_resolves_from_override(tmp_path: Path) -> None:
    cfg = MilknadoConfig(
        agent_family="claude",
        project_root=tmp_path,
        db_path=tmp_path / ".milknado" / "milknado.db",
        flavors={
            "spike": FlavorOverride(
                session_mode="resume",
                review_agent="claude -p --model opus",
                review_max_rounds=7,
                on_reject="warn",
            ),
        },
    )
    profile = resolve_flavor_profile(cfg, "spike")
    assert profile.session_mode == "resume"
    assert profile.review_agent == "claude -p --model opus"
    assert profile.review_max_rounds == 7
    assert profile.on_reject == "warn"


def test_flavor_preset_toml_examples_parse_and_use_codec_vocabulary() -> None:
    doc_path = (
        Path(__file__).parents[1]
        / "plugins"
        / "milknado"
        / "skills"
        / "milknado-config"
        / "references"
        / "flavor-presets.md"
    )
    snippets = re.findall(r"```toml\n(.*?)\n```", doc_path.read_text(encoding="utf-8"), re.DOTALL)
    assert snippets
    for snippet in snippets:
        raw = tomllib.loads(snippet)
        flavor_tables = raw.get("milknado", {}).get("flavor", {}).values()
        for flavor in flavor_tables:
            if "on_reject" in flavor:
                assert flavor["on_reject"] in {"block", "warn"}
