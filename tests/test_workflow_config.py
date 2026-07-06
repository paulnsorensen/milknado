"""Config + flavor-profile resolution for the native Workflow backend fields.

Covers the spec acceptance criteria for worker_agent_type / loop_mode /
max_iterations / max_turns: global defaults, global override, per-flavor override
(per-flavor wins), TOML parse, and save/load round-trip.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from milknado.domains.common.config import (
    DEFAULT_LOOP_MODE,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_TURNS,
    DEFAULT_WORKER_AGENT_TYPE,
    FlavorOverride,
    MilknadoConfig,
    load_config,
    save_config,
)
from milknado.domains.common.flavor_profile import resolve_flavor_profile


def _cfg(tmp_path: Path, **kwargs) -> MilknadoConfig:  # noqa: ANN003
    return MilknadoConfig(
        agent_family="claude",
        project_root=tmp_path,
        db_path=tmp_path / ".milknado" / "milknado.db",
        **kwargs,
    )


# ── defaults ─────────────────────────────────────────────────────────────────


def test_defaults_match_spec(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    assert cfg.worker_agent_type == DEFAULT_WORKER_AGENT_TYPE == "milknado:milknado-worker"
    assert cfg.loop_mode == DEFAULT_LOOP_MODE == "redispatch"
    assert cfg.max_iterations == DEFAULT_MAX_ITERATIONS == 8
    assert cfg.max_turns == DEFAULT_MAX_TURNS == 60


def test_profile_no_flavor_inherits_global_defaults(tmp_path: Path) -> None:
    profile = resolve_flavor_profile(_cfg(tmp_path), None)
    assert profile.worker_agent_type == "milknado:milknado-worker"
    assert profile.loop_mode == "redispatch"
    assert profile.max_iterations == 8
    assert profile.max_turns == 60


# ── global override ──────────────────────────────────────────────────────────


def test_global_worker_agent_type_override_honored(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, worker_agent_type="acme:custom-worker", max_iterations=3, max_turns=20)
    profile = resolve_flavor_profile(cfg, "implement")
    assert profile.worker_agent_type == "acme:custom-worker"
    assert profile.max_iterations == 3
    assert profile.max_turns == 20


# ── per-flavor override wins over global ─────────────────────────────────────


def test_per_flavor_agent_type_wins_over_global(tmp_path: Path) -> None:
    cfg = _cfg(
        tmp_path,
        worker_agent_type="global:worker",
        flavors={"spike": FlavorOverride(agent_type="milknado:spike-worker")},
    )
    spike = resolve_flavor_profile(cfg, "spike")
    assert spike.worker_agent_type == "milknado:spike-worker"
    # A flavor with no override still inherits the global.
    impl = resolve_flavor_profile(cfg, "implement")
    assert impl.worker_agent_type == "global:worker"


def test_per_flavor_loop_mode_and_caps_win(tmp_path: Path) -> None:
    cfg = _cfg(
        tmp_path,
        loop_mode="redispatch",
        max_iterations=8,
        max_turns=60,
        flavors={
            "spike": FlavorOverride(loop_mode="single", max_turns=25),
        },
    )
    profile = resolve_flavor_profile(cfg, "spike")
    assert profile.loop_mode == "single"
    assert profile.max_turns == 25
    # max_iterations not overridden per-flavor -> inherits global.
    assert profile.max_iterations == 8


def test_per_flavor_none_fields_inherit_global(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, max_iterations=12, flavors={"research": FlavorOverride()})
    assert resolve_flavor_profile(cfg, "research").max_iterations == 12


# ── TOML parse + defaults ────────────────────────────────────────────────────


def test_toml_parse_per_flavor_workflow_fields(tmp_path: Path) -> None:
    path = tmp_path / "milknado.toml"
    path.write_text(
        '[milknado]\nagent_family = "claude"\n'
        'worker_agent_type = "team:worker"\n\n'
        "[milknado.flavor.implement]\n"
        'loop_mode = "redispatch"\nmax_iterations = 5\nmax_turns = 40\n\n'
        "[milknado.flavor.spike]\n"
        'loop_mode = "single"\nmax_turns = 25\n'
        'agent_type = "milknado:milknado-spike-worker"\n'
    )
    cfg = load_config(path, include_global=False)
    assert cfg.worker_agent_type == "team:worker"
    impl = resolve_flavor_profile(cfg, "implement")
    assert (impl.loop_mode, impl.max_iterations, impl.max_turns) == ("redispatch", 5, 40)
    spike = resolve_flavor_profile(cfg, "spike")
    assert spike.loop_mode == "single"
    assert spike.max_turns == 25
    assert spike.worker_agent_type == "milknado:milknado-spike-worker"


def test_loop_mode_defaults_to_redispatch_when_absent(tmp_path: Path) -> None:
    path = tmp_path / "milknado.toml"
    path.write_text(
        '[milknado]\nagent_family = "claude"\n\n[milknado.flavor.spike]\nmax_turns = 25\n'
    )
    cfg = load_config(path, include_global=False)
    # Flavor entry present but no loop_mode -> profile falls back to the global default.
    assert resolve_flavor_profile(cfg, "spike").loop_mode == "redispatch"


def test_invalid_loop_mode_rejected(tmp_path: Path) -> None:
    path = tmp_path / "milknado.toml"
    path.write_text('[milknado]\nagent_family = "claude"\nloop_mode = "turbo"\n')
    with pytest.raises(ValueError, match="loop_mode must be one of"):
        load_config(path, include_global=False)


def test_invalid_max_iterations_rejected(tmp_path: Path) -> None:
    path = tmp_path / "milknado.toml"
    path.write_text(
        '[milknado]\nagent_family = "claude"\n\n[milknado.flavor.spike]\nmax_iterations = 0\n'
    )
    with pytest.raises(ValueError, match="max_iterations must be >= 1"):
        load_config(path, include_global=False)


def test_non_string_worker_agent_type_rejected(tmp_path: Path) -> None:
    """A non-string worker_agent_type is rejected, not silently str()-coerced."""
    path = tmp_path / "milknado.toml"
    path.write_text('[milknado]\nagent_family = "claude"\nworker_agent_type = 7\n')
    with pytest.raises(ValueError, match="worker_agent_type must be a string"):
        load_config(path, include_global=False)


def test_config_commit_footer_rejects_non_string(tmp_path: Path) -> None:
    """A non-string commit_footer is rejected, not silently str()-coerced."""
    path = tmp_path / "milknado.toml"
    path.write_text('[milknado]\nagent_family = "claude"\ncommit_footer = 7\n')
    with pytest.raises(ValueError, match="commit_footer must be a string"):
        load_config(path, include_global=False)


# ── save/load round-trip ─────────────────────────────────────────────────────


def test_save_load_round_trip_preserves_workflow_fields(tmp_path: Path) -> None:
    cfg = _cfg(
        tmp_path,
        worker_agent_type="team:worker",
        loop_mode="redispatch",
        max_iterations=9,
        max_turns=70,
        flavors={
            "spike": FlavorOverride(
                loop_mode="single",
                max_iterations=4,
                max_turns=25,
                agent_type="milknado:spike-worker",
            )
        },
    )
    path = tmp_path / "milknado.toml"
    save_config(cfg, path)
    reloaded = load_config(path, include_global=False)
    assert reloaded.worker_agent_type == "team:worker"
    assert reloaded.max_iterations == 9
    assert reloaded.max_turns == 70
    spike = reloaded.flavors["spike"]
    assert spike.loop_mode == "single"
    assert spike.max_iterations == 4
    assert spike.max_turns == 25
    assert spike.agent_type == "milknado:spike-worker"


def test_per_flavor_agent_type_must_be_string(tmp_path: Path) -> None:
    path = tmp_path / "milknado.toml"
    path.write_text(
        '[milknado]\nagent_family = "claude"\n\n[milknado.flavor.spike]\nagent_type = 7\n'
    )
    with pytest.raises(ValueError, match="agent_type must be a string"):
        load_config(path, include_global=False)


def test_per_flavor_max_iterations_must_be_integer(tmp_path: Path) -> None:
    path = tmp_path / "milknado.toml"
    path.write_text(
        '[milknado]\nagent_family = "claude"\n\n[milknado.flavor.spike]\nmax_iterations = "lots"\n'
    )
    with pytest.raises(ValueError, match="max_iterations must be an integer"):
        load_config(path, include_global=False)


def test_global_max_iterations_rejects_non_positive(tmp_path: Path) -> None:
    """The global [milknado] cap is validated like the per-flavor one: a
    zero/negative max_iterations would make the redispatch loop never dispatch a
    worker, so it must be rejected at parse, not silently coerced."""
    path = tmp_path / "milknado.toml"
    path.write_text('[milknado]\nagent_family = "claude"\nmax_iterations = 0\n')
    with pytest.raises(ValueError, match=r"\[milknado\] max_iterations must be >= 1"):
        load_config(path, include_global=False)


def test_global_max_turns_rejects_negative(tmp_path: Path) -> None:
    path = tmp_path / "milknado.toml"
    path.write_text('[milknado]\nagent_family = "claude"\nmax_turns = -3\n')
    with pytest.raises(ValueError, match=r"\[milknado\] max_turns must be >= 1"):
        load_config(path, include_global=False)


def test_global_max_iterations_rejects_bool(tmp_path: Path) -> None:
    """TOML booleans are ints in Python; the cap must not accept ``true`` as 1."""
    path = tmp_path / "milknado.toml"
    path.write_text('[milknado]\nagent_family = "claude"\nmax_iterations = true\n')
    with pytest.raises(ValueError, match=r"\[milknado\] max_iterations must be an integer"):
        load_config(path, include_global=False)
