"""Config + flavor-profile resolution for the max_iterations retry budget.

Covers global defaults, global override, per-flavor override (per-flavor wins),
TOML parse, validation, save/load round-trip, and the retired Workflow keys.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict, Unpack

import pytest

from milknado.domains.common.config import (
    DEFAULT_MAX_ITERATIONS,
    FlavorOverride,
    MilknadoConfig,
    decode_milknado_section,
    load_config,
    save_config,
)
from milknado.domains.common.flavor_profile import resolve_flavor_profile


class _ConfigOverrides(TypedDict, total=False):
    max_iterations: int
    flavors: dict[str, FlavorOverride]


def _cfg(tmp_path: Path, **kwargs: Unpack[_ConfigOverrides]) -> MilknadoConfig:
    return MilknadoConfig(
        agent_family="claude",
        project_root=tmp_path,
        db_path=tmp_path / ".milknado" / "milknado.db",
        **kwargs,
    )


# ── defaults ─────────────────────────────────────────────────────────────────


def test_defaults_match_spec(tmp_path: Path) -> None:
    assert _cfg(tmp_path).max_iterations == DEFAULT_MAX_ITERATIONS == 8


def test_profile_no_flavor_inherits_global_defaults(tmp_path: Path) -> None:
    assert resolve_flavor_profile(_cfg(tmp_path), None).max_iterations == 8


# ── global override ──────────────────────────────────────────────────────────


def test_global_max_iterations_override_honored(tmp_path: Path) -> None:
    profile = resolve_flavor_profile(_cfg(tmp_path, max_iterations=3), "implement")
    assert profile.max_iterations == 3


# ── per-flavor override wins over global ─────────────────────────────────────


def test_per_flavor_max_iterations_wins_over_global(tmp_path: Path) -> None:
    cfg = _cfg(
        tmp_path,
        max_iterations=8,
        flavors={"spike": FlavorOverride(max_iterations=2)},
    )
    assert resolve_flavor_profile(cfg, "spike").max_iterations == 2
    assert resolve_flavor_profile(cfg, "implement").max_iterations == 8


def test_per_flavor_none_fields_inherit_global(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, max_iterations=12, flavors={"research": FlavorOverride()})
    assert resolve_flavor_profile(cfg, "research").max_iterations == 12


# ── TOML parse + defaults ────────────────────────────────────────────────────


def test_toml_parse_per_flavor_max_iterations(tmp_path: Path) -> None:
    path = tmp_path / "milknado.toml"
    _ = path.write_text(
        '[milknado]\nagent_family = "claude"\nmax_iterations = 6\n\n'
        + "[milknado.flavor.implement]\nmax_iterations = 5\n"
    )
    cfg = load_config(path, include_global=False)
    assert cfg.max_iterations == 6
    assert resolve_flavor_profile(cfg, "implement").max_iterations == 5


def test_retired_workflow_keys_are_ignored(tmp_path: Path) -> None:
    """Stale Workflow keys decode as unknown fields: no error, no effect, not persisted."""
    path = tmp_path / "milknado.toml"
    _ = path.write_text(
        '[milknado]\nagent_family = "claude"\nworker_agent_type = "x"\n'
        + 'loop_mode = "turbo"\nmax_turns = 9\n\n'
        + '[milknado.flavor.spike]\nagent_type = 7\nloop_mode = "single"\nmax_turns = 3\n'
    )
    cfg = load_config(path, include_global=False)
    assert resolve_flavor_profile(cfg, "spike").max_iterations == DEFAULT_MAX_ITERATIONS
    saved = tmp_path / "saved.toml"
    save_config(cfg, saved)
    text = saved.read_text()
    assert not {"worker_agent_type", "agent_type", "loop_mode", "max_turns"} & set(
        key for line in text.splitlines() for key in [line.split("=")[0].strip()]
    )


def test_invalid_max_iterations_rejected(tmp_path: Path) -> None:
    path = tmp_path / "milknado.toml"
    _ = path.write_text(
        '[milknado]\nagent_family = "claude"\n\n[milknado.flavor.spike]\nmax_iterations = 0\n'
    )
    with pytest.raises(ValueError, match="max_iterations must be >= 1"):
        _ = load_config(path, include_global=False)


def test_config_commit_footer_rejects_non_string(tmp_path: Path) -> None:
    """A non-string commit_footer is rejected, not silently str()-coerced."""
    path = tmp_path / "milknado.toml"
    _ = path.write_text('[milknado]\nagent_family = "claude"\ncommit_footer = 7\n')
    with pytest.raises(ValueError, match="commit_footer must be a string"):
        _ = load_config(path, include_global=False)


# ── save/load round-trip ─────────────────────────────────────────────────────


def test_save_load_round_trip_preserves_max_iterations(tmp_path: Path) -> None:
    cfg = _cfg(
        tmp_path,
        max_iterations=9,
        flavors={"spike": FlavorOverride(max_iterations=4)},
    )
    path = tmp_path / "milknado.toml"
    save_config(cfg, path)
    reloaded = load_config(path, include_global=False)
    assert reloaded.max_iterations == 9
    assert reloaded.flavors["spike"].max_iterations == 4


def test_per_flavor_max_iterations_must_be_integer(tmp_path: Path) -> None:
    path = tmp_path / "milknado.toml"
    _ = path.write_text(
        '[milknado]\nagent_family = "claude"\n\n[milknado.flavor.spike]\nmax_iterations = "lots"\n'
    )
    with pytest.raises(ValueError, match="max_iterations must be an integer"):
        _ = load_config(path, include_global=False)


def test_global_max_iterations_rejects_non_positive(tmp_path: Path) -> None:
    """The global [milknado] cap is validated like the per-flavor one: a
    zero/negative max_iterations would make the redispatch loop never dispatch a
    worker, so it must be rejected at parse, not silently coerced."""
    path = tmp_path / "milknado.toml"
    _ = path.write_text('[milknado]\nagent_family = "claude"\nmax_iterations = 0\n')
    with pytest.raises(ValueError, match=r"\[milknado\] max_iterations must be >= 1"):
        _ = load_config(path, include_global=False)


def test_global_max_iterations_rejects_bool(tmp_path: Path) -> None:
    """TOML booleans are ints in Python; the cap must not accept ``true`` as 1."""
    path = tmp_path / "milknado.toml"
    _ = path.write_text('[milknado]\nagent_family = "claude"\nmax_iterations = true\n')
    with pytest.raises(ValueError, match=r"\[milknado\] max_iterations must be an integer"):
        _ = load_config(path, include_global=False)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"plan_reviewer_agent": 7}, "plan_reviewer_agent must be a string"),
        ({"plan_review_max_rounds": True}, "plan_review_max_rounds must be an integer"),
        ({"flavor": {"x": {"execution_agent": 7}}}, "execution_agent must be a string"),
        ({"flavor": {"x": {"brief_prepend": 7}}}, "brief_prepend must be a string"),
        ({"flavor": {"x": {"brief_prepend_path": ["ok", 7]}}}, "brief_prepend_path"),
        ({"flavor": {"x": {"session_mode": 7}}}, "session_mode must be a string"),
        ({"flavor": {"x": {"on_reject": 7}}}, "on_reject must be a string"),
        ({"flavor": {"x": {"review_max_rounds": True}}}, "review_max_rounds must be an integer"),
        ({"flavor": {"x": {"max_iterations": "many"}}}, "max_iterations must be an integer"),
    ],
)
def test_normalization_preserves_domain_validation_messages(
    payload: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _ = decode_milknado_section(payload)
