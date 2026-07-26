"""Tests for explicit configuration inheritance and inspection output."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from milknado.cli import app
from milknado.domains.common import load_config, load_config_details
from milknado.domains.common.config_view import explain_view, resolved_view
from milknado.domains.common.flavor_profile import resolve_flavor_profile

runner = CliRunner()


def _write(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


@pytest.fixture()
def xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "xdg"
    home.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
    return home


def test_inherit_global_false_skips_global_layer(xdg: Path, tmp_path: Path) -> None:
    _write(xdg / "milknado" / "milknado.toml", "[milknado]\nconcurrency_limit = 99\n")
    local = _write(
        tmp_path / "milknado.toml",
        '[milknado]\ninherit_global = false\nagent_family = "claude"\n',
    )

    assert load_config(local).concurrency_limit == 4


def test_flavor_inherit_false_replaces_global_table(xdg: Path, tmp_path: Path) -> None:
    _write(
        xdg / "milknado" / "milknado.toml",
        (
            "[milknado]\nmax_turns = 60\n\n"
            '[milknado.flavor.research]\nquality_gates = ["echo global"]\nmax_turns = 7\n'
        ),
    )
    local = _write(
        tmp_path / "milknado.toml",
        (
            '[milknado]\nagent_family = "claude"\n\n'
            '[milknado.flavor.research]\ninherit = false\ntools = ["Read"]\n'
        ),
    )

    profile = resolve_flavor_profile(load_config(local), "research")

    assert profile.quality_gates is None
    assert profile.max_turns == 60
    assert "Read" in profile.execution_agent


@pytest.mark.parametrize(
    "body, match",
    [
        ('[milknado]\ninherit_global = "false"\n', "inherit_global must be a boolean"),
        (
            '[milknado]\n[milknado.flavor.research]\ninherit = "false"\n',
            "inherit must be a boolean",
        ),
    ],
)
def test_invalid_inherit_controls_fail_at_config_boundary(
    tmp_path: Path, body: str, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        load_config(_write(tmp_path / "milknado.toml", body))


def test_resolved_flavor_output_is_the_runtime_profile(xdg: Path, tmp_path: Path) -> None:
    local = _write(
        tmp_path / "milknado.toml",
        (
            '[milknado]\nworker_agent_type = "worker"\n\n'
            "[milknado.flavor.research]\nmax_turns = 11\n"
        ),
    )
    details = load_config_details(local)
    expected = resolve_flavor_profile(details.config, "research").model_dump(mode="python")

    assert resolved_view(details, "research") == expected
    result = runner.invoke(
        app,
        ["config", "show", "--resolved", "--flavor", "research", "--project-root", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == expected


def test_explain_attributes_overridden_and_inherited_flavor_keys(
    xdg: Path, tmp_path: Path
) -> None:
    global_path = xdg / "milknado" / "milknado.toml"
    _write(
        global_path,
        (
            "[milknado]\nconcurrency_limit = 7\n\n"
            '[milknado.flavor.research]\nmax_turns = 9\nquality_gates = ["echo global"]\n'
        ),
    )
    local = _write(
        tmp_path / "milknado.toml",
        (
            "[milknado]\nconcurrency_limit = 12\n\n"
            '[milknado.flavor.research]\nbrief_prepend = "local brief"\n'
        ),
    )

    explained = explain_view(load_config_details(local), "research")

    assert explained["max_turns"]["source"] == f"global:{global_path} (flavor:research)"
    assert explained["brief_prepend"]["source"] == f"local:{local} (flavor:research)"


def test_config_show_resolved_and_explain_are_json(xdg: Path, tmp_path: Path) -> None:
    global_path = xdg / "milknado" / "milknado.toml"
    _write(global_path, "[milknado]\nconcurrency_limit = 7\n")
    _write(tmp_path / "milknado.toml", "[milknado]\nconcurrency_limit = 12\n")

    resolved = runner.invoke(
        app, ["config", "show", "--resolved", "--project-root", str(tmp_path)]
    )
    explained = runner.invoke(
        app, ["config", "show", "--explain", "--project-root", str(tmp_path)]
    )

    assert resolved.exit_code == 0, resolved.output
    assert json.loads(resolved.output)["concurrency_limit"] == 12
    assert explained.exit_code == 0, explained.output
    assert json.loads(explained.output)["concurrency_limit"] == {
        "source": f"local:{tmp_path / 'milknado.toml'}",
        "value": 12,
    }


def test_explain_attributes_inline_prompt_to_its_local_layer(tmp_path: Path) -> None:
    local = _write(
        tmp_path / "milknado.toml",
        '[milknado]\n[milknado.prompts]\nplanning_prepend = "local prompt"\n',
    )

    explained = explain_view(load_config_details(local))

    assert explained["planning_prompt_prepend"] == {
        "source": f"local:{local}",
        "value": "local prompt",
    }


def test_config_show_requires_one_view_and_supports_defaults(tmp_path: Path) -> None:
    invalid = runner.invoke(app, ["config", "show", "--project-root", str(tmp_path)])
    resolved = runner.invoke(
        app, ["config", "show", "--resolved", "--project-root", str(tmp_path)]
    )

    assert invalid.exit_code != 0
    assert resolved.exit_code == 0, resolved.output
    assert json.loads(resolved.output)["db_path"] == str(tmp_path / ".milknado" / "milknado.db")
