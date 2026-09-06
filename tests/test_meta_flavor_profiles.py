"""Cross-seam verification for the configured flavor profiles."""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import msgspec
import pytest

from milknado.domains.common.config import Gate, LoadedConfig, load_config_details
from milknado.domains.common.config_view import resolved_view
from milknado.domains.common.flavor_profile import FlavorProfile, resolve_flavor_profile
from milknado.domains.common.types import BUILTIN_FLAVORS

REPO_ROOT = Path(__file__).resolve().parents[1]
LUNA_MODEL = "openai-codex/gpt-5.6-luna"
LUNA_XHIGH = (
    "omp -p --auto-approve --no-session --model openai-codex/gpt-5.6-luna --thinking xhigh"
)
LUNA_HIGH = "omp -p --auto-approve --no-session --model openai-codex/gpt-5.6-luna --thinking high"
LUNA_RESEARCH = (
    "omp -p --auto-approve --no-session --tools=read,grep,glob,lsp "
    + "--model openai-codex/gpt-5.6-luna --thinking xhigh"
)
OPUS_REVIEWER = (
    "claude --model opus -p --permission-mode plan --allowedTools "
    + "'Read,Glob,Grep,mcp__tilth__tilth_read,mcp__tilth__tilth_search_v2,"
    + "mcp__tilth__tilth_diff,mcp__milknado__milknado_deposit_result' "
    + "--append-system-prompt 'Review mode: severity-report. Read-only review. "
    + "Do not modify files.'"
)


def _profile(
    execution_agent: str,
    quality_gates: tuple[str, ...] | None,
    brief_prepend: str | None,
    **overrides: object,
) -> FlavorProfile:
    values: dict[str, object] = {
        "execution_agent": execution_agent,
        "quality_gates": (
            None if quality_gates is None else tuple(Gate(command=gate) for gate in quality_gates)
        ),
        "brief_prepend": brief_prepend,
        "worker_agent_type": "milknado:milknado-worker",
        "loop_mode": "redispatch",
        "max_iterations": 8,
        "max_turns": 60,
        "worktree": True,
        "session_mode": "fresh",
        "review": False,
        "review_agent": None,
        "review_max_rounds": 2,
        "on_reject": "block",
    }
    values.update(overrides)
    return msgspec.convert(values, type=FlavorProfile)


EXPECTED_PROFILES = {
    "implement": _profile(
        LUNA_XHIGH,
        ("just check-llm",),
        "Implement the scoped production change in the worktree, then prove it with "
        + "the required quality gate.",
        review=True,
        review_agent=OPUS_REVIEWER,
    ),
    "spec": _profile(
        LUNA_XHIGH,
        (),
        "Write a concrete, testable design or specification. Do not modify production code "
        + "outside the requested spec artifact.",
        review=True,
        review_agent=OPUS_REVIEWER,
    ),
    "spike": _profile(
        LUNA_HIGH,
        (),
        "Run a time-boxed spike. Optimize for answering the question, record evidence "
        + "and a recommendation, and identify follow-up work.",
    ),
    "prototype": _profile(
        LUNA_HIGH,
        ("just lint",),
        "Build a working prototype end to end. Keep the scope narrow; production hardening "
        + "and exhaustive edge cases are not required.",
    ),
    "research": _profile(
        LUNA_RESEARCH,
        (),
        "Research only. Do not modify files. Report evidence, options, a recommendation, "
        + "and confidence.",
    ),
    "review": _profile(
        OPUS_REVIEWER,
        (),
        "Review only. Do not modify files. Report severity-ranked, evidence-backed "
        + "findings and a clear merge recommendation.",
    ),
    "plate": _profile(
        LUNA_XHIGH,
        ("just check-llm",),
        "Prepare the finished change for publication. Preserve the scoped work, run the "
        + "required gate, and make no unrelated edits.",
    ),
    "triage": _profile(
        LUNA_XHIGH,
        (),
        "Single-issue rennet-style triage. Use evidence first from code, tests, decisions, "
        + "and upstream facts; do not edit code. Apply idempotent GitHub triage/* labels "
        + "and evidence comments. Never close an issue without a human gate. Deposit one "
        + "structured result with verdict, confidence, evidence, actions, and "
        + "implementation-ready next steps.",
    ),
}


def _isolated_global(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    xdg = tmp_path / "xdg"
    global_path = xdg / "milknado" / "milknado.toml"
    global_path.parent.mkdir(parents=True)
    _ = global_path.write_text("[milknado]\n", encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    return global_path


@pytest.fixture
def repository_details(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LoadedConfig:
    _ = _isolated_global(tmp_path, monkeypatch)
    return load_config_details(REPO_ROOT / "milknado.toml")


def _model(command: str) -> str:
    parts = shlex.split(command)
    return parts[parts.index("--model") + 1]


def test_repository_profiles_match_every_field_and_runtime_projection(
    repository_details: LoadedConfig,
) -> None:
    details = repository_details
    assert set(EXPECTED_PROFILES) == BUILTIN_FLAVORS | {"triage"}
    assert set(details.config.flavors) == set(EXPECTED_PROFILES)
    assert details.config.flavor_registry == frozenset(EXPECTED_PROFILES)

    for flavor, expected in EXPECTED_PROFILES.items():
        profile = resolve_flavor_profile(details.config, flavor)
        assert profile == expected, flavor
        assert resolved_view(details, flavor) == json.loads(msgspec.json.encode(expected)), flavor

    for flavor, expected in EXPECTED_PROFILES.items():
        if flavor == "review":
            assert _model(expected.execution_agent) == "opus"
            assert expected.review is False
            assert expected.review_agent is None
        else:
            assert _model(expected.execution_agent) == LUNA_MODEL, flavor

    assert EXPECTED_PROFILES["implement"].review_agent == OPUS_REVIEWER
    assert EXPECTED_PROFILES["spec"].review_agent == OPUS_REVIEWER
    assert EXPECTED_PROFILES["implement"].review_agent
    assert EXPECTED_PROFILES["spec"].review_agent


def _write_override_configs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _ = _isolated_global(tmp_path, monkeypatch)
    global_path = tmp_path / "xdg" / "milknado" / "milknado.toml"
    _ = global_path.write_text(
        "[milknado]\n"
        + "agent_family = 'claude'\n"
        + "execution_agent = 'claude --model haiku -p'\n"
        + "quality_gates = ['echo global']\n"
        + "worktree = false\n"
        + "worker_agent_type = 'global-agent'\n"
        + "loop_mode = 'single'\n"
        + "max_iterations = 2\n"
        + "max_turns = 3\n"
        + "\n[milknado.prompts]\nworker_brief_prepend = 'global brief'\n",
        encoding="utf-8",
    )
    local = tmp_path / "project" / "milknado.toml"
    local.parent.mkdir()
    _ = local.write_text(
        "[milknado]\nagent_family = 'claude'\n\n"
        + "[milknado.flavor.derived]\n"
        + "tools = ['Read']\nquality_gates = []\n"
        + "brief_prepend = 'derived brief'\nagent_type = 'derived-agent'\n"
        + "loop_mode = 'redispatch'\nmax_iterations = 7\nmax_turns = 11\n"
        + "worktree = true\nsession_mode = 'fresh'\nreview = true\n"
        + "review_agent = 'claude --model opus -p'\nreview_max_rounds = 4\non_reject = 'warn'\n\n"
        + "[milknado.flavor.explicit]\n"
        + "execution_agent = 'omp --model openai-codex/gpt-5.6-luna -p'\n"
        + "tools = ['Read']\n",
        encoding="utf-8",
    )
    return local


def test_flavor_overrides_beat_global_defaults_and_tools_derive_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = _write_override_configs(tmp_path, monkeypatch)
    details = load_config_details(local)
    derived = resolve_flavor_profile(details.config, "derived")
    assert derived == _profile(
        "claude --model sonnet -p --allowedTools 'Read'",
        (),
        "derived brief",
        worker_agent_type="derived-agent",
        max_iterations=7,
        max_turns=11,
        worktree=True,
        review=True,
        review_agent="claude --model opus -p",
        review_max_rounds=4,
        on_reject="warn",
    )

    explicit = resolve_flavor_profile(details.config, "explicit")
    assert explicit.execution_agent == "omp --model openai-codex/gpt-5.6-luna -p"
    assert "allowedTools" not in explicit.execution_agent
    assert explicit.quality_gates == (Gate(command="echo global"),)
    assert explicit.brief_prepend == "global brief"
    assert explicit.worker_agent_type == "global-agent"
    assert explicit.loop_mode == "single"
    assert explicit.max_iterations == 2
    assert explicit.max_turns == 3
    assert explicit.worktree is False


def test_flavor_inherit_false_replaces_global_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = _isolated_global(tmp_path, monkeypatch)
    global_path = tmp_path / "xdg" / "milknado" / "milknado.toml"
    _ = global_path.write_text(
        "[milknado]\nagent_family = 'claude'\n"
        + "execution_agent = 'claude --model haiku -p'\nquality_gates = ['echo root']\n\n"
        + "[milknado.flavor.shared]\nexecution_agent = 'claude --model opus -p'\n"
        + "quality_gates = ['echo flavor']\nmax_turns = 7\n",
        encoding="utf-8",
    )
    local = tmp_path / "project" / "milknado.toml"
    local.parent.mkdir()
    _ = local.write_text(
        "[milknado]\nagent_family = 'claude'\n\n"
        + "[milknado.flavor.shared]\ninherit = false\ntools = ['Read']\n",
        encoding="utf-8",
    )

    details = load_config_details(local)
    override = details.config.flavors["shared"]
    assert override.execution_agent is None
    assert override.quality_gates is None
    assert override.max_turns is None
    assert override.tools == ("Read",)
    assert resolve_flavor_profile(details.config, "shared") == _profile(
        "claude --model sonnet -p --allowedTools 'Read'",
        ("echo root",),
        None,
        max_turns=60,
    )


def test_declared_custom_flavor_is_accepted_by_real_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = _isolated_global(tmp_path, monkeypatch)
    _ = (tmp_path / "milknado.toml").write_text(
        "[milknado]\nagent_family = 'codex'\n\n[milknado.flavor.custom]\nquality_gates = []\n",
        encoding="utf-8",
    )

    details = load_config_details(tmp_path / "milknado.toml")
    assert details.config.flavor_registry == BUILTIN_FLAVORS | {"custom"}

    from milknado.mcp.todo_mutate import milknado_todo_add

    node = milknado_todo_add(
        description="custom flavor task",
        flavor="custom",
        project_root=str(tmp_path),
    )
    assert node.get("flavor") == "custom"
