from __future__ import annotations

import pytest

from milknado.loop.adapters import CLIAdapter, Invocation, select_adapter
from milknado.loop.adapters.omp import OmpAdapter


def test_matches_omp_binary_stem() -> None:
    adapter = OmpAdapter()
    assert adapter.matches(["omp"]) is True
    assert adapter.matches(["/usr/local/bin/omp", "-p"]) is True
    assert adapter.matches(["codex"]) is False
    assert adapter.matches([]) is False


def test_build_command_enables_json_events_and_preserves_user_flags() -> None:
    adapter = OmpAdapter()
    command = ["omp", "-p", "--model", "openai-codex/gpt-5.6-luna"]
    assert adapter.build_command(command) == [*command, "--mode", "json"]


def test_build_command_replaces_existing_output_mode_idempotently() -> None:
    adapter = OmpAdapter()
    command = ["omp", "-p", "--mode=text", "--mode", "rpc"]
    expected = ["omp", "-p", "--mode", "json"]
    assert adapter.build_command(command) == expected
    assert adapter.build_command(expected) == expected


def test_deliver_prompt_appends_one_argument_without_stdin() -> None:
    adapter = OmpAdapter()
    command = ["omp", "-p", "--auto-approve"]
    prompt = 'fix "quoted" input\nthen exit'
    assert adapter.deliver_prompt(command, prompt) == Invocation([*command, prompt], None)


def test_capabilities_match_json_event_output() -> None:
    adapter = OmpAdapter()
    assert (
        adapter.name,
        adapter.counts_what,
        adapter.supports_streaming,
        adapter.renders_structured_peek,
        adapter.supports_soft_wind_down,
        adapter.requires_full_stdout_for_completion,
    ) == ("omp", "tool_use", True, False, False, True)


def test_parses_omp_tool_events_for_turn_counting() -> None:
    raw = {"type": "tool_execution_start", "tool_name": "Bash", "args": {"command": "pwd"}}
    event = OmpAdapter().parse_event(
        '{"type":"tool_execution_start","tool_name":"Bash","args":{"command":"pwd"}}'
    )
    assert event is not None
    assert event.kind == "tool_use"
    assert event.name == "Bash"
    assert event.raw == raw


@pytest.mark.parametrize(
    "line",
    [
        "not JSON",
        '["tool_execution_start"]',
        '{"type":"message_update"}',
        '{"type":"tool_execution_start","tool_name":42}',
    ],
)
def test_ignores_non_tool_events_and_malformed_tool_events(line: str) -> None:
    assert OmpAdapter().parse_event(line) is None


def test_satisfies_protocol_and_is_registered() -> None:
    assert isinstance(OmpAdapter(), CLIAdapter)
    assert isinstance(select_adapter(["omp", "-p"]), OmpAdapter)
