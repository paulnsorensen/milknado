from __future__ import annotations

from milknado.loop.adapters import CLIAdapter, Invocation, select_adapter
from milknado.loop.adapters.omp import OmpAdapter


def test_matches_omp_binary_stem() -> None:
    adapter = OmpAdapter()
    assert adapter.matches(["omp"]) is True
    assert adapter.matches(["/usr/local/bin/omp", "-p"]) is True
    assert adapter.matches(["codex"]) is False
    assert adapter.matches([]) is False


def test_build_command_preserves_user_flags() -> None:
    adapter = OmpAdapter()
    command = ["omp", "-p", "--model", "openai-codex/gpt-5.6-luna"]
    assert adapter.build_command(command) == command


def test_deliver_prompt_appends_one_argument_without_stdin() -> None:
    adapter = OmpAdapter()
    command = ["omp", "-p", "--auto-approve"]
    prompt = 'fix "quoted" input\nthen exit'
    assert adapter.deliver_prompt(command, prompt) == Invocation([*command, prompt], None)


def test_capabilities_match_blocking_text_output() -> None:
    adapter = OmpAdapter()
    assert (
        adapter.name,
        adapter.counts_what,
        adapter.supports_streaming,
        adapter.renders_structured_peek,
        adapter.supports_soft_wind_down,
        adapter.requires_full_stdout_for_completion,
    ) == ("omp", "none", False, False, False, True)


def test_satisfies_protocol_and_is_registered() -> None:
    assert isinstance(OmpAdapter(), CLIAdapter)
    assert isinstance(select_adapter(["omp", "-p"]), OmpAdapter)
