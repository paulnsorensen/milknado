"""Fallback adapter for CLIs with no dedicated implementation.

Returned by :func:`milknado.loop.adapters.select_adapter` when no specific
adapter's ``matches`` returns True. The core loop treats these sessions as
blocking and untyped.
"""

from __future__ import annotations

from milknado.loop.adapters._protocol import (
    AdapterEvent,
    CountsWhat,
    Invocation,
    stdin_invocation,
    stdout_only_completion_signal,
)


class GenericAdapter:
    """No-op adapter: pass commands through unchanged, parse nothing."""

    name: str = "generic"
    counts_what: CountsWhat = "none"
    supports_streaming: bool = False
    # Untyped agents have no streaming result event; the engine must keep
    # the full stdout buffer if it wants promise detection.
    requires_full_stdout_for_completion: bool = True

    def build_command(self, cmd: list[str]) -> list[str]:
        return list(cmd)

    def deliver_prompt(self, cmd: list[str], prompt: str) -> Invocation:
        """Unknown CLIs are assumed to read the prompt from stdin."""
        return stdin_invocation(cmd, prompt)

    def parse_event(self, line: str) -> AdapterEvent | None:  # pyright: ignore[reportUnusedParameter]
        return None

    def extract_completion_signal(
        self,
        *,
        result_text: str | None,
        stdout: str | None,
        user_signal: str,
    ) -> bool:
        """Scan the full stdout for the promise tag.

        Unknown CLIs have no event schema to parse, so the whole-stdout
        regex scan is the only reliable path.  Matches the current
        engine-side behavior so switching to adapter-owned detection does
        not regress promise completion for untyped agents.

        *result_text* is unused (the blocking path does not populate it
        for unknown CLIs); the engine opts into
        ``requires_full_stdout_for_completion`` to make sure *stdout* is
        supplied when promise detection is requested.
        """
        del result_text
        return stdout_only_completion_signal(stdout=stdout, user_signal=user_signal)
