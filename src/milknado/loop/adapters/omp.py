"""Oh My Pi CLI adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from typing_extensions import override

from milknado.loop.adapters._generic import GenericAdapter
from milknado.loop.adapters._protocol import ADAPTERS, AdapterEvent, CountsWhat, Invocation


class OmpAdapter(GenericAdapter):
    """Launch Oh My Pi in its newline-delimited JSON event mode."""

    name: str = "omp"
    counts_what: CountsWhat = "tool_use"
    supports_streaming: bool = True

    def matches(self, cmd: list[str]) -> bool:
        return bool(cmd) and Path(cmd[0]).stem == "omp"

    @override
    def build_command(self, cmd: list[str]) -> list[str]:
        """Replace any configured output mode with Oh My Pi's JSON event stream."""
        command: list[str] = []
        emitted_mode = False
        index = 0
        while index < len(cmd):
            argument = cmd[index]
            if argument == "--mode":
                index += 2
            elif argument.startswith("--mode="):
                index += 1
            else:
                command.append(argument)
                index += 1
                continue
            if not emitted_mode:
                command.extend(("--mode", "json"))
                emitted_mode = True
        if not emitted_mode:
            command.extend(("--mode", "json"))
        return command

    @override
    def deliver_prompt(self, cmd: list[str], prompt: str) -> Invocation:
        return Invocation([*cmd, prompt], None)

    @override
    def parse_event(self, line: str) -> AdapterEvent | None:
        try:
            raw = cast(object, json.loads(line))
        except json.JSONDecodeError:
            return None
        if not isinstance(raw, dict):
            return None
        raw = cast(dict[str, object], raw)
        if raw.get("type") != "tool_execution_start":
            return None
        tool_name = raw.get("toolName")
        if not isinstance(tool_name, str):
            return None
        return AdapterEvent(kind="tool_use", name=tool_name, raw=raw)


ADAPTERS.append(OmpAdapter())
