"""Oh My Pi CLI adapter."""

from __future__ import annotations

from pathlib import Path

from milknado.loop.adapters._generic import GenericAdapter
from milknado.loop.adapters._protocol import ADAPTERS, Invocation


class OmpAdapter(GenericAdapter):
    """Deliver print-mode prompts as positional arguments."""

    name = "omp"

    def matches(self, cmd: list[str]) -> bool:
        return bool(cmd) and Path(cmd[0]).stem == "omp"

    def deliver_prompt(self, cmd: list[str], prompt: str) -> Invocation:
        return Invocation([*cmd, prompt], None)


ADAPTERS.append(OmpAdapter())
