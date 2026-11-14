"""Lazy attribute loader for the batching crust.

The crust re-exports this `__getattr__` so that importing it keeps ortools and tiktoken
out of CLI and MCP startup.
"""

from __future__ import annotations

_CRUST = "milknado.domains.batching"


def __getattr__(name: str) -> object:
    """Load the CP-SAT solver and tokenizer only when a caller asks for them."""
    if name == "plan_batches":
        from milknado.domains.batching.solver import plan_batches

        return plan_batches
    if name == "estimate_tokens":
        from milknado.domains.batching.weights import estimate_tokens

        return estimate_tokens
    raise AttributeError(f"module {_CRUST!r} has no attribute {name!r}")
