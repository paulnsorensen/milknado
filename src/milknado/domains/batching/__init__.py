from __future__ import annotations

from typing import TYPE_CHECKING

from milknado.domains.batching.change import (
    DUMB_ZONE_BUDGET,
    MEGA_BATCH_THRESHOLD,
    Batch,
    BatchPlan,
    ChangeDependency,
    EditKind,
    FileChange,
    HashAnchors,
    NewRelationship,
    RelationshipReason,
    SymbolRef,
    SymbolSpread,
)

if TYPE_CHECKING:
    from milknado.domains.batching.solver import plan_batches
    from milknado.domains.batching.weights import estimate_tokens

__all__ = [
    "DUMB_ZONE_BUDGET",
    "MEGA_BATCH_THRESHOLD",
    "Batch",
    "BatchPlan",
    "ChangeDependency",
    "EditKind",
    "FileChange",
    "HashAnchors",
    "NewRelationship",
    "RelationshipReason",
    "SymbolRef",
    "SymbolSpread",
    "estimate_tokens",
    "plan_batches",
]


def __getattr__(name: str) -> object:
    """Load the CP-SAT solver and tokenizer lazily so importing this crust keeps
    ortools and tiktoken out of CLI and MCP startup."""
    if name == "plan_batches":
        from milknado.domains.batching.solver import plan_batches

        return plan_batches
    if name == "estimate_tokens":
        from milknado.domains.batching.weights import estimate_tokens

        return estimate_tokens
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
