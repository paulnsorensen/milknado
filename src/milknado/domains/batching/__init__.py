from __future__ import annotations

from typing import TYPE_CHECKING

# The private loader keeps ortools and tiktoken out of CLI and MCP startup.
from milknado.domains.batching._lazy import __getattr__ as __getattr__
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
