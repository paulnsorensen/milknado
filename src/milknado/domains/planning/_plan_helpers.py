from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from milknado.domains.common.protocols import CrgPort
    from milknado.domains.graph import MikadoGraph

_logger = logging.getLogger(__name__)


def hash_spec(spec_text: str) -> str:
    return hashlib.sha256(spec_text.encode()).hexdigest()


def read_spec(spec_path: Path | None) -> str | None:
    if spec_path is None:
        return None
    if not spec_path.exists():
        raise FileNotFoundError(f"spec_path does not exist: {spec_path}")
    if not spec_path.is_file():
        raise ValueError(f"spec_path is not a file: {spec_path}")
    return spec_path.read_text(encoding="utf-8")


def warn_active_worktrees(project_root: Path) -> None:
    worktrees_dir = project_root / ".worktrees"
    found: list[Path] = []
    if worktrees_dir.is_dir():
        found.extend(worktrees_dir.glob("milknado-*"))
        claude_dir = worktrees_dir / "claude"
        if claude_dir.is_dir():
            found.extend(claude_dir.glob("milknado-*"))
    for wt in found:
        print(f"WARNING: active worktree not deleted: {wt}", file=sys.stderr)


def count_active_worktrees(project_root: Path) -> int:
    worktrees_dir = project_root / ".worktrees"
    if not worktrees_dir.is_dir():
        return 0
    found = list(worktrees_dir.glob("milknado-*"))
    claude_dir = worktrees_dir / "claude"
    if claude_dir.is_dir():
        found.extend(claude_dir.glob("milknado-*"))
    return len(found)


def resolve_plan_mode(existing_node_count: int, *, reset: bool) -> str:
    if existing_node_count == 0:
        return "fresh"
    if reset:
        return "reset"
    return "resume"


def detect_spec_hash_change(
    graph: MikadoGraph,
    spec_text: str | None,
    *,
    resuming: bool,
) -> bool | None:
    if not resuming or spec_text is None:
        return None
    stored = graph.get_spec_hash()
    if stored is None:
        return None
    return stored != hash_spec(spec_text)


def safe_ensure_crg(
    crg: CrgPort,
    project_root: Path,
) -> tuple[CrgPort, bool]:
    try:
        crg.ensure_graph(project_root)
        return crg, True
    except Exception as exc:
        _logger.warning("CRG unavailable, running without graph context: %s", exc)
        return crg, False


def guard_existing_plan(
    graph: MikadoGraph,
    *,
    resuming: bool,
    reset: bool,
    project_root: Path,
    spec_text: str | None,
) -> None:
    from milknado.domains.common.errors import ExistingPlanDetected
    from milknado.domains.common.types import NodeStatus

    nodes = graph.get_all_nodes()
    if not nodes:
        return

    total = len(nodes)
    done = sum(1 for n in nodes if n.status == NodeStatus.DONE)
    pending = sum(1 for n in nodes if n.status == NodeStatus.PENDING)
    running = sum(1 for n in nodes if n.status == NodeStatus.RUNNING)

    if not resuming and not reset:
        raise ExistingPlanDetected(total, done, pending, running)

    if reset:
        graph.drop_all()
        _logger.info("Dropped %d nodes before re-plan", total)
        warn_active_worktrees(project_root)
        return

    if spec_text is not None:
        new_hash = hash_spec(spec_text)
        stored_hash = graph.get_spec_hash()
        if stored_hash is not None and stored_hash != new_hash:
            _logger.warning(
                "Spec hash mismatch on --resume: stored=%s..., new=%s... — "
                "spec may have changed since last plan.",
                stored_hash[:8],
                new_hash[:8],
            )


def build_coverage_delta(orphan_paths: list[str], original_context: str) -> str:
    path_list = "\n".join(f"- {p}" for p in orphan_paths)
    return (
        f"{original_context}\n\n"
        "## Coverage gaps — please extend the manifest\n\n"
        "The following impl changes lack a corresponding tests/ change:\n\n"
        f"{path_list}\n\n"
        "For each impl file above, add a change with:\n"
        "- `path` starting with `tests/`\n"
        "- `depends_on` referencing the impl change id, OR the same US-NNN ref in its description"
        "\n\n"
        "Output the complete updated manifest in the same JSON format."
    )
