"""Seed the roadmap, running goal, pending review, and failed runs used by the dogfood capture."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from milknado.domains.common import MikadoNode, NodeKind, NodeSpec, RunResult
from milknado.domains.graph import GoalReviewRequest, MikadoGraph

DEFAULT_ROOT = Path("/tmp/milknado-web-round2")


def archive_nodes(graph: MikadoGraph, count: int, prefix: str) -> None:
    for index in range(count):
        node = graph.add_node(f"{prefix} {index}")
        _ = graph.mark_running(node.id)
        _ = graph.mark_done(node.id)
        _ = graph.archive_subtree(node.id)


def seed_graph(root: Path) -> None:
    db_path = root / ".milknado" / "milknado.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    graph = MikadoGraph(db_path)
    try:
        archive_nodes(graph, 42, "archived filler")
        _roadmap = graph.add_node("Parked roadmap 43", spec=NodeSpec(kind=NodeKind.ROADMAP))
        archive_nodes(graph, 43, "archived filler after roadmap")
        goal = graph.add_node("Active goal 87", spec=NodeSpec(kind=NodeKind.GOAL))
        _ = graph.request_goal_review(
            GoalReviewRequest(
                goal.id,
                "rev-active-goal",
                "Review evidence from the seeded dashboard run.",
                "Keep the active goal title and run controls visible.",
                reviewer="dogfood",
            )
        )
        active = graph.add_node("Live worker task", parent_id=goal.id)
        failed = [
            graph.add_node(f"Failed worker {index}", parent_id=goal.id) for index in range(4)
        ]
        seed_runs(graph, root, active.id, failed)
    finally:
        graph.close()


def seed_runs(graph: MikadoGraph, root: Path, active_id: int, failed: list[MikadoNode]) -> None:
    _ = graph.runs.start(
        "run-live-87", active_id, str(root / "live.log"), "2026-09-28T01:00:00+00:00", 60
    )
    for index, node in enumerate(failed):
        run_id = f"run-failed-{index}"
        _ = graph.runs.start(
            run_id, node.id, str(root / f"{run_id}.log"), "2026-09-28T01:00:00+00:00", 60
        )
        _ = graph.runs.finish(
            run_id,
            RunResult(
                status="failed",
                exit_code=1,
                timed_out=False,
                ended_at="2026-09-28T01:00:01+00:00",
                error="worker session gone",
                detail=None,
                rebased=None,
            ),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = cast(Path, args.root)
    seed_graph(root)
    print(root)


if __name__ == "__main__":
    main()
