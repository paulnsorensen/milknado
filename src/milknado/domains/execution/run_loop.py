from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from milknado.domains.common.types import NodeStatus
from milknado.domains.execution.executor import RebaseConflict, get_dispatchable_nodes

if TYPE_CHECKING:
    from rich.live import Live
    from rich.table import Table

    from milknado.domains.common.protocols import RalphPort
    from milknado.domains.execution.executor import ExecutionConfig, Executor
    from milknado.domains.graph import MikadoGraph


@dataclass(frozen=True)
class RunLoopResult:
    root_done: bool
    dispatched_total: int
    completed_total: int
    failed_total: int
    rebase_conflicts: tuple[RebaseConflict, ...] = ()


class RunLoop:
    def __init__(
        self,
        executor: Executor,
        graph: MikadoGraph,
        ralph: RalphPort,
    ) -> None:
        self._executor = executor
        self._graph = graph
        self._ralph = ralph
        self._active: dict[str, int] = {}

    def run(
        self,
        config: ExecutionConfig,
        feature_branch: str,
        concurrency_limit: int = 4,
    ) -> RunLoopResult:
        from rich.live import Live

        dispatched_total = 0
        completed_total = 0
        failed_total = 0
        conflicts: list[RebaseConflict] = []

        with Live(self._build_table(), refresh_per_second=2) as live:
            dispatched_total += self._dispatch_batch(config, concurrency_limit, live)
            live.update(self._build_table())

            while self._active:
                run_id, success = self._ralph.wait_for_next_completion(
                    set(self._active.keys()),
                )
                c, f, batch_conflicts = self._handle_completion(
                    run_id,
                    success,
                    feature_branch,
                    live,
                )
                completed_total += c
                failed_total += f
                conflicts.extend(batch_conflicts)
                dispatched_total += self._dispatch_batch(config, concurrency_limit, live)
                live.update(self._build_table())

        root = self._graph.get_root()
        root_done = root is not None and root.status == NodeStatus.DONE
        return RunLoopResult(
            root_done=root_done,
            dispatched_total=dispatched_total,
            completed_total=completed_total,
            failed_total=failed_total,
            rebase_conflicts=tuple(conflicts),
        )

    _SPINNER_FRAMES = ("◜", "◝", "◞", "◟")

    def _build_table(self) -> Table:
        from rich.table import Table

        self._tick = getattr(self, "_tick", 0) + 1
        frame = self._SPINNER_FRAMES[self._tick % len(self._SPINNER_FRAMES)]

        table = Table(
            title=f"milknado — {len(self._active)} active",
            show_header=True,
            header_style="bold",
        )
        table.add_column("", width=1, no_wrap=True)
        table.add_column("ID", style="cyan", width=4, no_wrap=True)
        table.add_column("Description")
        table.add_column("Branch", style="dim")

        for _run_id, node_id in self._active.items():
            node = self._graph.get_node(node_id)
            if node:
                table.add_row(
                    f"[cyan]{frame}[/cyan]",
                    str(node_id),
                    node.description,
                    node.branch_name or "",
                )

        return table

    def _dispatch_batch(
        self,
        config: ExecutionConfig,
        concurrency_limit: int,
        live: Live,
    ) -> int:
        available = concurrency_limit - len(self._active)
        if available <= 0:
            return 0
        dispatchable = get_dispatchable_nodes(self._graph)
        batch = dispatchable[:available]
        dispatched = 0
        for node_id in batch:
            node = self._graph.get_node(node_id)
            desc = node.description if node else str(node_id)
            try:
                result = self._executor.dispatch(node_id, config)
            except Exception as exc:
                live.console.print(
                    f"[red]✗[/red] [{node_id}] {desc} — dispatch failed: {exc}",
                )
                self._executor.fail(node_id)
                continue
            self._active[result.run_id] = node_id
            live.console.print(f"[cyan]→[/cyan] [{node_id}] {desc}")
            dispatched += 1
        return dispatched

    def _handle_completion(
        self,
        run_id: str,
        success: bool,
        feature_branch: str,
        live: Live,
    ) -> tuple[int, int, list[RebaseConflict]]:
        completed = 0
        failed = 0
        conflicts: list[RebaseConflict] = []

        node_id = self._active.pop(run_id)
        node = self._graph.get_node(node_id)
        desc = node.description if node else str(node_id)

        if success:
            result = self._executor.complete(node_id, feature_branch)
            if result.rebase_conflict:
                conflicts.append(result.rebase_conflict)
                files = ", ".join(result.rebase_conflict.conflicting_files)
                live.console.print(
                    f"[red]✗[/red] [{node_id}] {desc} — rebase conflict: {files}",
                )
                failed += 1
            else:
                live.console.print(f"[green]✓[/green] [{node_id}] {desc}")
                completed += 1
        else:
            self._executor.fail(node_id)
            live.console.print(f"[red]✗[/red] [{node_id}] {desc}")
            failed += 1

        return completed, failed, conflicts
