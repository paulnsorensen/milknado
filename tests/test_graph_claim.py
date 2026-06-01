"""Atomic optimistic node-claim helpers: claim_node / try_reclaim / mark_terminal
/ set_pid / set_worktree.

These are the cross-process mutual-exclusion + dead-owner-reclaim + zombie-write-fence
primitives that replace the in-process `_dispatch_lock`. The SQLite conditional
UPDATE is the mutex; the unique `run_id` on the node row is the fence.
"""

from __future__ import annotations

import os

from milknado.domains.common import NodeStatus
from milknado.domains.dispatch._runstate import now_iso
from milknado.domains.graph import MikadoGraph
from milknado.domains.graph import graph as graph_module

_DEAD_PID = 2**31 - 1  # no process can hold this pid; os.kill(_, 0) -> ProcessLookupError


def _add_pending(graph: MikadoGraph) -> int:
    return graph.add_node("claimable").id


class TestClaimNode:
    def test_claims_a_pending_node(self, graph: MikadoGraph) -> None:
        node_id = _add_pending(graph)
        assert graph.claim_node(node_id, "run-A", now=now_iso()) is True
        node = graph.get_node(node_id)
        assert node is not None
        assert node.status == NodeStatus.RUNNING
        assert node.run_id == "run-A"
        assert node.dispatched_at is not None

    def test_claims_a_failed_node(self, graph: MikadoGraph) -> None:
        node_id = _add_pending(graph)
        graph.mark_failed(node_id)
        assert graph.claim_node(node_id, "run-A", now=now_iso()) is True
        assert graph.get_node(node_id).status == NodeStatus.RUNNING

    def test_claims_a_blocked_node(self, graph: MikadoGraph) -> None:
        # The locked seam declares status IN ('pending','failed','blocked') claimable.
        # Narrowing _CLAIMABLE to drop 'blocked' would silently break that contract:
        # a blocked node that becomes runnable could never be re-dispatched.
        node_id = _add_pending(graph)
        graph.mark_blocked(node_id)
        assert graph.get_node(node_id).status == NodeStatus.BLOCKED
        assert graph.claim_node(node_id, "run-A", now=now_iso()) is True
        assert graph.get_node(node_id).status == NodeStatus.RUNNING

    def test_refuses_a_running_node(self, graph: MikadoGraph) -> None:
        node_id = _add_pending(graph)
        assert graph.claim_node(node_id, "run-A", now=now_iso()) is True
        # A second claimant loses: the row is no longer pending/failed/blocked.
        assert graph.claim_node(node_id, "run-B", now=now_iso()) is False
        node = graph.get_node(node_id)
        assert node.run_id == "run-A", "the original owner's run_id is the fence; not overwritten"

    def test_refuses_a_done_node(self, graph: MikadoGraph) -> None:
        node_id = _add_pending(graph)
        graph.claim_node(node_id, "run-A", now=now_iso())
        graph.mark_terminal(node_id, "run-A", NodeStatus.DONE)
        assert graph.claim_node(node_id, "run-B", now=now_iso()) is False

    def test_reclaim_after_failed_retry_does_not_release_fresh_claim(
        self, graph: MikadoGraph
    ) -> None:
        """A failed run leaves a stale (dead) pid on the node — mark_terminal(FAILED)
        clears run_id but not pid. Re-claiming the node (the retry path) must reset
        pid to NULL, or a concurrent try_reclaim in the window before set_pid would
        read the dead stale pid and release the brand-new legitimate claim, letting
        a second worker in. This guards the exactly-one-worker guarantee on retry."""
        node_id = _add_pending(graph)
        graph.claim_node(node_id, "run-A", now=now_iso())
        graph.set_pid(node_id, "run-A", _DEAD_PID)  # run-A records its pid, then dies
        graph.mark_terminal(node_id, "run-A", NodeStatus.FAILED)  # leaves pid stale

        assert graph.claim_node(node_id, "run-B", now=now_iso()) is True
        assert graph.get_node(node_id).pid is None, "fresh claim resets the stale pid"

        # The window before run-B's set_pid: a concurrent dispatch tries to reclaim.
        # pid is NULL (pid-unknown), so the fresh claim must be left intact.
        graph.try_reclaim(node_id, now=now_iso())
        node = graph.get_node(node_id)
        assert node.status == NodeStatus.RUNNING, "fresh claim not released by stale-pid reclaim"
        assert node.run_id == "run-B"


class TestSetPid:
    def test_records_pid_on_the_fence_owner(self, graph: MikadoGraph) -> None:
        node_id = _add_pending(graph)
        graph.claim_node(node_id, "run-A", now=now_iso())
        graph.set_pid(node_id, "run-A", 4321)
        assert graph.get_node(node_id).pid == 4321

    def test_is_a_noop_for_a_stale_run_id(self, graph: MikadoGraph) -> None:
        node_id = _add_pending(graph)
        graph.claim_node(node_id, "run-A", now=now_iso())
        graph.set_pid(node_id, "run-stale", 4321)  # CAS on the fence: wrong owner
        assert graph.get_node(node_id).pid is None


class TestTryReclaim:
    def test_releases_a_dead_owner(self, graph: MikadoGraph) -> None:
        node_id = _add_pending(graph)
        graph.claim_node(node_id, "run-A", now=now_iso())
        graph.set_pid(node_id, "run-A", _DEAD_PID)
        graph.try_reclaim(node_id, now=now_iso())
        node = graph.get_node(node_id)
        assert node.status == NodeStatus.PENDING, "a dead pid frees the node immediately"
        assert node.run_id is None
        assert node.pid is None

    def test_leaves_a_live_owner(self, graph: MikadoGraph) -> None:
        node_id = _add_pending(graph)
        graph.claim_node(node_id, "run-A", now=now_iso())
        graph.set_pid(node_id, "run-A", os.getpid())  # this very process: alive
        graph.try_reclaim(node_id, now=now_iso())
        assert graph.get_node(node_id).status == NodeStatus.RUNNING

    def test_leaves_a_pid_unknown_owner(self, graph: MikadoGraph) -> None:
        node_id = _add_pending(graph)
        graph.claim_node(node_id, "run-A", now=now_iso())  # no set_pid -> pid is None
        graph.try_reclaim(node_id, now=now_iso())
        assert graph.get_node(node_id).status == NodeStatus.RUNNING

    def test_leaves_an_other_user_owner(self, graph: MikadoGraph, monkeypatch) -> None:
        """A runner owned by another user is alive: os.kill(pid, 0) raises
        PermissionError (process exists, signalling refused). The daemon spawns
        detached runners, so a cross-uid live owner is real — it must NOT be
        reclaimed, or two workers run the node at once (the race the claim prevents)."""
        node_id = _add_pending(graph)
        graph.claim_node(node_id, "run-A", now=now_iso())
        graph.set_pid(node_id, "run-A", 31337)

        def _refuse(_pid: int, _sig: int) -> None:
            raise PermissionError

        monkeypatch.setattr(graph_module.os, "kill", _refuse)
        graph.try_reclaim(node_id, now=now_iso())
        assert graph.get_node(node_id).status == NodeStatus.RUNNING, (
            "a live cross-uid owner is protected from reclaim"
        )

    def test_releases_on_unexpected_oserror(self, graph: MikadoGraph, monkeypatch) -> None:
        """An unexpected OSError from os.kill (errno other than ESRCH/EPERM) is
        treated as 'process gone', per the documented contract, so the node is
        released and re-dispatchable rather than wedged RUNNING forever."""
        node_id = _add_pending(graph)
        graph.claim_node(node_id, "run-A", now=now_iso())
        graph.set_pid(node_id, "run-A", 31337)

        def _other_oserror(_pid: int, _sig: int) -> None:
            raise OSError(22, "EINVAL")

        monkeypatch.setattr(graph_module.os, "kill", _other_oserror)
        graph.try_reclaim(node_id, now=now_iso())
        assert graph.get_node(node_id).status == NodeStatus.PENDING, (
            "an unexpected os.kill error frees the node (treated as dead)"
        )

    def test_ignores_a_non_running_node(self, graph: MikadoGraph) -> None:
        node_id = _add_pending(graph)
        graph.try_reclaim(node_id, now=now_iso())  # still pending
        assert graph.get_node(node_id).status == NodeStatus.PENDING

    def test_does_not_resurrect_a_completed_node(self, graph: MikadoGraph) -> None:
        """A DONE node keeps its run_id AND its (now-dead) pid — mark_terminal(DONE)
        clears neither. Without the status guard on release, try_reclaim would read
        that dead pid, judge the owner dead, and walk the completed node back to
        PENDING — re-running work that already finished. The guard refuses it."""
        node_id = _add_pending(graph)
        graph.claim_node(node_id, "run-A", now=now_iso())
        graph.set_pid(node_id, "run-A", _DEAD_PID)
        graph.mark_terminal(node_id, "run-A", NodeStatus.DONE)  # completes; keeps run_id + pid
        graph.try_reclaim(node_id, now=now_iso())
        assert graph.get_node(node_id).status == NodeStatus.DONE, (
            "a completed node must never be reclaimed back to PENDING"
        )


class TestRelease:
    def test_releases_a_running_owner(self, graph: MikadoGraph) -> None:
        node_id = _add_pending(graph)
        graph.claim_node(node_id, "run-A", now=now_iso())
        graph.set_worktree(node_id, "run-A", "/tmp/wt", "milknado/1-x")
        assert graph.release(node_id, "run-A") is True
        node = graph.get_node(node_id)
        assert node.status == NodeStatus.PENDING
        assert node.run_id is None
        assert node.worktree_path is None

    def test_is_a_noop_for_a_stale_run_id(self, graph: MikadoGraph) -> None:
        node_id = _add_pending(graph)
        graph.claim_node(node_id, "run-A", now=now_iso())
        assert graph.release(node_id, "run-stale") is False
        assert graph.get_node(node_id).status == NodeStatus.RUNNING, "a fresh owner is protected"

    def test_refuses_a_completed_node(self, graph: MikadoGraph) -> None:
        """DONE keeps its run_id, so a same-run_id release would match the fence —
        the status='running' guard is what stops it walking DONE back to PENDING."""
        node_id = _add_pending(graph)
        graph.claim_node(node_id, "run-A", now=now_iso())
        graph.mark_terminal(node_id, "run-A", NodeStatus.DONE)
        assert graph.release(node_id, "run-A") is False
        assert graph.get_node(node_id).status == NodeStatus.DONE


class TestMarkTerminal:
    def test_marks_done_for_the_owner(self, graph: MikadoGraph) -> None:
        node_id = _add_pending(graph)
        graph.claim_node(node_id, "run-A", now=now_iso())
        assert graph.mark_terminal(node_id, "run-A", NodeStatus.DONE) is True
        node = graph.get_node(node_id)
        assert node.status == NodeStatus.DONE
        assert node.completed_at is not None

    def test_marks_failed_and_clears_metadata(self, graph: MikadoGraph) -> None:
        node_id = _add_pending(graph)
        graph.claim_node(node_id, "run-A", now=now_iso())
        graph.set_worktree(node_id, "run-A", "/tmp/wt", "milknado/1-x")
        assert graph.mark_terminal(node_id, "run-A", NodeStatus.FAILED) is True
        node = graph.get_node(node_id)
        assert node.status == NodeStatus.FAILED
        assert node.worktree_path is None
        assert node.branch_name is None
        assert node.run_id is None

    def test_rejects_a_zombie_write_after_reclaim(self, graph: MikadoGraph) -> None:
        """The core fence: once the node is reclaimed under a new run_id, the old
        owner's terminal write hits zero rows and is rejected."""
        node_id = _add_pending(graph)
        graph.claim_node(node_id, "run-A", now=now_iso())
        graph.set_pid(node_id, "run-A", _DEAD_PID)
        graph.try_reclaim(node_id, now=now_iso())  # dead owner released
        graph.claim_node(node_id, "run-B", now=now_iso())  # fresh owner

        assert graph.mark_terminal(node_id, "run-A", NodeStatus.DONE) is False
        assert graph.get_node(node_id).status == NodeStatus.RUNNING, "fresh run unharmed"
        assert graph.mark_terminal(node_id, "run-B", NodeStatus.DONE) is True
        assert graph.get_node(node_id).status == NodeStatus.DONE


class TestSetWorktree:
    def test_attaches_worktree_to_the_owner(self, graph: MikadoGraph) -> None:
        node_id = _add_pending(graph)
        graph.claim_node(node_id, "run-A", now=now_iso())
        graph.set_worktree(node_id, "run-A", "/tmp/wt", "milknado/1-x")
        node = graph.get_node(node_id)
        assert node.worktree_path == "/tmp/wt"
        assert node.branch_name == "milknado/1-x"
        assert node.status == NodeStatus.RUNNING, "no status transition"

    def test_is_a_noop_for_a_stale_run_id(self, graph: MikadoGraph) -> None:
        node_id = _add_pending(graph)
        graph.claim_node(node_id, "run-A", now=now_iso())
        graph.set_worktree(node_id, "run-stale", "/tmp/wt", "milknado/1-x")
        assert graph.get_node(node_id).worktree_path is None
