from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from milknado.domains.common.errors import GitOperationError
from milknado.domains.dispatch import GroupWorktreeRequest, setup_group_worktree
from milknado.domains.graph import GroupWorkspace, MikadoGraph, TaskOutcome


def _tasks(graph: MikadoGraph) -> tuple[int, int]:
    prerequisite = graph.add_node("prerequisite", files=("shared.py",))
    dependent = graph.add_node("dependent", files=("shared.py",))
    _ = graph.add_edge(dependent.id, prerequisite.id)
    return prerequisite.id, dependent.id


def _workspace(suffix: str) -> GroupWorkspace:
    return GroupWorkspace(f"/tmp/group-{suffix}", f"group-{suffix}", f"session-{suffix}")


def test_group_persists_membership_and_distinct_results(graph: MikadoGraph) -> None:
    first, second = _tasks(graph)
    groups = graph.groups
    group = groups.create("graph-a", (first, second), _workspace("a"))

    assert groups.get(group.id) == group
    assert groups.tasks(group.id) == (first, second)
    first_attempt = groups.start_task(group.id, first, "run-a")
    groups.finish_task(first_attempt, TaskOutcome("done", "first result"))
    second_attempt = groups.start_task(group.id, second, "run-b")
    groups.finish_task(second_attempt, TaskOutcome("failed", "second result"))

    assert groups.task_result(first) == ("done", "first result")
    assert groups.task_result(second) == ("failed", "second result")


@pytest.mark.parametrize(
    ("order", "message"),
    [
        ((1, 0), "dependency"),
        ((0, 0), "distinct"),
    ],
)
def test_invalid_membership_fails(
    graph: MikadoGraph, order: tuple[int, int], message: str
) -> None:
    nodes = _tasks(graph)
    groups = graph.groups
    with pytest.raises(ValueError, match=message):
        _ = groups.create("graph-a", tuple(nodes[index] for index in order), _workspace("a"))


def test_file_ownership_conflict_between_groups_fails(graph: MikadoGraph) -> None:
    first, second = _tasks(graph)
    groups = graph.groups
    source = groups.create("graph-a", (first,), _workspace("a"))
    attempt = groups.start_task(source.id, first, "run-a")
    groups.finish_task(attempt, TaskOutcome("done", "first result"))
    with pytest.raises(ValueError, match="file ownership"):
        _ = groups.create("graph-a", (second,), _workspace("b"))


def test_second_writer_is_rejected_without_losing_first_result(graph: MikadoGraph) -> None:
    first, second = _tasks(graph)
    groups = graph.groups
    group = groups.create("graph-a", (first, second), _workspace("a"))
    attempt = groups.start_task(group.id, first, "run-a")
    with pytest.raises(ValueError, match="active writer"):
        _ = groups.start_task(group.id, second, "run-b")
    groups.finish_task(attempt, TaskOutcome("done", "first result"))
    assert groups.task_result(first) == ("done", "first result")
    assert groups.task_result(second) is None


def test_task_cannot_skip_predecessor_or_rewrite_result(graph: MikadoGraph) -> None:
    first, second = _tasks(graph)
    groups = graph.groups
    group = groups.create("graph-a", (first, second), _workspace("a"))
    with pytest.raises(ValueError, match="predecessor"):
        _ = groups.start_task(group.id, second, "run-b")
    attempt = groups.start_task(group.id, first, "run-a")
    groups.finish_task(attempt, TaskOutcome("failed", "first failed"))
    with pytest.raises(ValueError, match="predecessor"):
        _ = groups.start_task(group.id, second, "run-b")
    with pytest.raises(ValueError, match="completed"):
        _ = groups.start_task(group.id, first, "run-again")


def test_wrong_run_cannot_clear_writer(graph: MikadoGraph) -> None:
    first, _ = _tasks(graph)
    groups = graph.groups
    group = groups.create("graph-a", (first,), _workspace("a"))
    attempt = groups.start_task(group.id, first, "run-a")
    with pytest.raises(ValueError, match="fence"):
        groups.finish_task(replace(attempt, run_id="run-other"), TaskOutcome("done", "wrong"))
    groups.finish_task(attempt, TaskOutcome("done", "right"))
    assert groups.task_result(first) == ("done", "right")


def test_fork_has_distinct_identities_and_preserves_source(graph: MikadoGraph) -> None:
    first, _ = _tasks(graph)
    groups = graph.groups
    source = groups.create("graph-a", (first,), _workspace("a"))
    attempt = groups.start_task(source.id, first, "run-a")
    groups.finish_task(attempt, TaskOutcome("done", "source result"))

    fork = groups.fork(source.id, _workspace("b"))
    assert len({source.graph_id, fork.graph_id}) == 2
    assert len({source.id, fork.id}) == 2
    assert len({source.worktree_path, fork.worktree_path}) == 2
    assert len({source.branch_name, fork.branch_name}) == 2
    assert len({source.provider_session_id, fork.provider_session_id}) == 2
    assert groups.tasks(fork.id) != groups.tasks(source.id)
    assert groups.get(source.id) == source
    assert groups.task_result(first) == ("done", "source result")


def test_group_worktree_uses_new_branch_and_preserves_source(
    graph: MikadoGraph, mock_git: MagicMock, tmp_path: Path
) -> None:
    first, _ = _tasks(graph)
    groups = graph.groups
    source = groups.create("graph-a", (first,), _workspace("a"))
    mock_git.current_branch.return_value = source.branch_name  # pyright: ignore[reportAny]
    mock_git.resolve_ref.return_value = "base-oid"  # pyright: ignore[reportAny]
    fork = setup_group_worktree(
        groups,
        mock_git,
        tmp_path,
        GroupWorktreeRequest("ignored", (), "session-b", "fork", source.id),
    )

    assert fork.source_group_id == source.id
    assert fork.branch_name != source.branch_name
    assert fork.worktree_path != source.worktree_path
    mock_git.create_worktree.assert_called_once_with(  # pyright: ignore[reportAny]
        Path(fork.worktree_path), fork.branch_name
    )
    assert groups.get(source.id) == source


def test_fork_rejects_source_branch_mismatch(
    graph: MikadoGraph, mock_git: MagicMock, tmp_path: Path
) -> None:
    first, _ = _tasks(graph)
    groups = graph.groups
    source = groups.create("graph-a", (first,), _workspace("a"))
    mock_git.current_branch.return_value = "other"  # pyright: ignore[reportAny]
    with pytest.raises(ValueError, match="source group branch"):
        _ = setup_group_worktree(
            groups,
            mock_git,
            tmp_path,
            GroupWorktreeRequest("ignored", (), "session-b", "fork", source.id),
        )
    mock_git.create_worktree.assert_not_called()  # pyright: ignore[reportAny]


def test_group_rejects_unknown_task_and_empty_identity(graph: MikadoGraph) -> None:
    first, _ = _tasks(graph)
    groups = graph.groups
    with pytest.raises(ValueError, match="unknown"):
        _ = groups.create("graph-a", (999999,), _workspace("a"))
    with pytest.raises(ValueError, match="nonempty"):
        _ = groups.create("", (first,), _workspace("a"))


def test_group_rejects_reused_workspace_identity(graph: MikadoGraph) -> None:
    first, _ = _tasks(graph)
    independent = graph.add_node("independent")
    groups = graph.groups
    _ = groups.create("graph-a", (first,), _workspace("a"))
    with pytest.raises(ValueError, match="identity"):
        _ = groups.create("graph-b", (independent.id,), _workspace("a"))


def test_fork_rejects_missing_source_and_reused_session(graph: MikadoGraph) -> None:
    first, _ = _tasks(graph)
    groups = graph.groups
    assert groups.get("missing") is None
    with pytest.raises(ValueError, match="source"):
        _ = groups.fork("missing", _workspace("b"))
    source = groups.create("graph-a", (first,), _workspace("a"))
    with pytest.raises(ValueError, match="distinct"):
        _ = groups.fork(source.id, GroupWorkspace("/tmp/group-b", "group-b", "session-a"))


def test_writer_rejects_missing_group_member_run_and_status(graph: MikadoGraph) -> None:
    first, _ = _tasks(graph)
    outside = graph.add_node("outside")
    groups = graph.groups
    group = groups.create("graph-a", (first,), _workspace("a"))
    with pytest.raises(ValueError, match="nonempty"):
        _ = groups.start_task(group.id, first, "")
    with pytest.raises(ValueError, match="does not exist"):
        _ = groups.start_task("missing", first, "run-a")
    with pytest.raises(ValueError, match="member"):
        _ = groups.start_task(group.id, outside.id, "run-a")
    attempt = groups.start_task(group.id, first, "run-a")
    with pytest.raises(ValueError, match="status"):
        groups.finish_task(attempt, TaskOutcome("unknown", "wrong"))
    groups.finish_task(attempt, TaskOutcome("done", "right"))


def test_new_group_worktree_persists_its_workspace(
    graph: MikadoGraph, mock_git: MagicMock, tmp_path: Path
) -> None:
    first, _ = _tasks(graph)
    mock_git.current_branch.return_value = "main"  # pyright: ignore[reportAny]
    mock_git.resolve_ref.return_value = "base-oid"  # pyright: ignore[reportAny]
    groups = graph.groups
    group = setup_group_worktree(
        groups, mock_git, tmp_path, GroupWorktreeRequest("graph-a", (first,), "session-a", "work")
    )
    assert groups.get(group.id) == group
    assert groups.tasks(group.id) == (first,)


def test_failed_group_checkout_removes_new_branch(
    graph: MikadoGraph, mock_git: MagicMock, tmp_path: Path
) -> None:
    first, _ = _tasks(graph)
    mock_git.current_branch.return_value = "main"  # pyright: ignore[reportAny]
    mock_git.resolve_ref.side_effect = ["base-oid", "changed"]  # pyright: ignore[reportAny]
    with pytest.raises(GitOperationError, match="checkout changed"):
        _ = setup_group_worktree(
            graph.groups,
            mock_git,
            tmp_path,
            GroupWorktreeRequest("graph-a", (first,), "session-a", "work"),
        )
    mock_git.force_remove_worktree.assert_called_once()  # pyright: ignore[reportAny]
    mock_git.delete_branch.assert_called_once()  # pyright: ignore[reportAny]


def test_group_checkout_rejects_missing_source(
    graph: MikadoGraph, mock_git: MagicMock, tmp_path: Path
) -> None:
    _ = _tasks(graph)
    with pytest.raises(ValueError, match="source execution group"):
        _ = setup_group_worktree(
            graph.groups,
            mock_git,
            tmp_path,
            GroupWorktreeRequest("ignored", (), "session-a", "fork", "missing"),
        )
    mock_git.create_worktree.assert_not_called()  # pyright: ignore[reportAny]
