"""milknado graph export CLI veneer."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from milknado.cli import app
from milknado.domains.common import default_config
from milknado.domains.graph import MikadoGraph

runner = CliRunner()


def _init_graph(tmp_path: Path) -> tuple[int, int]:
    runner.invoke(app, ["init", str(tmp_path)])
    config = default_config(tmp_path)
    graph = MikadoGraph(config.db_path)
    goal = graph.add_node("goal")
    task = graph.add_node("task", parent_id=goal.id)
    graph.close()
    return goal.id, task.id


def test_export_dot_to_stdout_contains_seeded_nodes_and_edge(tmp_path: Path) -> None:
    goal_id, task_id = _init_graph(tmp_path)

    result = runner.invoke(
        app, ["graph", "export", "--format", "dot", "--project-root", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    assert result.output.startswith("digraph mikado {\n")
    assert f'"{goal_id}"' in result.output
    assert f'"{task_id}"' in result.output
    assert f'    "{goal_id}" -> "{task_id}";' in result.output


def test_export_writes_dot_file_with_out_option(tmp_path: Path) -> None:
    _init_graph(tmp_path)
    out_path = tmp_path / "graph.dot"

    result = runner.invoke(
        app,
        [
            "graph",
            "export",
            "--format",
            "dot",
            "--project-root",
            str(tmp_path),
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    content = out_path.read_text()
    assert content.startswith("digraph mikado {\n")
    assert content.endswith("}\n")


def test_export_empty_graph_is_valid_digraph(tmp_path: Path) -> None:
    runner.invoke(app, ["init", str(tmp_path)])

    result = runner.invoke(
        app, ["graph", "export", "--format", "dot", "--project-root", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    assert result.output == (
        'digraph mikado {\n    rankdir="TB";\n'
        '    node [style="filled", fontname="sans-serif"];\n}\n'
    )


def test_export_out_write_failure_reports_error_and_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _init_graph(tmp_path)
    out_path = tmp_path / "graph.dot"

    def _raise(self: Path, *args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", _raise)

    result = runner.invoke(
        app,
        [
            "graph",
            "export",
            "--format",
            "dot",
            "--project-root",
            str(tmp_path),
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 1
    assert "disk full" in result.output


def test_export_archived_node_only_included_with_include_archived_flag(tmp_path: Path) -> None:
    goal_id, task_id = _init_graph(tmp_path)

    config = default_config(tmp_path)
    graph = MikadoGraph(config.db_path)
    graph.mark_running(task_id)
    graph.mark_done(task_id)
    graph.mark_running(goal_id)
    graph.mark_done(goal_id)
    graph.close()

    archived = runner.invoke(
        app, ["graph", "archive", str(goal_id), "--project-root", str(tmp_path)]
    )
    assert archived.exit_code == 0, archived.output

    hidden = runner.invoke(
        app, ["graph", "export", "--format", "dot", "--project-root", str(tmp_path)]
    )
    assert f'"{goal_id}"' not in hidden.output

    shown = runner.invoke(
        app,
        [
            "graph",
            "export",
            "--format",
            "dot",
            "--project-root",
            str(tmp_path),
            "--include-archived",
        ],
    )
    assert shown.exit_code == 0, shown.output
    assert 'style="filled,dashed"' in shown.output
