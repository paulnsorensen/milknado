# pyright: basic

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from milknado.cli import app
from milknado.cli.web import _Controller, _owner_capabilities, _ProjectGitInspection
from milknado.domains.common import SessionContext
from milknado.domains.graph import MikadoGraph
from milknado.web.login import LaunchToken

runner = CliRunner()


def test_web_command_accepts_port_and_no_open(tmp_path: Path) -> None:
    token = LaunchToken("fixed")
    with (
        patch(
            "milknado.cli.web.load_or_default",
            return_value=(SimpleNamespace(db_path=tmp_path / "db"), []),
        ),
        patch("milknado.cli.web.ensure_db") as ensure_db,
        patch("milknado.cli.web.LaunchToken", return_value=token),
        patch("milknado.cli.web.PolledSnapshotSource") as polling,
        patch("milknado.cli.web.run_server") as server,
    ):
        ensure_db.return_value.close = lambda: None
        polling.return_value.start.return_value = None
        result = runner.invoke(
            app, ["web", "--project-root", str(tmp_path), "--port", "8123", "--no-open"]
        )
    assert result.exit_code == 0, result.output
    polling.return_value.start.assert_called_once()
    server.assert_called_once()
    options = server.call_args.args[2]
    assert options.port == 8123
    assert options.no_open is True


def test_run_web_options_delegate_to_owner_host(tmp_path: Path) -> None:
    with (
        patch("milknado.cli.run._load_or_default", return_value=(object(), [])),
        patch("milknado.cli.web.run_owner_web") as host,
        patch("milknado.cli.run._print_run_result") as print_result,
    ):
        host.return_value = SimpleNamespace(strict_exit=False)
        result = runner.invoke(
            app,
            ["run", "--web", "--no-open", "--port", "8124", "--project-root", str(tmp_path)],
        )
    assert result.exit_code == 0, result.output
    context = host.call_args.args[0]
    assert context.project_root == tmp_path
    options = host.call_args.args[1]
    assert options.strict is False
    assert options.allow_protected is False
    assert options.port == 8124
    assert options.no_open is True
    print_result.assert_called_once_with(host.return_value)


def test_project_git_inspection_delegates_session_context(tmp_path: Path) -> None:
    context = SessionContext(family="ralph", cwd=str(tmp_path), base_oid="base")
    with patch("milknado.adapters.GitAdapter") as adapter_type:
        adapter = adapter_type.return_value
        adapter.session_changes.return_value = ("changed",)
        adapter.session_diff.return_value = "diff"
        inspection = _ProjectGitInspection(tmp_path)
        assert inspection.changes(context) == ("changed",)
        assert inspection.diff(context, "file") == "diff"
        adapter.session_changes.assert_called_once_with(context)
        adapter.session_diff.assert_called_once_with(context, "file")


def test_owner_capabilities_reads_active_run() -> None:
    controller = SimpleNamespace(
        snapshot=lambda: SimpleNamespace(active_runs=(SimpleNamespace(run_id="run"),))
    )
    capability = object()
    graph = SimpleNamespace(commands=SimpleNamespace(capabilities=lambda run_id: capability))
    assert (
        _owner_capabilities(cast(_Controller, controller), cast(MikadoGraph, graph)) is capability
    )


def test_owner_capabilities_resolves_requested_run_with_concurrent_runs() -> None:
    controller = SimpleNamespace(
        snapshot=lambda: SimpleNamespace(
            active_runs=(SimpleNamespace(run_id="run-1"), SimpleNamespace(run_id="run-2"))
        )
    )
    capabilities = {"run-2": object()}
    graph = SimpleNamespace(
        commands=SimpleNamespace(capabilities=lambda run_id: capabilities.get(run_id))
    )
    actual = _owner_capabilities(cast(_Controller, controller), cast(MikadoGraph, graph), "run-2")
    assert actual is capabilities["run-2"]


def test_apply_runnable_root_exclusions_excludes_invalid_subgoal(tmp_path: Path) -> None:
    from rich.console import Console

    from milknado.cli._helpers import apply_runnable_root_exclusions
    from milknado.domains.common import NodeKind, NodeSpec

    graph = MikadoGraph(tmp_path / "graph.db")
    root = graph.add_node("root goal", spec=NodeSpec(kind=NodeKind.GOAL))
    stub = graph.add_node("sub-goal", parent_id=root.id, spec=NodeSpec(kind=NodeKind.GOAL))

    result = apply_runnable_root_exclusions(graph, Console())

    assert result.has_errors is True
    assert stub.id in result.excluded
    assert stub.id in graph.dispatch_exclusions()
    graph.close()


def test_run_owner_web_applies_runnable_root_exclusions(tmp_path: Path) -> None:
    from milknado.cli.web import OwnerWebContext, run_owner_web
    from milknado.domains.common import default_config

    _ = runner.invoke(app, ["init", str(tmp_path)])
    config = default_config(tmp_path)

    class _StopAfterExclusions(Exception):
        pass

    graph = SimpleNamespace(reconcile_completed_goals=lambda: 0, close=lambda: None)
    with (
        patch("milknado.cli.web.ensure_db", return_value=graph),
        patch("milknado.cli._helpers.apply_runnable_root_exclusions") as apply_exclusions,
        patch(
            "milknado.app.run.build_execution_controller",
            side_effect=_StopAfterExclusions,
        ) as build_controller,
    ):
        with pytest.raises(_StopAfterExclusions):
            _ = run_owner_web(OwnerWebContext(tmp_path, config, []))

    apply_exclusions.assert_called_once()
    assert apply_exclusions.call_args.args[0] is graph
    build_controller.assert_called_once()
