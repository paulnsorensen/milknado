from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from milknado.domains.execution import ExecutionConfig
    from milknado.domains.execution.run_loop import RunLoopResult
    from milknado.domains.planning.planner import PlanResult

import typer
from rich.console import Console

from milknado.domains.common import (
    MikadoNode,
    MilknadoConfig,
    NodeStatus,
    default_config,
    load_config,
    save_config,
)
from milknado.domains.common.agent_argv import (
    build_planning_subprocess,
)
from milknado.domains.common.toolchain import (
    get_required_tool_status,
    install_missing_rust_tools,
)
from milknado.domains.graph import MikadoGraph, render_tree

app = typer.Typer(name="milknado", help="Mikado execution engine")
console = Console()

CONFIG_FILE = "milknado.toml"


def _ensure_plugins_loaded(config: MilknadoConfig) -> None:
    from milknado.plugins import discover_entry_point_plugins, load_plugins

    for plugin in load_plugins(config.plugins):
        console.print(f"  Plugin loaded: {plugin.meta.name}")
    for plugin in discover_entry_point_plugins():
        console.print(f"  Plugin loaded: {plugin.meta.name} (entry point)")


def _maybe_block_parent(graph: MikadoGraph, parent: int | None) -> None:
    if parent is None:
        return
    parent_node = graph.get_node(parent)
    if parent_node and parent_node.status.value == "running":
        graph.mark_blocked(parent)
        console.print(f"Parent node {parent} marked as blocked.")


def _find_config(project_root: Path) -> Path:
    return project_root / CONFIG_FILE


def _load_or_default(project_root: Path) -> MilknadoConfig:
    config_path = _find_config(project_root)
    if config_path.exists():
        config = load_config(config_path)
    else:
        config = default_config(project_root)
    _ensure_plugins_loaded(config)
    return config


def _ensure_db(config: MilknadoConfig) -> MikadoGraph:
    config.db_path.parent.mkdir(parents=True, exist_ok=True)
    return MikadoGraph(config.db_path)


@app.command()
def init(
    project_root: Annotated[
        Path, typer.Argument(help="Project root directory")
    ] = Path("."),
    install_rust_tools: Annotated[
        bool,
        typer.Option(
            "--install-rust-tools",
            help="Install missing tilth/mergiraf via cargo.",
        ),
    ] = False,
) -> None:
    """Initialize milknado in a project directory."""
    from milknado.adapters.crg import CrgAdapter

    project_root = project_root.resolve()
    config_path = _find_config(project_root)

    if config_path.exists():
        console.print(f"Config already exists: {config_path}")
        config = load_config(config_path)
    else:
        config = default_config(project_root)
        save_config(config, config_path)
        console.print(f"Created config: {config_path}")

    _ensure_plugins_loaded(config)
    graph = _ensure_db(config)
    graph.close()
    console.print(f"Database ready: {config.db_path}")

    crg = CrgAdapter(project_root)
    try:
        crg.ensure_graph(project_root)
        console.print("Code-review-graph ready.")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]CRG build failed (exit {e.returncode}).[/red]")
        if e.stderr:
            console.print(e.stderr)
        raise typer.Exit(code=1) from None

    if install_rust_tools:
        _install_rust_tools_or_exit()


@app.command()
def index(
    project_root: Annotated[
        Path, typer.Argument(help="Project root directory")
    ] = Path("."),
) -> None:
    """Rebuild the code-review-graph index."""
    from milknado.adapters.crg import CrgAdapter

    project_root = project_root.resolve()
    _load_or_default(project_root)
    crg = CrgAdapter(project_root)
    try:
        crg.build_graph(project_root)
        console.print("Code-review-graph rebuilt.")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]CRG build failed (exit {e.returncode}).[/red]")
        if e.stderr:
            console.print(e.stderr)
        raise typer.Exit(code=1) from None


@app.command()
def status(
    project_root: Annotated[
        Path, typer.Argument(help="Project root directory")
    ] = Path("."),
) -> None:
    """Show the current state of the Mikado graph."""
    project_root = project_root.resolve()
    config = _load_or_default(project_root)
    graph = _ensure_db(config)

    try:
        nodes = graph.get_all_nodes()
        if not nodes:
            console.print("No nodes in graph. Run [bold]milknado plan[/bold] to start.")
            return

        run_states = _fetch_run_states(nodes)
        output = render_tree(graph, run_states=run_states)
        console.print(output)
    finally:
        graph.close()


def _fetch_run_states(nodes: list[MikadoNode]) -> dict[str, str] | None:
    run_ids = [n.run_id for n in nodes if n.run_id and n.status == NodeStatus.RUNNING]
    if not run_ids:
        return None
    try:
        from milknado.adapters.ralphify import RalphifyAdapter

        ralph = RalphifyAdapter()
        states: dict[str, str] = {}
        for run_id in run_ids:
            run = ralph.get_run(run_id)
            if run:
                states[run_id] = getattr(run, "status", "unknown")
        return states or None
    except Exception:  # noqa: BLE001 — status enrichment is best-effort; never block render
        return None


@app.command()
def crg(
    project_root: Annotated[
        Path, typer.Argument(help="Project root directory")
    ] = Path("."),
) -> None:
    """Show the code-review-graph architecture overview."""
    import json

    from milknado.adapters.crg import CrgAdapter

    project_root = project_root.resolve()
    if not (project_root / ".code-review-graph" / "graph.db").exists():
        console.print("[dim]No CRG index. Run [bold]milknado index[/bold] to build.[/dim]")
        raise typer.Exit(code=1)

    adapter = CrgAdapter(project_root)
    overview = adapter.get_architecture_overview()
    formatted = json.dumps(overview, indent=2, default=str)
    console.print(formatted)


@app.command("add-node")
def add_node(
    description: Annotated[str, typer.Argument(help="Node description")],
    parent: Annotated[
        int | None, typer.Option("--parent", "-p", help="Parent node ID")
    ] = None,
    files: Annotated[
        list[str] | None,
        typer.Option("--files", "-f", help="Files this node will touch"),
    ] = None,
    project_root: Annotated[
        Path, typer.Option("--project-root", help="Project root directory")
    ] = Path("."),
) -> None:
    """Add a node to the Mikado graph."""
    project_root = project_root.resolve()
    config = _load_or_default(project_root)
    graph = _ensure_db(config)

    try:
        node = graph.add_node(description, parent_id=parent)
        if files:
            graph.set_file_ownership(node.id, files)

        _maybe_block_parent(graph, parent)
        console.print(f"Added node {node.id}: {node.description}")
    finally:
        graph.close()


_ISSUE_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _flatten_csv(items: list[str] | None) -> list[str]:
    if not items:
        return []
    out: list[str] = []
    for raw in items:
        for piece in raw.split(","):
            stripped = piece.strip()
            if stripped:
                out.append(stripped)
    return out


def _validate_spec_paths(raw_paths: list[str]) -> list[Path]:
    validated: list[Path] = []
    for p in raw_paths:
        path = Path(p).expanduser().resolve()
        if not path.exists() or not path.is_file():
            console.print(f"[red]--spec file not found: {p}[/red]")
            raise typer.Exit(code=1)
        if path.suffix.lower() != ".md":
            console.print(f"[red]--spec must point to a .md file, got: {path}[/red]")
            raise typer.Exit(code=1)
        validated.append(path)
    return validated


def _fetch_issue(issue_ref: str) -> dict[str, object]:
    """Fetch a single GitHub issue via `gh`; exits on error."""
    try:
        result = subprocess.run(
            ["gh", "issue", "view", issue_ref, "--json", "title,body,number,url"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        console.print(
            "[red]`gh` CLI not found. Install GitHub CLI to use --issue.[/red]"
        )
        raise typer.Exit(code=1) from None

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        console.print(f"[red]gh issue view {issue_ref} failed:[/red] {stderr}")
        raise typer.Exit(code=1)

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        console.print(f"[red]gh returned invalid JSON for {issue_ref}: {exc}[/red]")
        raise typer.Exit(code=1) from None


def _slug_for(refs: list[str], issues: list[dict[str, object]]) -> str:
    numbers = [str(i.get("number")) for i in issues if i.get("number") is not None]
    source = "-".join(numbers) if numbers else "-".join(refs) or "issue"
    return _ISSUE_SLUG_RE.sub("-", source).strip("-") or "issue"


def _materialize_issue_spec(issue_refs: list[str], project_root: Path) -> Path:
    """Fetch one or more GitHub issues and write them as a single spec .md file."""
    if not issue_refs:
        raise ValueError("issue_refs must not be empty")
    issues = [_fetch_issue(ref) for ref in issue_refs]

    issues_dir = project_root / ".milknado" / "issues"
    issues_dir.mkdir(parents=True, exist_ok=True)
    spec_path = issues_dir / f"issue-{_slug_for(issue_refs, issues)}.md"
    spec_path.write_text(_render_issue_spec(issues), encoding="utf-8")
    return spec_path


def _render_issue_spec(issues: list[dict[str, object]]) -> str:
    if len(issues) == 1:
        return _render_single_issue(issues[0])
    return _render_multi_issue(issues)


def _render_single_issue(issue: dict[str, object]) -> str:
    title = _issue_title(issue)
    url = issue.get("url") or ""
    body = str(issue.get("body") or "").rstrip()
    header = f"# {title}\n"
    if url:
        header += f"\n> Source: {url}\n"
    return f"{header}\n{body}\n"


def _render_multi_issue(issues: list[dict[str, object]]) -> str:
    refs = [f"#{i.get('number')}" for i in issues if i.get("number") is not None]
    combined_title = "Plan for issues " + ", ".join(refs) if refs else "Plan"
    sections = [f"# {combined_title}\n"]
    for issue in issues:
        sections.append(_render_issue_section(issue))
    return "\n".join(sections) + "\n"


def _render_issue_section(issue: dict[str, object]) -> str:
    title = _issue_title(issue)
    number = issue.get("number")
    url = issue.get("url") or ""
    body = str(issue.get("body") or "").rstrip()
    heading = f"## #{number}: {title}" if number is not None else f"## {title}"
    lines = [heading]
    if url:
        lines.append(f"\n> Source: {url}")
    lines.append(f"\n{body}")
    return "\n".join(lines) + "\n"


def _issue_title(issue: dict[str, object]) -> str:
    title = str(issue.get("title") or "").strip()
    if title:
        return title
    number = issue.get("number")
    return f"Issue {number}" if number is not None else "Issue"


def _materialize_combined_spec(
    spec_paths: list[Path],
    issue_refs: list[str],
    project_root: Path,
) -> Path:
    """Merge multiple specs and/or issues into one materialized spec .md."""
    issues = [_fetch_issue(ref) for ref in issue_refs]
    sections = [f"# {_combined_title(spec_paths, issues)}\n"]
    for sp in spec_paths:
        sections.append(_render_spec_section(sp))
    for issue in issues:
        sections.append(_render_issue_section(issue))

    issues_dir = project_root / ".milknado" / "issues"
    issues_dir.mkdir(parents=True, exist_ok=True)
    spec_path = issues_dir / f"plan-{_combined_slug(spec_paths, issues)}.md"
    spec_path.write_text("\n".join(sections) + "\n", encoding="utf-8")
    return spec_path


def _combined_title(spec_paths: list[Path], issues: list[dict[str, object]]) -> str:
    parts: list[str] = []
    if spec_paths:
        parts.append("specs " + ", ".join(sp.stem for sp in spec_paths))
    if issues:
        refs = [f"#{i.get('number')}" for i in issues if i.get("number") is not None]
        if refs:
            parts.append("issues " + ", ".join(refs))
    return "Plan for " + " + ".join(parts) if parts else "Plan"


def _combined_slug(spec_paths: list[Path], issues: list[dict[str, object]]) -> str:
    tokens: list[str] = [sp.stem for sp in spec_paths]
    tokens += [str(i.get("number")) for i in issues if i.get("number") is not None]
    source = "-".join(tokens) or "plan"
    return _ISSUE_SLUG_RE.sub("-", source).strip("-") or "plan"


def _render_spec_section(spec_path: Path) -> str:
    body = spec_path.read_text(encoding="utf-8").rstrip()
    return f"## Spec: {spec_path.stem}\n\n> Source: {spec_path}\n\n{body}\n"


def _resolve_plan_spec(
    spec_paths: list[Path],
    issue_refs: list[str],
    project_root: Path,
) -> Path:
    """Pick the right spec to feed the planner, materializing as needed."""
    if len(spec_paths) == 1 and not issue_refs:
        return spec_paths[0]
    if not spec_paths and issue_refs:
        return _materialize_issue_spec(issue_refs, project_root)
    return _materialize_combined_spec(spec_paths, issue_refs, project_root)


def _derive_goal(spec_path: Path) -> str:
    try:
        text = spec_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise typer.BadParameter(
            f"--spec file is not valid UTF-8 text: {spec_path} ({exc})"
        ) from None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            heading = stripped[2:].strip()
            if heading:
                return heading
    return spec_path.stem


def _plan_summary(result: PlanResult) -> str:
    return (
        f"Planned {result.change_count} changes → {result.batch_count} batches"
        f" ({result.oversized_count} oversized), solver={result.solver_status};"
        f" {result.nodes_created} Mikado nodes created"
    )


def _plan_exit_code(result: PlanResult) -> int:
    if result.solver_status == "INFEASIBLE":
        return 1
    if result.solver_status == "NO_MANIFEST":
        return 1
    if result.solver_status in ("OPTIMAL", "FEASIBLE"):
        return 0
    if result.solver_status == "UNKNOWN" and result.batch_count >= 1:
        return 0
    if not result.success:
        return result.exit_code
    return 0


@app.command()
def plan(
    spec: Annotated[
        list[str] | None,
        typer.Option(
            "--spec",
            help=(
                "Spec .md file(s). Repeat the flag or pass comma-separated paths "
                "to combine multiple specs."
            ),
        ),
    ] = None,
    issue: Annotated[
        list[str] | None,
        typer.Option(
            "--issue",
            help=(
                "GitHub issue ref (number, owner/repo#123, or URL) fetched via gh. "
                "Repeat the flag or pass comma-separated refs to combine issues."
            ),
        ),
    ] = None,
    project_root: Annotated[
        Path, typer.Option("--project-root", help="Project root directory")
    ] = Path("."),
) -> None:
    """Launch interactive planning session to decompose one or more specs/issues."""
    from milknado.adapters.crg import CrgAdapter
    from milknado.domains.planning import Planner

    spec_paths = _validate_spec_paths(_flatten_csv(spec))
    issue_refs = _flatten_csv(issue)
    if not spec_paths and not issue_refs:
        console.print("[red]Provide --spec or --issue.[/red]")
        raise typer.Exit(code=1)

    project_root = project_root.resolve()
    effective_spec = _resolve_plan_spec(spec_paths, issue_refs, project_root)

    goal = _derive_goal(effective_spec)
    config = _load_or_default(project_root)
    graph = _ensure_db(config)

    try:
        crg = CrgAdapter(project_root)
        planner = Planner(graph, crg, config.planning_agent)
        console.print(f"[bold]Planning:[/bold] {goal}")
        result = planner.launch(goal, project_root, spec_path=effective_spec)
        console.print(_plan_summary(result))
        exit_code = _plan_exit_code(result)
        if result.solver_status == "UNKNOWN" and result.batch_count >= 1:
            Console(stderr=True).print(
                "[yellow]Warning: solver returned UNKNOWN — results may be suboptimal[/yellow]"
            )
        if exit_code != 0:
            raise typer.Exit(code=exit_code)
    finally:
        graph.close()


def _build_exec_config(
    config: MilknadoConfig, project_root: Path,
) -> ExecutionConfig:
    from milknado.domains.execution import ExecutionConfig

    return ExecutionConfig(
        execution_agent=config.execution_agent,
        quality_gates=config.quality_gates,
        worktree_pattern=config.worktree_pattern,
        project_root=project_root,
    )


def _print_run_result(result: RunLoopResult) -> None:
    if result.root_done:
        console.print("[green]All nodes complete. Root goal achieved.[/green]")
    else:
        console.print(
            f"[yellow]Loop ended: {result.completed_total} completed, "
            f"{result.failed_total} failed.[/yellow]"
        )

    for conflict in result.rebase_conflicts:
        console.print(
            f"\n[red bold]Rebase conflict — node {conflict.node_id}:[/red bold] "
            f"{conflict.description}",
        )
        if conflict.conflicting_files:
            for f in conflict.conflicting_files:
                console.print(f"  [red]•[/red] {f}")
        if conflict.detail:
            console.print(f"  [dim]{conflict.detail}[/dim]")


@app.command()
def run(
    project_root: Annotated[
        Path, typer.Option("--project-root", help="Project root directory")
    ] = Path("."),
) -> None:
    """Execute ready leaf nodes as parallel ralph loops."""
    from milknado.adapters import CrgAdapter, GitAdapter, RalphifyAdapter
    from milknado.domains.execution import Executor, RunLoop, get_dispatchable_nodes

    project_root = project_root.resolve()
    config = _load_or_default(project_root)
    graph = _ensure_db(config)

    try:
        if not get_dispatchable_nodes(graph):
            console.print("No nodes ready for execution.")
            return

        git = GitAdapter(project_root)
        ralph = RalphifyAdapter()
        crg = CrgAdapter(project_root)
        executor = Executor(graph=graph, git=git, ralph=ralph, crg=crg)

        feature_branch = git.current_branch()
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)
        console.print(f"Starting execution loop on [bold]{feature_branch}[/bold]...")
        result = loop.run(
            config=_build_exec_config(config, project_root),
            feature_branch=feature_branch,
            concurrency_limit=config.concurrency_limit,
        )
        _print_run_result(result)
    finally:
        graph.close()


@app.command()
def doctor(
    project_root: Annotated[
        Path, typer.Argument(help="Project root directory")
    ] = Path("."),
) -> None:
    """Run health checks on the milknado installation."""
    from milknado.domains.common.doctor import render_report, run_doctor

    project_root = project_root.resolve()
    config_path = _find_config(project_root)
    config = _load_or_default(project_root)
    report = run_doctor(config_path, config)
    text, issue_count = render_report(report)
    typer.echo(text)
    if issue_count > 0:
        raise typer.Exit(code=1)


agents_app = typer.Typer(name="agents", help="Agent CLI compatibility")
app.add_typer(agents_app)


@agents_app.command("check")
def agents_check(
    project_root: Annotated[
        Path, typer.Option("--project-root", help="Project root directory")
    ] = Path("."),
) -> None:
    """Print resolved planning/execution commands and sample planning argv."""

    project_root = project_root.resolve()
    config = _load_or_default(project_root)
    console.print(f"[bold]agent_family[/bold]: {config.agent_family}")
    console.print(f"[bold]planning[/bold]: {config.planning_agent}")
    console.print(f"[bold]execution (ralphify)[/bold]: {config.execution_agent}")

    sample = project_root / ".milknado" / ".agent-check-sample.md"
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_text("# sample planning context\n", encoding="utf-8")
    try:
        argv, extra = build_planning_subprocess(
            sample, config.planning_agent,
        )
        redacted = {k: ("<stdin>" if k == "input" else v) for k, v in extra.items()}
        console.print(f"[bold]planning argv[/bold]: {argv}")
        if redacted:
            console.print(f"[bold]planning extras[/bold]: {redacted}")
    finally:
        sample.unlink(missing_ok=True)


plugin_app = typer.Typer(name="plugin", help="Plugin management commands")
app.add_typer(plugin_app)

tools_app = typer.Typer(name="tools", help="Toolchain commands")
app.add_typer(tools_app)


def _print_tool_status() -> list[tuple[str, bool]]:
    statuses = get_required_tool_status()
    rows: list[tuple[str, bool]] = []
    for status in statuses:
        state = "ok" if status.installed else "missing"
        details = f" ({status.path})" if status.path else ""
        console.print(f"{status.name}: {state}{details}")
        rows.append((status.name, status.installed))
    return rows


def _install_rust_tools_or_exit() -> None:
    installed, failed = install_missing_rust_tools()
    for name in installed:
        console.print(f"[green]Installed {name}[/green]")
    if failed:
        console.print("[red]Failed to install:[/red] " + ", ".join(failed))
        console.print("Install Rust/cargo, then run: [bold]milknado tools install[/bold]")
        raise typer.Exit(code=1)


@tools_app.command("check")
def tools_check() -> None:
    """Check whether required Rust tools are available."""
    rows = _print_tool_status()
    if not all(installed for _, installed in rows):
        raise typer.Exit(code=1)


@tools_app.command("install")
def tools_install() -> None:
    """Install missing Rust tools used by milknado workflows."""
    _install_rust_tools_or_exit()
    _print_tool_status()


@plugin_app.command("init")
def plugin_init(
    name: Annotated[str, typer.Argument(help="Plugin name")],
    target_dir: Annotated[
        Path, typer.Option("--target-dir", "-d", help="Directory to create plugin in")
    ] = Path("."),
) -> None:
    """Scaffold a new milknado plugin."""
    from milknado.plugins import scaffold_plugin

    try:
        result = scaffold_plugin(name, target_dir.resolve())
        console.print(f"Created plugin [bold]{name}[/bold] at {result.plugin_dir}")
        for f in result.files_created:
            console.print(f"  {f}")
    except FileExistsError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from None
