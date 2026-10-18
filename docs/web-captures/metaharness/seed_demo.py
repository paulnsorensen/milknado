"""Seed a demo milknado project that exercises every metaharness surface.

Usage: uv run python seed_demo.py <project_dir>
"""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from milknado.domains.common import NodeKind, NodeSpec, RunResult, SessionContext, SessionEvent
from milknado.domains.graph import GoalReviewRequest, MikadoGraph

NOW = datetime.now(UTC)


def ts(minutes_ago: float) -> str:
    return (NOW - timedelta(minutes=minutes_ago)).isoformat()


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


BASE_QUEUE = """export function queuePrompt(session, note) {
  const entry = toEntry(note);
  pending.push(entry);
  return entry.id;
}
"""
NEW_QUEUE = """export async function queuePrompt(session, note) {
  const entry = toEntry(note);
  await store.append(session.id, entry);
  emit("queued", { session: session.id, id: entry.id });
  return entry.id;
}

export async function drain(session) {
  const due = await store.take(session.id, POLL_MS);
  return due.map(toPrompt);
}
"""
BASE_STORE = "const memory = new Map();\nexport const append = (id, e) => memory.set(id, e);\n"
NEW_STORE = (
    'const db = openStore(".tern/sessions.db");\n'
    "export const append = (id, e) => db.push(`queue:${id}`, e);\n"
)
TEST = """test("queued prompt survives a restart", async () => {
  await queuePrompt(s, "tighten the table");
  await restart();
  expect(await drain(s)).toHaveLength(1);
});
"""

WIKI = {
    "conventions/adoption-contract.md": "# Adoption contract\n\nWhat every worker must respect: gates, file length, no migration code.\n",
    "architecture/sliced-bread.md": "# Sliced Bread architecture\n\nVertical slices under domains/, adapters at the edge.\n",
    "architecture/review-protocol.md": "# Lavish review protocol\n\nArtifacts open beside the graph; queued prompts reach the worker on its next poll.\n",
    "history/queue-server-side.md": "# Decision: queue lives server-side\n\nADR adversarial-review-loops-002. The prompt queue persists in the session store.\n",
}


def init_repo(root: Path) -> tuple[str, Path]:
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "demo@example.com")
    git(root, "config", "user.name", "Demo")
    write(root / "src/review/queue-prompt.js", BASE_QUEUE)
    write(root / "src/session-store.js", BASE_STORE)
    write(root / ".gitignore", ".milknado/\n")
    write(
        root / "milknado.toml",
        '[milknado]\nquality_gates = ["just check-llm"]\nconcurrency_limit = 4\n',
    )
    for rel, text in WIKI.items():
        write(root / ".hallouminate/wiki" / rel, text)
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "base")
    base = git(root, "rev-parse", "HEAD")
    worktree = root / ".milknado/worktrees/milknado-8-annotation"
    git(root, "worktree", "add", "-q", "-b", "tern/8-annotation-queued-prompt", str(worktree))
    write(worktree / "src/review/queue-prompt.js", NEW_QUEUE)
    write(worktree / "src/session-store.js", NEW_STORE)
    write(worktree / "test/queue-prompt.test.js", TEST)
    git(worktree, "add", "-N", "test/queue-prompt.test.js")
    return base, worktree


def node(
    graph: MikadoGraph,
    text: str,
    parent: int | None,
    kind: NodeKind = NodeKind.TASK,
    flavor: str | None = None,
) -> int:
    spec = NodeSpec(kind=kind, flavor=flavor) if flavor else NodeSpec(kind=kind)
    return graph.add_node(text, parent, spec).id


def run(
    graph: MikadoGraph,
    run_id: str,
    node_id: int,
    started: float,
    context: SessionContext | None = None,
) -> None:
    graph.runs.start(run_id, node_id, f"/tmp/{run_id}.log", ts(started), 3600)
    if context is not None:
        graph.sessions.start(run_id, context)


def say(graph: MikadoGraph, run_id: str, *events: tuple[str, str]) -> None:
    for i, (kind, text) in enumerate(events):
        state = "requested" if kind == "permission" else ""
        graph.sessions.append(
            run_id, SessionEvent(kind=kind, text=text, event_id=f"{run_id}-e{i}", state=state)
        )


def finish(
    graph: MikadoGraph, run_id: str, status: str, ended: float, error: str | None = None
) -> None:
    graph.runs.finish(
        run_id,
        RunResult(
            status=status,
            exit_code=0 if status == "done" else 1,
            timed_out=False,
            ended_at=ts(ended),
            error=error,
        ),
    )


def done(graph: MikadoGraph, node_id: int, run_id: str, started: float) -> None:
    graph.mark_running(node_id)
    run(graph, run_id, node_id, started)
    finish(graph, run_id, "done", started - 9)
    graph.mark_done(node_id)


def seed(root: Path) -> None:
    base, worktree = init_repo(root)
    graph = MikadoGraph(root / ".milknado/milknado.db")
    goal = NodeKind.GOAL
    tern = node(graph, "Tern browser UI", None, goal)
    graph_view = node(graph, "Graph view", tern, goal)
    review = node(graph, "Review integration", tern, goal)
    stats = node(graph, "Stats dashboard", tern, goal)
    detail = node(graph, "Node detail panel", graph_view, flavor="implement")
    drag = node(graph, "Drag-to-depend", graph_view, flavor="implement")
    sandbox = node(graph, "Artifact viewer sandbox", review, flavor="implement")
    annotation = node(
        graph,
        "Annotation → queued prompt\n\nQueued prompts persist in the session store and reach the worker on its next poll.",
        review,
        flavor="implement",
    )
    inbox = node(graph, "Layout issues inbox", review, flavor="implement")
    fingerprint = node(graph, "Issue fingerprinting", inbox, flavor="implement")
    _ = node(graph, "Inbox drawer UI", inbox, flavor="prototype")
    strip = node(graph, "Session stats strip", stats, flavor="implement")
    core = node(graph, "Milknado core", None, goal)
    pool = node(graph, "Worktree pool", core, flavor="implement")
    adapters = node(graph, "Harness adapters", core, goal)
    omp = node(graph, "omp adapter", adapters, flavor="implement")
    claude = node(graph, "Claude Code adapter", adapters, flavor="implement")
    lavish = node(graph, "Lavish AXI", None, goal)
    renderer = node(graph, "Sandboxed renderer", lavish, flavor="implement")
    _ = node(graph, "Artifact protocol v2", lavish, flavor="spec")

    for node_id, run_id, started in (
        (detail, "run-5a1c", 300),
        (drag, "run-6b2d", 260),
        (sandbox, "run-7c3e", 180),
        (pool, "run-21aa", 400),
        (renderer, "run-32bb", 500),
    ):
        done(graph, node_id, run_id, started)
    graph.mark_running(graph_view)
    graph.mark_done(graph_view)

    for goal_id in (tern, review, stats, core, adapters):
        graph.mark_running(goal_id)

    ctx = SessionContext(family="claude", cwd=str(worktree), base_oid=base)
    graph.mark_running(annotation)
    run(graph, "run-8f4a9c21", annotation, 42, ctx)
    import sqlite3

    with sqlite3.connect(root / ".milknado/milknado.db") as conn:
        conn.execute("UPDATE nodes SET run_id = ? WHERE id = ?", ("run-8f4a9c21", annotation))
    graph.set_worktree(
        annotation, "run-8f4a9c21", str(worktree), "tern/8-annotation-queued-prompt"
    )
    say(
        graph,
        "run-8f4a9c21",
        ("assistant", "Read the brief and the Lavish review protocol page."),
        ("tool", 'tilth_search "queuePrompt" → 4 hits'),
        (
            "assistant",
            "Queued prompts now persist in the session store and reach the worker on its next poll. Tests written first.",
        ),
        ("tool", "just check-llm · PASS · 97% coverage"),
        ("permission", "git push origin tern/8-annotation-queued-prompt"),
    )
    graph.runs.insert_review(
        annotation,
        "approve",
        "Two low findings: reject root-prefixed artifact paths explicitly, and name the poll interval constant.",
        ts(6),
    )

    graph.mark_blocked(inbox)
    graph.mark_running(fingerprint)
    run(
        graph,
        "run-13d7e0b2",
        fingerprint,
        18,
        SessionContext(
            family="codex", cwd=str(root / ".milknado/worktrees/milknado-13"), base_oid=base
        ),
    )
    say(
        graph,
        "run-13d7e0b2",
        (
            "assistant",
            "Fingerprinting issues by selector and viewport so repeats collapse into one row.",
        ),
        ("tool", "edit src/issues/fingerprint.js (+64)"),
        ("assistant", "Writing tests for viewport changes. 3 of 5 passing."),
    )

    graph.mark_running(strip)
    run(graph, "run-10e2f7aa", strip, 95)
    finish(
        graph,
        "run-10e2f7aa",
        "failed",
        61,
        "just check-llm · coverage 88% < 95% after 8 of 8 iterations",
    )
    graph.mark_failed(strip)

    graph.mark_blocked(claude)
    graph.mark_running(omp)
    run(
        graph,
        "run-23c90d11",
        omp,
        25,
        SessionContext(
            family="omp", cwd=str(root / ".milknado/worktrees/milknado-23"), base_oid=base
        ),
    )
    say(
        graph,
        "run-23c90d11",
        ("assistant", "Mapping omp JSON stream events onto session events."),
        ("tool", "edit src/adapters/omp.py (+120)"),
    )

    graph.request_goal_review(
        GoalReviewRequest(
            review,
            "rev-3",
            "Fingerprinting (13) changes the issue schema the drawer reads.",
            "Start Inbox drawer UI (12) now on the last free worker, or hold it until Issue fingerprinting (13) lands so it builds on the final issue schema.",
            reviewer="coordinator",
        )
    )
    graph.close()
    print(f"seeded {root}")


if __name__ == "__main__":
    target = Path(sys.argv[1]).resolve()
    target.mkdir(parents=True, exist_ok=True)
    seed(target)
