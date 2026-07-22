"""Project a wiki-origin roadmap onto a GitHub Project (one-time bind, ADR-6).

Binding uses a durable, immutable correlation marker in each Issue body. The
marker lets retries recover an Issue/Project item after any remote call or
local-ref write may have succeeded without ever inferring identity from a
mutable title.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from milknado.domains.common import NodeKind
from milknado.domains.github._fields import (
    HARVEST_FIELD_NAME,
    STATUS_FIELD_NAME,
    STATUS_OPTIONS,
    find_field,
)
from milknado.domains.github._intent import goal_file_map, goal_intent
from milknado.domains.github.ports import GithubProjectPort
from milknado.domains.graph import MikadoGraph
from milknado.domains.wiki import load_frontmatter, locate_roadmap_dir, read_text


@dataclass(frozen=True)
class GithubBindResult:
    roadmap_node_id: int
    issues_created: int
    items_added: int
    field_created: bool  # whether the Status field was created this bind


def bind_github_project(
    graph: MikadoGraph,
    roadmap_node_id: int,
    wiki_root: Path,
    github: GithubProjectPort,
    *,
    owner: str | None = None,
    number: int | None = None,
) -> GithubBindResult:
    """Create Issues + items for each goal and store the GitHub refs."""
    roadmap = graph.get_node(roadmap_node_id)
    if roadmap is None or roadmap.kind != NodeKind.ROADMAP:
        raise ValueError(f"node {roadmap_node_id} is not a roadmap node")
    if roadmap.wiki_ref is None:
        raise ValueError(
            f"roadmap node {roadmap_node_id} has no wiki_ref; import it from the wiki first"
        )
    github.preflight()
    roadmap_dir, _slug = locate_roadmap_dir(wiki_root, roadmap.wiki_ref)
    frontmatter = load_frontmatter(read_text(wiki_root, roadmap_dir / "index.md"))
    owner, number = _resolve_project(owner, number, frontmatter)
    issue_owner, issue_repo = _resolve_repo(frontmatter)
    file_map = goal_file_map(wiki_root, roadmap.wiki_ref)

    project = github.project_view(owner, number)
    _bind_roadmap_ref(graph, roadmap_node_id, roadmap.github_ref, project["id"])
    issues_created = 0
    items_added = 0
    items = github.item_list(owner, number)
    for goal in graph.get_children(roadmap_node_id):
        if goal.kind != NodeKind.GOAL or goal.github_ref is not None:
            continue
        intent = goal_intent(goal, file_map, wiki_root)
        marker = _correlation_marker(roadmap.wiki_ref, goal.wiki_ref or str(goal.id))
        matches = _correlated_items(items, marker)
        attempt = graph.get_github_bind_attempt(goal.id)

        if attempt is not None:
            if len(matches) > 1:
                raise RuntimeError(
                    f"cannot safely recover goal {goal.id}: correlation marker "
                    f"matched {len(matches)} project items"
                )
            if len(matches) == 1:
                graph.set_github_ref(goal.id, matches[0]["id"])
                graph.clear_github_bind_attempt(goal.id)
                continue
            issue_url = attempt.get("issue_url")
            if not issue_url:
                raise RuntimeError(
                    f"cannot safely recover goal {goal.id}: issue creation outcome "
                    "is unknown and no project item carries its correlation marker"
                )
            item_id = github.item_add(owner, number, issue_url)
            graph.set_github_ref(goal.id, item_id)
            graph.clear_github_bind_attempt(goal.id)
            items_added += 1
            items.append({"id": item_id, "body": f"{marker}\n", "url": issue_url})
            continue
        if matches:
            if len(matches) != 1:
                raise RuntimeError(
                    f"cannot safely recover goal {goal.id}: correlation marker "
                    f"matched {len(matches)} project items"
                )
            graph.set_github_ref(goal.id, matches[0]["id"])
            continue

        graph.set_github_bind_attempt(goal.id, marker, None, datetime.now(UTC).isoformat())
        body = f"{intent.rstrip()}\n\n{marker}\n"
        url = github.issue_create(issue_owner, issue_repo, goal.description, body)
        graph.set_github_bind_attempt(goal.id, marker, url, datetime.now(UTC).isoformat())
        item_id = github.item_add(owner, number, url)
        graph.set_github_ref(goal.id, item_id)
        graph.clear_github_bind_attempt(goal.id)
        issues_created += 1
        items_added += 1
        items.append({"id": item_id, "body": body, "url": url})

    field_created = _ensure_fields(github, owner, number)
    return GithubBindResult(
        roadmap_node_id=roadmap_node_id,
        issues_created=issues_created,
        items_added=items_added,
        field_created=field_created,
    )


def _bind_roadmap_ref(
    graph: MikadoGraph, roadmap_node_id: int, current_ref: str | None, project_id: str
) -> None:
    """Set the roadmap's github_ref once; guard against silently rebinding it."""
    if current_ref is None:
        graph.set_github_ref(roadmap_node_id, project_id)
        return
    if current_ref != project_id:
        raise ValueError(
            f"roadmap node {roadmap_node_id} is already bound to github_ref "
            f"{current_ref!r}, cannot rebind to {project_id!r}"
        )


def _correlation_marker(roadmap_ref: str, goal_ref: str) -> str:
    return f"<!-- milknado-bind:roadmap={roadmap_ref};goal={goal_ref} -->"


def _correlated_items(items: list[dict], marker: str) -> list[dict]:
    return [item for item in items if marker in str(item.get("body", ""))]


def _resolve_project(owner: str | None, number: int | None, frontmatter: dict) -> tuple[str, int]:
    """Owner/number from explicit args, else `github_project: owner/number` frontmatter."""
    if owner is not None and number is not None:
        return owner, number
    raw = frontmatter.get("github_project")
    if not isinstance(raw, str) or "/" not in raw:
        raise ValueError(
            "bind requires owner/number args or a `github_project: owner/number` "
            "field in the roadmap index.md frontmatter"
        )
    fm_owner, _, fm_number = raw.partition("/")
    if not fm_owner.strip() or not fm_number.strip().isdigit():
        raise ValueError(f"malformed `github_project` frontmatter: {raw!r}")
    return fm_owner.strip(), int(fm_number.strip())


def _resolve_repo(frontmatter: dict) -> tuple[str, str]:
    """Issue repo from the `github_repo: owner/repo` frontmatter (Issues need a repo)."""
    raw = frontmatter.get("github_repo")
    if not isinstance(raw, str) or "/" not in raw:
        raise ValueError(
            "bind requires a `github_repo: owner/repo` field in the roadmap "
            "index.md frontmatter (Issues are created against a repo)"
        )
    repo_owner, _, repo_name = raw.partition("/")
    if not repo_owner.strip() or not repo_name.strip():
        raise ValueError(f"malformed `github_repo` frontmatter: {raw!r}")
    return repo_owner.strip(), repo_name.strip()


def _ensure_fields(github: GithubProjectPort, owner: str, number: int) -> bool:
    """Create the Status + harvest fields once; return whether Status was created."""
    fields = github.field_list(owner, number)
    status_created = False
    if find_field(fields, STATUS_FIELD_NAME) is None:
        github.field_create(number, owner, STATUS_FIELD_NAME, STATUS_OPTIONS)
        status_created = True
    if find_field(fields, HARVEST_FIELD_NAME) is None:
        github.field_create_text(number, owner, HARVEST_FIELD_NAME)
    return status_created
