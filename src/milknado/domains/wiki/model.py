"""Canonical frozen models for roadmap wiki documents."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal, cast

import msgspec
from msgspec import structs

from milknado.domains.common.types import NodeStatus
from milknado.domains.wiki._locate import read_text, roadmap_markdown_files
from milknado.domains.wiki._markdown import _frontmatter_and_body, _parse_body
from milknado.domains.wiki._serialize import parse_wikilink


class Lifecycle(StrEnum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"


class _WikiModel(msgspec.Struct, frozen=True, kw_only=True):
    pass


def _validate_slug_value(value: str) -> None:
    if not value or ".." in value or "/" in value or "\\" in value:
        raise ValueError(f"invalid slug {value!r}")


def _list_of_slugs(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("must be a list of goal slugs")
    return tuple(parse_wikilink(item.strip()) for item in value)


def _normalize_last_synced(value: object) -> object:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


class GoalDocument(_WikiModel, frozen=True, kw_only=True):
    slug: str
    roadmap: str
    kind: Literal["goal"]
    created: date
    title: str
    status: NodeStatus | None = None
    lifecycle: Lifecycle = Lifecycle.ACTIVE
    last_synced: str | None = None
    last_verified: date | None = None
    prereqs: tuple[str, ...] = ()
    up: tuple[str, ...] = ()
    down: tuple[str, ...] = ()
    intent: str | None = None
    acceptance: str | None = None
    outcome: str | None = None

    def __post_init__(self) -> None:
        _validate_slug_value(self.slug)
        _validate_slug_value(self.roadmap)
        if self.lifecycle is Lifecycle.ACTIVE:
            missing = [name for name in ("intent", "acceptance") if not getattr(self, name)]
            if missing:
                raise ValueError(f"active goal requires sections: {', '.join(missing)}")

    @property
    def edges(self) -> tuple[str, ...]:
        """Return prereq-first, order-preserving unique edge targets."""
        return tuple(dict.fromkeys((*self.prereqs, *self.down)))


class RoadmapDocument(_WikiModel, frozen=True, kw_only=True):
    slug: str
    kind: Literal["roadmap"]
    created: date
    github_project: str | None = None

    def __post_init__(self) -> None:
        _validate_slug_value(self.slug)
        if self.github_project is None:
            return
        owner, separator, number = self.github_project.partition("/")
        if not separator or not owner.strip() or not number.strip().isdigit():
            raise ValueError("github_project must be owner/number")
        structs.force_setattr(self, "github_project", f"{owner.strip()}/{number.strip()}")


class RoadmapModel(_WikiModel, frozen=True, kw_only=True):
    roadmap: RoadmapDocument
    goals: dict[str, GoalDocument]

    def __post_init__(self) -> None:
        _validate_graph(self)


def _validate_graph(model: RoadmapModel) -> None:
    for key, goal in model.goals.items():
        filename = f"{key}.md"
        if key != goal.slug:
            raise ValueError(f"{filename}: slug must match filename ({key!r} != {goal.slug!r})")
        if goal.roadmap != model.roadmap.slug:
            raise ValueError(
                f"{filename}: roadmap must be {model.roadmap.slug!r}, got {goal.roadmap!r}"
            )
        for field, targets in (
            ("prereqs", goal.prereqs),
            ("down", goal.down),
            ("up", goal.up),
        ):
            for target in targets:
                if target not in model.goals:
                    raise ValueError(f"{filename}: {field} target {target!r} does not resolve")
        for target in goal.down:
            if key not in model.goals[target].up:
                raise ValueError(f"{filename}: down/{target} is not mirrored by {target}.up")
        for target in goal.up:
            if key not in model.goals[target].down:
                raise ValueError(f"{filename}: up/{target} is not mirrored by {target}.down")
    _ensure_acyclic(model)


def _ensure_acyclic(model: RoadmapModel) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(slug: str) -> None:
        if slug in visiting:
            raise ValueError(f"{slug}.md: graph contains a cycle")
        if slug in visited:
            return
        visiting.add(slug)
        for target in model.goals[slug].edges:
            visit(target)
        visiting.remove(slug)
        visited.add(slug)

    for slug in model.goals:
        visit(slug)


def _goal_data(data: dict[str, object]) -> dict[str, object]:
    normalized = dict(data)
    for field in ("slug", "roadmap"):
        value = normalized.get(field)
        if isinstance(value, str):
            try:
                _validate_slug_value(value)
            except ValueError as exc:
                raise ValueError(f"{field}: {exc}") from exc
    for field in ("prereqs", "up", "down"):
        normalized[field] = _list_of_slugs(normalized.get(field))
    normalized["last_synced"] = _normalize_last_synced(normalized.get("last_synced"))
    return normalized


def parse_goal_document(text: str, *, slug: str) -> GoalDocument:
    """Parse one goal markdown document, naming validation errors by filename."""
    filename = f"{slug}.md"
    try:
        frontmatter, body = _frontmatter_and_body(text, filename)
        data = dict(frontmatter)
        if data.get("slug") != slug:
            raise ValueError(f"{filename}: slug must match filename")
        title, sections = _parse_body(body, filename)
        data.update(
            title=title,
            intent=sections.get("Intent"),
            acceptance=sections.get("Acceptance"),
            outcome=sections.get("Outcome"),
        )
        return msgspec.convert(_goal_data(data), type=GoalDocument, strict=True)
    except (msgspec.ValidationError, ValueError) as exc:
        if str(exc).startswith(f"{filename}:"):
            raise
        compact = " ".join(str(exc).split())
        raise ValueError(f"{filename}: {compact}") from exc


def _parse_roadmap_document(text: str, filename: str) -> RoadmapDocument:
    try:
        frontmatter, _ = _frontmatter_and_body(text, filename)
        return msgspec.convert(frontmatter, type=RoadmapDocument, strict=True)
    except (msgspec.ValidationError, ValueError) as exc:
        compact = " ".join(str(exc).split())
        raise ValueError(f"{filename}: {compact}") from exc


def load_roadmap(roadmap_dir: Path) -> RoadmapModel:
    """Load index.md and all goal markdown files into one validated model."""
    confined_root = roadmap_dir.parent
    index_path = roadmap_dir / "index.md"
    try:
        roadmap = _parse_roadmap_document(read_text(confined_root, index_path), "index.md")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"roadmap index not found: {index_path}") from exc
    if roadmap.slug != roadmap_dir.name:
        raise ValueError(f"index.md: slug must match roadmap directory {roadmap_dir.name!r}")
    goals = {
        path.stem: parse_goal_document(read_text(confined_root, path), slug=path.stem)
        for path in roadmap_markdown_files(confined_root, roadmap_dir)
        if path.name != "index.md"
    }
    try:
        return msgspec.convert(
            {"roadmap": roadmap, "goals": goals}, type=RoadmapModel, strict=True
        )
    except (msgspec.ValidationError, ValueError) as exc:
        raise ValueError(" ".join(str(exc).split())) from exc


def roadmap_schema() -> dict[str, object]:
    """Return the canonical roadmap JSON Schema."""
    generated = cast(dict[str, object], msgspec.json.schema(RoadmapModel))
    definitions = cast(dict[str, object], generated["$defs"])
    root = cast(dict[str, object], definitions["RoadmapModel"])
    return {**root, "$defs": definitions}


def roadmap_json(model: RoadmapModel) -> dict[str, object]:
    """Return the canonical JSON-compatible roadmap projection."""
    built = cast(dict[str, object], msgspec.to_builtins(model))
    goals = cast(dict[str, object], built["goals"])
    return {
        "roadmap": built["roadmap"],
        "goals": {slug: goals[slug] for slug in sorted(goals)},
        "edges": [
            {"from": slug, "to": dependency}
            for slug in sorted(model.goals)
            for dependency in model.goals[slug].edges
        ],
    }
