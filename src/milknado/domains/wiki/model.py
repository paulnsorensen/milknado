"""Canonical pydantic models for roadmap wiki documents."""

from __future__ import annotations

import re
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from milknado.domains.common.types import NodeStatus
from milknado.domains.wiki._markdown import _frontmatter_and_body, _parse_body


class Lifecycle(StrEnum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"


class _WikiModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


_SLUG_RE = re.compile(r"^[^/\\]+$")
_WIKILINK_RE = re.compile(r"^\[\[([^|\]]+)(?:\|[^\]]+)?\]$")


def _validate_slug_value(value: str) -> str:
    if not value or ".." in value or not _SLUG_RE.fullmatch(value):
        raise ValueError(f"invalid slug {value!r}")
    return value


def _wikilink_slug(value: str) -> str:
    match = _WIKILINK_RE.fullmatch(value.strip())
    return match.group(1).strip() if match else value.strip()


def _list_of_slugs(value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("must be a list of goal slugs")
    return [_wikilink_slug(item) for item in value]


class GoalDocument(_WikiModel):
    slug: str
    roadmap: str
    kind: Literal["goal"]
    created: date
    status: NodeStatus | None = None
    lifecycle: Lifecycle = Lifecycle.ACTIVE
    last_synced: str | None = None
    last_verified: date | None = None
    prereqs: list[str] = []
    up: list[str] = []
    down: list[str] = []
    title: str
    intent: str | None = None
    acceptance: str | None = None
    outcome: str | None = None

    _slug = field_validator("slug", "roadmap")(_validate_slug_value)
    _edges = field_validator("prereqs", "up", "down", mode="before")(_list_of_slugs)

    @field_validator("last_synced", mode="before")
    @classmethod
    def _normalize_last_synced(cls, value: object) -> str | None:
        if value is None or isinstance(value, str):
            return value
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        raise ValueError("last_synced must be a string")

    @model_validator(mode="after")
    def _require_active_sections(self) -> GoalDocument:
        if self.lifecycle is Lifecycle.ACTIVE:
            missing = [name for name in ("intent", "acceptance") if not getattr(self, name)]
            if missing:
                raise ValueError(f"active goal requires sections: {', '.join(missing)}")
        return self

    @property
    def edges(self) -> list[str]:
        """Return prereq-first, order-preserving unique edge targets."""
        return list(dict.fromkeys((*self.prereqs, *self.down)))


class RoadmapDocument(_WikiModel):
    slug: str
    kind: Literal["roadmap"]
    created: date
    github_project: str | None = None

    _slug = field_validator("slug")(_validate_slug_value)

    @field_validator("github_project")
    @classmethod
    def _validate_github_project(cls, value: str | None) -> str | None:
        if value is None:
            return None
        owner, separator, number = value.partition("/")
        if not separator or not owner.strip() or not number.strip().isdigit():
            raise ValueError("github_project must be owner/number")
        return f"{owner.strip()}/{number.strip()}"


class RoadmapModel(_WikiModel):
    roadmap: RoadmapDocument
    goals: dict[str, GoalDocument]

    @model_validator(mode="after")
    def _validate_graph(self, info: ValidationInfo) -> RoadmapModel:
        filenames = _filename_context(info.context)
        for key, goal in self.goals.items():
            filename = filenames.get(key, f"{key}.md")
            if key != goal.slug:
                raise ValueError(
                    f"{filename}: slug must match filename ({key!r} != {goal.slug!r})"
                )
            if Path(filename).stem != key:
                raise ValueError(f"{filename}: slug must match filename")
            if goal.roadmap != self.roadmap.slug:
                raise ValueError(
                    f"{filename}: roadmap must be {self.roadmap.slug!r}, got {goal.roadmap!r}"
                )
            for field, targets in (
                ("prereqs", goal.prereqs),
                ("down", goal.down),
                ("up", goal.up),
            ):
                for target in targets:
                    if target not in self.goals:
                        raise ValueError(f"{filename}: {field} target {target!r} does not resolve")
            for target in goal.down:
                if key not in self.goals[target].up:
                    raise ValueError(f"{filename}: down/{target} is not mirrored by {target}.up")
            for target in goal.up:
                if key not in self.goals[target].down:
                    raise ValueError(f"{filename}: up/{target} is not mirrored by {target}.down")
        self._ensure_acyclic(filenames)
        return self

    def _ensure_acyclic(self, filenames: dict[str, str]) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(slug: str) -> None:
            if slug in visiting:
                filename = filenames.get(slug, f"{slug}.md")
                raise ValueError(f"{filename}: graph contains a cycle")
            if slug in visited:
                return
            visiting.add(slug)
            for target in self.goals[slug].edges:
                visit(target)
            visiting.remove(slug)
            visited.add(slug)

        for slug in self.goals:
            visit(slug)


def _filename_context(context: dict[str, object] | None) -> dict[str, str]:
    if not context:
        return {}
    for key in ("filenames", "goal_filenames"):
        value = context.get(key)
        if isinstance(value, dict):
            return {str(slug): str(filename) for slug, filename in value.items()}
    return {}


def parse_goal_document(text: str, *, slug: str) -> GoalDocument:
    """Parse one goal markdown document, naming validation errors by filename."""
    filename = f"{slug}.md"
    try:
        frontmatter, body = _frontmatter_and_body(text, filename)
        data = dict(frontmatter)
        declared_slug = data.get("slug")
        if declared_slug != slug:
            raise ValueError(f"{filename}: slug must match filename")
        title, sections = _parse_body(body, filename)
        data.update(
            title=title,
            intent=sections.get("Intent"),
            acceptance=sections.get("Acceptance"),
            outcome=sections.get("Outcome"),
        )
        return GoalDocument.model_validate(data)
    except (ValidationError, ValueError) as exc:
        if str(exc).startswith(f"{filename}:"):
            raise
        compact = " ".join(str(exc).split())
        raise ValueError(f"{filename}: {compact}") from exc


def _parse_roadmap_document(text: str, filename: str) -> RoadmapDocument:
    try:
        frontmatter, _ = _frontmatter_and_body(text, filename)
        return RoadmapDocument.model_validate(frontmatter)
    except (ValidationError, ValueError) as exc:
        if str(exc).startswith(f"{filename}:"):
            raise
        compact = " ".join(str(exc).split())
        raise ValueError(f"{filename}: {compact}") from exc


def load_roadmap(roadmap_dir: Path) -> RoadmapModel:
    """Load ``index.md`` and all goal markdown files into one validated model."""
    index_path = roadmap_dir / "index.md"
    try:
        roadmap = _parse_roadmap_document(index_path.read_text(encoding="utf-8"), "index.md")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"roadmap index not found: {index_path}") from exc
    if roadmap.slug != roadmap_dir.name:
        raise ValueError(f"index.md: slug must match roadmap directory {roadmap_dir.name!r}")
    goals: dict[str, GoalDocument] = {}
    filenames: dict[str, str] = {}
    for path in sorted(roadmap_dir.glob("*.md")):
        if path.name == "index.md":
            continue
        goal = parse_goal_document(path.read_text(encoding="utf-8"), slug=path.stem)
        goals[path.stem] = goal
        filenames[path.stem] = path.name
    try:
        return RoadmapModel.model_validate(
            {"roadmap": roadmap, "goals": goals},
            context={"filenames": filenames},
        )
    except ValidationError as exc:
        compact = " ".join(str(exc).split())
        raise ValueError(compact) from exc
