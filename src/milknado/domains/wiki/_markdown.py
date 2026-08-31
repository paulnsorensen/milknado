"""Markdown parsing helpers for roadmap wiki documents."""

from __future__ import annotations

import re
from typing import cast

import yaml

_HEADING_RE = re.compile(r"^(#{1,6})(?:[ \t]+)(.*?)\s*$")
_ALLOWED_SECTIONS = frozenset({"Intent", "Acceptance", "Outcome"})


def frontmatter_and_body(text: str, filename: str) -> tuple[dict[str, object], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{filename}: frontmatter is required")
    try:
        end = next(index for index, line in enumerate(lines[1:], 1) if line.strip() == "---")
    except StopIteration as exc:
        raise ValueError(f"{filename}: unterminated frontmatter") from exc
    try:
        frontmatter: object = cast(object, yaml.safe_load("\n".join(lines[1:end])))
        if not frontmatter:
            frontmatter = {}
    except yaml.YAMLError as exc:
        raise ValueError(f"{filename}: frontmatter: invalid YAML") from exc
    if not isinstance(frontmatter, dict):
        raise ValueError(f"{filename}: frontmatter must be a mapping")
    return cast(dict[str, object], frontmatter), "\n".join(lines[end + 1 :]).strip()


def _parse_heading(
    line: str,
    *,
    in_fence: bool,
) -> tuple[int, str] | None:
    if in_fence:
        return None
    match = _HEADING_RE.fullmatch(line)
    if not match:
        return None
    heading = match.group(2).strip()
    if not heading:
        return None
    return len(match.group(1)), heading


def parse_body(body: str, filename: str) -> tuple[str, dict[str, str | None]]:  # noqa: PLR0915
    title: str | None = None
    sections: dict[str, str | None] = {}
    current: str | None = None
    content: list[str] = []
    in_fence = False
    fence = chr(96) * 3

    def finish_section() -> None:
        if current is not None:
            value = "\n".join(content).strip()
            sections[current] = value or None

    for line in body.splitlines():
        if line.lstrip().startswith((fence, "~~~")):
            in_fence = not in_fence
            content.append(line)
            continue
        parsed = _parse_heading(line, in_fence=in_fence)
        if parsed is None:
            content.append(line)
            continue
        level, heading = parsed
        if title is None:
            if level == 1:
                title = heading
                current = None
                content = []
            else:
                content.append(line)
            continue
        if level == 1:
            content.append(line)
            continue
        if level != 2:
            content.append(line)
            continue
        if heading not in _ALLOWED_SECTIONS or heading in sections:
            finish_section()
            current = None
            content = []
            continue
        finish_section()
        content = []
        current = heading
        sections[heading] = None
    if in_fence:
        raise ValueError(f"{filename}: unterminated fenced code")
    finish_section()
    if title is None:
        raise ValueError(f"{filename}: missing H1 title")
    return title, sections
