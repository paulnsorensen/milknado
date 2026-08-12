"""Pure serialization helpers for the wiki <-> milknado roadmap crossover.

No graph, no filesystem side effects — just deterministic key derivation and
surgical markdown rewrites. The surgical rewrites are the membrane mechanism:
milknado overwrites only the harvest block and the status/last_synced
frontmatter fields, leaving every other byte (notably human-authored Intent and
Acceptance) untouched.
"""

from __future__ import annotations

import re
import uuid
from datetime import date, datetime
from typing import cast

import yaml

# Fixed package-level namespace so wiki_ref keys are identical on every machine.
NAMESPACE_MILKNADO_WIKI = uuid.UUID("d9347c3c-3e8d-5029-91ee-de7c720f8dd8")

HARVEST_START = "<!-- milknado:harvest:start -->"
HARVEST_END = "<!-- milknado:harvest:end -->"

_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)

Created = str | date | datetime


def compute_goal_ref(roadmap_slug: str, goal_slug: str, created: Created) -> str:
    """Deterministic goal key from git-resident frontmatter alone."""
    return str(uuid.uuid5(NAMESPACE_MILKNADO_WIKI, f"{roadmap_slug}/{goal_slug}@{created}"))


def compute_roadmap_ref(roadmap_slug: str, created: Created) -> str:
    """Deterministic roadmap key — disjoint from any goal key by construction."""
    return str(uuid.uuid5(NAMESPACE_MILKNADO_WIKI, f"{roadmap_slug}@{created}"))


def load_frontmatter(text: str) -> dict[str, object]:
    """Parse the leading `---` YAML block into a dict, or {} when absent."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}
    data = yaml.safe_load(m.group(1))
    return cast(dict[str, object], data) if isinstance(data, dict) else {}


def extract_section(text: str, heading: str) -> str | None:
    """Return the stripped body under `## <heading>`, up to the next `## ` or EOF."""
    pattern = re.compile(
        rf"^## {re.escape(heading)}[^\n]*\n(.*?)(?=^## |\Z)",
        re.DOTALL | re.MULTILINE,
    )
    m = pattern.search(text)
    return m.group(1).strip() if m else None


def set_frontmatter_field(text: str, key: str, value: str) -> str:
    """Rewrite (or insert) one frontmatter `key: value` line, surgically.

    Every byte outside the matched frontmatter line is preserved, so the body —
    and any other frontmatter key — is byte-identical afterwards.
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        raise ValueError("cannot set frontmatter field: file has no frontmatter block")
    block = m.group(1)
    new_line = f"{key}: {value}"
    key_re = re.compile(rf"^{re.escape(key)}:.*$", re.MULTILINE)
    if key_re.search(block):
        new_block = key_re.sub(lambda _m: new_line, block, count=1)
    else:
        new_block = f"{block}\n{new_line}"
    return text[: m.start(1)] + new_block + text[m.end(1) :]


def replace_harvest_block(text: str, inner: str) -> str:
    """Overwrite only the bytes between the harvest markers.

    When the markers are absent, append a fresh `## Outcome` section carrying
    them — the original text is preserved verbatim as the prefix.
    """
    if HARVEST_START in text and HARVEST_END in text:
        pattern = re.compile(re.escape(HARVEST_START) + r".*?" + re.escape(HARVEST_END), re.DOTALL)
        return pattern.sub(lambda _m: f"{HARVEST_START}\n{inner}\n{HARVEST_END}", text, count=1)
    suffix = "" if text.endswith("\n") else "\n"
    return f"{text}{suffix}## Outcome\n{HARVEST_START}\n{inner}\n{HARVEST_END}\n"


def validate_slug(slug: str) -> str:
    """Reject a roadmap slug that would escape the roadmaps dir once path-joined.

    The slug names a directory under `roadmaps/`; a `..` or path separator in it
    (CLI argument or MCP parameter) would walk outside the wiki. Fail fast.
    """
    if not slug or "/" in slug or "\\" in slug or ".." in slug:
        raise ValueError(f"unsafe roadmap slug: {slug!r}")
    return slug


_WIKILINK_RE = re.compile(r"^\[\[([^|\]]+)(?:\|[^\]]+)?\]\]$")


def parse_wikilink(value: str) -> str:
    """Extract the link target from a wikilink string.

    "[[target]]" -> "target"; "[[target|alias]]" -> "target"; bare string passes through.
    """
    m = _WIKILINK_RE.match(value)
    return m.group(1).strip() if m else value
