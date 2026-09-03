"""Parse YAML frontmatter from RALPH.md files.

A ralph is a directory containing a ``RALPH.md`` file.  The frontmatter
configures the agent command, commands to run, and accepted arguments.
The body is the prompt template.

HTML comments in the body are stripped so users can leave notes that
don't leak into the assembled prompt.
"""

from __future__ import annotations

import re
from typing import cast

import yaml

# Single source of truth for the ralph marker filename.
RALPH_MARKER = "RALPH.md"

# Frontmatter field names — used in parsing (cli.py) and placeholder
# resolution (_resolver.py).  Centralised here so renames stay in sync.
FIELD_AGENT = "agent"
FIELD_COMMANDS = "commands"
FIELD_ARGS = "args"
FIELD_RALPH = "ralph"

# YAML frontmatter delimiter line.
_FRONTMATTER_DELIMITER = "---"

# Valid characters for identifier names (commands, args, ralph context) —
# letters, digits, hyphens, underscores.  Used by cli.py (validation) and
# _resolver.py (placeholders).
NAME_RE = re.compile(r"[a-zA-Z0-9_-]+")


# UTF-8 BOM character — files saved on Windows may start with this.
_UTF8_BOM = "\ufeff"

# Matches fenced code blocks (``` or ~~~, captured in group 1) OR HTML
# comments.  Used by _strip_html_comments to remove comments while
# preserving any that appear inside code fences.
#
# Backreferences (\2 / \3) ensure the closing fence has *at least* as
# many characters as the opening, so a ```` fence is not broken by an
# inner ``` — the inner ``` is treated as content, not a fence boundary.
# The trailing ```*`` / ``~*`` after each backreference consumes any
# *extra* fence characters on the closing line so they are not
# misinterpreted as a new opening fence (CommonMark allows the closing
# fence to be longer than the opening).
# The ``\n`` before each backreference requires the closing fence to
# start on a new line, preventing inline backticks/tildes from being
# mistaken for a closing fence (matching CommonMark's rule that closing
# code fences must be on their own line).
_FENCE_OR_COMMENT_RE = re.compile(
    r"((`{3,}).*?\n[ \t]*\2`*|(~{3,}).*?\n[ \t]*\3~*)|<!--.*?-->",
    re.DOTALL,
)


def _strip_html_comments(body: str) -> str:
    """Remove HTML comments from *body*, preserving those inside fenced code blocks."""
    return _FENCE_OR_COMMENT_RE.sub(
        lambda m: m.group(1) if m.group(1) else "",
        body,
    )


def _extract_frontmatter_block(text: str) -> tuple[str, str]:
    """Split text into raw YAML frontmatter and body at ``---`` delimiters.

    The opening ``---`` must be the very first line (standard YAML
    frontmatter convention).  Returns ``("", text)`` when no valid
    frontmatter block is found.
    """
    lines = text.split("\n")
    if lines[0].rstrip() != _FRONTMATTER_DELIMITER:
        return "", text

    for i, line in enumerate(lines[1:], start=1):
        if line.rstrip() == _FRONTMATTER_DELIMITER:
            fm_raw = "\n".join(lines[1:i])
            body = "\n".join(lines[i + 1 :]).strip()
            return fm_raw, body
    return "", text


def parse_frontmatter(text: str) -> tuple[dict[str, object], str]:
    """Parse a RALPH.md file with YAML frontmatter.

    Frontmatter is delimited by ``---`` lines at the start of the file.
    Full YAML is supported (nested lists, dicts).  HTML comments are
    stripped from the body so they don't leak into the assembled prompt.

    A leading UTF-8 BOM (``\\ufeff``) is stripped so files saved with a
    BOM (common on Windows) are handled transparently.

    Returns ``(frontmatter_dict, body_text)``.
    """
    text = text.removeprefix(_UTF8_BOM)
    fm_raw, body = _extract_frontmatter_block(text)
    if fm_raw:
        try:
            raw_frontmatter: object = cast(object, yaml.safe_load(fm_raw))
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in frontmatter: {exc}") from exc
        if raw_frontmatter is None:
            frontmatter: dict[str, object] = {}
        elif not isinstance(raw_frontmatter, dict):
            raise ValueError(
                f"Frontmatter must be a YAML mapping, got {type(raw_frontmatter).__name__}"
            )
        else:
            frontmatter = cast(dict[str, object], raw_frontmatter)
    else:
        frontmatter = {}
    body = _strip_html_comments(body).strip()
    return frontmatter, body
