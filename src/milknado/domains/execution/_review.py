from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from milknado.domains.common.paths import slugify
from milknado.domains.common.types import MikadoNode


@dataclass(frozen=True)
class ReviewNotification:
    audit_succeeded: bool
    notification_succeeded: bool


def build_review_prompt(
    node: MikadoNode,
    worktree: Path,
    project_root: Path,
    diff: str,
) -> str:
    try:
        brief = (worktree / "RALPH.md").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        brief = node.description
    spec_path = node.artifact_path
    if spec_path:
        candidate = Path(spec_path).expanduser()
        if not candidate.is_absolute():
            candidate = (project_root / candidate).resolve()
        spec_path = str(candidate)
    return (
        f"# Adversarial review: {node.description}\n\n"
        "## Generated node brief/context\n\n"
        f"{brief.rstrip()}\n\n"
        f"## Spec\n\n{spec_path or '(no spec path)'}\n\n"
        "Review the pinned worktree against the brief and spec. Report "
        "easy-cheese severity/dimension findings with evidence and fixes, "
        "then emit exactly one verdict tag, with no other verdict tags:\n\n"
        "<verdict>approve</verdict>\n\nor\n\n"
        "<verdict>reject</verdict>\n\n"
        f"## Full base-to-worktree diff\n\n```diff\n{diff}\n```\n"
    )


def persist_review_findings(
    node: MikadoNode,
    worktree: Path,
    findings_md: str,
) -> None:
    slug = slugify(node.description, max_length=30) or str(node.id)
    path = worktree / ".cheese" / "age" / f"{slug}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(findings_md.rstrip() + "\n", encoding="utf-8")
