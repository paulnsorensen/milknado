"""Wiki crossover slice: import roadmap markdown into milknado nodes and harvest
execution outcomes back into the wiki goal files (one-way milknado -> wiki dep)."""

from milknado.domains.wiki._locate import (
    goal_file_map,
    locate_roadmap_dir,
    read_text,
    resolve_roadmap_dir,
    wiki_root,
)
from milknado.domains.wiki._serialize import extract_section, load_frontmatter
from milknado.domains.wiki.exporter import (
    ExportResult,
    export_roadmap,
    resolve_roadmap_node,
)
from milknado.domains.wiki.importer import ImportResult, import_roadmap
from milknado.domains.wiki.model import (
    GoalDocument,
    Lifecycle,
    RoadmapDocument,
    RoadmapModel,
    load_roadmap,
    parse_goal_document,
    roadmap_json,
    roadmap_schema,
)
from milknado.domains.wiki.ports import WikiIndexerPort, WikiIndexResult, WikiIndexStatus
from milknado.domains.wiki.render import render_html, render_mermaid

__all__ = [
    "ExportResult",
    "GoalDocument",
    "ImportResult",
    "Lifecycle",
    "RoadmapDocument",
    "RoadmapModel",
    "WikiIndexerPort",
    "WikiIndexResult",
    "WikiIndexStatus",
    "export_roadmap",
    "extract_section",
    "goal_file_map",
    "import_roadmap",
    "load_frontmatter",
    "load_roadmap",
    "locate_roadmap_dir",
    "parse_goal_document",
    "roadmap_json",
    "roadmap_schema",
    "read_text",
    "render_html",
    "render_mermaid",
    "resolve_roadmap_node",
    "resolve_roadmap_dir",
    "wiki_root",
]
