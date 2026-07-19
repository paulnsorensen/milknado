"""Wiki crossover slice: import roadmap markdown into milknado nodes and harvest
execution outcomes back into the wiki goal files (one-way milknado -> wiki dep)."""

from milknado.domains.wiki._locate import (
    goal_file_map,
    locate_roadmap_dir,
    read_text,
    wiki_root,
)
from milknado.domains.wiki._serialize import extract_section, load_frontmatter
from milknado.domains.wiki.exporter import (
    ExportResult,
    export_roadmap,
    resolve_roadmap_node,
)
from milknado.domains.wiki.importer import ImportResult, import_roadmap
from milknado.domains.wiki.ports import WikiIndexerPort, WikiIndexResult, WikiIndexStatus

__all__ = [
    "ExportResult",
    "ImportResult",
    "export_roadmap",
    "WikiIndexerPort",
    "WikiIndexResult",
    "WikiIndexStatus",
    "extract_section",
    "goal_file_map",
    "import_roadmap",
    "load_frontmatter",
    "locate_roadmap_dir",
    "read_text",
    "resolve_roadmap_node",
    "wiki_root",
]
