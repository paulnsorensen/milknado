from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

INIT_TEMPLATE = """\
from {name}.plugin import {class_name}

__all__ = ["{class_name}"]
"""

PLUGIN_TEMPLATE = '''\
from milknado.domains.common import MikadoNode, NodeStatus, PluginHook, PluginMeta


class {class_name}:
    """Hello-world milknado plugin."""

    @property
    def meta(self) -> PluginMeta:
        return PluginMeta(
            name="{name}",
            version="0.1.0",
            description="A hello-world milknado plugin",
        )

    def on_node_status_change(
        self, node: MikadoNode, old_status: NodeStatus, new_status: NodeStatus,
    ) -> None:
        print(
            f"[{{self.meta.name}}] Node {{node.id}} "
            f"{{old_status.value}} -> {{new_status.value}}"
        )
'''

README_TEMPLATE = """\
# {name}

A milknado plugin.

## Installation

Add `"{name}"` to the `plugins` list in your `milknado.toml`:

```toml
[milknado]
plugins = ["{name}"]
```
"""


def _to_class_name(name: str) -> str:
    return "".join(part.capitalize() for part in name.replace("-", "_").split("_"))


@dataclass(frozen=True)
class ScaffoldResult:
    plugin_dir: Path
    files_created: list[str]


def scaffold_plugin(name: str, target_dir: Path) -> ScaffoldResult:
    plugin_dir = target_dir / name
    if plugin_dir.exists():
        msg = f"Directory already exists: {plugin_dir}"
        raise FileExistsError(msg)

    class_name = _to_class_name(name)
    plugin_dir.mkdir(parents=True)

    files = []
    for filename, template in [
        ("__init__.py", INIT_TEMPLATE),
        ("plugin.py", PLUGIN_TEMPLATE),
        ("README.md", README_TEMPLATE),
    ]:
        path = plugin_dir / filename
        path.write_text(template.format(name=name, class_name=class_name))
        files.append(filename)

    return ScaffoldResult(plugin_dir=plugin_dir, files_created=files)
