"""Config sync domain — per-flavor agent-def generation.

Public API (crust):
  FlavorIdentity    — resolved per-flavor identity (tools, model, max_turns).
  RenderedDef       — a generated agent-def file (harness, path, content).
  SyncFilter        — (harnesses, flavors) filter passed to sync().
  resolve_flavor_identity — resolve a TaskFlavor against a MilknadoConfig.
  render_claude     — render a Claude Code agent-def Markdown file.
  render_opencode   — render an opencode agent-def Markdown file.
  render_codex      — render a Codex TOML agent-def file.
  sync              — generate and write all requested flavor×harness defs.
"""

from milknado.domains.config_sync._identity import (
    FlavorIdentity,
    RenderedDef,
    resolve_flavor_identity,
)
from milknado.domains.config_sync._render import (
    SyncFilter,
    render_claude,
    render_codex,
    render_opencode,
    sync,
)

__all__ = [
    "FlavorIdentity",
    "RenderedDef",
    "SyncFilter",
    "render_claude",
    "render_codex",
    "render_opencode",
    "resolve_flavor_identity",
    "sync",
]
