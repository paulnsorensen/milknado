# Roadmap graph schema

Roadmap wiki documents are canonical Pydantic models: active goals require `Intent` and `Acceptance`; deprecated goals map to `MikadoNode.archived_at`; execution `status` remains the existing `NodeStatus` vocabulary.[^1]

The wiki importer validates the whole roadmap before graph import, while exporter writes remain surgical so human-authored markdown survives.[^2]

The CLI and MCP expose schema, JSON interchange, and deterministic Mermaid/inline-SVG HTML graph rendering. JSON edges are `{from: goal_slug, to: dependency_slug}` in canonical goal-edge order.[^3]

[^1]: src/milknado/domains/wiki/model.py; src/milknado/domains/wiki/importer.py
[^2]: src/milknado/domains/wiki/importer.py; src/milknado/domains/wiki/exporter.py
[^3]: src/milknado/cli/roadmap.py; src/milknado/mcp/wiki.py; src/milknado/domains/wiki/render.py
