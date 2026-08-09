# Roadmap graph schema

Roadmap wiki documents are frozen msgspec boundary models validated by strict conversion: active goals require `Intent` and `Acceptance`; deprecated goals remove existing goal-prerequisite edges and skip new dependency wiring before their task-only subtree is archived, leaving prerequisites active while preserving archived-subtree invariants; execution `status` remains the existing `NodeStatus` vocabulary.[^1]

The wiki importer validates the whole roadmap before graph import, while exporter writes remain surgical so human-authored markdown survives.[^2]

The wiki slice owns JSON Schema and canonical JSON projection for both CLI and MCP. JSON edges are `{from: goal_slug, to: dependency_slug}` in canonical goal-edge order; rendering remains deterministic Mermaid/inline-SVG HTML.[^3]

[^1]: src/milknado/domains/wiki/model.py; src/milknado/domains/wiki/importer.py
[^2]: src/milknado/domains/wiki/importer.py; src/milknado/domains/wiki/exporter.py
[^3]: src/milknado/cli/roadmap.py; src/milknado/mcp/wiki.py; src/milknado/domains/wiki/render.py
