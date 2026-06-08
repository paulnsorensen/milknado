status: ok
next: press
artifact: /home/paul/.local/share/cheese/paulnsorensen-milknado/specs/hallouminate-milknado-crossover.md
Built the milknado<->hallouminate wiki crossover: a new domains/wiki slice (importer+exporter), additive wiki_ref column, roadmap CLI + MCP tools, and two plugin skills — all 8 acceptance criteria covered, full gate green.

## Package-ready report

### Answer

The one-way milknado→wiki roadmap crossover is implemented and the full gate
(`just check-llm`) is green: lint, format, all tests, and project + diff coverage
≥95%. Wiki markdown under `<project>/.hallouminate/wiki/roadmaps/<slug>/` is now
the durable store for roadmap + goal intent; milknado imports it (deterministic
`wiki_ref` keys, idempotent) and harvests execution outcomes back into harvest-marker
blocks without touching human-authored Intent/Acceptance.

### Files changed (and why)

**DB migration (additive, rides auto-migrate-on-load):**

- `domains/graph/_persistence.py` — `wiki_ref TEXT` column on `nodes`, partial
  `UNIQUE` index `idx_nodes_wiki_ref` (created in `ensure_schema` after the column
  is guaranteed, so NULLs stay unrestricted), `row_to_node` reads it gracefully.
- `domains/common/types.py` — `MikadoNode.wiki_ref` field.
- `domains/graph/graph.py` — `add_node(wiki_ref=…)`, `find_node_by_wiki_ref()`,
  `set_wiki_ref()` (the last needed so an orphan goal node filed during export
  round-trips on the next export).

**New `domains/wiki/` slice (Sliced Bread crust re-exports import/export):**

- `_serialize.py` — `NAMESPACE_MILKNADO_WIKI`, `compute_goal_ref` /
  `compute_roadmap_ref` (uuid5, stdlib), frontmatter load + surgical
  `set_frontmatter_field`, `extract_title` / `extract_section`,
  `replace_harvest_block` (byte-preserving), `slugify`.
- `importer.py` — `import_roadmap()` → `ImportResult`; find-or-create by wiki_ref,
  prereq edges, `created` stamping, goal-layer-only, fail-fast.
- `exporter.py` — `export_roadmap()` → `ExportResult`; recovers the roadmap dir by
  matching the roadmap node's wiki_ref, harvest-block rebuild (status, task rollup,
  `deposit_result` summaries, branch links), orphan-goal file creation, best-effort
  `hallouminate index`. Also `resolve_roadmap_node()` (shared by CLI + MCP).

**Veneer:**

- `cli_roadmap.py` + `cli.py` wire — `milknado roadmap import|export <slug>`.
- `mcp_wiki.py` + `mcp_server.py:main` wire — `milknado_roadmap_import|export`.
- `plugins/milknado/skills/{load-roadmap,harvest}/SKILL.md` — thin skill veneers.

### Tests run

`just check-llm` → PASS (lint + format + full suite + project & diff coverage ≥95%).
56 new tests across `test_wiki_serialize`, `test_wiki_importer`, `test_wiki_exporter`,
`test_graph_wiki_ref`, `test_cli_roadmap`, `test_mcp_wiki`. Updated the MCP
tool-registration pin test (`test_mcp_server`) to include the two new tools.

### Open questions — resolved during /cook

- **PR/branch links** `<certain>` — no PR field exists anywhere in the schema;
  `MikadoNode.branch_name` is the only signal. Harvest renders a `branches:` line
  when present, omits a `PRs:` line entirely.
- **Task-summary source** `<certain>` — `RunDict.summary` is always None (a log
  tail); the real deliverable is `graph.latest_run_message(run_id, "result")` over
  terminal runs of each descendant task node.
- **`created` stamping** — import stamps `created` (injectable `today` for tests)
  when absent; determinism after git-commit is a documented non-goal.

### Residual risk

- `branches:`/summary harvest content is not pinned by the spec's acceptance
  criteria (only structure is), so the exact rollup wording is a judgment call
  `<certain>` — easily adjusted if review wants a different shape.
- tilth MCP was rooted at a different project (`n8n-mcp`) this session, so all
  reads/edits used host tools — no impact on the artifact `<certain>`.
- Not yet committed; no PR exists for this branch.
