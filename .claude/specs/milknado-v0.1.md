---
title: Milknado v0.1 — Mikado Execution Engine
created: 2026-04-13
status: draft
stakeholders: [Paul Sorensen]
related: [code-review-graph, ralphify]
---

# Milknado v0.1 — Mikado Execution Engine

## Executive Summary

Milknado is a Python CLI that decomposes development goals into Mikado dependency graphs and executes them as parallel ralph loops in isolated git worktrees. It wraps ralphify's library API for loop execution, uses code-review-graph for structural codebase awareness during planning, and stores the Mikado DAG in SQLite. The key design decision: ralphify is the execution engine, code-review-graph is the planning brain, milknado is the orchestrator that connects them. File-level ownership tracking prevents parallel nodes from producing merge conflicts.

## Business Context

- **Domain**: AI-assisted autonomous software development
- **Primary entities**: Goals (root nodes), Mikado nodes (atoms of work), ralph loops (execution units), worktrees (isolation boundaries)
- **This feature's role**: Replace flat agent loops with structured, dependency-aware, parallelizable execution trees
- **Success looks like**: A developer states a goal, milknado decomposes it into a Mikado graph, and leaf nodes execute as parallel ralph loops that converge into a feature branch via auto-merged commits — with no merge conflicts because milknado tracks file ownership per node

## Problem Statement

Ralphify runs flat, single-track agent loops. Real development work has dependency structure — you can't refactor the auth adapter until you extract the interface, and both of those are independent from updating the tests. Flat loops serialize inherently parallel work and can't express "do X before Y." The Mikado Method solves this on paper; milknado automates it with agent loops.

## Design Principles

1. **Ralphify is the spine** — milknado extends ralphify, never reimplements it. All loop execution flows through ralphify's library API.
2. **Leaf-first, always** — execution order follows Mikado discipline: leaves before parents, never work in a broken state.
3. **Workers can be dumb** — ralph loops are simple executors. Graph intelligence lives in the orchestrator.
4. **Dynamic discovery over upfront planning** — the graph grows as workers discover prerequisites, not just from initial decomposition.
5. **File ownership prevents conflicts** — each node knows which files it touches (via code-review-graph). Parallel nodes with overlapping files are serialized, not parallelized.
6. **YAGNI the orchestration loop** — v0.1 dispatches and merges. If workers need a smarter supervisor, that's v0.2 (see Future: Orchestration Loop).

## Goals

- [ ] Wrap ralphify as a dependency, build a new CLI shell around its library API
- [ ] Integrate code-review-graph as infrastructure for planning-time structural analysis
- [ ] Implement Mikado DAG in SQLite (nodes, edges, status, file ownership)
- [ ] Interactive planning phase: launch full Claude session with code-review-graph context
- [ ] Parallel leaf execution via ralph loops in isolated git worktrees
- [ ] File-level ownership tracking to prevent parallel nodes from touching the same files
- [ ] Dynamic graph growth: running ralphs can register new prerequisite nodes via CLI
- [ ] Auto-merge completed worktrees into feature branch (auto-rebase, notify on conflict)
- [ ] Quality gates: milknado defaults + per-ralph overrides
- [ ] Hello-world plugin scaffold
- [ ] Runs on uv (pyproject.toml, uv-native toolchain)

## Non-Goals

- PR creation and GitHub integration (deferred)
- PAUL-style orchestration/unify loop (see Future section)
- Complex plugin system beyond hello-world scaffold
- Backward compatibility with any existing tool
- GUI or TUI (CLI only for v0.1)
- MCP tool injection into ralph loops (v0.2, see Future section)
- Single-shot planning prompt mode (v0.2, see Future section)

## Context

### Existing Landscape

**ralphify (v0.4.0b3)** — Python 3.11+, Hatchling build. Core library API:
- `run_loop(config, state, emitter)` — blocking single-run loop
- `RunManager` — thread-safe multi-run orchestrator
- `ManagedRun` — per-run handle with event listeners
- `EventEmitter` protocol — structural typing, implement `.emit()` and `.wants_agent_output_lines()`
- `FanoutEmitter` — broadcasts to multiple listeners
- RALPH.md format: YAML frontmatter (`agent`, `commands`, `args`, `credit`) + markdown prompt template
- `{{ commands.<name> }}` and `{{ args.<name> }}` placeholders
- Discovery: current dir, `.agents/ralphs/`, `~/.agents/ralphs/`

**code-review-graph (v2.3.1)** — Python 3.10+, Hatchling build. Key APIs:
- Tree-sitter parsing for 19 languages into SQLite-backed graph
- `graph.py` — SQLite graph store (`.code-review-graph/` directory)
- `changes.py` — blast-radius / impact analysis
- `flows.py` — execution flow tracing
- `communities.py` — Leiden algorithm clustering
- `parser.py` — incremental SHA-256 diff/re-parse
- 22 MCP tools exposed via `tools/` directory
- CLI: `code-review-graph build`, `update`, `detect-changes`, etc.
- Key deps: `fastmcp`, `tree-sitter`, `networkx`, `watchdog`

**Mikado Method** — No canonical execution tooling exists. The method is documented (Manning 2014, Ellnestam & Brolund) but practiced on paper/whiteboards. Milknado would be the first tool to execute a Mikado graph as autonomous agent loops.

### Constraints

- Python 3.11+ (ralphify's minimum)
- Must run on uv (`uv run`, `uv tool install`)
- SQLite is already a transitive dependency via code-review-graph
- MIT license

### Dependencies

**Milknado depends on:**
- `ralphify` — loop execution, RunManager, event system
- `code-review-graph` — structural analysis, blast radius, graph queries
- `typer` — CLI framework (align with ralphify's choice)
- `rich` — terminal output (align with ralphify's choice)
- `pyyaml` — RALPH.md frontmatter parsing

**Nothing depends on milknado** (greenfield).

## Quality Gates

Commands that must pass for every user story:
- `uv run pytest` — unit and integration tests
- `uv run ruff check` — linting
- `uv run ty check` — type checking

## User Stories

### US-001: Project Scaffolding

**Description:** As a developer, I want to run `milknado init` in a project so that milknado is configured and ready to use.

**Acceptance Criteria:**
- [ ] `milknado init` creates a `milknado.toml` config file with sensible defaults
- [ ] Config includes: default agent command, quality gate commands, worktree naming pattern, concurrency limit
- [ ] If code-review-graph is not built, prompts to build it (or auto-builds)
- [ ] Creates SQLite database for Mikado graph storage
- [ ] Idempotent — running twice doesn't clobber existing config

### US-002: Mikado Graph Data Model

**Description:** As milknado's core, I need a SQLite-backed Mikado DAG that tracks nodes, edges, status, file ownership, and metadata.

**Acceptance Criteria:**
- [ ] Nodes table: id, description, status (pending/running/done/blocked/failed), parent_id, worktree_path, branch_name, run_id, created_at, completed_at
- [ ] Edges table: parent_id, child_id (child must complete before parent)
- [ ] File ownership table: node_id, file_path (which files this node is expected to touch)
- [ ] A node is "ready" when it is pending and all its children are status=done
- [ ] Root node is the overall goal
- [ ] Leaf nodes are nodes with no children
- [ ] Status transitions are enforced (e.g., can't go from done to pending)
- [ ] Graph operations: add_node, add_edge, get_leaves, get_ready_nodes, mark_done, mark_failed
- [ ] Parallel-safety check: two nodes can run in parallel only if their file ownership sets don't overlap
- [ ] Cycle detection on add_edge — reject edges that would create cycles

### US-003: Interactive Planning

**Description:** As a developer, I want to run `milknado plan "extract payment service"` so that milknado launches a full interactive Claude session with code-review-graph context to decompose the goal into a Mikado graph.

**Acceptance Criteria:**
- [ ] Launches a full interactive agent session (default: `claude` with injected context)
- [ ] Session context includes: goal description, code-review-graph blast radius, architecture overview, existing graph state (if resuming)
- [ ] The agent's first action is to call a milknado skill/tool to propose the decomposition
- [ ] Agent proposes a set of nodes with dependency edges and file ownership per node
- [ ] User sees the proposed graph and can approve, edit, or reject within the interactive session
- [ ] Approved graph is persisted to SQLite
- [ ] If code-review-graph data is stale or missing, auto-updates before planning

### US-004: Leaf-First Parallel Execution

**Description:** As a developer, I want to run `milknado run` so that ready leaf nodes execute as parallel ralph loops in isolated worktrees.

**Acceptance Criteria:**
- [ ] Identifies all "ready" leaf nodes (no incomplete children)
- [ ] Filters ready nodes for parallel safety (no overlapping file ownership)
- [ ] For each dispatchable leaf, creates a git worktree: `milknado-<node-id>-<slug>`
- [ ] Generates a RALPH.md in the worktree with: node description as prompt, code-review-graph context scoped to that node's files, quality gate commands
- [ ] Starts a ralph loop (via ralphify's `RunManager`) for each worktree
- [ ] Runs leaves in parallel (bounded by configurable concurrency limit from `milknado.toml`)
- [ ] Monitors loop completion via ralphify's event system
- [ ] On completion: auto-rebases worktree branch onto feature branch, creates a discrete commit
- [ ] On rebase conflict: notify user, mark node as failed (see Future: mergiraf integration)
- [ ] On merge success: marks node as done, cleans up worktree, checks if parent is now ready
- [ ] Newly ready nodes are immediately dispatched (continuous leaf-first execution)
- [ ] Execution completes when root node is done (all descendants merged)

### US-005: Dynamic Graph Growth

**Description:** As a running ralph loop, I want to signal milknado that I've discovered a prerequisite so that a new node is added to the graph.

**Acceptance Criteria:**
- [ ] `milknado add-node --parent <id> "description of prerequisite"` adds a child node
- [ ] Optionally accepts `--files <path1> <path2>` to declare file ownership for the new node
- [ ] If the parent was running, it is paused/marked as blocked
- [ ] The new child becomes a leaf and is eligible for execution
- [ ] The parent won't resume until the new child completes
- [ ] Multiple children can be added to the same parent
- [ ] Graph integrity is maintained (no cycles, enforced by cycle detection)

### US-006: Graph Status

**Description:** As a developer, I want to run `milknado status` so that I can see the current state of the Mikado graph.

**Acceptance Criteria:**
- [ ] Shows tree-formatted graph with node IDs, descriptions, and status (color-coded via Rich)
- [ ] Shows active worktrees and their ralph loop status
- [ ] Shows completion percentage (done nodes / total nodes)
- [ ] Shows which nodes are ready for execution
- [ ] Shows file ownership conflicts (nodes that can't run in parallel)

### US-007: Hello-World Plugin Scaffold

**Description:** As a developer, I want to run `milknado plugin init <name>` so that a plugin directory is created with the minimal structure.

**Acceptance Criteria:**
- [ ] Creates `<name>/` directory with: `__init__.py`, `plugin.py`, `README.md`
- [ ] `plugin.py` contains a hello-world plugin class with a single hook point
- [ ] Plugin is discoverable by milknado via entry points or a config reference in `milknado.toml`
- [ ] Running milknado with the plugin installed shows a log line confirming it loaded

### US-008: Code-Review-Graph Integration

**Description:** As milknado's planning and execution phases, I need to query code-review-graph for structural context and file-level dependency information.

**Acceptance Criteria:**
- [ ] If `.code-review-graph/` doesn't exist, auto-runs build
- [ ] If graph is stale (files changed since last build), auto-runs incremental update
- [ ] Planning prompt includes: blast radius for files mentioned in the goal, architecture overview (communities), relevant execution flows
- [ ] File ownership per node is informed by code-review-graph's blast radius — if a node touches `auth.py`, its ownership includes files in `auth.py`'s blast radius
- [ ] Graph context is injected into each ralph's RALPH.md (scoped to that node's file ownership set)
- [ ] Code-review-graph is called via its Python API, not CLI subprocess

## Functional Requirements

- FR-1: Milknado must use ralphify's `RunManager` for all loop execution — no reimplementation
- FR-2: The Mikado DAG must be persisted in SQLite and survive process restarts
- FR-3: Worktree creation and cleanup must be atomic — partial failures must not leave orphan worktrees
- FR-4: Node completion triggers an immediate check for newly ready nodes (event-driven, not polling)
- FR-5: The planning agent command must be configurable via `milknado.toml`
- FR-6: Quality gates are defined in `milknado.toml` (global defaults) and can be overridden per-node in RALPH.md frontmatter
- FR-7: `milknado add-node` must be callable from within a worktree subprocess (the ralph loop's agent)
- FR-8: Two nodes with overlapping file ownership must never execute simultaneously
- FR-9: Auto-rebase uses standard git rebase; on conflict, node is marked failed and user is notified

## Proposed Approach

### Option A: Ralphify Library + SQLite DAG (Recommended)

Milknado depends on ralphify as a library, uses `RunManager` to orchestrate parallel loops, and maintains the Mikado graph in SQLite with file ownership tracking. Code-review-graph is called via Python API for planning context and blast radius. The CLI is built with Typer (matching ralphify's choice).

**Evidence**: ralphify exports a clean library API (`run_loop`, `RunManager`, `EventEmitter` protocol). Code-review-graph uses SQLite internally. Both are Hatchling builds, compatible with uv. No new architectural patterns needed — just composition.

**Tradeoffs**: Couples milknado to ralphify's event model. If ralphify changes its API, milknado breaks. Acceptable given early development and no backward compat concerns.

### Option B: Fork Both, Monolith

Fork ralphify and code-review-graph source into milknado, build a unified codebase.

**Why not**: Violates YAGNI. We don't need to fork — the library APIs are sufficient.

### Option C: Do Nothing

Continue using ralphify flat loops and code-review-graph separately.

**Why not**: The whole point is automating Mikado decomposition and parallel execution. The status quo requires the developer to be the orchestrator.

## Risks & Mitigations

- **Risk**: Ralphify's library API is beta (v0.4.0b3) and may change → **Mitigation**: Pin version, wrap in a thin adapter layer. Early development = no backward compat concerns.
- **Risk**: Merge conflicts when auto-rebasing parallel worktrees → **Mitigation**: File ownership tracking prevents parallel nodes from touching the same files. Auto-rebase with conflict detection. Fail loudly on conflict. (Future: mergiraf for AST-aware merge resolution.)
- **Risk**: Ralph loops go off-rails and produce garbage → **Mitigation**: Quality gates run before merge. Failed gates = failed node, not merged. (Future: orchestration loop can add smarter supervision.)
- **Risk**: Dynamic graph growth creates cycles → **Mitigation**: Enforce DAG constraint on `add-node`. Reject edges that would create cycles.
- **Risk**: Code-review-graph Python API is not documented (MCP tools are the public surface) → **Mitigation**: Import and call internal modules directly. Pin version. Acceptable for early development.

## Implementation Notes

- **Package structure** (Sliced Bread):
  ```
  src/milknado/
  ├── __init__.py
  ├── cli.py              # Typer CLI entry point
  ├── domains/
  │   ├── graph/           # Mikado DAG (SQLite, node/edge/ownership operations)
  │   ├── planning/        # Agent session launch, code-review-graph queries
  │   ├── execution/       # Worktree management, ralph loop orchestration
  │   └── common/          # Shared types, config
  ├── adapters/
  │   ├── ralphify.py      # Thin wrapper around ralphify's RunManager
  │   ├── crg.py           # code-review-graph Python API adapter
  │   └── git.py           # Git/worktree operations
  └── app/
      └── cli_app.py       # DI wiring, Typer app assembly
  ```
- **Entry point**: `milknado = "milknado.cli:app"`
- **Build**: Hatchling (matching both upstream deps)
- **Config**: `milknado.toml` in project root
- Worktree naming: `milknado-<node-id>-<slug>` (e.g., `milknado-003-fix-auth-adapter`)

## Key Patterns (from research)

- **Mikado leaf-first execution** is a topological sort of the DAG executed in reverse — multiple sources confirm this (Manning book, Ellnestam & Brolund, multiple blog posts). [Evidence: 5+ independent sources]
- **Ralphify's RunManager** already supports multi-run orchestration with thread-safe start/stop/pause — exactly what milknado needs for parallel worktree loops. [Evidence: ralphify source code]
- **Code-review-graph's blast radius** (`get_impact_radius_tool`) identifies exactly which files/functions are affected by a change — ideal for scoping Mikado nodes, file ownership tracking, and detecting potential merge conflicts. [Evidence: code-review-graph MCP tool surface]
- **No existing Mikado execution tooling** — the method is universally practiced on paper/whiteboards. Milknado is novel here. [Evidence: Tavily search, no production tools found]

## Future Work (NOT v0.1)

> **NOTE — Preserve this section across implementation. It captures architectural intent that should not die with the spec.**

### Orchestration Loop (PAUL-style Unify)

Ralph workers are simple — they execute a prompt loop and signal completion or new prerequisites. If workers consistently fail to self-manage (bad decomposition, circular discoveries, poor merge quality), milknado may need a **smarter orchestration node** similar to the PAUL unify loop (github.com/ChristopherKahler/paul).

The PAUL pattern: **Plan → Apply → Unify**, where Unify is a mandatory reconciliation step that closes each work loop. Applied to milknado, this would mean:

- An orchestration ralph (smarter model, more context) that supervises worker ralphs
- After each node completion, the orchestrator reviews the merge, validates against the overall goal, and decides: continue, re-plan, or intervene
- The orchestrator could also handle re-decomposition when `add-node` signals indicate the original plan was wrong

**When to build this**: When v0.1 ships and we observe worker ralphs producing low-quality results or making poor `add-node` decisions. The SQLite DAG and event system are designed to support this — adding an orchestrator means adding a special node type, not restructuring.

**Design constraint**: The orchestrator must be a ralph loop itself (eating our own dog food), not a bespoke execution path.

### MCP Tool Injection

Instead of ralph loops calling `milknado add-node` via CLI, expose milknado's graph operations as MCP tools that are injected into the agent's tool set. This enables richer interaction: the agent can query graph state, check file ownership, and add nodes without shelling out.

### Single-Shot Planning Mode

Add `milknado plan --auto "goal"` that runs a single-shot prompt instead of a full interactive session. Useful for CI/CD pipelines and headless environments.

### Mergiraf Integration

When auto-rebase fails due to merge conflicts, use mergiraf (AST-aware structural merge) to attempt resolution before failing the node. This would significantly reduce manual conflict resolution.

### PR Creation

After all nodes merge into the feature branch, auto-create PR(s) via GitHub CLI. May want to split into multiple PRs based on Mikado subtree boundaries.

## Success Metrics

- A developer can go from goal → Mikado graph → parallel execution → merged feature branch with `milknado plan` + `milknado run`
- Leaf nodes execute truly in parallel (observable via `milknado status`)
- File ownership prevents merge conflicts between parallel nodes
- Dynamic graph growth works: a ralph adds a prerequisite, milknado handles it without human intervention
- Quality gates prevent bad merges

### Red/Green Paths

- **Green**: User runs `milknado plan "add caching layer"` → interactive session proposes 4-node Mikado graph with file ownership → user approves → `milknado run` → 2 leaves execute in parallel (non-overlapping files) → both merge → parent becomes ready → executes → root merges → done
- **Green (dynamic)**: During leaf execution, ralph discovers a missing interface → calls `milknado add-node --parent 2 --files src/ports/cache.py "extract CachePort interface"` → new leaf spawns → completes → original parent resumes
- **Green (ownership)**: Two leaves both need `auth.py` → milknado detects overlap → serializes them (one runs, other waits) → no merge conflict
- **Red**: Ralph loop fails quality gates → node marked failed → parent stays blocked → `milknado status` shows the failure → user investigates
- **Red**: `milknado add-node` would create a cycle → rejected with error → ralph loop continues without the new node
- **Red**: Auto-rebase hits conflict → node marked failed → user notified with conflict details

## Resolved Questions

- `milknado plan` launches a full interactive Claude session (not single-shot). Single-shot mode is future work.
- Planning agent uses CLI (`milknado add-node`) for graph operations. MCP tool injection is future work.
- Merge conflicts: auto-rebase, fail and notify on conflict. Mergiraf integration is future work.
- File ownership tracking is core to v0.1 — it's the mechanism that makes parallel execution safe.

## Assumptions

- ralphify's library API (`RunManager`, `run_loop`, event emitters) is stable enough for v0.1 (if wrong, wrap in adapter)
- code-review-graph's Python internals can be imported directly (if wrong, shell out to CLI)
- Git worktree operations are fast enough for interactive use (if wrong, pre-create worktree pool)
- A single machine can handle 4-8 parallel ralph loops without resource exhaustion (if wrong, add concurrency config)
- code-review-graph's blast radius accurately reflects file-level dependencies (if wrong, fall back to explicit user-declared ownership)
