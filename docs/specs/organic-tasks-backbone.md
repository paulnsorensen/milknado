---
slug: organic-tasks-backbone
status: draft
created: 2026-07-06
confidence: high
gates_overridden: []
agent_introduced_scope:
  - "worktree profile knob (per-node worktree policy)"  # approved via grill round 2 ledger
  - "milknado_goal_claim / milknado_goal_release tools"  # approved via sketch round
  - "artifact-existence completion evidence"             # approved via grill round 2 verdict
  - "session owner token via env var"                    # AGENT-DECIDED, presented, not vetoed
  - "wiring node"                                        # derived from user's "wire them together in the current checkout"
entity_referent_bindings:
  - {noun: "session", verdict: bound, referent: "goal_claims machinery", citation: "src/milknado/domains/graph/_persistence.py:513-569", note: "claim/release/pid-liveness exist; MCP exposure is new"}
  - {noun: "artifact", verdict: NEW ENTITY, referent: "MikadoNode.artifact_path (new field)", citation: "src/milknado/domains/common/types.py:55-74", note: "precedent: wiki_ref (types.py:73-74)"}
  - {noun: "flavor registry", verdict: bound, referent: "_parse_flavor_tables", citation: "src/milknado/domains/common/config.py:561-586", note: "keys widen TaskFlavor -> str"}
  - {noun: "prereq edge", verdict: bound, referent: "MikadoGraph.add_edge / edges table", citation: "src/milknado/domains/graph/graph.py:171-179", note: "cycle-checked; MCP exposure is new"}
  - {noun: "completion verification", verdict: bound, referent: "milknado_node_verify / _build_completion_verifier", citation: "src/milknado/mcp_node.py:191-230, src/milknado/adapters/loop.py:216-248"}
  - {noun: "preset pack", verdict: NEW ENTITY, referent: "TOML example in adoption contract doc", citation: "", note: "ships as documentation, not code"}
  - {noun: "adoption contract", verdict: NEW ENTITY, referent: "new markdown doc in this repo", citation: "", note: "easy-cheese-facing interface promises"}
---

# Organic tasks backbone

## Problem

milknado is wired into the easy-cheese ecosystem only as an optional parallel-execution backend (`/ultracook`, `/cheese-factory`); every other skill tracks work in flat `.cheese/<phase>/<slug>.md` files, and the session-level task surface is Claude Code's native Tasks — which are graph-shaped (blocks/blockedBy) but type-less and artifact-less (verified by observation, v2.1.201). There is no shared, typed, narrative-bearing task lifecycle: a session cannot express "this node is a review, its deliverable is this report, it depends on those two cook curds" anywhere. milknado's graph already has kind/flavor/status/prereq machinery; it lacks the small surface extensions and the documented contract to serve as the session task backbone.

## Goals

- Flavor registry: retire the `TaskFlavor` enum; `flavor` becomes a free string validated against BUILTIN ∪ TOML-declared names, rejected fail-fast at every write path.
- Artifact narratives: add `artifact_path` to nodes (opaque repo-relative markdown reference), settable at add-time and post-hoc.
- Prereq edges on the MCP surface: `prereqs=[node_ids]` on `todo_add`/`track_follow_up`, plus an emergency graph-ops CLI (`milknado edge add`).
- Session-as-GOAL: expose `milknado_goal_claim`/`milknado_goal_release` over the existing goal_claims machinery, with dead-pid claims reclaimable.
- Per-node worktree policy: new `worktree` flavor-profile knob + claim-time override; `todo_claim` provisions accordingly.
- Flavor-appropriate completion verification: `node_verify` runs in `worktree_path or project_root`; completion evidence is a stageable change (worktree nodes) or artifact existence (in-place artifact nodes); `None` gates stay fail-closed.
- Adoption contract doc: the easy-cheese-facing interface — native-Tasks mapping, phase-flavor preset pack, ultracook worked example, ingestion conventions, presence-gated adoption.

## Non-goals

- tmux status panel / needs-input signal (deferred sibling roadmap goal; known interim visibility gap accepted and named).
- `up:`/`down:`-prose wiki relations as graph edges (deferred to the roadmap-sync follow-up).
- Mirroring/syncing to the native Tasks list (rejected: lossy, undocumented env var, drift surface).
- `agent_type` parity for the subprocess dispatcher (native Workflow backend only, as today).
- `milknado_ground` / MCP proxying (issue #154).
- Migration code, deprecation shims (pre-release).
- easy-cheese-side implementation (bound by the contract doc; implemented in that repo later).

## Approach

Extend, don't rebuild: every capability lands as a thin widening of machinery that already exists. Flavor semantics are already profile-driven (only 3 enum-member references in src, none business-logic); the registry change is enum→str + one validation set. Dependencies already exist as containment + extra cycle-checked edges (the batching bridge writes them today); `prereqs` just exposes that. Sessions ride goal_claims (claim/release/pid functions all present). Narratives follow the `wiki_ref` precedent: an opaque reference to markdown on disk — the DB owns graph/status, files own prose. The vocabulary of phase flavors (brainstorm/explore/agent-review/harden/fix + re-profiled spec/implement/research) ships as an easy-cheese preset-pack TOML documented in the adoption contract, not as milknado built-ins — milknado stays workflow-agnostic. Replacement of native Tasks is presence-gated: milknado present → sole task surface; absent → existing fallbacks.

## Decisions

- Replace native Tasks outright (presence-gated) — mirror is lossy + rides undocumented surface; coexist splits truth. (ADR-001)
- Narratives = `artifact_path` markdown references, opaque, no existence enforcement — cross-checkout truth is unknowable; wiki_ref precedent. (ADR-002)
- Session-as-GOAL on goal_claims — machinery exists; per-session lists would add a second "which list" indirection. (ADR-003)
- Flavor vocabulary: mechanism in milknado, vocabulary in easy-cheese preset pack — milknado stays workflow-agnostic. (ADR-004)
- Per-node worktree policy (profile knob + claim override); every agent dispatch is a claimed node — user-corrected from session-wide mode. (ADR-005)
- Completion evidence is flavor-appropriate: change-based for worktree nodes, artifact-existence for in-place artifact nodes. (ADR-006)
- _Minor decisions:_ prereqs as `todo_add` param (not a standalone MCP edge tool) + emergency CLI per user; findings ingestion = per-finding leaves severity-gated (blocker/high) under a cure GOAL; artifact settable at add + `edit_node`; owner token inherited by sub-agents via env var (MILKNADO_NODE_ID precedent); graph hygiene = keep-all after GOAL closes, prune via graph-ops CLI; ultracook mapping is the contract's worked example, normative binding lands easy-cheese-side; `worktree` is the 9th FlavorOverride knob.

## Acceptance

- WHEN a TOML declares `[milknado.flavor.<custom-name>]` THE SYSTEM SHALL accept `flavor="<custom-name>"` on todo_add/track_follow_up/edit_node and resolve its profile with the existing precedence (defaults → global → local).
- WHEN a write path receives a flavor not in BUILTIN ∪ declared THE SYSTEM SHALL reject with an error naming the unknown flavor and the valid set.
- WHEN `todo_add`/`track_follow_up` receive `artifact` and/or `prereqs` THE SYSTEM SHALL persist `artifact_path` verbatim and create cycle-checked prereq edges.
- WHEN `edit_node` receives `artifact` THE SYSTEM SHALL update `artifact_path` without existence validation.
- WHEN `milknado_goal_claim(goal_id, owner)` succeeds THE SYSTEM SHALL fence the subtree via existing goal_claims semantics; the claim releases on the goal's terminal transition or explicit `milknado_goal_release`.
- WHEN a goal claim's recorded pid is no longer alive THE SYSTEM SHALL treat the claim as reclaimable on the next claim attempt.
- WHEN a node's resolved profile has `worktree=false` (or the claim call overrides) THE SYSTEM SHALL claim it RUNNING without provisioning a worktree.
- WHEN `node_verify` runs for a node without a worktree THE SYSTEM SHALL execute gates in the project root.
- WHEN `node_verify` runs with explicitly-empty gates for an in-place node bearing `artifact_path` THE SYSTEM SHALL pass iff the artifact file exists and is non-empty; worktree nodes retain the stageable-change requirement.
- WHEN `quality_gates` is None THE SYSTEM SHALL fail verification closed (unchanged behavior).
- WHEN `milknado edge add <parent> <child>` runs THE SYSTEM SHALL create the edge with cycle validation.
- WHEN the adoption contract doc ships THE SYSTEM SHALL contain: the native-Tasks→milknado mapping table, the phase-flavor preset-pack TOML, the ultracook worked example (multi-curd worktrees + in-place wiring node), wheypoint + severity-gated findings ingestion conventions, presence-gated adoption language, and the named interim visibility gap.
- [prose-fallback] Spike (verification item): a scratch `.claude/agents/<name>.md` agent def dispatches successfully when a claimed node's `agent_type=<name>` reaches Workflow `agent()` — proves bare-name resolution end-to-end before the preset pack relies on it.

## Interface sketches

```pseudocode
# types.py — enum retired (pre-release, no shims)
BUILTIN_FLAVORS = frozenset({"implement", "spec", "spike", "prototype", "research"})
MikadoNode.flavor: str | None          # TASK default "implement"
MikadoNode.artifact_path: str | None   # NEW; opaque repo-relative markdown ref

# config.py
_parse_flavor_tables(...) -> dict[str, FlavorOverride]   # str keys
MilknadoConfig.flavor_registry: frozenset[str]           # BUILTIN | declared
FlavorOverride.worktree: bool | None                     # NEW 9th knob
FlavorProfile.worktree: bool                             # default True (implement-style)

# MCP surface
milknado_todo_add(description, kind, parent_id, files, flavor, artifact=None, prereqs=[])
milknado_track_follow_up(..., artifact=None, prereqs=[])
milknado_edit_node(..., artifact=...)
milknado_goal_claim(goal_id, owner) / milknado_goal_release(goal_id, owner)   # NEW; wraps claim_goal_row/release_goal_row; owner env-inherited by sub-agents
milknado_todo_claim(node_id, worktree=None)   # None -> profile default; False -> in-place claim (no worktree, runs row + fence kept)
milknado_node_verify(run_id)                  # cwd = worktree_path or project_root; evidence per ADR-006

# CLI (emergency graph-ops)
milknado edge add <parent_id> <child_id>
```

## Risks

- Interim visibility gap: no harness UI renders the graph until the panel sibling goal lands; mitigation — `todo_tree` summaries at session boundaries, named in the contract.
- Registry rejection in repos lacking the preset pack; mitigation — global `~/.config/milknado/milknado.toml` merge distributes presets once.
- Dangling `artifact_path` (worktree-written files may never reach the main checkout); accepted — opaque semantics, documented convention.
- Enum retirement breaks the Python API for external callers; accepted — pre-release.
- Two coordinator types (interactive session, detached run_loop) share goal fencing; existing CAS semantics arbitrate, but concurrent-claim UX (loser behavior) is defined as fail-loud.

## Open questions

- [TBD] Should `graph_summary`/`todo_tree` annotate prereq edges (tree view shows containment only)? Default: brief-level rendering suffices; revisit on use.
- [TBD] Preset-pack flavor names final vocabulary (brainstorm/explore/agent-review/harden/fix) — settled at contract-doc write, easy-cheese may rename.

## Quality gates

- `just check-llm`: PASS line (lint + format + tests + project & diff coverage ≥95%).

## References

- Verification research (four verdicts, claim tables): `.cheese/research/organic-tasks-verification/organic-tasks-verification.md`
- Prior research (typed skill-chain patterns): `.cheese/research/skills-lifecycle-task-replacement/skills-lifecycle-task-replacement.md`
- Session handoff: `.cheese/notes/milknado-organic-tasks-exploration.md`
