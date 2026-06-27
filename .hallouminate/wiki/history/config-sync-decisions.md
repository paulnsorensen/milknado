# Config Sync — Design Decisions

Date: 2026-06-27 · Status: all four accepted; ADR-001 spike confirmed.

Per-flavor agent-def generation (`milknado config sync`) rests on four decisions
that were made together during the config-sync spec session. See
[[architecture/config-sync]] for what shipped; [[workflow-executor-decision]] for
the native Workflow backend these defs target.

---

## ADR-001: Proceed on doc-prior; defer (then confirm) the Workflow spike

**Status: accepted → spike CONFIRMED (2026-06-27)**

### Context

The whole generator only matters if the Workflow `agent({agentType})` primitive
actually honors the generated def's `tools`, `model`, and `maxTurns` frontmatter.
At spec time this was empirically unconfirmed. Claude Code's agent-teams docs
explicitly state *"the teammate honors that definition's tools allowlist and
model"* and `maxTurns` is a documented frontmatter field — a strong doc-prior.

### Decision

Build on the doc-prior; treat the empirical Workflow-path spike as a fast-follow,
not a blocker.

### Spike result — CONFIRMED

The spike ran after the implementation shipped. Deterministic result: PASS.

**Confirmed chain (no code change needed to use it):**

- `[milknado.flavor.<f>].agent_type` in milknado.toml → `FlavorOverride.agent_type`
  (`src/milknado/domains/common/config.py`) → `resolve_flavor_profile`
  (`src/milknado/domains/common/flavor_profile.py`) → `milknado_todo_claim` payload
  returns `agent_type` + restricted `tools`.
- `node-runner.js:97` spawns `agent({ agentType: c.agent_type || 'milknado:milknado-worker' })`,
  passing NO `tools` parameter — so the def file the `agentType` resolves to IS the
  effective allowlist.

**Spike procedure:** scratch project with
`[milknado.flavor.implement] agent_type="milknado-worker-implement"` + a
restrictive tools list (Write excluded). `config sync` emitted
`.claude/agents/milknado-worker-implement.md` (tools: 6, Write absent; model:
sonnet; maxTurns: 60). `milknado_todo_claim` returned `agent_type=
"milknado-worker-implement"` with exactly those restricted tools. All assertions
green.

**Harness honoring verified against Claude Code docs
(<https://code.claude.com/docs/en/sub-agents>):** project-local `agentType` = the
bare `name` field in frontmatter (no namespace; plugin agents are scoped, project-
local aren't). `tools`/`model`/`maxTurns` are strictly honored — Write is
unavailable to the worker if not listed.

### Caveats

1. **Session-start discovery.** Subagent defs load at session start; a mid-session
   disk write requires a restart or `/agents` reload before it takes effect.
   `milknado config sync` already prints this hint on every non-dry-run write
   (`src/milknado/cli_config.py:94-96`).
2. **`model` can be overridden per `agent()` invocation** (per the CC resolution
   order), but `tools` has no such override path — the def fully governs the
   allowlist. Flavor profiles keep `model` consistent via `execution_agent` parsing.

### Actionable follow-up (not yet done)

`config sync` generates the def files but does NOT auto-set `agent_type` in
milknado.toml — so the node-runner still defaults to the static
`milknado:milknado-worker` and the generated defs go unused until
`agent_type = "milknado-worker-<flavor>"` is set per flavor. Two closure options:

- (a) Users set it manually after running `config sync`.
- (b) `sync()` auto-writes `agent_type` into the flavor tables — roughly a 1-line
  enhancement, recommended.

---

## ADR-002: Generated def owns tools; prompt stays via MCP

**Status: accepted**

### Context

`agent()` takes no `tools` parameter and has no honored prompt channel. The def
frontmatter is the only place per-flavor tools/turns can live on the native
Workflow path. However, the prompt/persona arrives dynamically via the MCP
claim-payload and `FlavorOverride.brief_prepend` — baking it into the def would
create a stale duplicate and prevent per-node customization.

### Decision

The generated def carries only the tool allowlist, `model`, and `maxTurns`. The
prompt stays dynamic via MCP. Codex carries a generic `developer_instructions`
stub because its TOML format requires the field; the real prompt still arrives via
MCP at dispatch.

### Consequences

Defs stay thin and stable — resync is only needed when tool lists or model
preferences change, not when prompt wording evolves. Codex defs are slightly
heavier than CC/opencode by necessity, not preference.

---

## ADR-003: Per-harness fidelity tiers (resolve → classify → project)

**Status: accepted**

### Context

Only Claude Code supports a named-tool allowlist in its agent def. opencode uses
permission categories (`edit`/`write`/`bash`); Codex has no per-tool allowlist at
all — only OS-level `sandbox_mode`. The milknado/CC tool names
(`mcp__serena__*`, `Bash(rtk:*)`) do not translate to those harnesses.

### Decision

Resolve each flavor's tool list → classify write-vs-read → project onto each
harness's available channel:

| Harness | Fidelity | Mechanism |
|---|---|---|
| Claude Code | exact | literal tool names in `tools:` frontmatter |
| opencode | capability | `permission.{edit,write,bash}` allow/deny |
| Codex | coarse | `sandbox_mode = "read-only"` or `"workspace-write"` |

**Write-capable classifier** (`_is_write_capable`, `_identity.py:41-52`): true if
the resolved tool list contains any of: `Edit`, `Write`, `tilth_edit`,
`write_file`, `edit_file`, `Bash(...)`, `ShellTool(...)`, or any Serena
write-symbol op (`replace_symbol_body`, `insert_before_symbol`,
`insert_after_symbol`, `replace_content`, `rename_symbol`, `safe_delete_symbol`)
in either bare or `mcp__serena__<op>` form.

`danger-full-access` is never emitted. Codex restriction is coarse by design —
there is no per-tool allowlist available.

### Related

`pyyaml` renders CC and opencode YAML frontmatter; `tomli_w` renders Codex TOML.
Both were already direct milknado deps (vendored via the ralphify loop engine —
see [[nih-decisions]] and [[loop-vendor-in-decision]]).

---

## ADR-004: Default `--harness` = all three

**Status: accepted**

### Context

CC is the only def milknado's native Workflow path consumes; opencode/codex are
user-facing portability. A CC-only default would leave cross-harness users with an
extra flag on every sync.

### Decision

Default emits all three harnesses; `--harness` narrows. One command yields
portable identities across CC, opencode, and Codex.

### Consequences

A `config sync` with no flags may write coarse-fidelity Codex TOML files the user
never invokes. The `--harness` flag is the escape valve. Detect-installed was
rejected for added detection logic and implicit behavior.
