---
name: milknado-config
description: >
  Configure milknado's per-flavor worker execution by interviewing the user and
  writing milknado.toml — agent_family, [milknado.worker.tools] allowlists, and
  [milknado.flavor.*] tables for named flavor presets (implement, spec, spike,
  prototype, research). Use when the user wants to set up or tune how milknado
  runs workers — phrases like "configure milknado flavors", "set up
  milknado.toml", "tune the research/spec/spike agent", "which model should
  milknado use for prototypes", "skip quality gates for spikes", "milknado
  worker tools", "per-flavor agent config". Interviews for agent family, then
  per-flavor model/agent, tools, quality gates, and brief prepend; writes the
  tables and validates them with `milknado agents check` plus a flavor
  resolution probe. Do NOT use for running graphs (milknado run) or planning
  (milknado plan).
---

# milknado-config

Set up `milknado.toml` so each **task flavor name** runs with the right agent, tools,
quality gates, briefing, and worktree policy. Interview the user, write the tables,
validate.

## When to use

- The user wants milknado workers tuned per flavor (e.g. research reads-only,
  prototypes skip strict gates, specs run a stronger model).
- The user is bootstrapping `milknado.toml` and wants more than the bare default.
- The user asks "what knobs do flavors have" or "how do I make spikes faster".

**Not for:** running the graph (`milknado run`), planning (`milknado plan`), or
graph/node edits. This skill only writes config.

## The model in 60 seconds

A milknado **task** node carries a string `flavor` name: `implement` (the default),
`spec`, `spike`, `prototype`, or `research`. Custom names are also valid. At dispatch
time milknado resolves a **flavor profile** — `execution_agent`, `quality_gates`,
`brief_prepend`, and `worktree` — and runs the worker with it. The profile comes
from `[milknado.flavor.<flavor>]` in `milknado.toml`, falling back to the global
worker config when a flavor sets nothing.

Resolution precedence (per field — see `flavor_profile.py`):

- **execution_agent**: flavor `execution_agent` (explicit full CLI) → else flavor
  `tools` derived against the **family built-in default** → else the config-level
  `execution_agent`.
- **quality_gates**: flavor value if set (`[]` means *skip gates*) → else config
  default.
- **brief_prepend**: flavor value if set (**replaces**, not concatenates) → else
  config `prompts.worker_brief_prepend`.
- **worktree**: flavor value if set → else `[milknado] worktree` (default `true`).

`agent_family` is one of `claude | cursor | gemini | codex`; it picks the default
planning + execution commands and (for `claude`/`gemini`) the tool allowlist.

## Config location & layering

- Project config: `milknado.toml` at the project root (`milknado init` creates a
  default if missing).
- Optional user-global: `$XDG_CONFIG_HOME/milknado/milknado.toml` (else
  `~/.config/milknado/milknado.toml`), merged **under** the project file
  (project wins). Global `brief_prepend_path` is resolved against the global config
  dir. Local `brief_prepend_path` and prompt `*_prepend_path` values are confined
  to `project_root`; an absolute path that escapes it raises a clear `ValueError`.
  `project_root`, `db_path`, `plugins` are ignored if set globally.

Decide with the user whether a flavor preset is **project-specific** (most cases
— gates are project commands) or a **personal default** for the global file.

## Interview flow

Work top-down. Prefer a structured prompt (e.g. AskUserQuestion) so the user
picks from options; accept free-form too. Don't ask about flavors the user
doesn't want to touch — `implement` usually needs no override at all.

1. **Locate / create the config.** If `milknado.toml` is absent, run
   `milknado init` (or hand-write a minimal `[milknado]` table). Read the current
   file first — you will edit it in place, preserving comments and existing keys.
2. **Agent family.** Confirm `agent_family` (`claude` default). This sets the
   default models and which families can restrict tools (only `claude` and
   `gemini` enforce a CLI tool allowlist; `cursor` and `codex` have **no**
   per-tool restriction — their `tools` list won't gate anything, so steer those
   users to `execution_agent`/`quality_gates` instead).
3. **Baseline worker tools** (optional). `[milknado.worker.tools.<family>]` is a
   single list. Bare list **replaces** the built-in allowlist; include the `"..."`
   sentinel once to mean "the built-in default, plus these". Use this for tools
   *every* flavor should have (e.g. add `"Bash(just:*)"` so workers can run the
   project's recipes).
4. **Per flavor** (`spec`, `spike`, `prototype`, `research`, and `implement` only
   if they want to change the default): for each one the user wants to tune, ask:
   - **Model / agent** — keep the family default, or override? A model bump
     (e.g. opus for `spec`) needs a full `execution_agent` string, which **opts
     out of the tool allowlist**. To keep the allowlist, change `tools` instead
     (model stays the family default).
   - **Tools** — inherit, or a flavor-specific list? (`research` → read-only +
     `WebSearch`, dropping edit tools; `spike` → add `WebSearch` via `"..."`).
   - **Quality gates** — inherit, a custom list, or `[]` to skip (typical for
     `spec`/`research`/`spike` where there's no code to test).
   - **Brief prepend** — inline text or `brief_prepend_path` (a file, or a list of
     files concatenated) that tells the worker how to behave for this flavor.
5. **Write** the `[milknado.worker.tools]` and `[milknado.flavor.*]` tables into
   `milknado.toml` (see "Writing" below).
6. **Validate** (see "Validation"). Show the user the resolved per-flavor profile
   and confirm it matches intent.

See `references/flavor-presets.md` for recommended presets per flavor and a full
worked `milknado.toml`. Offer those as the starting point, then tailor.

## The grammar (quick reference)

```toml
[milknado]
agent_family = "claude"

# Family-wide worker allowlist. Bare list REPLACES the built-in default;
# "..." expands in place to the built-in default (at most one sentinel).
[milknado.worker.tools]
claude = ["...", "Bash(just:*)"]

# Per-flavor override. Every field optional; absent = inherit.
[milknado.flavor.research]
quality_gates = []                       # [] = skip; omit = inherit
tools = ["mcp__serena__get_symbols_overview", "mcp__serena__find_symbol", "Read", "Grep", "Glob", "WebSearch"]  # replaces default
brief_prepend = "Research only. Investigate and report; do not modify code."
# brief_prepend_path = "agents/research.md"   # mutually exclusive with brief_prepend

[milknado.flavor.spec]
execution_agent = "claude --model opus -p"   # explicit full CLI; opts out of allowlist
quality_gates = []
```

## Writing

- **Edit `milknado.toml` in place** — add/replace the `[milknado.worker.tools]`
  and `[milknado.flavor.*]` tables; keep the user's other keys and comments. Do
  not regenerate the whole file (that would strip comments).
- One `[milknado.flavor.<flavor>]` table per flavor you're configuring. Only emit
  fields that differ from the inherited default — an empty table is a no-op.
- `brief_prepend` and `brief_prepend_path` are **mutually exclusive** in the same
  table. `brief_prepend_path` may be a string or a list of strings (files joined
  with a blank line).

## Validation

After writing, confirm the file parses and the profiles resolve as intended:

```bash
# Base resolution + that the file loads (family / planning / execution command):
milknado agents check

# Per-flavor resolution probe — prints the resolved profile for every flavor:
uv run python - <<'PY'
from pathlib import Path
from milknado.domains.common.config import load_config
from milknado.domains.common.flavor_profile import resolve_flavor_profile
from milknado.domains.common.types import BUILTIN_FLAVORS

cfg = load_config(Path("milknado.toml"))
for name in [*sorted(BUILTIN_FLAVORS), "custom"]:
    p = resolve_flavor_profile(cfg, name)
    print(f"{name:10} agent={p.execution_agent!r}")
    print(
        f"{'':10} gates={list(p.quality_gates)} "
        f"brief={'set' if p.brief_prepend else 'none'} worktree={p.worktree}"
    )
PY
```

`load_config` raises on malformed tables (unknown flavor key, disallowed
executable, bad sentinel count, missing `brief_prepend_path`, non-list gates) —
a clean run means the config is well-formed. If it raises, read the message, fix
the offending table, re-run.

## Gotchas / invariants

- **`execution_agent` opts out of the allowlist.** A full CLI string is run
  verbatim; the `tools` allowlist no longer applies. Must start with one of
  `claude`, `codex`, `cursor-agent`, `gemini` (basename checked).
- **Use flavor names, not a retired enum.** The runtime accepts the string registry
  (`BUILTIN_FLAVORS`) plus project-specific names; documentation and probes should
  iterate names and call `resolve_flavor_profile`.
- **`tools`-derived model is the family default**, not opus — derivation uses the
  built-in family command (sonnet for claude, flash for gemini). Want a bigger
  model *and* a tool list? You must spell the whole command in `execution_agent`.
- **`tools` only restricts `claude` and `gemini`.** `cursor-agent` headless and
  `codex` have no CLI tool-allowlist flag, so a `tools` list is inert for them.
- **`quality_gates = []` skips gates; omitting the key inherits.** They're
  different — be explicit about which the user wants.
- **No `quality_gates` at the config level means fail-closed.** If `[milknado]
  quality_gates` is absent from `milknado.toml`, every node run is rejected with
  a message explaining what to add. `milknado init` auto-detects the project type
  and writes explicit gates (see "Auto-detection table" below). Use `quality_gates
  = []` to intentionally skip; never leave the key absent expecting a skip.
- **`brief_prepend` replaces**, it does not append to the config-level
  `worker_brief_prepend`.
- **Worktree fallback is config-level.** `[milknado] worktree = false` disables
  worktrees for flavors that do not override it; set `worktree = true` or `false`
  in a flavor table for an explicit per-flavor policy.
- **At most one `"..."` sentinel** per tool list.
- **A flavor's `"..."` expands the BUILT-IN family allowlist, not your
  `[milknado.worker.tools]` override.** So if you added `"Bash(just:*)"`
  family-wide and a flavor re-specifies `tools = ["...", "WebSearch"]`, that
  flavor does **not** inherit `Bash(just:*)` — re-add it in the flavor's list if
  it's needed there.

## Gate grammar

Each entry in `quality_gates` is either a bare string (command only) or a
`{command, fail_on_stdout}` inline table. The inline-table form fails the gate
even on exit 0 when combined stdout+stderr matches the regex — for tools like
Godot headless that always exit 0:

```toml
[milknado]
quality_gates = [
  "uv run pytest",                                    # bare string — fails on non-zero exit
  "uv run ruff check",
  {command = "godot --headless --quit", fail_on_stdout = "SCRIPT ERROR|FAILED"},
]
```

`fail_on_stdout` is a Python `re.search` pattern matched against the combined
stdout+stderr output. The gate fails when it matches.

## Auto-detection table

`milknado init` probes for project-type markers (first match wins) and writes
explicit gates into `milknado.toml`. The interview step should confirm or tune
what detection chose:

| Marker file | Detected gates |
|---|---|
| `pyproject.toml` | `uv run pytest`, `uv run ruff check`, `uv run basedpyright` |
| `Cargo.toml` | `cargo test`, `cargo clippy -- -D warnings` |
| `package.json` | `npm test` |
| `go.mod` | `go test ./...`, `go vet ./...` |
| `project.godot` | `godot --headless --quit` with `fail_on_stdout = "SCRIPT ERROR\|FAILED"` |
| (none matched) | No gates written — runs fail-closed until you add them |

For mixed-stack repos (e.g. Rust backend + JS frontend), check which markers
exist and propose combining rows — the detection takes the first match, so the
skill may need to add the second language's gates manually.
