# Flavor presets & worked config

Recommended starting points for each `TaskFlavor`. These are defaults to offer,
not law — tailor models, tools, and gates to the user's project.

`[milknado] quality_gates` is **required** — if absent, every node run fails
closed with a message telling you to add it. Run `milknado init` to auto-detect
gates from the project type (pyproject.toml → uv/ruff/ty; Cargo.toml → cargo
test + clippy; etc.), then tune from there. Quality-gate commands in the examples
below are from the milknado project itself; swap in the consuming project's real
commands.

## What each flavor is for

| Flavor | Intent | Produces | Gates | Tools | Model |
|---|---|---|---|---|---|
| `implement` | Production change to satisfy a task | Tested code | **Full** (inherit) | Default allowlist | Family default (sonnet) |
| `spec` | Turn a fuzzy goal into a written spec/design | A markdown spec, not code | **Skip** (`[]`) | Read + Write (default ok) | Often a stronger model (opus) |
| `spike` | Time-boxed throwaway to answer a question | Throwaway code + an answer | **Skip** or lint-only | Default `+ WebSearch` | Family default |
| `prototype` | Rough working code, not production-grade | Working-but-loose code | **Loosened** (lint only) | Default `+ WebSearch` | Family default |
| `research` | Investigate and report; no code changes | Findings / a written report | **Skip** (`[]`) | **Read-only** + `WebSearch` | Family default (or opus) |

Key trade-off to surface every time: **a model bump requires a full
`execution_agent` string, which opts out of the tool allowlist.** If keeping the
deny-by-default allowlist matters more than the model, change `tools` instead and
leave the model at the family default.

Second subtlety: a flavor's `tools` `"..."` sentinel expands the **built-in**
family allowlist (`WORKER_ALLOWED_TOOLS[family]`), **not** the
`[milknado.worker.tools]` family override. A `Bash(just:*)` you added family-wide
is therefore *not* inherited by a flavor that re-specifies `tools` — re-list it
in that flavor if the worker needs it.

## Per-flavor TOML (claude family)

### implement — usually no override

Leave it out entirely. It inherits the config-level execution agent and gates.

### spec — stronger reasoning, no test gates

```toml
[milknado.flavor.spec]
# opus for design reasoning. Full CLI ⇒ allowlist no longer applies, so the
# worker can read/write freely to produce the spec doc.
execution_agent = "claude --model opus -p"
quality_gates = []
brief_prepend = """
You are writing a SPEC, not production code. Produce a clear, testable design
document (problem, approach, public seams, acceptance criteria). Do not modify
application code beyond the spec file.
"""
```

Allowlist-preserving alternative (stays on sonnet):

```toml
[milknado.flavor.spec]
quality_gates = []
brief_prepend_path = "agents/spec-brief.md"
```

### spike — throwaway exploration

```toml
[milknado.flavor.spike]
quality_gates = []                  # throwaway; don't block on the test suite
tools = ["...", "WebSearch"]        # family default + web lookup
brief_prepend = """
This is a time-boxed SPIKE. Optimize for learning, not polish. Answer the
question, leave a short note on what you found, and flag anything that should
become a real task.
"""
```

### prototype — rough working code, loosened gates

```toml
[milknado.flavor.prototype]
quality_gates = ["uv run ruff check"]   # lint only; skip strict types + full tests
tools = ["...", "WebSearch"]
brief_prepend = "Prototype quality: make it work end-to-end. Skip exhaustive edge cases and polish."
```

### research — read-only, report, no edits

```toml
[milknado.flavor.research]
quality_gates = []
# Bare list REPLACES the default allowlist — drops Edit/Write so the worker
# cannot modify code. Keep deposit_result so it can hand findings back.
tools = [
  "mcp__tilth__tilth_read",
  "mcp__tilth__tilth_search",
  "mcp__serena__get_symbols_overview",
  "mcp__serena__find_symbol",
  "mcp__serena__find_referencing_symbols",
  "Read",
  "Grep",
  "Glob",
  "WebSearch",
  "mcp__milknado__milknado_deposit_result",
]
brief_prepend = """
RESEARCH ONLY. Investigate and report — do not modify code. Deposit your
findings (sources, options, recommendation with a confidence level) as the
result.
"""
```

## Full worked `milknado.toml`

A complete project config (claude family, uv/just Python project). The user
substitutes their own gate commands and brief text.

```toml
[milknado]
agent_family = "claude"
concurrency_limit = 4
protected_branches = ["main", "master"]
# Required — absent = fail-closed. Use `milknado init` to auto-detect.
quality_gates = ["uv run pytest", "uv run ruff check", "uv run ty check"]

# Every worker also gets the project's just recipes.
[milknado.worker.tools]
claude = ["...", "Bash(just:*)"]

# implement: inherits everything (no table needed).

[milknado.flavor.spec]
execution_agent = "claude --model opus -p"
quality_gates = []
brief_prepend = "Write a testable spec; do not modify application code."

[milknado.flavor.spike]
quality_gates = []
tools = ["...", "WebSearch"]
brief_prepend = "Time-boxed spike. Optimize for learning; leave a findings note."

[milknado.flavor.prototype]
quality_gates = ["uv run ruff check"]
tools = ["...", "WebSearch"]

[milknado.flavor.research]
quality_gates = []
tools = ["mcp__tilth__tilth_read", "mcp__tilth__tilth_search", "Read", "Grep", "Glob", "WebSearch", "mcp__milknado__milknado_deposit_result"]
brief_prepend = "Research only — investigate and report; do not modify code."
```

## Family notes

- **claude** — default execution `claude --model sonnet -p --allowedTools '<csv>'`;
  default planning opus. `tools` allowlist is enforced. Tool names are MCP-prefixed
  (`mcp__tilth__*`, `mcp__serena__find_symbol`, …) plus host `Read`/`Edit`/`Write`/
  `Glob`/`Grep`/`MultiEdit`, `Bash(rtk:*)`, `WebSearch`.
- **gemini** — default execution flash with `--allowed-tools`; planning
  `gemini-3.1-pro-preview`. Allowlist enforced; tool names are **bare**
  (`tilth_search`, `find_symbol`, …) and shell is `ShellTool(<pattern>)`.
- **cursor** — `cursor-agent --model sonnet -p`. **No** CLI tool allowlist
  (headless `--allowedTools` unsupported); a `tools` list is inert. Use
  `execution_agent` / `quality_gates` to differentiate flavors.
- **codex** — `codex exec --model gpt-5.4-mini --sandbox workspace-write`. Sandbox
  scopes **files**, not tools; no tool-allowlist flag, so `tools` is inert. Use
  `execution_agent` / `quality_gates`.

## Source of truth

- Flavors: `src/milknado/domains/common/types.py` (`TaskFlavor`).
- Parsing + validation + save/load: `src/milknado/domains/common/config.py`
  (`FlavorOverride`, `_parse_flavor_entry`, `save_config`).
- Resolution precedence: `src/milknado/domains/common/flavor_profile.py`
  (`resolve_flavor_profile`).
- Family defaults + allowlists: `src/milknado/domains/common/agent_argv.py`
  (`WORKER_ALLOWED_TOOLS`, `_default_execution_command`).
- Dispatch-time application: `run_loop/__init__.py` `_dispatch_batch`,
  `mcp_run.py`, `mcp_todo.py`.
