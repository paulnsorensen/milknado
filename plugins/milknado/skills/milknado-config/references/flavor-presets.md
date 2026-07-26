# Flavor presets & worked config

Recommended starting points for each built-in flavor name. These are defaults to
offer, not law — tailor models, tools, gates, and worktree policy to the project.

`[milknado] quality_gates` is **required** — if absent, every node run fails
closed with a message telling you to add it. Run `milknado init` to auto-detect
gates from the project type (pyproject.toml → uv/ruff/ty; Cargo.toml → cargo
test + clippy; etc.), then tune from there. Quality-gate commands in the examples
below are from the milknado project itself; swap in the consuming project's real
commands.

## What each flavor is for

| Flavor name | Intent | Produces | Gates | Tools | Model | Worktree |
|---|---|---|---|---|---|---|
| `implement` | Production change to satisfy a task | Tested code | **Full** (inherit) | Default allowlist | Family default (sonnet) | Inherit |
| `spec` | Turn a fuzzy goal into a written spec/design | A markdown spec, not code | **Skip** (`[]`) | Read + Write (default ok) | Often a stronger model (opus) | Inherit |
| `spike` | Time-boxed throwaway to answer a question | Throwaway code + an answer | **Skip** or lint-only | Default `+ WebSearch` | Family default | Inherit |
| `prototype` | Rough working code, not production-grade | Working-but-loose code | **Loosened** (lint only) | Default `+ WebSearch` | Family default | Inherit |
| `research` | Investigate and report; no code changes | Findings / a written report | **Skip** (`[]`) | **Read-only** + `WebSearch` | Family default (or opus) | Inherit |
| `review` | Read-only adversarial review | Severity-grouped findings | **Skip** (`[]`) | Read-only + deposit | Family default (or opus) | Usually false |
| `plate` | Terminal commit / publish | Commit SHA or PR URL | **Full** (project gates) | Write + GitHub | Family default | Usually true |

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

`[milknado] worktree` is the fallback for every flavor. Set `worktree = false`
in the project table to avoid detached worktrees by default, or set it explicitly
inside a flavor table when one flavor needs a different policy. Local
`brief_prepend_path` values and prompt `*_prepend_path` values must resolve under
`project_root`; escaping absolute paths fail with `ValueError`. Global path values
are resolved relative to the global config directory.

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

### review — read-only findings

```toml
[milknado.flavor.review]
agent_type = "reviewer"
session_mode = "fresh"
review = true
review_max_rounds = 2
on_reject = "block"
worktree = false
quality_gates = []
brief_prepend = """
Review the work against its brief and spec. Produce severity-grouped findings with
path/line evidence and a fix recommendation. Do not modify application code.
Deposit the complete findings report.
"""
```

### plate — terminal commit and publish

```toml
[milknado.flavor.plate]
agent_type = "plate"
session_mode = "fresh"
review = false
on_reject = "block"
worktree = true
quality_gates = ["just check-llm"]
brief_prepend = """
This is terminal plating. Run the configured gate, commit with a Conventional
Commit message, and open or update the pull request. Deposit the commit SHA or
PR URL and a shipping summary.
"""
```

`review` and `plate` are milknado built-in names. Easy-cheese's `agent-review`
and `fix` names are external phase vocabulary from `docs/adoption-contract.md`;
they are not aliases for these runtime profiles.

The codec also accepts `review_agent` for an explicit review command,
`review_max_rounds` as a positive integer, `session_mode = "fresh"|"resume"`, and
on_reject = "block"|"warn"`. Use `session_mode = "resume"` only with a
verified resumable agent.

## Full worked `milknado.toml`

A complete project config (claude family, uv/just Python project). The user
substitutes their own gate commands and brief text.

```toml
[milknado]
agent_family = "claude"
concurrency_limit = 4
worktree = true                 # fallback; flavors may override it
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
tools = ["mcp__serena__get_symbols_overview", "mcp__serena__find_symbol", "Read", "Grep", "Glob", "WebSearch", "mcp__milknado__milknado_deposit_result"]
brief_prepend = "Research only — investigate and report; do not modify code."
```

## Family notes

- **claude** — default execution `claude --model sonnet -p --allowedTools '<csv>'`;
  default planning opus. `tools` allowlist is enforced. Tool names are MCP-prefixed
  (`mcp__serena__get_symbols_overview`, `mcp__serena__find_symbol`, …) plus host
  `Read`/`Edit`/`Write`/`Glob`/`Grep`/`MultiEdit`, `Bash(rtk:*)`, `WebSearch`.
- **gemini** — default execution flash with `--allowed-tools`; planning
  `gemini-3.1-pro-preview`. Allowlist enforced; tool names are **bare**
  (`get_symbols_overview`, `find_symbol`, …) and shell is `ShellTool(<pattern>)`.
- **cursor** — `cursor-agent --model sonnet -p`. **No** CLI tool allowlist
  (headless `--allowedTools` unsupported); a `tools` list is inert. Use
  `execution_agent` / `quality_gates` to differentiate flavors.
- **codex** — `codex exec --model gpt-5.4-mini --sandbox workspace-write`. Sandbox
  scopes **files**, not tools; no tool-allowlist flag, so `tools` is inert. Use
  `execution_agent` / `quality_gates`.

## Source of truth

- Built-in flavor names: `src/milknado/domains/common/types.py` (`BUILTIN_FLAVORS`);
  custom project flavor names are also accepted.
- Parsing + validation + save/load: `src/milknado/domains/common/config.py`
  (`FlavorOverride`, `_parse_flavor_entry`, `save_config`).
- Resolution precedence: `src/milknado/domains/common/flavor_profile.py`
  (`resolve_flavor_profile`).
- Family defaults + allowlists: `src/milknado/domains/common/agent_argv.py`
  (`WORKER_ALLOWED_TOOLS`, `_default_execution_command`).
- Dispatch-time application: `run_loop/__init__.py` `_dispatch_batch`,
  `mcp_run.py`, `mcp_todo.py`.
