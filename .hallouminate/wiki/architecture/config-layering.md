# Config Layering — defaults → global → local

How `milknado.toml` cascades, and where the merge is per-field vs whole-value.
Load order: **built-in defaults → global → project-local**, local wins
(`domains/common/config.py:109-140`). Global path:
`$XDG_CONFIG_HOME/milknado/milknado.toml`, else `~/.config/milknado/milknado.toml`
(`config.py:109-116`).

## The merge is a recursive deep-merge

`_merge` = `deep_merge(base, override, list_mode="replace")`
(`config.py:274-276`, `domains/common/merge.py:11-32`):

- **Dicts recurse at every nesting level** — merge happens per-key all the way down.
- **Scalars and lists replace outright** — no list concatenation.

Consequences by table shape:

| Table | Merge behaviour |
|---|---|
| top-level scalars (`concurrency_limit`, `execution_agent`, …) | per-key; local wins, global fills unset keys |
| `[milknado.flavor.*]` (dict of flavor-name → field dict) | **per-flavor-name, then per-field**: a local flavor table does not wipe global flavors; a local `[milknado.flavor.spec]` setting only `quality_gates` inherits the global `spec` table's other fields |
| `[milknado.worker.tools]` (family → list) | per-family key, but the **list replaces wholesale** (`tests/test_config_layered.py:130-154`) |
| `prompts.*` | custom slot-clearing: a local override in one key form clears the global's alternate form (`config.py:303-318`) |

> The per-flavor-name/per-field claim follows from the merge algorithm but has
> **no dedicated regression test** — `tests/test_config_layered.py` exercises only
> scalars, `worker.tools`, and `prompts`. Add one before relying on it in anger.

## Layer hygiene rules

- `_LOCAL_ONLY_KEYS` (`project_root`, `db_path`, `plugins`) are stripped from the
  global layer with a warning (`config.py:53,321-328`) — machine-local identity
  never cascades.
- Relative `prompts.*_path` values in the global layer are absolutized against
  the **global config dir before the merge** (`config.py:282-300`). Rule of
  thumb (from the PR #96 bug): *resolve relative paths against the layer they
  came from, not the active dir* (`history/review-lessons.md`).

## The one fail-closed key

`quality_gates` defaults to `None`, and `None` fails dispatch closed with
`NO_GATES_CONFIGURED_MESSAGE` — it is the only key with no working default.
Explicit `quality_gates = []` means "no gates" and passes. Every other config
field has a coded default (`config.py:65-96,362-399`).

## Inherit vs full override

Deep-merge **is** the inherit mode. There is currently no full-override switch —
a project file cannot say "ignore the global layer" short of redefining every
key it cares about. If a global flavor table sets a field you want *unset*
locally, there is no way to express that (`None` in TOML doesn't exist; omitting
the key inherits). Tracked as a design gap in the issue tracker.


### Path confinement and worktree fallback

Prompt and flavor `brief_prepend_path` values from the project layer resolve under `project_root`; absolute values that escape it raise `ValueError` naming the config key and root. Global-layer path values are resolved against the global config directory before merging and are explicitly trusted.

`[milknado] worktree` is the config-level fallback for every flavor and defaults to `true`. A flavor's `worktree` value overrides that fallback only when explicitly set; an omitted flavor value inherits the config-level default.

## Preset packs (the "reasonable defaults" story)

- Milknado's own worked global config:
  `plugins/milknado/skills/milknado-config/references/flavor-presets.md`
  (existence gated by `tests/test_plugin_manifests.py:185-187`).
- The easy-cheese phase-flavor pack (brainstorm/explore/spec/implement/research/
  agent-review/harden/fix) ships as a copy-paste global config in
  `docs/adoption-contract.md:62-112` — intended for
  `~/.config/milknado/milknado.toml`, with repo-local files overriding
  individual knobs per the cascade above (ADR-004).
