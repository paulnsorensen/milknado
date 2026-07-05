# Distribution — Portable Plugin Packaging (Skills + MCP) Across Harnesses

How to ship milknado's MCP server (`milknado-mcp`, see [[entry-points]]) plus a set of skills as one installable artifact for Claude Code, OpenAI Codex, and opencode. Researched 2026-06-06 from primary vendor docs; full claim tables with citations live in local research artifacts under `.cheese/research/` (`portable-plugin-{claude-code,codex,opencode}/`, `opencode-python-plugin/`, `agent-skills-standard/` — local scratch, not committed).

## The portable unit is SKILL.md, not the plugin

The [agentskills.io specification](https://agentskills.io/specification) (Anthropic-originated open standard, Dec 2025) is the only layer all three harnesses share: a directory with `SKILL.md` whose frontmatter requires only `name` + `description`. Everything above that — plugin manifests, marketplaces, MCP registration — is per-harness. The MCP server is explicitly **not** part of the skills spec; it lives at the plugin/config layer everywhere.

Keep skills spec-minimal: `allowed-tools` is honored only by Claude Code, and `${CLAUDE_PLUGIN_ROOT}` does not expand inside skill Markdown ([claude-code#9354](https://github.com/anthropics/claude-code/issues/9354), open as of June 2026).

## Asymmetric compat: Codex reads Claude's paths, not vice versa

The critical, non-obvious finding that shapes the layout:

- **Codex** (plugins shipped March 2026) natively reads `$REPO_ROOT/.agents/plugins/marketplace.json` and `~/.agents/plugins/marketplace.json`, **and** reads `.claude-plugin/marketplace.json` as a legacy-compat path ([Codex plugin docs](https://developers.openai.com/codex/plugins/build)). Skills load from `.agents/skills/` (repo) and `~/.agents/skills/` (user).
- **Claude Code reads no `.agents/` path at all.** Ruled out across five surfaces (skills discovery list, `--add-dir`, `additionalDirectories`, env vars, settings keys — see [settings](https://code.claude.com/docs/en/settings), [skills](https://code.claude.com/docs/en/skills), [env-vars](https://code.claude.com/docs/en/env-vars)) and absent from the changelog through v2.1.167. The feature request is [claude-code#31005](https://github.com/anthropics/claude-code/issues/31005) (3,000+ upvotes, no Anthropic response after 7 months). Only bridge: `extraKnownMarketplaces` in `.claude/settings.json` — per-repo opt-in, not portable.

So the "generic-looking" `.agents/plugins/marketplace.json` covers **Codex only**, while `.claude-plugin/marketplace.json` covers **both Claude Code and Codex**. The neutral path is the narrower one.

## Marketplace schemas differ — don't symlink one file to both paths

Claude's marketplace.json (`name`, `owner`, `plugins[]` with source types: relative path / github / url / git-subdir / npm) and Codex's native schema (`interface.displayName`, `source.source`/`source.path`, `policy.installation` ∈ NOT_AVAILABLE|AVAILABLE|INSTALLED_BY_DEFAULT, `policy.authentication` ∈ ON_INSTALL|ON_USE, `category`) are different formats. A considered-and-rejected idea: make `.claude-plugin` a symlink to `.agents/plugins` so one file serves both. Rejected because:

1. **Schema collision** — with the symlink, Codex's *native* reader (expecting Codex schema) and its *legacy* reader (expecting Claude schema) both resolve to the same file. Since verified (locked decision 1): there is only one reader and it does tolerate Claude schema — so this reason alone no longer rejects the symlink, but reasons 2–3 still do.
2. **Symlink survival is fetch-dependent** — git clone preserves symlinks on Unix, but GitHub zipball downloads convert them to text files containing the target path; and Claude Code's plugin-install pipeline is confirmed to mishandle symlinks inconsistently across platforms (locked decision 2).
3. **Windows** — committed symlinks need developer mode + `core.symlinks`.

**Decision: two small real marketplace files, one shared plugin payload.** marketplace.json is ~20 lines; duplication is cheaper than the symlink's failure modes, and each file can be schema-correct (Codex's `policy.*` fields have no Claude equivalent).

```
repo/
├── .claude-plugin/marketplace.json   # Claude schema → Claude Code native + Codex legacy
├── .agents/plugins/marketplace.json  # Codex schema → Codex native
└── plugins/milknado/                 # shared payload (subdir — source.path "./" is rejected, openai/codex#17066)
    ├── .claude-plugin/plugin.json    # mcpServers → ${CLAUDE_PLUGIN_ROOT}-anchored stdio command
    ├── skills/…                      # SKILL.md dirs
    └── .mcp.json
```

Exposure model is pull-only on both: the git repo *is* the marketplace (`/plugin marketplace add owner/repo` on Claude, `codex plugin marketplace add owner/repo` on Codex). Codex auto-discovers `~/.agents/plugins/marketplace.json` on start (local-dev path); the repo-level one needs a restart + manual install from the `/plugins` directory — no auto-prompt.

## opencode: no marketplace, no Python plugin runtime — two install paths

opencode has no marketplace format (open requests: [#23800](https://github.com/anomalyco/opencode/issues/23800), [#7467](https://github.com/anomalyco/opencode/issues/7467)) and its plugin loader globs `*.{ts,js}` only (Bun runtime). Python runs fine as the MCP *server* — only the plugin *glue* must be JS. The local-MCP spawn passes `command` straight to `StdioClientTransport` with full `process.env` inherited, so `uvx --from milknado milknado-mcp` resolves from the user's PATH (CLI launch; the Desktop app may not inherit shell profiles).

- **Path A — npm shim** (one-line install, `"plugin": ["milknado-opencode"]`): the `config` hook mutates the Config object — `c.mcp.milknado = { type: "local", command: ["uvx", "--from", "milknado", "milknado-mcp"] }` and `c.skills.paths.push(<skills dir inside the npm pkg>)`. Precedent: [@sveltejs/opencode](https://www.npmjs.com/package/@sveltejs/opencode) ships MCP + skills + a subagent this way. Known fragility in the Desktop app ([#30130](https://github.com/anomalyco/opencode/issues/30130)); CLI is reliable.
- **Path B — no shim** (zero JS, most reliable): documented `opencode.json` mcp block + skills copied into `.agents/skills/` or `~/.config/opencode/skills/`. opencode natively scans `.opencode/skills/`, `.agents/skills/`, **and** `.claude/skills/` (project + global).

## Releasing the MCP to PyPI — trusted publishing makes the uvx launch resolve

The plugin is pull-from-git and needs no registry release, but every harness's config launches the server through `uvx`. **The invocation must pass a `--from <distribution>` matching the channel — `uvx --from milknado==X.Y.Z milknado-mcp` on tag/stable, `uvx --from git+…@main milknado-mcp` on main — never the bare `uvx milknado-mcp`.** This is the one easy-to-get-wrong gotcha: `uvx <name>` resolves `<name>` as a *package*, and there is no `milknado-mcp` package — `milknado-mcp` is a console script *inside* the `milknado` package ([[entry-points]]), so a `--from <distribution>` arg is required to name it. (Bare `--from milknado` floats the latest PyPI release independent of the installed ref — the pre-channels form the manifests now reject; see `TestLauncherFormAcceptance`.) The bare form fails with `milknado-mcp was not found in the package registry` (verified against the live 0.1.0 release, 2026-06-14). The earlier machine-local `--from /home/paul/Dev/milknado` arg worked *because* `--from <path>` also names a distribution; the portable fix had to keep a `--from`, swapping the path for the package name — dropping `--from` entirely (an intermediate `fix/mcp-portable-uvx` attempt) broke resolution. Until `milknado` is on PyPI, a freshly-installed plugin's launch fails to resolve. **Publishing the package — not re-releasing the plugin — is the release step.**

Release is automated by `.github/workflows/release.yml` via PyPI **trusted publishing** (OIDC — no stored token anywhere). The `vX.Y.Z` tag is no longer the trigger — it is an *output* of a successful publish (`release.yml` at commit `b49fdcc`, PR #157):

- **Trigger:** `on: push: branches: [main]` plus `workflow_dispatch` (a manual escape hatch to re-run a failed release without a dummy commit). **Not** `push: tags` — pushing a `v*` tag no longer starts anything. No `paths` filter, no `concurrency` guard.
- **Two jobs, version-gated.** A read-only `detect` job reads `[project].version` from `pyproject.toml` and curls `https://pypi.org/pypi/milknado/<version>/json`; it outputs `release=false` when that version is already on PyPI and `release=true` otherwise. The `build-and-publish` job is gated on `if: needs.detect.outputs.release == 'true'`. Net effect: a **version-bump merge to main publishes; every other push to main no-ops** (build-and-publish is skipped). No manual dispatch is needed for a normal release.
- **`build-and-publish`** (only when `detect` says release) builds an off-main commit that pins `plugins/milknado/.mcp.json` to `["--from", "milknado==<version>", "milknado-mcp"]`, runs `uv build` (sdist + wheel), publishes via `pypa/gh-action-pypi-publish`, and only *then* — last step, after a successful upload — tags `vX.Y.Z` on that off-main commit and repoints the `stable` branch at it with `git push --force-with-lease`. A failed publish therefore never leaves a dangling tag or a `stable` pinned to an unpublished version. `main` itself keeps the `git+…@main` launcher; only the tag and `stable` carry the PyPI-pinned `.mcp.json`.
- `id-token: write` + `environment: pypi` let the action mint a short-lived PyPI token at run time — nothing secret lives in the repo or in CI secrets. `contents: write` on `build-and-publish` is what lets it push the tag and move `stable`; `detect` runs `contents: read` only.
- **Recursion is impossible by construction.** The workflow pushes a tag and the `stable` branch, never `main`; and pushes made with the default `GITHUB_TOKEN` do not spawn new workflow runs.

**One-time manual step that cannot be automated — register the pending publisher on PyPI.** Because `milknado` did not yet exist on PyPI, the first publish needed a *pending publisher* created at pypi.org → Account → Publishing → "Add a pending publisher", with fields that must match the workflow exactly:

| Field | Value |
|---|---|
| PyPI Project Name | `milknado` |
| Owner | `paulnsorensen` |
| Repository name | `milknado` |
| Workflow filename | `release.yml` |
| Environment name | `pypi` |

Once registered, cutting a release is just: bump `version` in `pyproject.toml` → merge that change to `main`. The `detect` job sees the new version is not on PyPI and publishes; the tag and `stable` move follow as job outputs. No tag push, no manual dispatch. `v0.1.0` was the first release (published 2026-06-14). There is **no `just publish` recipe by design** — releases go through the CI publish path, never a local `uv publish`, so a PyPI token never touches a dev machine.

**Pinning the publish action is a trap worth recording.** `pypa/gh-action-pypi-publish` is a Docker container action whose image is tagged by *commit* SHA at `ghcr.io/pypa/gh-action-pypi-publish:<sha>`. Two distinct ways the first release broke: (1) a *corrupted* SHA in the pin (correct 10-char prefix, garbage tail) → "unable to resolve action" at setup; (2) pinning to the *annotated-tag object* SHA instead of the commit SHA → action resolves but the Docker pull fails with `manifest unknown`. Resolve the real commit SHA with `gh api repos/pypa/gh-action-pypi-publish/commits/<tag> --jq .sha` (the `/commits/<tag>` endpoint dereferences to the commit; `/git/ref/tags/<tag>` returns the tag object for annotated tags — the wrong one). v1.12.4 = `76f52bc884231f62b9a034ebfe128415bbaabdfc`.

PyPI versions are immutable and cannot be reused — a botched `0.1.0` burns that version (not the name). Run `uv build` locally and sanity-check the wheel before bumping the version. (A failed *publish* does not consume the version, so re-running the workflow — push another commit to `main`, or use `workflow_dispatch` — can retry the same version; only a successful upload locks it.)

## Locked decisions — the layout is LOCKED (resolved 2026-07-02)

The five open questions this section used to carry are resolved from primary sources (source read of `openai/codex` at commit [`129ea2a`](https://github.com/openai/codex/blob/129ea2aaf5fb426d8ba683ee53f290742f41dd31/codex-rs/core-plugins/src/marketplace.rs), live fetches of `code.claude.com/docs`, issue evidence on `anthropics/claude-code`). All five converge: **the shipped two-file, no-symlink, skills-inside-plugin-dir layout is correct as-is.** No marketplace or payload file changed when locking.

### 1. Codex's native reader tolerates Claude-schema marketplace.json — YES (certain)

There is no separate "native" vs "legacy" reader — one shared parser serves a priority-ordered path list (`.agents/plugins/marketplace.json`, then `.agents/plugins/api_marketplace.json`, then `.claude-plugin/marketplace.json`; **first match wins**, so when both files exist Codex reads only the `.agents` one). The per-plugin `source` deserializes as an untagged serde enum whose `Path(String)` variant is exactly Claude's `"source": "./plugins/milknado"`; `policy` is `#[serde(default)]` (defaults `AVAILABLE`/`ON_INSTALL`); there is no `deny_unknown_fields`, so Claude's top-level `owner` object is silently ignored.

Citation: [`codex-rs/core-plugins/src/marketplace.rs`](https://github.com/openai/codex/blob/129ea2aaf5fb426d8ba683ee53f290742f41dd31/codex-rs/core-plugins/src/marketplace.rs) — lines 20–24 (path list), 961–1008 (schema), 991–999 (policy defaults).

**Simplification declined — two files kept, each schema-correct.** Q1 makes a unified Claude-shape content possible (Codex would parse it at either path), but the priority order means `.agents/plugins/marketplace.json` is the file Codex actually reads when both exist — keeping it Codex-schema keeps `policy.*` explicit and preserves `category`/`interface.displayName`, which the Claude shape cannot carry. Unifying on the Claude shape would discard that information and save nothing: both discovery paths must exist regardless (path discovery is separate from schema tolerance — Claude Code still reads only `.claude-plugin/marketplace.json`). Recorded as: **confirmed required (two discovery paths), kept as-is.**

### 2. Claude Code following a symlinked `.claude-plugin/` on git-clone installs — DO NOT RELY ON IT (speculating)

No primary doc addresses symlinking the marketplace-root directory, and no direct test was run — but the same plugin-install pipeline is confirmed unreliable with symlinks across platforms. [anthropics/claude-code#41392](https://github.com/anthropics/claude-code/issues/41392) ("Plugin symlinks not dereferenced when copying to cache"; filed against 2.1.37, reproduced on 2.1.88, open) shows symlinks preserved as machine-specific absolute paths into the marketplace checkout; a follow-up on 2.1.117/Ubuntu 24.04 found Linux now *drops* symlinks entirely (macOS keeps them as absolute-path symlinks; Copilot CLI dereferences them). Moot under the two-file decision — recorded as a constraint, not a change: **no symlinks anywhere in the plugin layout.** The current layout uses none.

### 3. `plugin.json` `skills` referencing paths outside the plugin dir — NO (certain)

Directly documented at [code.claude.com/docs/en/plugins-reference](https://code.claude.com/docs/en/plugins-reference) § "Path traversal limitations": "Installed plugins cannot reference files outside their directory. Paths that traverse outside the plugin root (such as `../shared-utils`) will not work after installation because those external files are not copied to the cache." And on the `skills` manifest field: "All paths must be relative to the plugin root and start with `./`". The documented escape hatch — symlinks *inside* the plugin dir, dereferenced at cache-copy time — is the exact behavior [#41392](https://github.com/anthropics/claude-code/issues/41392) shows failing in practice; treat it as unavailable. Consequence: skills stay physically inside `plugins/milknado/skills/`; no repo-neutral shared skills path. The layout already complies.

### 4. Repo-level Codex marketplace pickup gated by `trust_level` — NO (certain, scoped to discovery/listing)

Scope checked: whether Codex reads/parses/surfaces `$REPO_ROOT/.agents/plugins/marketplace.json` when the project's `trust_level` is unset or untrusted. Ruled out per candidate: `codex-rs/core-plugins/src/marketplace.rs` (`discover_marketplace_paths_from_roots`), `loader.rs`, and the TUI plugin catalog (`tui/src/chatwidget/plugin_catalog.rs`) have zero `trust` references; `manager.rs`'s only hit is an unrelated product-restriction comment; a repo-wide `trust_level` search (21 hits) lands entirely in config/exec-policy/session code — the only user-facing gate is the session trust prompt in `app-server/src/request_processors/thread_processor.rs`, which never touches `core-plugins::marketplace`. Not ruled out (not checked): a UI-only confirmation before *installing* from an untrusted repo, distinct from discovery/listing. Citations: same [`openai/codex` permalink base](https://github.com/openai/codex/blob/129ea2aaf5fb426d8ba683ee53f290742f41dd31/codex-rs/core-plugins/src/marketplace.rs) as decision 1.

### 5. HTTP-transport MCP inside a Claude Code plugin — YES at the schema level (certain); no shipped worked example found (speculating)

[code.claude.com/docs/en/mcp](https://code.claude.com/docs/en/mcp) documents `type: "http"` (alias `streamable-http`) as a first-class `.mcp.json` transport, and [plugins-reference](https://code.claude.com/docs/en/plugins-reference) ties plugin-provided MCP config to the same schema family ("Plugin-provided MCP configurations substitute `${CLAUDE_PROJECT_DIR}` directly and don't need the default") with no stdio-only restriction stated anywhere. But every worked plugin example in the docs is stdio, and a GitHub code search found no confirmed in-plugin `"type": "http"` example — inconclusive, not a negative. Irrelevant to the current layout: milknado's server is stdio-only today.

### Risk note — Cowork rejects stdio MCP in plugins (watch for spread)

[anthropics/claude-code#69940](https://github.com/anthropics/claude-code/issues/69940) (open; labels `bug, regression, area:cowork, area:mcp, area:plugins`): Claude **Cowork** refuses plugins declaring local/stdio MCP servers — "Plugins may only declare remote (http/sse/ws) or MCPB servers." Scoped to Cowork, not the CLI milknado targets (`claude plugin install` / `--plugin-dir` still load stdio plugins, and a commenter confirms the `.mcpb` bundle workaround: interior `manifest.json` with `server.type: "binary"` — not `"uv"`, which silently fails schema validation). If the remote/MCPB-only restriction spreads from Cowork to the general plugin installer, milknado's stdio `uvx` launcher would need MCPB wrapping.
