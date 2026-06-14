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

1. **Schema collision** — with the symlink, Codex's *native* reader (expecting Codex schema) and its *legacy* reader (expecting Claude schema) both resolve to the same file; whether the native reader tolerates Claude schema is unverified.
2. **Symlink survival is fetch-dependent** — git clone preserves symlinks on Unix, but GitHub zipball downloads convert them to text files containing the target path; whether Claude Code follows a symlinked `.claude-plugin/` is unverified.
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

The plugin is pull-from-git and needs no registry release, but every harness's config launches the server through `uvx`. **The invocation must be `uvx --from milknado milknado-mcp`, not the bare `uvx milknado-mcp`.** This is the one easy-to-get-wrong gotcha: `uvx <name>` resolves `<name>` as a *package*, and there is no `milknado-mcp` package — `milknado-mcp` is a console script *inside* the `milknado` package ([[entry-points]]), so `--from milknado` is required to name the distribution. The bare form fails with `milknado-mcp was not found in the package registry` (verified against the live 0.1.0 release, 2026-06-14). The earlier machine-local `--from /home/paul/Dev/milknado` arg worked *because* `--from <path>` also names a distribution; the portable fix had to keep a `--from`, swapping the path for the package name — dropping `--from` entirely (an intermediate `fix/mcp-portable-uvx` attempt) broke resolution. Until `milknado` is on PyPI, a freshly-installed plugin's launch fails to resolve. **Publishing the package — not re-releasing the plugin — is the release step.**

Release is automated by `.github/workflows/release.yml` via PyPI **trusted publishing** (OIDC — no stored token anywhere):

- **Trigger:** pushing a `v*` tag.
- The job asserts the tag matches `pyproject.toml` `[project].version` (a `vX.Y.Z` tag against a mismatched manifest fails fast), runs `uv build` (sdist + wheel), then `pypa/gh-action-pypi-publish`.
- `id-token: write` + `environment: pypi` let the action mint a short-lived PyPI token at run time — nothing secret lives in the repo or in CI secrets.

**One-time manual step that cannot be automated — register the pending publisher on PyPI.** Because `milknado` did not yet exist on PyPI, the first publish needed a *pending publisher* created at pypi.org → Account → Publishing → "Add a pending publisher", with fields that must match the workflow exactly:

| Field | Value |
|---|---|
| PyPI Project Name | `milknado` |
| Owner | `paulnsorensen` |
| Repository name | `milknado` |
| Workflow filename | `release.yml` |
| Environment name | `pypi` |

Once registered, cutting a release is: bump `version` in `pyproject.toml` → commit → tag `vX.Y.Z` → push the tag. `v0.1.0` was the first release (published 2026-06-14). There is **no `just publish` recipe by design** — releases go through the tag-triggered CI path, never a local `uv publish`, so a PyPI token never touches a dev machine.

**Pinning the publish action is a trap worth recording.** `pypa/gh-action-pypi-publish` is a Docker container action whose image is tagged by *commit* SHA at `ghcr.io/pypa/gh-action-pypi-publish:<sha>`. Two distinct ways the first release broke: (1) a *corrupted* SHA in the pin (correct 10-char prefix, garbage tail) → "unable to resolve action" at setup; (2) pinning to the *annotated-tag object* SHA instead of the commit SHA → action resolves but the Docker pull fails with `manifest unknown`. Resolve the real commit SHA with `gh api repos/pypa/gh-action-pypi-publish/commits/<tag> --jq .sha` (the `/commits/<tag>` endpoint dereferences to the commit; `/git/ref/tags/<tag>` returns the tag object for annotated tags — the wrong one). v1.12.4 = `76f52bc884231f62b9a034ebfe128415bbaabdfc`.

PyPI versions are immutable and cannot be reused — a botched `0.1.0` burns that version (not the name). Run `uv build` locally and sanity-check the wheel before tagging. (A failed *publish* does not consume the version, so a re-pointed tag can retry the same version — only a successful upload locks it.)

## Open questions (verify before locking the layout)

- Does Codex's native `.agents/plugins` reader tolerate Claude-schema marketplace.json? (Determines whether the two-file split is strictly required.)
- Does Claude Code follow a symlinked `.claude-plugin/` on git-clone installs? (Moot under the two-file decision, but cheap to test.)
- Can Claude's `plugin.json` `skills` field reference paths outside the plugin dir (e.g. `../../.agents/skills/`), which would let skills live once at the repo's neutral path?
- Is repo-level Codex marketplace pickup gated by `[projects."…"].trust_level`? Not documented either way.
- HTTP-transport MCP inside a Claude Code plugin — all doc examples are stdio.
