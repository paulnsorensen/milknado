# milknado

Mikado execution engine — decomposes goals into dependency graphs and executes them as parallel ralph loops.

## Installation

milknado ships as a plugin — skills plus an MCP server launcher — on three channels.
Each channel's skills **and** server come from the same ref, so the two never drift:

| Channel | Skills | Server | Use when |
|---|---|---|---|
| `@stable` (recommended) | latest release tag | PyPI, pinned to that release | you want vetted releases that auto-follow the newest |
| `@vX.Y.Z` | that tag | PyPI `==X.Y.Z` | you need a reproducible pin |
| `@main` | `main` HEAD | git `@main` | you want unreleased / bleeding-edge work |

### Claude Code

```
/plugin marketplace add paulnsorensen/milknado@stable   # or @v0.2.0, or @main
/plugin install milknado
```

`@stable` auto-advances: when a release moves the `stable` branch, `/plugin
marketplace update` pulls the new HEAD and `/plugin update` installs it — no re-add
needed. (Releases force-repoint `stable`; in the rare case `update` reports a
conflict instead of fast-forwarding, re-run the `marketplace add …@stable` line to
reset it.)

### Codex

```
codex plugin marketplace add paulnsorensen/milknado@stable   # or @v0.2.0, or @main
```

Restart Codex, then install from the `/plugins` directory. Repo-level marketplace
entries do not auto-prompt on start — select and install after restart. To pick up a
new `@stable` HEAD, run `codex plugin marketplace upgrade`; whether that advances an
explicit branch ref is undocumented for Codex, so if it does not move, re-run the
`marketplace add …@stable` line.

### opencode

opencode has no plugin marketplace — wire the server into `opencode.json` and copy
the skills from a checkout at the ref you want. Set `--from` to match your channel:

```json
{
  "mcp": {
    "milknado": {
      "type": "local",
      "command": ["uvx", "--from", "milknado", "milknado-mcp"]
    }
  }
}
```

- a release: `"--from", "milknado==0.2.0"` (exact pin) or `"--from", "milknado"` (latest).
- `@main`: `"--from", "git+https://github.com/paulnsorensen/milknado@main"`.

Then copy the skills:

```sh
cp -r plugins/milknado/skills/ .agents/skills/
# or globally:
cp -r plugins/milknado/skills/ ~/.config/opencode/skills/
```

> **uvx caching caveat:** the `milknado-mcp` command ships *inside* the `milknado`
> PyPI package, so a `--from` target (pin, latest, or git ref) is required — a bare
> `uvx milknado-mcp` looks up a nonexistent package and fails. uvx caches the
> resolved server: PyPI pins (`==X.Y.Z`) carry a fresh version string so each
> release busts the cache, but the `@main` git ref caches by commit SHA — pull a
> newer `main` server with `uvx --refresh` / `uv tool upgrade milknado`.

## Configuration

### quality_gates (required — fail-closed if absent)

`[milknado] quality_gates` must be set in `milknado.toml`. If the key is absent,
every node run is rejected with a message telling you to add it. Run `milknado
init` to auto-detect gates from your project type:

| Project marker | Auto-detected gates |
|---|---|
| `pyproject.toml` | `uv run pytest`, `uv run ruff check`, `uv run ty check` |
| `Cargo.toml` | `cargo test`, `cargo clippy -- -D warnings` |
| `package.json` | `npm test` |
| `go.mod` | `go test ./...`, `go vet ./...` |
| `project.godot` | `godot --headless --quit` (with `fail_on_stdout`) |

To skip gates for a flavor (e.g. spec tasks), use `quality_gates = []` in the
flavor table — the empty list is an explicit skip, not the same as omitting the
key.

### fail_on_stdout

Each gate entry is either a bare string or a `{command, fail_on_stdout}` inline
table. The second form fails the gate even on exit 0 when the combined
stdout+stderr matches the regex:

```toml
[milknado]
quality_gates = [
  "uv run pytest",
  {command = "godot --headless --quit", fail_on_stdout = "SCRIPT ERROR|FAILED"},
]
```
