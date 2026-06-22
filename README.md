# milknado

Mikado execution engine — decomposes goals into dependency graphs and executes them as parallel ralph loops.

## Installation

### Claude Code

```
/plugin marketplace add paulnsorensen/milknado
/plugin install milknado
```

### Codex

```
codex plugin marketplace add paulnsorensen/milknado
```

Then restart Codex and install from the `/plugins` directory. Note: repo-level marketplace entries do not auto-prompt on start — you must select and install after restart.

> **Note:** the plugin launches its MCP server with `uvx --from milknado milknado-mcp`. The `milknado-mcp` command ships *inside* the `milknado` PyPI package, so the `--from milknado` is required — a bare `uvx milknado-mcp` resolves the argument as a package name and fails.

### opencode

Add the MCP server to `opencode.json` in your project or home directory:

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

Then copy the skill into your skills directory:

```sh
cp -r plugins/milknado/skills/ .agents/skills/
# or globally:
cp -r plugins/milknado/skills/ ~/.config/opencode/skills/
```

### Run from an unreleased git ref

To run `main` (or any branch) instead of the published PyPI release:

```
uvx --from git+https://github.com/paulnsorensen/milknado milknado-mcp
```

> **Note — this wires the MCP server only; the plugin skills do not load.** The
> skills (`milknado-config`, `harvest`, `load-roadmap`) ship through the plugin
> install path, not raw MCP registration, so a bare git-ref MCP gives you the
> tools without the skills. To get the skills too:
>
> - **Released MCP + skills:** install the plugin (`/plugin install milknado`),
>   which runs the published-release MCP.
> - **Git-ref MCP + skills:** keep the git-ref MCP command above, clone/check out
>   this repo so `plugins/milknado/skills/` exists locally, then copy the
>   skills in manually from the checkout root, as the opencode section does:
>
>   ```sh
>   cp -r plugins/milknado/skills/ ~/.claude/skills/   # personal
>   # or project-scoped: cp -r plugins/milknado/skills/ .claude/skills/
>   ```

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
