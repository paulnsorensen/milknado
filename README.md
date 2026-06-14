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
