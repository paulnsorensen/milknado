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

### opencode

Add the MCP server to `opencode.json` in your project or home directory:

```json
{
  "mcp": {
    "milknado": {
      "type": "local",
      "command": ["uvx", "milknado-mcp"]
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

### Pre-release fallback

If `uvx milknado-mcp` is unavailable (before the first PyPI release), use:

```
uvx --from git+https://github.com/paulnsorensen/milknado milknado-mcp
```

The PyPI release is currently blocked by a git direct-reference dependency on a
`ralphify` fork (fork-only features not yet available on PyPI). The git fallback
above is the working install path until that dependency is resolved.
