# Milknado — Gemini CLI extension

See [Gemini CLI extensions](https://google-gemini.github.io/gemini-cli/docs/extensions/).

## Layout

- `gemini-extension.json` — name, version, optional `mcpServers` for `milknado-mcp`.
- `commands/*.toml` — custom slash commands that shell out to `uv run milknado …`.

## Install (local dev)

From this repo root:

```bash
gemini extensions link integrations/gemini-milknado
```

Or install from Git after publish:

```bash
gemini extensions install https://github.com/<org>/<repo>
```

Restart Gemini CLI after install or update.

## Requirements

- `uv` on `PATH`.
- `milknado` available to `uv run` (path dependency or published wheel).

## Variables

`gemini-extension.json` uses `${workspacePath}` for `MILKNADO_PROJECT_ROOT` per extension variable rules. If your CLI version differs, set the env in user `settings.json` instead.

## Generated commands

Command TOML files in `commands/` are generated from `integrations/spec/commands.yaml`:

```bash
uv run python integrations/spec/generate_integrations.py
```
