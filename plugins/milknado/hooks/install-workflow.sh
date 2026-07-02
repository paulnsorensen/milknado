#!/usr/bin/env bash
# SessionStart hook: install the ultracode Workflow script into the project.
#
# `workflows/` is not a recognized plugin component, so installing the milknado
# plugin does not make node-runner.js discoverable. This hook closes that gap
# with an idempotent copy into .claude/workflows/ (the project-local discovery
# path): copy when absent, no-op when identical, refresh when the plugin copy
# changed, never delete.
set -euo pipefail

# Outside plugin context (e.g. invoked manually) this is a silent no-op.
[[ -n ${CLAUDE_PLUGIN_ROOT:-} ]] || exit 0

src="${CLAUDE_PLUGIN_ROOT}/workflows/node-runner.js"
dest_dir="${CLAUDE_PROJECT_DIR:-$PWD}/.claude/workflows"
dest="${dest_dir}/node-runner.js"

[[ -f $src ]] || exit 0
diff -q "$src" "$dest" >/dev/null 2>&1 && exit 0
mkdir -p "$dest_dir"
cp "$src" "$dest"
