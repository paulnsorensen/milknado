# OMP workers: MCP request cap versus node_verify

Source: dogfood pass 3 (2026-09-27), run `node-92-20260927T221720Z-6335422c`. Confidence: certain
(reproduced twice in the worker log; fixed and re-verified).

## Gotcha

OMP caps every MCP tool call at 30 000 ms unless the server entry sets `timeout`. `milknado_node_verify`
runs the resolved quality gates synchronously (`just check-llm` takes minutes), so a Codex/OMP worker
always saw `Request timeout after 30000ms`, refused to declare done, and burned iterations.
The worker also fell back to reading `.milknado/milknado.db` directly to find out what happened.

## Rule

- Both registrations carry `"timeout": 1800000` (the gate ceiling `_GATE_TIMEOUT_SECONDS`):
  `.mcp.json` (project server `milknado`) and `plugins/milknado/.mcp.json` (plugin server
  `milknado:milknado`). Claude Code accepts a positive `timeout`; it rejects `0`.
- The OMP plugin cache (`~/.omp/plugins/cache/plugins/milknado___*/.mcp.json`) refreshes from GitHub
  `main` at run start, so the plugin path only picks the fix up after a push.
- A running worker keeps the MCP config it started with; copying the fixed `.mcp.json` into its
  worktree only helps the next iteration.
- If a worker is stuck on verify, the coordinator can call `milknado_node_verify` for the run id;
  the verdict persists as a `role=verify` run message and satisfies the mark-terminal gate.

## Related engine behavior seen in the same pass

- A coordinator restart sweeps runs whose recorded pid is dead to `failed` ("worker session gone")
  and never re-queues them; reset such nodes to `pending` by hand (node 94 tracks the decision).
- The run loop rescans dispatchable nodes only at start and after a merge-back
  (`executor.py:1214`), so nodes reset or added mid-run wait for the next completion.
- Before this pass's fix, the CLI run loop (`milknado run`) spawned workers with no
  `MILKNADO_NODE_ID`/`MILKNADO_RUN_ID`/`MILKNADO_PROJECT_ROOT` at all, so `milknado-mcp`
  could not resolve the run and workers fell back to reading `.milknado/milknado.db` directly.
- Before the same fix, the CLI run loop's per-node flavor `dataclasses.replace`
  (`run_loop/__init__.py`) copied the review settings but dropped `max_iterations`,
  `attempt_timeout_seconds`, and the derived completion timeout. `milknado run`
  attempts had no 1800 s cap and ran for 3 to 5 hours; `runs.timeout_seconds` was NULL.
  The MCP node runner (`_loop_node_runner.py`) always set all three.
