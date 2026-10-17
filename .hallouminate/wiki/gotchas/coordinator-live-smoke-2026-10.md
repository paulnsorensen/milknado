# Coordinator live smoke 2026-10

The first live run of the meta-harness coordinator (PRs #517–#521) proves session continuity. It also finds four defects. The run used `milknado launch` on a disposable clone at `4af3a962`, one Claude provider session, and the OMP planner from the local `milknado.toml`.

## What works

- `start_goal`, `plan_goal`, `decide_plan_proposal`, `create_group`, `dispatch_task`, `acknowledge_launch`, `start_turn`, and `finish_task` all return `accepted` over the web API.
- A native Claude turn in the group worktree commits the requested file in 12 seconds.
- A second `start_turn` in the same group resumes the same provider session. Without tools, the agent repeats its earlier commit message. This is the one capability that ordinary nodes do not have: several nodes share one worktree and one live agent context.

## Defects found

1. **Claude planner cannot authenticate on macOS.** `build_minimal_mcp_env` (`src/milknado/domains/common/agent_argv.py:298`) dropped `USER`. Claude Code reads its OAuth login from the keychain through `USER`, so every Claude planning launch fails with `Failed to authenticate: OAuth session expired`. This affects `milknado plan` and coordinator `plan_goal`. Worker dispatch already forwards `USER` and `LOGNAME` (`src/milknado/domains/dispatch/runner.py:75`). Fixed on branch `fix/planner-env-keeps-user`.
2. **Planner failures are opaque.** `Planner.propose` raises `planner did not produce a valid proposal` and discards stdout, stderr, and the exit code (`src/milknado/domains/planning/planner.py:132-135`). Diagnosis needed a wrapper around `_PlanningSubprocess.run_agent`.
3. **`finish_task` trusts the caller.** `finish_task_attempt` sets `done` from the caller's `success` flag (`src/milknado/domains/coordinator/_task_attempt_workflow.py:100-134`). It runs no quality gate, no completion verifier, and no merge-back. Compare the inline-dispatch gate fix in #541.
4. **OMP planning probably writes to the graph.** A direct OMP planning run created graph node 123 three seconds before it returned, with no coordinator event. The OMP planner loads the project `.mcp.json`, which registers `milknado-mcp`. Evidence is timing and wording only; no tool-call log exists. If confirmed, read-only planning can bypass plan approval.

## Product gaps

- The cockpit exposes only start, plan, approve, and recover. Groups, dispatch, launch acknowledgement, turns, and finish exist only in the HTTP API.
- `plan_goal` uses the configured `planning_agent`, not the provider the user chose for the goal.
- The dock bar labelled "Coordinator" predates the feature and shows only the last graph event.

## Test-setup gotchas

- The local `milknado.toml` has the git skip-worktree bit. `git status` hides local edits, and a clone gets the committed file, whose `planning_agent` is the Claude default.
- The web API rejects a POST without an `Origin` header that matches the server (`src/milknado/web/guards.py:46`).
- `milknado launch` opens the bare `/` URL. Only the terminal shows the token URL.
- A fresh worktree needs `just install` before `just check-llm`; without `web/node_modules`, the two stale-build browser tests fail.
