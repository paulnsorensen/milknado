# Owner run exits when nothing is dispatchable at start

Source: dogfood pass 3 (2026-09-28), coordinator run `run-20260928T192138378532Z`. Confidence: certain
(process sampled: execution thread gone, log holds only `Run started`, nodes re-queued 20 min later never
dispatched).

## Gotcha

`RunLoop._execute_run` dispatched once, then looped `while self._active`. A restart made while every
goal node was running (orphan-swept to failed) or done found nothing dispatchable, so the loop returned
at once. `milknado run --web` kept serving the dashboard, which showed the run as active, but nodes the
owner set back to pending afterwards sat there forever. The final telemetry line was also invisible
because it was emitted after the run-log context closed.

## Rule

- An owner-attached run (`run_controller` in `web/server.py` passes `await_owner_work=True`) idles in
  `_wait_for_owner_work`: drain controls, rescan every `IDLE_RESCAN_SECONDS`, dispatch what became
  ready, and return only when the owner stops scheduling, the root is done, or strict mode tripped.
- Batch runs (`milknado run` without `--web`, `run_execution_loop`) keep the old behaviour and end when
  nothing is dispatchable.
- Restarting the coordinator to load a code fix is safe now; re-queue the swept nodes with
  `milknado_todo_set_status(pending)` and the idle loop picks them up.

## Follow-up: the attempt cap ended runs after one attempt (2026-09-29)

The restarted coordinator dispatched 52, 53, 55, 56 and every one failed 1800 s after dispatch with
`session stopped` mid-turn. `LoopAdapter.create_run` sets `stop_on_error=True`, and
`_run_iteration` treated a timed-out attempt like a crashed agent command, so the per-attempt cap
(`attempt_timeout_seconds`, 1800 by default) ended the whole run on the first attempt and
`max_iterations` never applied. A timed-out attempt now spends one iteration and the loop retries
with a fresh session while the budget has room; the last allowed attempt, and any non-zero exit,
still fail the run. Bound it with `max_iterations` and the
`max_consecutive_failures` cap, not with `stop_on_error`. Worker logs live under the worktree's
`.loop-logs`, which fail-closed teardown removes, so the failure detail in `runs.detail` was the
only evidence (node 106 tracks persisting it).
