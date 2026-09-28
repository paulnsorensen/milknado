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
