# Worker Evidence on a Slow Disk

Worker startup has one 8-second evidence budget (`src/milknado/loop/_process_startup.py`).
Each worker evidence operation opens its own SQLite connection (`LoopWorkerEvidence._store`).
On a slow disk, the fsync count per operation decides whether startup fits that budget.

## Cause found in October 2026

Three `real_process` tests failed on a host with fsync at 200-1400 ms.
The pane process raised `TimeoutError: worker evidence deadline expired`.
The tmux receipt stayed `PREPARED`.

A connection that is the last one open on a WAL database checkpoints and deletes the WAL on close.
A fresh connection also fsyncs the directory and the WAL header on its first commit.
With `synchronous=FULL`, one evidence operation cost up to five fsyncs.
A clean tmux run made 45 fsyncs. At 0.5 s or more per fsync, startup exceeded 8 s.

The tmux pane process (`milknado.loop._tmux_exec`) holds no graph connection.
That makes every evidence close a last close.

## Decision

- `WorkerEvidenceStore` sets `PRAGMA synchronous=NORMAL`. A WAL commit then skips fsync.
  An OS crash can drop the newest evidence rows. Those rows describe processes that died with the OS.
  A process crash drops no committed row.
- `WorkerEvidence.held()` keeps one idle connection open, so per-operation closes skip the checkpoint.
  `spawn_protected` holds it for the startup sequence.
  `_tmux_exec._run` and `ProcessAdapter._run_protected` hold it for the whole worker lifetime.
- The 8-second deadline stays. Making it wider would only hide the fsync amplification.

## Reproduce without host disk load

Inject latency only into your own process tree with strace. Do not run `dd` or `fio` on a shared host.

```bash
strace -f --seccomp-bpf -qq -o /dev/null -e trace=fsync,fdatasync \
  -e inject=fsync,fdatasync:delay_enter=500000 \
  python -m milknado.loop._tmux_exec nonce receipt.json - python -c 'raise SystemExit(7)'
```

`tests/test_tmux_exec_receipts.py::test_tmux_receipt_confirms_exit_when_database_fsync_is_slow` runs this check.
It skips when `strace` is not on `PATH`.

## Gotchas

- SIGSTOP on a ptraced process gives a tracing stop, not `STATUS_STOPPED`.
  `test_lifeline_supervisor_gap_recovery_verifies_before_release` therefore fails under strace even at zero delay.
- Test waits of 10-15 s cannot hold a full run when every fsync takes 0.7 s.
  Durable graph writes and receipt publication still fsync, and they are outside the evidence budget.
- A `PREPARED` receipt reads as unknown (`read_exec_evidence` returns `None`), which fails closed.
  `_tmux_exec` does not write `FAILED` after a startup exception, because the gate may already be open.
