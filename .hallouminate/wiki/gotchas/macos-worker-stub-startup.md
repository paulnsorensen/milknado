# macOS worker stub startup

Fresh executable agent stubs can stall before shell code runs on the tested Mac.
This makes short subprocess tests fail without proving a Milknado dispatch defect.
Use a stable interpreter to execute test script data.
Do not increase production timeouts to hide this fixture cost.

## Measured failure

The combined meta gate fails three consecutive runs with different async and worker-environment tests.[^1]
Affected tests also fail without the new meta tests when coverage remains enabled.
A stalled shell has an empty output log.
Its process sample remains at `_dyld_start +0` before shell code, with a 96K memory footprint.[^2]
The exact macOS service responsible is not identified.

## Controlled comparison

Each variant uses 24 trials, four concurrent workers, and a three-second timeout.[^3]
Successful trials must return exit zero and exactly `probe\n`.

| Launch shape | Timeouts | Mean elapsed time |
| --- | ---: | ---: |
| Fresh executable shell stub named `claude` | 8/24 | 1.883 s |
| One reused executable shell stub | 0/24 | 0.145 s |
| Fresh `claude` symlink to `/bin/cat` | 0/24 | 0.013 s |
| Explicit `/bin/sh` plus script path | 0/24 | 0.019 s |
| Real `ProcessAdapter` plus fresh executable shell stub | 0/24 | 0.757 s |

A follow-up tests the selected fixture shape directly.[^4]
A fresh `claude` symlink points to `/bin/sh`.
A separate script has no executable bits.
The command supplies the script path before the original worker arguments.
All 24 trials return the exact output, with no timeout and a 5.872 ms mean.

Fresh scripts named `worker-probe` also pass 24 trials, with a 356.747 ms mean.
Thus, the evidence does not prove that every fresh shell script stalls.
Executable name, cache state, or both can affect the result.

## Fixture contract

Keep the first command token as the validated bare agent name.
Quote the script path.
Keep real subprocess execution and the existing timeout assertions.
Preserve stdin delivery for Claude-style commands.
Preserve the trailing positional brief and empty stdin for OMP.
Preserve stdout, stderr, exit status, cancellation, and environment checks.



## Detached Python runner fixtures

A later fixture run exposes a separate five-second polling failure.[^5]
Its Python runner imports `open_graph` from `milknado.mcp._core`.
That module also initializes FastMCP, although the runner only needs the graph helper.
The helper is a direct re-export from `milknado.app.project`.
Use that owning module in the fixture to avoid unrelated MCP startup.
Keep the real detached process, database write, and poll assertions.

A four-worker comparison opens and closes real graphs through each import path.
Twelve MCP-path trials take 1.597–2.918 seconds, with a 2.290-second mean.
Twelve application-path trials take 0.170–0.329 seconds, with a 0.233-second mean.
Every trial exits zero.
These measurements establish avoidable startup cost, not the exact cause of the departed failed process.

[^5]: `/tmp/worker-fixtures-final-followup.log`, `TestUnifiedRunSchema.test_run_loop_poll_returns_superset_schema`, PID 40428. The unchanged rerun passes. Parent process probes compare the identical `open_graph` function through both modules.



The same owning-module import now applies to the production detached runner and both matching Python runner stubs.
Tests that patch `open_graph` must patch `milknado.app.project`, where the runner now resolves it.
A logging test initially retains the old patch target and fails the combined gate.
Its unchanged assertions pass after the patch target is corrected.
This failure is a test seam missed by the import change, not a production gate-policy defect.

## Evidence

[^1]: `.context/meta-final-check.log`; `.context/async-repro-*.log`; `.context/async-meta-repro-*.log`; `.context/async-cov-control-*.log`. The affected three-file subset passes 5/5 without coverage, fails 2/5 with meta tests and coverage, and fails 1/5 with coverage alone.
[^2]: Live `sample 3975 1 1 -file /dev/stdout` captures 889 samples at `_dyld_start +0` on macOS 26.5.1, arm64. No loaded-image details appear in that sample.
[^3]: `.context/startup-probe.json`, produced by `.context/async-startup-probe.py`. The adapter comparison uses the real `ProcessAdapter`, not a mock.
[^4]: `.context/startup-probe-followup.json`, produced with `--followup`. Temporary directories remain inside `.context` and are removed after the probe.


The combined repaired source passes `just check-llm` at `b3a1a8a`.
Project coverage is 97.66%, and diff coverage meets 95%.
The final Opus review approves the preserved subprocess and prompt contracts.

