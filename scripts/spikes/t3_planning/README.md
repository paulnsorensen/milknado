# T3 planning reuse spike

This CLI returns a batch plan. It does not create goal nodes, start workers, or call T3.
Run this command from the Milknado checkout with the locked Python dependencies installed:

```sh
fixture=$(mktemp -d) && git init -q "$fixture" && \
  UV_PYTHON=3.12 uv run --no-sync python -m scripts.spikes.t3_planning \
    --project-root "$fixture" scripts/spikes/t3_planning/dependent.json "$fixture/plan.json"
```

Open `plan.json` in the temporary repo. Batch 0 contains `api`; batch 1 contains `use` and depends on batch 0.
The budget of 1 marks both batches oversized. The flag `--force-single-batch` permits a mega-batch; it does not force batches to merge.
Bad input exits 1 without a plan. The output file must not exist.

Version 1 is a spike envelope around the MCP change lists. Milknado validates those lists as part of its own v2 manifest.
The CLI composes public slice functions: `decode_manifest`, the batching solver, and its mega-batch threshold.
It does not call the [application planner](../../../src/milknado/app/plan_batches.py#L100-L116), because that call refreshes the index and appends telemetry.
The `plan` object has the same fields as the MCP `milknado_plan_batches` result.

## Intelligence and writes

The CLI writes one file: its output. It writes nothing in the project root.
It writes the JSON to a temporary file in the output directory.
Then it hard-links that file to the output path and deletes the temporary file.
The output has mode 0600, less the umask, so only its owner can read it. The output directory must support hard links.
A failed write deletes the temporary file, so the output is never partial or empty.
An existing output stops the run before planning.
The link never replaces a file. If another run publishes a plan to the same path during the solve, the CLI exits 1 and keeps that plan.

If `.code-review-graph/graph.db` exists, the CLI copies it, and its `-wal` file if present, to a temporary directory.
The CRG adapter reads that copy, and the CLI then deletes the directory.
Opening a CRG `GraphStore` sets WAL mode and runs schema migrations, so a read in place writes files next to the database.
The solver reads the copy during planning. Damage can be in any page, so a read can fail at the header or at a later file lookup.
If a read fails, the CLI prints one warning and plans again from the explicit dependencies only.
The CLI never runs the CRG executable, so a missing or stale index stays missing or stale.
Do not run a CRG build during planning, because a copy taken during a write can be inconsistent.
The result field `crg_index_present` reports that an index file existed and the plan used it. It does not report freshness or file coverage.
The CLI does not append telemetry to `.milknado/calibration.jsonl`.
See [CRG adapter](../../../src/milknado/adapters/crg.py#L61-L135).

In one manual run, `code-review-graph build` stored absolute file paths, and the solver queried relative change paths.
No query matched, so that plan used only the explicit dependencies.

## Next extraction cut

One Python 3.12 process with `-X importtime` and no CRG executable measured cumulative import times of
14 ms for the initial application seam, 212 ms for OR-Tools CP-SAT, and 152 ms for the deferred CRG adapter.
These are one-run measurements of an earlier version, not a performance benchmark. The CLI now imports the CRG adapter at startup.

This CLI is the cut in script form. It keeps manifest validation, the solver, and serialization together.
It takes intelligence from an existing index or from no index, and it leaves index refresh and telemetry to Milknado orchestration.
A library extraction moves that composition into a callable that accepts a caller-supplied CRG port.
That extraction must also resolve the path mismatch between change paths and the index.
Leave runtime launch, graph application, admission, and acceptance outside the call.
This spike reuses installed Milknado code; it does not extract a standalone library or implement the external CRG engine.