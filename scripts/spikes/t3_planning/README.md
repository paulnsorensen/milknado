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
The application seam validates, batches, and serializes the plan without a graph application call.
See [application planner](../../../src/milknado/app/plan_batches.py#L44-L116).

## Intelligence and writes

The JSON goes to a file because the external CRG CLI prints build progress to stdout.
The result field `crg_index_present` reports file presence, not index freshness, successful use, or coverage of any file.
The application logs its fallback if CRG setup fails. A missing Python package is an import failure, not this fallback.
See [CRG adapter](../../../src/milknado/adapters/crg.py#L61-L135).

The command above produced `.code-review-graph/graph.db` and its companion `.gitignore`.
The empty fixture confirms index setup, not impact enrichment. To exercise fallback, invoke the same module with
`env PATH="$fixture" .venv/bin/python` in place of the `uv run` prefix on a fresh fixture.
That run produced no CRG index, logged the missing executable, and preserved the explicit dependency.
Both runs appended one record to `.milknado/calibration.jsonl`. A mega-batch rejection happens before that append, but after possible CRG writes.
The CLI creates the output file before planning, so an existing output stops the run before any index or telemetry write.
A planning failure deletes that empty output file.
See [telemetry](../../../src/milknado/domains/planning/telemetry.py#L19-L41).

## Next extraction cut

One Python 3.12 process with `-X importtime` and no CRG executable measured cumulative import times of
14 ms for the initial application seam, 212 ms for OR-Tools CP-SAT, and 152 ms for the deferred CRG adapter.
These are one-run measurements, not a performance benchmark. Missing the executable does not remove the Python adapter dependency.

Keep manifest validation, the solver, and result serialization together in a callable that accepts caller-supplied intelligence.
Keep external index refresh and telemetry on the Milknado orchestration side of that cut.
Leave runtime launch, graph application, admission, and acceptance outside the call.
This spike reuses installed Milknado code; it does not extract a standalone library or implement the external CRG engine.
