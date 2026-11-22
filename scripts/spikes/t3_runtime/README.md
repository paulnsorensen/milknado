# T3 execution adapter spike

Run `UV_PYTHON=3.12 uv run python -m scripts.spikes.t3_runtime.demo` from the repository root.
The demo uses a loopback HTTP server and a temporary SQLite journal.
The fixture drops the first launch reply after it records the request.
A new journal instance opens the same database and retries with the same key.
The demo accepts A before it starts B.

The adapter uses the tool names and consumed result fields from T3 commit
`6ab633f6d6e8273f08a060137ab64463638787ce`. The HTTP fixture is not an MCP server.
It does not verify live authentication, launch deduplication, provider outcomes, or process shutdown.
There is no configured live launch path.

Use one journal for one workspace under one operator. Do not share that workspace with another controller.
SQLite serializes launch binding and protects acceptance with an attempt and run identity check.
Run completion does not accept work. An interrupt reply does not establish that processes exited.
Unknown runs do not trigger fallback, cleanup, or a new request key.
The operator must reconcile the workspace before explicitly superseding an unaccepted completed run.
The spike refuses requests that require permission approval.

Defer production transport, approval handling, workspace ownership enforcement, and process reconciliation.
No production path imports this spike.
