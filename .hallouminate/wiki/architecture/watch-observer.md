# Durable run observer

`milknado watch --project-root <repo>` is a read-only Textual observer for runs started by another process.[^1]

## Snapshot boundary

Each poll uses one SQLite read transaction through `read_observer_snapshot`. The connection uses `mode=ro` and `query_only=ON`.[^2]

The hot path does not run integrity checks, migrations, or database backups.[^2]

The graph slice owns `DurableRun` and `ObserverSnapshot`. It returns bounded runs with descriptions, the root description, and exact availability.[^3]

One bounded SQL query applies ready-node rules and file ownership precedence. It does not build the graph conflict-pair projection.[^3]

## Durable-state limits

The runs table stores only `running`, `done`, and `failed`. It does not store a separate stopped state.

The observer reports zero stopped runs and maps `done` to completed.[^4]

The runs table does not store live progress, ETA, attempt data, or pending guidance.

The observer preserves missing attempts and guidance as `None`. The shared view renders these values as unavailable.[^5]

The observer displays at most 50 recent runs and 20 status lines.

It resolves stored paths and selected directory children inside the project root.

A portable secure-open sequence uses `lstat`, one open, `fstat`, and stable identity comparison.[^6]

The sequence rejects non-regular files and changed identities before reading. It fails closed on symlink swap races.[^6]

The bounded tail reads through the verified descriptor. An inode, size, and modification-time cache avoids unchanged reads.[^6]

## Presentation

`ExecutionSnapshotApp` accepts the minimal `ExecutionSnapshotSource` protocol and owns snapshot-only presentation.[^7]

`ExecutionApp` adds commands and requires `ExecutionController`. `WatchApp` uses the snapshot base through a protocol-compatible adapter.[^7]

Watch mode removes guidance, cancel, and force-stop bindings. Quit exits only the observer process.[^7]

[^1]: src/milknado/cli/run.py:97-115; src/milknado/cli/__init__.py:25,61-62
[^2]: src/milknado/domains/graph/observer.py:73-79,107-121
[^3]: src/milknado/domains/graph/observer.py:14-46,49-70,82-117; src/milknado/app/watch.py:48-68
[^4]: src/milknado/domains/graph/_persistence.py:219-233; src/milknado/app/watch.py:59-67
[^5]: src/milknado/app/watch.py:70-101; src/milknado/app/run_view.py:68-95
[^6]: src/milknado/app/watch.py:28-34,103-146; tests/test_watch.py:81-105
[^7]: src/milknado/app/run_source.py:1-12; src/milknado/app/run_view_app.py:26-73; src/milknado/app/run_tui.py:23-60; src/milknado/app/watch_tui.py:20-71
