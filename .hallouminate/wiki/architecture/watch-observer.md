# Durable run observer

`milknado watch --project-root <repo>` is a read-only Textual observer for runs started by another process.[^1]

## Snapshot boundary

The observer polls every second. Each poll calls `MikadoGraph.open_snapshot`. This method opens SQLite in read-only mode and copies the database into memory. The observer reads and closes that copy. It never opens the normal writer session or changes the project database.[^2]

`WatchSnapshotSource` validates each runs-table row with `DurableRun`. It combines recent run rows with `get_execution_overview`. It then converts the data into the `ExecutionSnapshot` models that the execution TUI uses.[^3]

## Durable-state limits

The runs table stores only `running`, `done`, and `failed`. It does not store a separate stopped state. The observer therefore reports zero stopped runs and maps `done` to completed.[^4]

The runs table does not store live progress, ETA, attempt data, or pending guidance. The observer marks these values as unavailable. It uses stored file or directory paths for bounded log tails.[^5]

The observer displays at most 50 recent runs and 20 status lines. It accepts only log paths inside the project root. This check prevents a changed database from making the observer read unrelated files.[^6]

## Presentation

`WatchApp` reuses the execution panels and formatting. It removes guidance, cancel, and force-stop bindings. Quit exits only the observer process.[^7]

[^1]: src/milknado/cli/run.py:97-115; src/milknado/cli/__init__.py:25,61-62
[^2]: src/milknado/app/watch.py:56-68; src/milknado/domains/graph/graph.py:58-85
[^3]: src/milknado/app/watch.py:30-88
[^4]: src/milknado/domains/graph/_persistence.py:537,569-613; src/milknado/app/watch.py:79-87
[^5]: src/milknado/domains/graph/_persistence.py:219-233; src/milknado/app/watch.py:90-128
[^6]: src/milknado/app/watch.py:48-54,87,123-128
[^7]: src/milknado/app/watch_tui.py:37-76
