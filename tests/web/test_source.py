from milknado.app.run_source import ExecutionSnapshot
from tests.web.support import FixtureSnapshotSource


def test_fixture_source_publishes_and_unsubscribes() -> None:
    fixture = FixtureSnapshotSource()
    seen: list[str] = []
    unsubscribe = fixture.subscribe(lambda snapshot: seen.append(snapshot.goal))
    fixture.publish(fixture.snapshot())
    unsubscribe()
    fixture.publish(
        ExecutionSnapshot(
            goal="second",
            active_runs=(),
            terminal_runs=(),
            completed=0,
            failed=0,
            stopped=0,
            available=0,
            event_lines=(),
            listener_errors=(),
            graph=None,
            node=None,
        )
    )
    assert seen == ["fixture goal"]
