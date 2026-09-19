from milknado.app.run_source import ExecutionSnapshot
from tests.web.support import FixtureSnapshotSource, client, headers


def test_snapshot_returns_goal_and_capabilities() -> None:
    response = client()[0].get("/api/snapshot", headers=headers())  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    assert response.status_code == 200  # pyright: ignore[reportUnknownMemberType]
    payload = response.json()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    assert payload["goal"] == "fixture goal"
    assert payload["listener_errors"] == ["fixture listener error"]
    assert response.json()["capabilities"]["force_stop"]["available"] is False  # pyright: ignore[reportUnknownMemberType]


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
