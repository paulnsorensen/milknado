from dataclasses import replace
from http.client import RemoteDisconnected
from pathlib import Path
from sqlite3 import connect

import pytest

from scripts.spikes.t3_runtime.backend import Journal, Unresolved
from scripts.spikes.t3_runtime.demo import Fixture, serve


def test_dependency_and_recovery(tmp_path: Path) -> None:
    fixture = Fixture()
    path = tmp_path / "journal.db"
    with serve(fixture) as backend:
        journal = Journal(path, backend)
        journal.initialize()
        with pytest.raises(Unresolved, match="A is not accepted"):
            _ = journal.start("B")
        with pytest.raises(RemoteDisconnected):
            _ = journal.start("A")
        assert len(fixture.calls) == 1
        key = fixture.calls[0].arguments["clientRequestId"]
        with connect(path) as db:
            assert db.execute(
                "SELECT request_key, backend, run FROM attempt WHERE task='A'"
            ).fetchone() == (key, backend.identity, None)
        with pytest.raises(Unresolved, match="refusing fallback"):
            _ = Journal(path, replace(backend, environment="other")).start("A")
        recovered = Journal(path, backend)
        a = recovered.start("A")
        assert a.key == key
        assert fixture.calls[0] == fixture.calls[1]
        assert len(fixture.runs) == 1
        assert backend.wait(a.ref).status == "completed"
        with pytest.raises(Unresolved, match="A is not accepted"):
            _ = recovered.start("B")
        recovered.accept(a)
        b = recovered.start("B")
        assert b.ref.run != a.ref.run
        recovered.accept(b)


def test_stale_completion_cannot_accept_successor(tmp_path: Path) -> None:
    with serve(Fixture()) as backend:
        journal = Journal(tmp_path / "journal.db", backend)
        journal.initialize()
        with pytest.raises(RemoteDisconnected):
            _ = journal.start("A")
        old = journal.start("A")
        journal.supersede_reconciled(old)
        new = journal.start("A")
        assert new.key != old.key
        with pytest.raises(Unresolved, match="Stale"):
            journal.accept(old)
        with pytest.raises(Unresolved, match="A is not accepted"):
            _ = journal.start("B")
        journal.accept(new)
        assert journal.start("B").task == "B"


def test_cancel_ack_is_not_completion_and_approval_is_unsupported(tmp_path: Path) -> None:
    fixture = Fixture()
    fixture.drop_reply = False
    fixture.status = "running"
    with serve(fixture) as backend:
        journal = Journal(tmp_path / "journal.db", backend)
        journal.initialize()
        with pytest.raises(Unresolved, match="Permission approval"):
            _ = journal.start("A", needs_approval=True)
        assert not fixture.calls
        a = journal.start("A")
        assert backend.request_cancel(a.ref) == "interrupt_requested"
        assert fixture.calls[-1].arguments == a.ref.arguments()
        assert backend.inspect(a.ref) == "running"
        with pytest.raises(Unresolved, match="Expected a completed run"):
            journal.supersede_reconciled(a)
        with pytest.raises(Unresolved, match="Expected a completed run"):
            journal.accept(a)
        with pytest.raises(Unresolved, match="A is not accepted"):
            _ = journal.start("B")
