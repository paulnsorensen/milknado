from __future__ import annotations

import hashlib
import os
import sqlite3
from pathlib import Path
from typing import cast

import pytest

from milknado.domains.common import CONTROLLER_MASTER_ENV, WORKER_CONTEXT_ENV
from milknado.domains.graph import ControllerAuthorizationError, MikadoGraph
from milknado.domains.graph import controller_capability as capability


def _store(state: Path) -> Path:
    """Return the controller credential store beneath a state root."""
    return state / "milknado" / "controllers"


def _raise_denied(*_args: object, **_kwargs: object) -> object:
    """Simulate an operating-system permission failure."""
    raise OSError("denied")


def _raise_replace(*_args: object, **_kwargs: object) -> Path:
    """Simulate failure while publishing a credential record."""
    raise OSError("replace denied")


def test_startup_rejects_worker_blank_long_and_relative_contexts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject unsafe controller registration inputs and worker contexts."""
    graph = MikadoGraph(tmp_path / "graph.db")
    try:
        monkeypatch.setenv(WORKER_CONTEXT_ENV, "1")
        with pytest.raises(ControllerAuthorizationError, match="worker context"):
            graph.register_controller_master()
        monkeypatch.delenv(WORKER_CONTEXT_ENV)
        monkeypatch.setenv(CONTROLLER_MASTER_ENV, " ")
        with pytest.raises(ControllerAuthorizationError, match="set but empty"):
            graph.register_controller_master()
        monkeypatch.setenv(CONTROLLER_MASTER_ENV, "x" * 4097)
        with pytest.raises(ControllerAuthorizationError, match="4096-byte"):
            graph.register_controller_master()
        monkeypatch.delenv(CONTROLLER_MASTER_ENV)
        monkeypatch.setenv("XDG_STATE_HOME", "relative-state")
        with pytest.raises(ControllerAuthorizationError, match="absolute"):
            graph.register_controller_master()
    finally:
        graph.close()


def test_legacy_missing_and_corrupt_records_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail closed when a registered credential record is absent or corrupt."""
    state = tmp_path / "state"
    monkeypatch.setenv("XDG_STATE_HOME", str(state))
    monkeypatch.setenv(CONTROLLER_MASTER_ENV, "legacy-secret")
    graph = MikadoGraph(tmp_path / "graph.db")
    graph.register_controller_master()
    graph.close()
    record = next(_store(state).iterdir())
    record.unlink()

    monkeypatch.delenv(CONTROLLER_MASTER_ENV)
    graph = MikadoGraph(tmp_path / "graph.db")
    try:
        with pytest.raises(RuntimeError, match="original secret"):
            graph.register_controller_master()
    finally:
        graph.close()

    monkeypatch.setenv(CONTROLLER_MASTER_ENV, "legacy-secret")
    graph = MikadoGraph(tmp_path / "graph.db")
    _ = graph.register_controller_master()
    graph.close()
    _ = record.write_bytes(b"corrupt")
    monkeypatch.delenv(CONTROLLER_MASTER_ENV)
    graph = MikadoGraph(tmp_path / "graph.db")
    try:
        with pytest.raises(RuntimeError, match="corrupt"):
            graph.register_controller_master()
    finally:
        graph.close()


@pytest.mark.skipif(os.name == "nt", reason="permission and symlink checks need POSIX")
def test_credential_record_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject credential records with unsafe type, ownership, or permissions."""
    state = tmp_path / "state"
    monkeypatch.setenv("XDG_STATE_HOME", str(state))
    monkeypatch.setenv(CONTROLLER_MASTER_ENV, "record-secret")
    graph = MikadoGraph(tmp_path / "graph.db")
    _ = graph.register_controller_master()
    graph.close()
    record = next(_store(state).iterdir())
    target = tmp_path / "target"
    _ = target.write_bytes(b"record-secret")
    record.unlink()
    record.symlink_to(target)
    monkeypatch.delenv(CONTROLLER_MASTER_ENV)
    graph = MikadoGraph(tmp_path / "graph.db")
    try:
        with pytest.raises(RuntimeError, match="symlink"):
            graph.register_controller_master()
    finally:
        graph.close()

    record.unlink()
    record.mkdir()
    graph = MikadoGraph(tmp_path / "graph.db")
    try:
        with pytest.raises(RuntimeError, match="owner-owned"):
            graph.register_controller_master()
    finally:
        graph.close()
    record.rmdir()

    _ = record.write_bytes(b"record-secret")
    record.chmod(0o644)
    graph = MikadoGraph(tmp_path / "graph.db")
    try:
        with pytest.raises(RuntimeError, match="unsafe permissions"):
            graph.register_controller_master()
    finally:
        graph.close()


def test_store_directory_routes_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setenv("XDG_STATE_HOME", r"C:\\state")
    with pytest.raises(ControllerAuthorizationError, match="requires pywin32"):
        _ = capability._store_dir()  # pyright: ignore[reportPrivateUsage]


def test_store_directory_permissions_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject credential stores that are shared or traverse symlinks."""
    state = tmp_path / "state"
    namespace = state / "milknado"
    namespace.mkdir(parents=True)
    os.chmod(namespace, 0o750)
    monkeypatch.setenv("XDG_STATE_HOME", str(state))
    with pytest.raises(RuntimeError, match="owner-only"):
        _ = capability._store_dir()  # pyright: ignore[reportPrivateUsage]

    (namespace / "controllers").rmdir()
    namespace.rmdir()
    namespace.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink"):
        _ = capability._store_dir()  # pyright: ignore[reportPrivateUsage]


def test_malformed_hash_and_storage_errors_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Convert malformed hashes and storage errors into authorization failures."""
    state = tmp_path / "state"
    monkeypatch.setenv("XDG_STATE_HOME", str(state))
    monkeypatch.setenv(CONTROLLER_MASTER_ENV, "hash-secret")
    db = tmp_path / "graph.db"
    graph = MikadoGraph(db)
    graph.register_controller_master()
    graph.close()
    with sqlite3.connect(db) as conn:
        _ = conn.execute("UPDATE controller_master SET master_hash = 'bad'")
        conn.commit()
    monkeypatch.delenv(CONTROLLER_MASTER_ENV)
    graph = MikadoGraph(db)
    try:
        with pytest.raises(RuntimeError, match="malformed"):
            graph.register_controller_master()
    finally:
        graph.close()

    monkeypatch.setattr(os, "open", _raise_denied)
    with pytest.raises(RuntimeError, match="cannot read"):
        _ = capability._load_credential("a" * 64)  # pyright: ignore[reportPrivateUsage]

    monkeypatch.undo()
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "write-state"))
    monkeypatch.setattr(Path, "replace", _raise_replace)
    with pytest.raises(RuntimeError, match="cannot publish"):
        _ = capability._publish_if_missing(  # pyright: ignore[reportPrivateUsage]
            hashlib.sha256(b"write-secret").hexdigest(), b"write-secret"
        )


def test_relative_home_rejects_registration_before_storage_or_graph_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "graph.db"
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.setenv("HOME", "relative-home")
    monkeypatch.chdir(tmp_path)
    graph = MikadoGraph(db_path)
    try:
        with pytest.raises(ControllerAuthorizationError, match="HOME must be an absolute path"):
            graph.register_controller_master()
    finally:
        graph.close()

    assert not (tmp_path / "relative-home").exists()
    with sqlite3.connect(db_path) as conn:
        registration = cast(
            tuple[int], conn.execute("SELECT COUNT(*) FROM controller_master").fetchone()
        )
    assert registration == (0,)


def test_surrogate_override_rejects_without_storage_or_graph_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "graph.db"
    graph = MikadoGraph(db_path)
    monkeypatch.setattr(os, "environ", {CONTROLLER_MASTER_ENV: "\ud800"})
    try:
        with pytest.raises(ControllerAuthorizationError, match="cannot be encoded"):
            graph.register_controller_master()
    finally:
        graph.close()

    with sqlite3.connect(db_path) as conn:
        registration = cast(
            tuple[int], conn.execute("SELECT COUNT(*) FROM controller_master").fetchone()
        )
    assert registration == (0,)


def test_blob_controller_hash_rejects_without_type_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "graph.db"
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    graph = MikadoGraph(db_path)
    graph.register_controller_master()
    graph.close()
    with sqlite3.connect(db_path) as conn:
        _ = conn.execute(
            "UPDATE controller_master SET master_hash = ?", (sqlite3.Binary(b"a" * 64),)
        )
        conn.commit()
    monkeypatch.delenv(CONTROLLER_MASTER_ENV)
    graph = MikadoGraph(db_path)
    try:
        with pytest.raises(ControllerAuthorizationError, match="malformed"):
            graph.register_controller_master()
    finally:
        graph.close()
