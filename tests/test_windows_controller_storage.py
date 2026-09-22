from __future__ import annotations

import contextlib
import hashlib
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from milknado.domains.common import CONTROLLER_MASTER_ENV
from milknado.domains.graph import ControllerAuthorizationError, MikadoGraph
from milknado.domains.graph import _windows_controller_storage as windows_storage
from milknado.domains.graph import controller_capability as capability

pytestmark = pytest.mark.skipif(os.name != "nt", reason="requires native Windows security APIs")


def _record(state: Path) -> Path:
    return next((state / "milknado" / "controllers").iterdir())


def _registered_hash(db_path: Path) -> str:
    with contextlib.closing(sqlite3.connect(db_path)) as conn:
        row = cast(
            tuple[str] | None,
            conn.execute(
                "SELECT master_hash FROM controller_master WHERE singleton = 1"
            ).fetchone(),
        )
    assert row is not None
    return row[0]


def _junction(link: Path, target: Path) -> None:
    _ = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True)


def test_windows_store_round_trips_4096_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "state"
    monkeypatch.setenv("XDG_STATE_HOME", str(state))
    master = b"x" * 4096
    master_hash = hashlib.sha256(master).hexdigest()

    windows_storage.publish_if_missing(
        windows_storage.store_dir() / master_hash, master_hash, master
    )

    assert (
        windows_storage.load_credential(windows_storage.store_dir() / master_hash, master_hash)
        == master
    )


def test_windows_allows_concurrent_reads_across_graphs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import win32con  # pyright: ignore[reportMissingModuleSource]

    state = tmp_path / "state"
    master = "concurrent-read-master"
    monkeypatch.setenv("XDG_STATE_HOME", str(state))
    monkeypatch.setenv(CONTROLLER_MASTER_ENV, master)
    first_db = tmp_path / "first.db"
    first_graph = MikadoGraph(first_db)
    try:
        first_graph.register_controller_master()
    finally:
        first_graph.close()

    record = _record(state)
    credential = record.read_bytes()
    handle = windows_storage._open(  # pyright: ignore[reportPrivateUsage]
        record, win32con.GENERIC_READ
    )
    second_db = tmp_path / "second.db"
    try:
        second_graph = MikadoGraph(second_db)
        try:
            second_graph.register_controller_master()
        finally:
            second_graph.close()
    finally:
        windows_storage._close(handle, record)  # pyright: ignore[reportPrivateUsage]

    assert credential == master.encode()
    expected_hash = hashlib.sha256(master.encode()).hexdigest()
    assert _registered_hash(first_db) == expected_hash
    assert _registered_hash(second_db) == expected_hash
    assert hashlib.sha256(record.read_bytes()).hexdigest() == expected_hash
    assert record.read_bytes() == credential


def test_windows_rejects_unsafe_acl_and_corrupt_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import win32con  # pyright: ignore[reportMissingModuleSource]
    import win32security  # pyright: ignore[reportMissingModuleSource]

    state = tmp_path / "state"
    monkeypatch.setenv("XDG_STATE_HOME", str(state))
    graph = MikadoGraph(tmp_path / "graph.db")
    graph.register_controller_master()
    graph.close()
    record = _record(state)
    windows_storage._native()  # pyright: ignore[reportPrivateUsage]
    sid = windows_storage._token_user_sid()  # pyright: ignore[reportPrivateUsage]
    dacl = win32security.ACL()
    dacl.AddAccessAllowedAce(
        win32security.ACL_REVISION,
        win32con.GENERIC_ALL,
        sid,
    )
    dacl.AddAccessAllowedAce(
        win32security.ACL_REVISION,
        win32con.GENERIC_READ,
        win32security.CreateWellKnownSid(  # pyright: ignore[reportUnknownMemberType]
            win32security.WinWorldSid, None
        ),
    )
    descriptor = win32security.GetFileSecurity(
        str(record), win32security.DACL_SECURITY_INFORMATION
    )
    descriptor.SetSecurityDescriptorDacl(1, dacl, 0)
    win32security.SetFileSecurity(str(record), win32security.DACL_SECURITY_INFORMATION, descriptor)

    graph = MikadoGraph(tmp_path / "graph.db")
    try:
        with pytest.raises(ControllerAuthorizationError, match="current-user-only"):
            graph.register_controller_master()
    finally:
        graph.close()


def test_windows_rejects_root_and_record_junctions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    link = tmp_path / "root-link"
    _junction(link, root)
    monkeypatch.setenv("XDG_STATE_HOME", str(link))
    with pytest.raises(ControllerAuthorizationError, match="unsafe object"):
        _ = capability._store_dir()  # pyright: ignore[reportPrivateUsage]

    state = tmp_path / "state"
    monkeypatch.setenv("XDG_STATE_HOME", str(state))
    graph = MikadoGraph(tmp_path / "graph.db")
    graph.register_controller_master()
    graph.close()
    record = _record(state)
    record.unlink()
    target = tmp_path / "record-target"
    target.mkdir()
    _junction(record, target)
    graph = MikadoGraph(tmp_path / "graph.db")
    try:
        with pytest.raises(ControllerAuthorizationError):
            graph.register_controller_master()
    finally:
        graph.close()


def test_windows_worker_context_cannot_register(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    monkeypatch.setenv("MILKNADO_WORKER_CONTEXT", "1")
    try:
        with pytest.raises(ControllerAuthorizationError, match="worker context"):
            graph.register_controller_master()
    finally:
        graph.close()


def test_windows_rejects_oversized_and_corrupt_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "state"
    monkeypatch.setenv("XDG_STATE_HOME", str(state))
    graph = MikadoGraph(tmp_path / "graph.db")
    graph.register_controller_master()
    graph.close()
    record = _record(state)

    _ = record.write_bytes(b"x" * 4097)
    graph = MikadoGraph(tmp_path / "graph.db")
    try:
        with pytest.raises(ControllerAuthorizationError, match="unsafe size"):
            graph.register_controller_master()
    finally:
        graph.close()

    _ = record.write_bytes(b"corrupt")
    graph = MikadoGraph(tmp_path / "graph.db")
    try:
        with pytest.raises(ControllerAuthorizationError, match="corrupt"):
            graph.register_controller_master()
    finally:
        graph.close()


def test_windows_cli_import_does_not_require_posix_locking() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import milknado.cli"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
