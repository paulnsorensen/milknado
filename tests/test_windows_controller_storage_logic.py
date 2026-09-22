from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path, PosixPath
from typing import NoReturn, cast

import pytest

from milknado.domains.graph import ControllerAuthorizationError
from milknado.domains.graph import _windows_controller_storage as storage
from milknado.domains.graph import controller_capability as capability
from tests._windows_storage_fakes import (
    Dacl,
    Descriptor,
    FileModule,
    NativeError,
    NativeModules,
    NativeState,
)


def _raise_native(*_args: object) -> NoReturn:
    raise NativeError(5)


def _ignore_native(*_args: object) -> None:
    return None


def _set_attr(module: object, name: str, value: object) -> None:
    setattr(module, name, value)


@pytest.fixture
def native(monkeypatch: pytest.MonkeyPatch) -> NativeState:
    state = NativeState()
    _ = NativeModules(monkeypatch, state)
    return state


def _root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "nested" / "state"
    monkeypatch.setenv("XDG_STATE_HOME", str(root))
    return root


def _record(state: NativeState, root: Path, value: bytes) -> Path:
    path = root / "milknado" / "controllers" / hashlib.sha256(value).hexdigest()
    state.records[path] = value
    state.descriptors[path] = Descriptor(state.sid, Dacl(state.sid, 7), control=(8, 0))
    return path


def test_store_dir_creates_missing_absolute_ancestors(
    native: NativeState, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path, monkeypatch)
    store = storage.store_dir()
    assert store == root / "milknado" / "controllers"
    assert {root, store.parent, store} <= native.directories
    assert all(path not in native.reparse for path in (root, store.parent, store))


def test_public_storage_round_trip_uses_shared_read_only_record_access(
    native: NativeState, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path, monkeypatch)
    store = storage.store_dir()
    master = b"secret"
    path = store / hashlib.sha256(master).hexdigest()
    storage.publish_if_missing(path, path.name, master)
    assert storage.load_credential(path, path.name) == master
    storage.publish_if_missing(path, path.name, master)
    reads = [call for call in native.calls if call[1] == 1]
    assert reads and all(call[2] == 1 for call in reads)
    assert path.parent == root / "milknado" / "controllers"


def test_controller_capability_routes_windows_credentials(
    native: NativeState, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path, monkeypatch)
    master = b"secret"
    digest = hashlib.sha256(master).hexdigest()
    capability._publish_if_missing(digest, master)  # pyright: ignore[reportPrivateUsage]
    path = capability._credential_path(digest)  # pyright: ignore[reportPrivateUsage]

    assert path == root / "milknado" / "controllers" / digest
    assert capability._load_credential(digest) == master  # pyright: ignore[reportPrivateUsage]
    assert native.records[path] == master


def test_load_rejects_reparse_acl_size_and_hash_failures(
    native: NativeState, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path, monkeypatch)
    value = b"secret"
    path = _record(native, root, value)
    descriptor = native.descriptor(path)
    monkeypatch.setattr(descriptor, "control", ())
    with pytest.raises(ControllerAuthorizationError, match="invalid security control"):
        _ = storage.load_credential(path, hashlib.sha256(value).hexdigest())
    descriptor.control = 8, 0
    native.reparse.add(path)
    with pytest.raises(ControllerAuthorizationError, match="unsafe object"):
        _ = storage.load_credential(path, hashlib.sha256(value).hexdigest())
    native.reparse.clear()
    descriptor = native.descriptor(path)
    descriptor.dacl.extra = True
    with pytest.raises(ControllerAuthorizationError, match="current-user-only"):
        _ = storage.load_credential(path, hashlib.sha256(value).hexdigest())
    descriptor.dacl.extra = False
    native.records[path] = b"x" * 4097
    with pytest.raises(ControllerAuthorizationError, match="unsafe size"):
        _ = storage.load_credential(path, "0" * 64)
    native.records[path] = b"wrong"
    with pytest.raises(ControllerAuthorizationError, match="corrupt"):
        _ = storage.load_credential(path, hashlib.sha256(value).hexdigest())


def test_load_translates_native_read_and_missing_record_errors(
    native: NativeState, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path, monkeypatch)
    store = storage.store_dir()
    missing = store / "missing"
    assert storage.load_credential(missing, "0" * 64) is None
    path = _record(native, root, b"secret")
    file = cast(FileModule, sys.modules["win32file"])
    original = file.CreateFile
    _set_attr(file, "CreateFile", _raise_native)
    with pytest.raises(ControllerAuthorizationError, match="cannot read"):
        _ = storage.load_credential(path, hashlib.sha256(b"secret").hexdigest())
    _set_attr(file, "CreateFile", original)
    native.fail_read = 5
    with pytest.raises(ControllerAuthorizationError, match="cannot read"):
        _ = storage.load_credential(path, hashlib.sha256(b"secret").hexdigest())


def test_publish_rejects_oversize_and_short_writes(
    native: NativeState, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path, monkeypatch)
    store = storage.store_dir()
    with pytest.raises(ControllerAuthorizationError, match="4096-byte"):
        storage.publish_if_missing(store / "large", "0" * 64, b"x" * 4097)
    native.short_write = True
    master = b"secret"
    path = store / hashlib.sha256(master).hexdigest()
    with pytest.raises(ControllerAuthorizationError, match="cannot publish"):
        storage.publish_if_missing(path, path.name, master)
    assert path not in native.records
    assert root in native.directories


def test_publish_validates_first_writer_conflict_destination(
    native: NativeState, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = _root(tmp_path, monkeypatch)
    store = storage.store_dir()
    master = b"secret"
    path = store / hashlib.sha256(master).hexdigest()
    file = cast(FileModule, sys.modules["win32file"])

    def conflict(_source: str, destination: str, _flags: int) -> None:
        native.records[PosixPath(destination)] = b"corrupt"
        native.descriptors[PosixPath(destination)] = Descriptor(
            native.sid, Dacl(native.sid, 7), control=(8, 0)
        )
        raise NativeError(183)

    _set_attr(file, "MoveFileEx", conflict)
    with pytest.raises(ControllerAuthorizationError, match="corrupt"):
        storage.publish_if_missing(path, path.name, master)

    _ = native.records.pop(path)
    _ = native.descriptors.pop(path)

    def matching(source: str, destination: str, _flags: int) -> None:
        source_path = PosixPath(source)
        destination_path = PosixPath(destination)
        native.records[destination_path] = native.records[source_path]
        native.descriptors[destination_path] = native.descriptors[source_path]
        raise NativeError(183)

    _set_attr(file, "MoveFileEx", matching)
    storage.publish_if_missing(path, path.name, master)
    assert storage.load_credential(path, path.name) == master


def test_publish_translates_native_write_move_and_cleanup_failures(
    native: NativeState, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = _root(tmp_path, monkeypatch)
    store = storage.store_dir()
    master = b"secret"
    path = store / hashlib.sha256(master).hexdigest()
    native.fail_write = 5
    with pytest.raises(ControllerAuthorizationError, match="cannot publish"):
        storage.publish_if_missing(path, path.name, master)
    native.fail_write = None
    file = cast(FileModule, sys.modules["win32file"])
    _set_attr(file, "MoveFileEx", _raise_native)
    with pytest.raises(ControllerAuthorizationError, match="cannot publish"):
        storage.publish_if_missing(path, path.name, master)
    native.fail_delete = 5
    with pytest.raises(ControllerAuthorizationError, match="cannot clean"):
        storage.publish_if_missing(path, path.name, master)


def test_store_dir_rejects_missing_relative_and_non_windows_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    with pytest.raises(ControllerAuthorizationError, match="LOCALAPPDATA"):
        _ = storage.store_dir()
    monkeypatch.setenv("XDG_STATE_HOME", "relative")
    with pytest.raises(ControllerAuthorizationError, match="absolute"):
        _ = storage.store_dir()
    monkeypatch.setattr(os, "name", "posix")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    with pytest.raises(ControllerAuthorizationError, match="requires Windows"):
        _ = storage.store_dir()


def test_native_import_failure_is_authorized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setenv("XDG_STATE_HOME", "/state")
    monkeypatch.setattr(storage, "Path", PosixPath)
    monkeypatch.setitem(sys.modules, "win32file", None)
    with pytest.raises(ControllerAuthorizationError, match="requires pywin32"):
        _ = storage.store_dir()


def test_token_user_failure_closes_open_token(
    native: NativeState, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _ = _root(tmp_path, monkeypatch)
    _ = storage.store_dir()
    assert native.closed_handles == [99]
    security = sys.modules["win32security"]
    _set_attr(security, "OpenProcessToken", _raise_native)
    with pytest.raises(ControllerAuthorizationError, match="current Windows user"):
        _ = storage.store_dir()


def test_directory_validation_rejects_native_open_and_info_failures(
    native: NativeState, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _root(tmp_path, monkeypatch)
    _ = storage.store_dir()
    file = cast(FileModule, sys.modules["win32file"])
    original = file.CreateFile

    def denied(*_args: object) -> NoReturn:
        raise NativeError(5)

    _set_attr(file, "CreateFile", denied)
    with pytest.raises(ControllerAuthorizationError, match="cannot validate"):
        _ = storage.store_dir()
    _set_attr(file, "CreateFile", original)
    _set_attr(file, "GetFileInformationByHandle", denied)
    with pytest.raises(ControllerAuthorizationError, match="cannot validate"):
        _ = storage.store_dir()
    assert root in native.directories


@pytest.mark.usefixtures("native")
def test_store_dir_translates_directory_creation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = _root(tmp_path, monkeypatch)
    file = cast(FileModule, sys.modules["win32file"])
    _set_attr(file, "CreateDirectory", _raise_native)
    with pytest.raises(ControllerAuthorizationError, match="cannot create"):
        _ = storage.store_dir()


def test_close_translates_native_handle_failure(
    native: NativeState, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path, monkeypatch)
    path = _record(native, root, b"secret")
    native.fail_close = 5
    with pytest.raises(ControllerAuthorizationError, match="cannot close"):
        _ = storage.load_credential(path, hashlib.sha256(b"secret").hexdigest())


def test_publish_rejects_lost_destination(
    native: NativeState, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = native
    _ = _root(tmp_path, monkeypatch)
    store = storage.store_dir()
    master = b"secret"
    path = store / hashlib.sha256(master).hexdigest()
    file = cast(FileModule, sys.modules["win32file"])
    _set_attr(file, "MoveFileEx", _ignore_native)
    with pytest.raises(ControllerAuthorizationError, match="publication"):
        storage.publish_if_missing(path, path.name, master)
