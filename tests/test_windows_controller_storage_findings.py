from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

from milknado.domains.graph import ControllerAuthorizationError
from milknado.domains.graph import _windows_controller_storage as storage
from tests._windows_storage_fakes import Dacl, Descriptor, NativeError, NativeState
from tests.test_windows_controller_storage_logic import (
    native as native,
)


def _assert_restrictive(state: NativeState, path: Path) -> None:
    assert path in state.directories or path in state.records
    descriptor = state.descriptors[path]
    assert descriptor.sid == state.sid
    assert descriptor.dacl.GetAceCount() == 1
    _ace_type, mask, ace_sid = descriptor.dacl.GetAce(0)
    assert mask == 7
    assert ace_sid == state.sid
    assert descriptor.control[0] & 8


def test_creation_uses_supplied_restrictive_security_attributes(
    native: NativeState, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "nested" / "state"
    monkeypatch.setenv("XDG_STATE_HOME", str(root))
    unsafe = Descriptor("other", Dacl("other", 0, extra=True), control=(0, 0))
    native.descriptors[root] = unsafe
    store = storage.store_dir()
    master = b"secret"
    path = store / hashlib.sha256(master).hexdigest()
    native.descriptors[path] = unsafe
    storage.publish_if_missing(path, path.name, master)
    for created in (root, store.parent, store, path):
        _assert_restrictive(native, created)


@pytest.mark.usefixtures("native")
def test_security_attribute_construction_failure_is_authorized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    security = sys.modules["win32security"]
    failure = NativeError(5)

    def raise_native(*_args: object) -> None:
        raise failure

    monkeypatch.setattr(security, "ACL", raise_native)
    with pytest.raises(ControllerAuthorizationError, match="security attributes") as raised:
        _ = storage.store_dir()
    assert raised.value.__cause__ is failure


@pytest.mark.usefixtures("native")
def test_security_attribute_construction_does_not_translate_unrelated_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    security = sys.modules["win32security"]

    def raise_unrelated(*_args: object) -> None:
        raise ValueError("unexpected constructor failure")

    monkeypatch.setattr(security, "ACL", raise_unrelated)
    with pytest.raises(ValueError, match="unexpected constructor failure"):
        _ = storage.store_dir()
