from __future__ import annotations

import hashlib
import importlib
import os
import secrets
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import ntsecuritycon  # pyright: ignore[reportMissingModuleSource]
    import pywintypes  # pyright: ignore[reportMissingModuleSource]
    import win32con  # pyright: ignore[reportMissingModuleSource]
    import win32file  # pyright: ignore[reportMissingModuleSource]
    import win32process  # pyright: ignore[reportMissingModuleSource]
    import win32security  # pyright: ignore[reportMissingModuleSource]
    from _win32typing import (  # pyright: ignore[reportMissingModuleSource]
        PyACL,
        PyHANDLE,
        PySECURITY_ATTRIBUTES,
        PySECURITY_DESCRIPTOR,
        PySID,
    )


_MAX_CREDENTIAL_BYTES = 4096
_ERROR_FILE_NOT_FOUND = 2
_ERROR_FILE_CONFLICT = (80, 183)


def store_dir() -> Path:
    explicit = os.environ.get("XDG_STATE_HOME", "").strip()
    local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
    if not explicit and not local_app_data:
        raise _authorization("LOCALAPPDATA is required for controller credential storage")
    root = Path(explicit or local_app_data)
    label = "XDG_STATE_HOME" if explicit else "LOCALAPPDATA"
    if not root.is_absolute():
        raise _authorization(f"{label} must be an absolute path")
    _native()
    sid = _token_user_sid()
    attributes = _security_attributes(sid)
    _ensure_directory(root, sid, attributes, protected=False)
    namespace = root / "milknado"
    _ensure_directory(namespace, sid, attributes, protected=True)
    store = namespace / "controllers"
    _ensure_directory(store, sid, attributes, protected=True)
    return store


def load_credential(path: Path, master_hash: str) -> bytes | None:
    _native()
    sid = _token_user_sid()
    try:
        handle = _open(path, win32con.GENERIC_READ)
    except pywintypes.error as exc:
        if _winerror(exc) == _ERROR_FILE_NOT_FOUND:
            return None
        raise _authorization(f"cannot read controller credential store at {path}: {exc}") from exc
    try:
        _validate_handle(handle, path, directory=False)
        _validate_protection(handle, path, sid)
        _status, raw = win32file.ReadFile(handle.handle, _MAX_CREDENTIAL_BYTES + 1)
        value = raw if isinstance(raw, bytes) else raw.encode()
        if len(value) > _MAX_CREDENTIAL_BYTES:
            raise _authorization(f"controller credential record has unsafe size: {path}")
    except pywintypes.error as exc:
        raise _authorization(f"cannot read controller credential store at {path}: {exc}") from exc
    finally:
        _close(handle, path)
    digest = hashlib.sha256(value).hexdigest()
    if not value or digest != master_hash:
        raise _authorization(f"controller credential record is corrupt: {path}")
    return value


def publish_if_missing(path: Path, master_hash: str, master: bytes) -> None:
    if len(master) > _MAX_CREDENTIAL_BYTES:
        raise _authorization("controller credential exceeds the 4096-byte limit")
    if load_credential(path, master_hash) is not None:
        return
    _native()
    sid = _token_user_sid()
    temporary = path.with_name(f".credential-{secrets.token_hex(16)}")
    try:
        _write_temporary(temporary, master, sid)
        try:
            move_file = cast(Callable[[str, str, int], None], win32file.MoveFileEx)
            move_file(str(temporary), str(path), win32file.MOVEFILE_WRITE_THROUGH)
        except pywintypes.error as exc:
            if _winerror(exc) not in _ERROR_FILE_CONFLICT:
                raise
        final = load_credential(path, master_hash)
        if final is None:
            raise _authorization(f"controller credential publication lost its destination: {path}")
    except pywintypes.error as exc:
        raise _authorization(
            f"cannot publish controller credential store at {path.parent}: {exc}"
        ) from exc
    finally:
        _delete_temporary(temporary)


def _native() -> None:
    if os.name != "nt":
        raise _authorization("Windows credential storage requires Windows")
    try:
        global ntsecuritycon, pywintypes, win32con, win32file, win32process, win32security
        import ntsecuritycon  # pyright: ignore[reportMissingModuleSource]
        import pywintypes  # pyright: ignore[reportMissingModuleSource]
        import win32con  # pyright: ignore[reportMissingModuleSource]
        import win32file  # pyright: ignore[reportMissingModuleSource]
        import win32process  # pyright: ignore[reportMissingModuleSource]
        import win32security  # pyright: ignore[reportMissingModuleSource]
    except ImportError as exc:
        raise _authorization("Windows credential storage requires pywin32") from exc


def _token_user_sid() -> PySID:
    token = None
    try:
        open_token = cast(Callable[[int, int], int], win32security.OpenProcessToken)
        token = open_token(win32process.GetCurrentProcess(), win32con.TOKEN_QUERY)
        get_token_information = cast(
            "Callable[[int, int], tuple[PySID, ...]]", win32security.GetTokenInformation
        )
        return get_token_information(token, win32security.TokenUser)[0]
    except pywintypes.error as exc:
        raise _authorization(f"cannot read current Windows user: {exc}") from exc
    finally:
        if token is not None:
            _close(token, Path("<process-token>"))


def _security_attributes(sid: PySID) -> PySECURITY_ATTRIBUTES:
    try:
        dacl = win32security.ACL()
        dacl.AddAccessAllowedAce(win32security.ACL_REVISION, win32con.GENERIC_ALL, sid)
        descriptor = win32security.SECURITY_DESCRIPTOR()
        descriptor.Initialize()
        descriptor.SetSecurityDescriptorOwner(sid, False)
        descriptor.SetSecurityDescriptorDacl(1, dacl, 0)
        set_control = cast(Callable[[int, int], None], descriptor.SetSecurityDescriptorControl)
        set_control(win32security.SE_DACL_PROTECTED, win32security.SE_DACL_PROTECTED)
        attributes = win32security.SECURITY_ATTRIBUTES()
        attributes.SECURITY_DESCRIPTOR = descriptor
        return attributes
    except pywintypes.error as exc:
        raise _authorization(f"cannot construct controller security attributes: {exc}") from exc


def _ensure_directory(
    path: Path, sid: PySID, attributes: PySECURITY_ATTRIBUTES, *, protected: bool
) -> None:
    if path.parent != path:
        _ensure_directory(path.parent, sid, attributes, protected=False)
    try:
        win32file.CreateDirectory(str(path), attributes)
    except pywintypes.error as exc:
        if _winerror(exc) not in _ERROR_FILE_CONFLICT:
            raise _authorization(
                f"cannot create controller credential store at {path}: {exc}"
            ) from exc
    _validate_path(path, sid, directory=True, protected=protected)


def _open(path: Path, access: int, *, directory: bool = False) -> PyHANDLE:
    flags = win32con.FILE_ATTRIBUTE_NORMAL | win32file.FILE_FLAG_OPEN_REPARSE_POINT
    if directory:
        flags |= win32file.FILE_FLAG_BACKUP_SEMANTICS
    return win32file.CreateFile(
        str(path), access, win32file.FILE_SHARE_READ, None, win32con.OPEN_EXISTING, flags, None
    )


def _write_temporary(path: Path, value: bytes, sid: PySID) -> None:
    handle = win32file.CreateFile(
        str(path),
        win32con.GENERIC_WRITE,
        0,
        _security_attributes(sid),
        win32con.CREATE_NEW,
        win32con.FILE_ATTRIBUTE_NORMAL,
        None,
    )
    try:
        _status, written = win32file.WriteFile(handle.handle, value)
        if written != len(value):
            raise _authorization(f"cannot publish controller credential store at {path.parent}")
        win32file.FlushFileBuffers(handle.handle)
    finally:
        _close(handle, path)


def _delete_temporary(path: Path) -> None:
    try:
        win32file.DeleteFile(str(path))
    except pywintypes.error as exc:
        if _winerror(exc) != _ERROR_FILE_NOT_FOUND:
            raise _authorization(
                f"cannot clean controller credential temporary at {path.parent}: {exc}"
            ) from exc


def _validate_path(path: Path, sid: PySID, *, directory: bool, protected: bool) -> None:
    try:
        handle = _open(path, win32con.READ_CONTROL, directory=directory)
    except pywintypes.error as exc:
        raise _authorization(
            f"cannot validate controller credential store at {path}: {exc}"
        ) from exc
    try:
        _validate_handle(handle, path, directory=directory)
        if protected:
            _validate_protection(handle, path, sid)
    except pywintypes.error as exc:
        raise _authorization(
            f"cannot validate controller credential store at {path}: {exc}"
        ) from exc
    finally:
        _close(handle, path)


def _validate_handle(handle: PyHANDLE, path: Path, *, directory: bool) -> None:
    get_file_info = cast(Callable[[int], tuple[int, ...]], win32file.GetFileInformationByHandle)
    attributes = get_file_info(handle.handle)[0]
    reparse = win32con.FILE_ATTRIBUTE_REPARSE_POINT
    is_directory = bool(attributes & win32con.FILE_ATTRIBUTE_DIRECTORY)
    if attributes & reparse or is_directory is not directory:
        raise _authorization(f"controller credential store has unsafe object: {path}")


def _validate_protection(handle: PyHANDLE, path: Path, sid: PySID) -> None:
    get_security_info = cast(
        "Callable[[int, int, int], PySECURITY_DESCRIPTOR]", win32security.GetSecurityInfo
    )
    descriptor: PySECURITY_DESCRIPTOR = get_security_info(
        handle.handle,
        win32security.SE_FILE_OBJECT,
        win32security.OWNER_SECURITY_INFORMATION | win32security.DACL_SECURITY_INFORMATION,
    )
    owner = descriptor.GetSecurityDescriptorOwner()
    dacl: PyACL | None = descriptor.GetSecurityDescriptorDacl()
    control_values = descriptor.GetSecurityDescriptorControl()
    if not control_values or not isinstance(control_values[0], int):
        raise _authorization("controller credential store has invalid security control")
    control = control_values[0]
    if (
        win32security.ConvertSidToStringSid(owner) != win32security.ConvertSidToStringSid(sid)
        or not dacl
        or not control & win32security.SE_DACL_PROTECTED
        or not _is_current_user_only_dacl(dacl, sid)
    ):
        raise _authorization(f"controller credential store is not current-user-only: {path}")


def _is_current_user_only_dacl(dacl: PyACL, sid: PySID) -> bool:
    if dacl.GetAceCount() != 1:
        return False
    (ace_type, ace_flags), mask, ace_sid = dacl.GetAce(0)
    mapped_mask = mask
    for generic, specific in (
        (win32con.GENERIC_READ, ntsecuritycon.FILE_GENERIC_READ),
        (win32con.GENERIC_WRITE, ntsecuritycon.FILE_GENERIC_WRITE),
        (win32con.GENERIC_EXECUTE, ntsecuritycon.FILE_GENERIC_EXECUTE),
        (win32con.GENERIC_ALL, ntsecuritycon.FILE_ALL_ACCESS),
    ):
        if mask & generic:
            mapped_mask = (mapped_mask & ~generic) | specific
    file_all = ntsecuritycon.FILE_ALL_ACCESS
    full_access = mapped_mask & file_all == file_all
    return (
        ace_type == win32security.ACCESS_ALLOWED_ACE_TYPE
        and ace_flags == 0
        and win32security.ConvertSidToStringSid(ace_sid)
        == win32security.ConvertSidToStringSid(sid)
        and full_access
    )


def _close(handle: PyHANDLE | int, path: Path) -> None:
    try:
        if isinstance(handle, int):
            win32file.CloseHandle(handle)
        else:
            handle.Close()
    except pywintypes.error as exc:
        raise _authorization(
            f"cannot close controller credential handle at {path}: {exc}"
        ) from exc


def _winerror(exc: BaseException) -> int | None:
    return cast(int | None, getattr(exc, "winerror", None))


def _authorization(message: str) -> RuntimeError:
    module = importlib.import_module("milknado.domains.graph.controller_capability")
    return cast(type[RuntimeError], module.ControllerAuthorizationError)(message)
