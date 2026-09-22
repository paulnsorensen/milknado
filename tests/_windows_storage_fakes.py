from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path, PosixPath
from types import ModuleType, SimpleNamespace
from typing import cast, final

import pytest

from milknado.domains.graph import _windows_controller_storage as storage
from milknado.domains.graph import controller_capability as capability


@final
class NativeError(Exception):
    def __init__(self, winerror: int) -> None:
        super().__init__(winerror)
        self.winerror = winerror


def _set_attr(module: ModuleType, name: str, value: object) -> None:
    setattr(module, name, value)


@dataclass
class Handle:
    state: NativeState
    path: Path
    handle: int
    closed: bool = False

    def Close(self) -> None:
        if self.state.fail_close is not None:
            raise NativeError(self.state.fail_close)
        self.closed = True
        self.state.closed.append(self.path)


@dataclass
class Dacl:
    sid: object
    mask: int
    extra: bool = False

    def GetAceCount(self) -> int:
        return 2 if self.extra else 1

    def GetAce(self, _index: int) -> tuple[tuple[int, int], int, object]:
        return ((1, 0), self.mask, self.sid)

    def AddAccessAllowedAce(self, _revision: int, mask: int, sid: object) -> None:
        self.mask = mask
        self.sid = sid


@dataclass
class Descriptor:
    sid: object
    dacl: Dacl
    control: tuple[int, int] = (0, 0)

    def Initialize(self) -> None:
        return None

    def SetSecurityDescriptorOwner(self, sid: object, _defaulted: bool) -> None:
        self.sid = sid

    def SetSecurityDescriptorDacl(self, _present: int, dacl: Dacl, _defaulted: int) -> None:
        self.dacl = dacl

    def SetSecurityDescriptorControl(self, interest: int, value: int) -> None:
        self.control = (self.control[0] & ~interest | value, self.control[1])

    def GetSecurityDescriptorOwner(self) -> object:
        return self.sid

    def GetSecurityDescriptorDacl(self) -> Dacl:
        return self.dacl

    def GetSecurityDescriptorControl(self) -> tuple[int, int]:
        return self.control


@dataclass
class NativeState:
    sid: str = "user"
    directories: set[Path] = field(default_factory=lambda: {PosixPath("/")})
    records: dict[Path, bytes] = field(default_factory=dict)
    descriptors: dict[Path, Descriptor] = field(default_factory=dict)
    calls: list[tuple[str, object, object]] = field(default_factory=list)
    closed: list[Path] = field(default_factory=list)
    closed_handles: list[int] = field(default_factory=list)
    fail_read: int | None = None
    fail_write: int | None = None
    fail_delete: int | None = None
    fail_close: int | None = None
    short_write: bool = False
    reparse: set[Path] = field(default_factory=set)
    next_handle: int = 10
    handles: dict[int, Handle] = field(default_factory=dict)

    def descriptor(self, path: Path) -> Descriptor:
        return self.descriptors.setdefault(path, Descriptor(self.sid, Dacl(self.sid, 0)))

    def handle(self, path: Path) -> Handle:
        result = Handle(self, path, self.next_handle)
        self.handles[result.handle] = result
        self.next_handle += 1
        return result

    def modules(self) -> dict[str, ModuleType]:
        con = _constants()
        return {
            "ntsecuritycon": _ntsecuritycon(),
            "pywintypes": _pywintypes(),
            "win32con": con,
            "win32file": _file_module(self, con),
            "win32process": _process_module(),
            "win32security": _security_module(self),
        }


@final
class _Constants(ModuleType):
    GENERIC_READ = 1
    GENERIC_WRITE = 2
    GENERIC_EXECUTE = 4
    GENERIC_ALL = 7
    READ_CONTROL = 8
    TOKEN_QUERY = 16
    FILE_ATTRIBUTE_NORMAL = 0
    FILE_ATTRIBUTE_REPARSE_POINT = 0x400
    FILE_ATTRIBUTE_DIRECTORY = 0x10
    OPEN_EXISTING = 3
    CREATE_NEW = 1


def _constants() -> _Constants:
    return _Constants("win32con")


def _copy_descriptor(attributes: object) -> Descriptor:
    source = getattr(attributes, "SECURITY_DESCRIPTOR", None)
    assert isinstance(source, Descriptor), "security descriptor is required for creation"
    return Descriptor(
        source.sid, Dacl(source.dacl.sid, source.dacl.mask, source.dacl.extra), source.control
    )


@final
class FileModule(ModuleType):
    def __init__(self, state: NativeState, con: _Constants) -> None:
        super().__init__("win32file")
        self.state: NativeState = state
        self.con: _Constants = con
        self.FILE_FLAG_OPEN_REPARSE_POINT = 0x400
        self.FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
        self.FILE_SHARE_READ = 1
        self.MOVEFILE_WRITE_THROUGH = 8
        self.CreateDirectory = self.create_directory
        self.CreateFile = self.create_file
        self.GetFileInformationByHandle = self.file_info
        self.ReadFile = self.read_file
        self.WriteFile = self.write_file
        self.FlushFileBuffers = self.flush
        self.MoveFileEx = self.move_file
        self.DeleteFile = self.delete_file
        self.CloseHandle = self.close_handle

    def create_directory(self, value: str, attributes: object) -> None:
        path = PosixPath(value)
        if path.parent == path:
            raise NativeError(5)
        if path in self.state.directories:
            raise NativeError(183)
        descriptor = _copy_descriptor(attributes)
        self.state.directories.add(path)
        self.state.descriptors[path] = descriptor

    def create_file(self, *args: object) -> Handle:
        value, access, share, security, disposition, _, _ = cast(
            tuple[str, int, int, object, int, object, object], args
        )
        path = PosixPath(value)
        self.state.calls.append(("open", access, share))
        exists = path in self.state.directories or path in self.state.records
        if disposition == self.con.OPEN_EXISTING and not exists:
            raise NativeError(2)
        if disposition == self.con.CREATE_NEW and exists:
            raise NativeError(80)
        if disposition == self.con.CREATE_NEW:
            descriptor = _copy_descriptor(security)
            self.state.records[path] = b""
            self.state.descriptors[path] = descriptor
        return self.state.handle(path)

    def _resolve(self, handle: int) -> Handle:
        return self.state.handles[handle]

    def file_info(self, handle: int) -> tuple[int]:
        target = self._resolve(handle)
        attributes = (
            self.con.FILE_ATTRIBUTE_DIRECTORY if target.path in self.state.directories else 0
        )
        if target.path in self.state.reparse:
            attributes |= self.con.FILE_ATTRIBUTE_REPARSE_POINT
        return (attributes,)

    def read_file(self, handle: int, _size: int) -> tuple[int, bytes]:
        if self.state.fail_read is not None:
            raise NativeError(self.state.fail_read)
        return 0, self.state.records[self._resolve(handle).path]

    def write_file(self, handle: int, value: bytes) -> tuple[int, int]:
        if self.state.fail_write is not None:
            raise NativeError(self.state.fail_write)
        self.state.records[self._resolve(handle).path] = value
        count = len(value) - 1 if self.state.short_write else len(value)
        return 0, count

    def move_file(self, source: str, destination: str, _flags: int) -> None:
        src, dest = PosixPath(source), PosixPath(destination)
        if dest in self.state.records:
            raise NativeError(183)
        self.state.records[dest] = self.state.records.pop(src)
        self.state.descriptors[dest] = self.state.descriptors.pop(src)

    def delete_file(self, value: str) -> None:
        if self.state.fail_delete is not None:
            raise NativeError(self.state.fail_delete)
        _ = self.state.records.pop(PosixPath(value), None)

    def flush(self, _handle: int) -> None:
        return None

    def close_handle(self, handle: int) -> None:
        self.state.closed_handles.append(handle)


def _file_module(state: NativeState, con: _Constants) -> ModuleType:
    return FileModule(state, con)


def _security_module(state: NativeState) -> ModuleType:
    module = ModuleType("win32security")

    def open_process_token(_process: object, _access: object) -> int:
        return 99

    def get_token_information(_token: object, _kind: object) -> tuple[str]:
        return (state.sid,)

    def get_security_info(handle: int, _kind: object, _info: object) -> Descriptor:
        return state.descriptor(state.handles[handle].path)

    def convert_sid(value: object) -> str:
        return str(value)

    _set_attr(module, "ACL_REVISION", 1)
    _set_attr(module, "SE_DACL_PROTECTED", 8)
    _set_attr(module, "SE_FILE_OBJECT", 1)
    _set_attr(module, "OWNER_SECURITY_INFORMATION", 2)
    _set_attr(module, "DACL_SECURITY_INFORMATION", 4)
    _set_attr(module, "ACCESS_ALLOWED_ACE_TYPE", 1)
    _set_attr(module, "TokenUser", 3)
    _set_attr(module, "ACL", lambda: Dacl("other", 0))
    _set_attr(module, "SECURITY_DESCRIPTOR", lambda: Descriptor("other", Dacl("other", 0)))
    _set_attr(module, "SECURITY_ATTRIBUTES", lambda: type("Attributes", (), {})())
    _set_attr(module, "OpenProcessToken", open_process_token)
    _set_attr(module, "GetTokenInformation", get_token_information)
    _set_attr(module, "GetSecurityInfo", get_security_info)
    _set_attr(module, "ConvertSidToStringSid", convert_sid)
    return module


def _process_module() -> ModuleType:
    module = ModuleType("win32process")
    _set_attr(module, "GetCurrentProcess", lambda: 1)
    return module


def _ntsecuritycon() -> ModuleType:
    module = ModuleType("ntsecuritycon")
    _set_attr(module, "FILE_GENERIC_READ", 1)
    _set_attr(module, "FILE_GENERIC_WRITE", 2)
    _set_attr(module, "FILE_GENERIC_EXECUTE", 4)
    _set_attr(module, "FILE_ALL_ACCESS", 7)
    return module


def _pywintypes() -> ModuleType:
    module = ModuleType("pywintypes")
    _set_attr(module, "error", NativeError)
    return module


@final
class NativeModules:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, state: NativeState) -> None:
        self.state = state
        native_os = SimpleNamespace(name="nt", environ=os.environ)
        monkeypatch.setattr(storage, "os", native_os)
        monkeypatch.setattr(capability, "os", native_os)
        monkeypatch.setattr(storage, "Path", PosixPath)
        for name, module in state.modules().items():
            monkeypatch.setitem(sys.modules, name, module)
