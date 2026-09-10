"""Wire framing and JSON helpers for OMP RPC sessions."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping
from typing import cast

import msgspec


class OmpMessage(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    role: str
    content: str | list[dict[str, object]] = ""
    id: str | None = None
    tool_call_id: str | None = None
    stop_reason: str | None = None
    custom_type: str | None = None
    is_error: bool = False
    error_message: str | None = None


class OmpAssistantEvent(msgspec.Struct, frozen=True, kw_only=True):
    type: str
    delta: str | None = None
    content: str | None = None
    message: OmpMessage | None = None
    error: object = None


class OmpResponse(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    session_id: str | None = None
    agent_invoked: bool | None = None


class OmpFrame(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    type: str
    id: str | None = None
    command: str | None = None
    success: bool | None = None
    data: OmpResponse | str | None = None
    message: OmpMessage | str | None = None
    messages: list[OmpMessage] | None = None
    assistant_message_event: OmpAssistantEvent | None = None
    is_terminal: bool | None = None
    agent_invoked: bool | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    result: object = None
    partial_result: object = None
    delta: str | None = None
    is_error: bool = False
    error: object = None
    error_message: str | None = None
    status_text: str | None = None
    method: str | None = None
    title: str | None = None
    options: list[str] | None = None
    supported_protocol_versions: list[int] | None = None
    max_reassembled_frame_bytes: int | None = None
    chunk_id: str | None = None
    index: int | None = None
    count: int | None = None
    byte_length: int | None = None


def as_dict(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    raw = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in raw):
        return None
    return cast(dict[str, object], raw)


def text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, OmpMessage):
        return text(value.content)
    if isinstance(value, list):
        return "".join(text(item) for item in cast(list[object], value))
    data = as_dict(value)
    if data is None:
        return ""
    for key in ("text", "output", "content", "message", "error", "errorMessage"):
        if key in data and (result := text(data[key])):
            return result
    return ""


def encode(payload: Mapping[str, object]) -> bytes:
    return msgspec.json.encode(payload) + b"\n"


def rpc_command(argv: tuple[str, ...]) -> tuple[str, ...]:
    command: list[str] = []
    mode_written = False
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "--mode":
            index += 2
        elif argument.startswith("--mode="):
            index += 1
        else:
            command.append(argument)
            index += 1
            continue
        if not mode_written:
            command.extend(("--mode", "rpc"))
            mode_written = True
    if not mode_written:
        command.extend(("--mode", "rpc"))
    return tuple(command)


class ChunkDecoder:
    """Validate and reassemble negotiated lossless ``rpc_chunk`` frames."""

    def __init__(self, max_bytes: int = 64 * 1024 * 1024) -> None:
        self.max_bytes: int = max_bytes
        self._chunk_id: str | None = None
        self._count: int = 0
        self._next: int = 0
        self._length: int = 0
        self._data: bytearray = bytearray()

    def set_limit(self, advertised: object) -> None:
        if isinstance(advertised, int) and advertised > 0:
            self.max_bytes = min(advertised, self.max_bytes)

    def consume(self, frame: OmpFrame) -> tuple[bytes | None, str | None]:
        chunk_id, index, count, length, data = (
            frame.chunk_id,
            frame.index,
            frame.count,
            frame.byte_length,
            frame.data,
        )
        valid_ints = all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (index, count, length)
        )
        if not isinstance(chunk_id, str) or not valid_ints or not isinstance(data, str):
            self.reset()
            return None, "Invalid OMP rpc_chunk metadata"
        index = cast(int, index)
        count = cast(int, count)
        length = cast(int, length)
        if index < 0 or count <= 0 or index >= count or length < 0 or length > self.max_bytes:
            self.reset()
            return None, "OMP rpc_chunk exceeds advertised limits"
        if self._chunk_id is None:
            if index != 0:
                return None, "OMP rpc_chunk sequence must start at index 0"
            self._chunk_id, self._count, self._length = chunk_id, count, length
        elif (chunk_id, count, length, index) != (
            self._chunk_id,
            self._count,
            self._length,
            self._next,
        ):
            self.reset()
            return None, "OMP rpc_chunk sequence was interleaved or interrupted"
        try:
            piece = base64.b64decode(data, validate=True)
        except (ValueError, binascii.Error) as exc:
            self.reset()
            return None, f"Invalid OMP rpc_chunk data: {exc}"
        self._data.extend(piece)
        if len(self._data) > self._length:
            self.reset()
            return None, "OMP rpc_chunk byte length mismatch"
        self._next += 1
        if self._next < self._count:
            return None, None
        raw = bytes(self._data)
        if len(raw) != self._length:
            self.reset()
            return None, "OMP rpc_chunk ended with wrong byte length"
        self.reset()
        try:
            _ = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            return None, f"OMP rpc_chunk is not UTF-8: {exc}"
        return raw, None

    def reset(self) -> None:
        self._chunk_id = None
        self._count = self._next = self._length = 0
        self._data.clear()
