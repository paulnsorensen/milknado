"""Private wire schema for invocation-bound tmux execution receipts."""

from __future__ import annotations

from typing import Annotated, TypeAlias

import msgspec

_PositivePid = Annotated[int, msgspec.Meta(gt=0)]


class PreparedReceipt(
    msgspec.Struct, frozen=True, tag="PREPARED", tag_field="state", forbid_unknown_fields=True
):
    nonce: str


class FailedReceipt(
    msgspec.Struct, frozen=True, tag="EXEC_FAILED", tag_field="state", forbid_unknown_fields=True
):
    nonce: str
    error: str


class ExecutedReceipt(
    msgspec.Struct, frozen=True, tag="EXECUTED", tag_field="state", forbid_unknown_fields=True
):
    nonce: str
    pid: _PositivePid
    start_token: float | None


class FinishedReceipt(
    msgspec.Struct, frozen=True, tag="FINISHED", tag_field="state", forbid_unknown_fields=True
):
    nonce: str
    pid: _PositivePid
    returncode: int


Receipt: TypeAlias = PreparedReceipt | FailedReceipt | ExecutedReceipt | FinishedReceipt
