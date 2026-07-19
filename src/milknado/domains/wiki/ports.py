"""Outbound wiki-indexing contract owned by the wiki domain."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol


class WikiIndexStatus(StrEnum):
    REFRESHED = "refreshed"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


@dataclass(frozen=True)
class WikiIndexResult:
    status: WikiIndexStatus
    detail: str | None = None


class WikiIndexerPort(Protocol):
    def refresh(self, wiki_root: Path) -> WikiIndexResult: ...
