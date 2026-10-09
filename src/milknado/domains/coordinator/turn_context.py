"""Trusted identity shared by one coordinator turn's internal operations."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TurnContext:
    conn: sqlite3.Connection
    session_id: str
    command_id: str
