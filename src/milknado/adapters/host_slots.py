"""Host-wide worker slot pool built on POSIX advisory file locks.

The kernel drops a lock when its holder closes the descriptor or dies, so a
killed supervisor never strands a slot. Windows has no ``fcntl``; the pool is
disabled there with a logged warning.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from milknado.domains.common import SlotLease
from milknado.domains.graph import HostCapacityFull

_logger = logging.getLogger(__name__)


def slot_directory() -> Path:
    """Return the pool directory under ``$XDG_STATE_HOME`` (default ``~/.local/state``)."""
    base = os.environ.get("XDG_STATE_HOME", "").strip() or str(Path.home() / ".local" / "state")
    return Path(base) / "milknado" / "worker-slots"


class _NoLease:
    def release(self) -> None:
        return None


class _FlockLease:
    def __init__(self, fd: int) -> None:
        self._fd: int | None = fd

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            os.ftruncate(fd, 0)
        except OSError:
            _logger.debug("slot body clear failed", exc_info=True)
        os.close(fd)


class FlockSlotPool:
    """``limit`` slot files, each guarded by an exclusive non-blocking ``flock``."""

    def __init__(self, limit: int) -> None:
        self._limit: int = limit
        self._dir: Path = slot_directory()

    @property
    def directory(self) -> Path:
        return self._dir

    def acquire(self, run_id: str, node_id: int, project_root: Path) -> SlotLease:
        if sys.platform == "win32":
            _logger.warning("host worker pool is disabled: the platform has no flock")
            return _NoLease()
        import fcntl

        # The parent doubles as the owner-only controller credential namespace.
        self._dir.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._dir.mkdir(mode=0o700, exist_ok=True)
        body = json.dumps(
            {
                "pid": os.getpid(),
                "run_id": run_id,
                "node_id": node_id,
                "project_root": str(project_root),
                "acquired_at": datetime.now(UTC).isoformat(),
            }
        ).encode()
        for index in range(self._limit):
            fd = os.open(self._dir / f"slot-{index}", os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(fd)
                continue
            try:
                os.ftruncate(fd, 0)
                _ = os.pwrite(fd, body, 0)
            except OSError:
                os.close(fd)
                raise
            return _FlockLease(fd)
        raise HostCapacityFull(self._limit, self._limit)

    def holders(self) -> list[dict[str, object]]:
        """Return the diagnostic body of every slot a live process holds.

        Serves the doctor report only. The probe takes ``LOCK_SH`` so two
        concurrent reports never see each other as holders. It holds the lock
        for one syscall, so a racing ``acquire`` at worst sees one slot as
        taken and tries the next.
        """
        if sys.platform == "win32":
            return []
        import fcntl

        found: list[dict[str, object]] = []
        for index in range(self._limit):
            path = self._dir / f"slot-{index}"
            try:
                fd = os.open(path, os.O_RDONLY)
            except OSError:
                continue
            try:
                fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except OSError:
                try:
                    found.append(
                        cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
                    )
                except (OSError, ValueError):
                    found.append({})
            finally:
                os.close(fd)
        return found
