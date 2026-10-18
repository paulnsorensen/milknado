"""Size xdist worker pools from free cores so concurrent gates share one host."""

from __future__ import annotations

import os

import psutil


def free_core_workers(ceiling: int | None = None) -> int:
    """Return the ``-n auto`` pool size for this host.

    An explicit ``PYTEST_XDIST_AUTO_NUM_WORKERS`` pin wins. Otherwise the pool is the
    free logical cores, at least 2, capped by ``TEST_WORKERS_MAX`` and by ``ceiling``.
    A missing ``ceiling`` reads ``TEST_WORKERS_CEILING``, which the serial real-process
    lane sets to ``REAL_PROCESS_WORKERS``.
    ``TEST_WORKERS_MAX`` defaults to xdist's own physical-core count, so the pool only
    shrinks. psutil reads the load average on Linux and macOS and emulates it on Windows.
    """
    if pinned := os.environ.get("PYTEST_XDIST_AUTO_NUM_WORKERS"):
        return int(pinned)
    cpus = psutil.cpu_count() or 1
    cap = int(os.environ.get("TEST_WORKERS_MAX", psutil.cpu_count(logical=False) or cpus))
    if ceiling is None and (env_ceiling := os.environ.get("TEST_WORKERS_CEILING")):
        ceiling = int(env_ceiling)
    if ceiling is not None:
        cap = min(cap, ceiling)
    return min(cap, max(2, int(cpus - psutil.getloadavg()[0])))
