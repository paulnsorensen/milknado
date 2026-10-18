from __future__ import annotations

from dataclasses import dataclass, field

import psutil
import pytest

from tests.worker_pool import free_core_workers


@dataclass(frozen=True)
class PoolCase:
    load: float
    workers: int
    ceiling: int | None = None
    env: dict[str, str] = field(default_factory=dict)


@pytest.mark.parametrize(
    "case",
    [
        PoolCase(load=0.5, workers=12),
        PoolCase(load=19.0, workers=5),
        PoolCase(load=80.0, workers=2),
        PoolCase(load=0.5, workers=4, ceiling=4),
        PoolCase(load=0.5, workers=3, env={"TEST_WORKERS_MAX": "3"}),
        PoolCase(load=0.5, workers=3, ceiling=4, env={"TEST_WORKERS_MAX": "3"}),
        PoolCase(load=0.5, workers=3, env={"TEST_WORKERS_CEILING": "3"}),
        PoolCase(load=0.5, workers=2, env={"TEST_WORKERS_CEILING": "3", "TEST_WORKERS_MAX": "2"}),
        PoolCase(load=19.0, workers=5, env={"TEST_WORKERS_CEILING": "8"}),
        PoolCase(load=80.0, workers=2, env={"TEST_WORKERS_CEILING": "3"}),
        PoolCase(load=0.5, workers=4, ceiling=4, env={"TEST_WORKERS_CEILING": "9"}),
        PoolCase(load=80.0, workers=1, env={"TEST_WORKERS_MAX": "1"}),
        PoolCase(
            load=80.0,
            workers=7,
            ceiling=4,
            env={"PYTEST_XDIST_AUTO_NUM_WORKERS": "7", "TEST_WORKERS_MAX": "3"},
        ),
    ],
)
def test_pool_follows_free_cores_cap_ceiling_and_pin(
    monkeypatch: pytest.MonkeyPatch, case: PoolCase
) -> None:
    for name in ("PYTEST_XDIST_AUTO_NUM_WORKERS", "TEST_WORKERS_MAX", "TEST_WORKERS_CEILING"):
        monkeypatch.delenv(name, raising=False)
    for name, value in case.env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(psutil, "cpu_count", lambda logical=True: 24 if logical else 12)
    monkeypatch.setattr(psutil, "getloadavg", lambda: (case.load, case.load, case.load))

    assert free_core_workers(case.ceiling) == case.workers


def test_missing_physical_core_count_falls_back_to_logical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("PYTEST_XDIST_AUTO_NUM_WORKERS", "TEST_WORKERS_MAX", "TEST_WORKERS_CEILING"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(psutil, "cpu_count", lambda logical=True: 8 if logical else None)
    monkeypatch.setattr(psutil, "getloadavg", lambda: (0.0, 0.0, 0.0))

    assert free_core_workers() == 8
