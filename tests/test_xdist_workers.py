from __future__ import annotations

import os

import psutil
import pytest

from tests.conftest import pytest_xdist_auto_num_workers


@pytest.mark.parametrize(
    ("env", "load", "workers"),
    [
        ({}, 0.5, 12),
        ({}, 19.0, 5),
        ({}, 80.0, 2),
        ({"TEST_WORKERS_MAX": "3"}, 0.5, 3),
        ({"TEST_WORKERS_MAX": "1"}, 80.0, 1),
        ({"PYTEST_XDIST_AUTO_NUM_WORKERS": "7", "TEST_WORKERS_MAX": "3"}, 80.0, 7),
    ],
)
def test_auto_workers_follow_free_cores_cap_and_pin(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], load: float, workers: int
) -> None:
    for name in ("PYTEST_XDIST_AUTO_NUM_WORKERS", "TEST_WORKERS_MAX"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(os, "cpu_count", lambda: 24)
    monkeypatch.setattr(psutil, "cpu_count", lambda logical=True: 24 if logical else 12)
    monkeypatch.setattr(os, "getloadavg", lambda: (load, load, load))

    assert pytest_xdist_auto_num_workers() == workers
