from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path


class SystemToolchainAdapter:
    def find_executable(self, command: str) -> str | None:
        return shutil.which(command)

    def run(self, argv: list[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
        return subprocess.run(argv, check=check, capture_output=True, text=True)

    def system(self) -> str:
        return platform.system()

    def machine(self) -> str:
        return platform.machine()

    def environment(self, name: str) -> str | None:
        return os.environ.get(name)

    def home(self) -> Path:
        return Path.home()
