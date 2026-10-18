"""Project facts shown by the metaharness view."""

from __future__ import annotations

from pathlib import Path

from milknado.domains.common import MilknadoConfig, resolve_flavor_profile
from milknado.web.commands import MetaharnessInfo


def build_metaharness_info(config: MilknadoConfig, project_root: Path) -> MetaharnessInfo:
    return MetaharnessInfo(
        project_root=project_root,
        concurrency=config.concurrency_limit,
        flavor_gates={
            name: tuple(
                gate.command for gate in resolve_flavor_profile(config, name).quality_gates or ()
            )
            for name in sorted(config.flavor_registry)
        },
    )
