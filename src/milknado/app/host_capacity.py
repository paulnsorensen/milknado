"""Doctor report for the host-wide worker slot pool."""

from __future__ import annotations

from milknado.adapters import FlockSlotPool


def describe_host_pool(limit: int) -> str:
    """Render the pool path, its limit, and the live slot holders."""
    pool = FlockSlotPool(limit)
    holders = pool.holders()
    lines = [f"host worker pool: {pool.directory} (limit {limit}, {len(holders)} held)"]
    lines.extend(
        f"  pid={h.get('pid')} run_id={h.get('run_id')} node_id={h.get('node_id')} "
        + f"project_root={h.get('project_root')} since={h.get('acquired_at')}"
        for h in holders
    )
    return "\n".join(lines)
