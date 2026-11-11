"""Aircraft assignment.

**Phase 1 placeholder.** This is a greedy nearest-aircraft allocator with no
consensus, no bidding and no conflict resolution. Phase 2 replaces it with CBBA
behind the same ``allocate`` signature; the point of keeping the interface
narrow now is that swapping in CBBA should not touch the tick loop.

Per rule 7 the plan never names aircraft, so allocation happens here and only
here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from planguard.config import WorldConfig, dist_m
from planguard.dsl.schema import MissionPlan
from planguard.sim.drone import DroneState


def allocate(
    plan: MissionPlan,
    ready_task_ids: Sequence[str],
    drones: Mapping[str, DroneState],
    busy_drone_ids: Sequence[str],
    cfg: WorldConfig,
) -> dict[str, list[str]]:
    """Assign idle aircraft to ready, unassigned tasks.

    Tasks are considered in priority order then by id, so the result is
    deterministic (rule 5). A task that cannot be fully staffed is left
    unassigned rather than partly staffed — a half-crewed sweep would quietly
    take twice as long and corrupt the time-margin signal Phase 4 learns from.
    """
    idle = [d for d in sorted(drones) if d not in set(busy_drone_ids) and not drones[d].grounded]
    tasks = [t for t in plan.tasks if t.id in set(ready_task_ids)]
    tasks.sort(key=lambda t: (t.priority, t.id))

    assignment: dict[str, list[str]] = {}
    for task in tasks:
        if len(idle) < task.aircraft_count:
            continue
        feature = cfg.world_map.get(task.target)
        if feature is None:  # unreachable for a validated plan
            continue
        target = feature.centroid_m
        idle.sort(key=lambda d: (dist_m(drones[d].pos_m, target), d))
        chosen = idle[: task.aircraft_count]
        assignment[task.id] = sorted(chosen)
        idle = [d for d in idle if d not in set(chosen)]
    return assignment
