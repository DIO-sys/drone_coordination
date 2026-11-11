"""Aircraft assignment.

Per rule 7 the plan never names aircraft, so allocation happens here and only
here. The tick loop calls ``allocate`` and does not know which algorithm ran.

Two allocators share one signature, selected by ``cfg.cbba.allocator``:

* ``cbba`` (default) -- the real thing, in ``coord/cbba.py``.
* ``greedy`` -- the Phase 1 placeholder, nearest idle aircraft, no bidding or
  consensus. Kept rather than deleted so the two can be compared on identical
  scenarios, which is what makes "CBBA was worth implementing" a measurement
  instead of an assertion.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from planguard.config import WorldConfig, dist_m
from planguard.coord.cbba import allocate_cbba
from planguard.dsl.schema import MissionPlan
from planguard.sim.drone import DroneState


def allocate_greedy(
    plan: MissionPlan,
    ready_task_ids: Sequence[str],
    drones: Mapping[str, DroneState],
    busy_drone_ids: Sequence[str],
    cfg: WorldConfig,
) -> dict[str, list[str]]:
    """Assign idle aircraft to ready, unassigned tasks, nearest first.

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


def allocate(
    plan: MissionPlan,
    ready_task_ids: Sequence[str],
    drones: Mapping[str, DroneState],
    busy_drone_ids: Sequence[str],
    cfg: WorldConfig,
) -> dict[str, list[str]]:
    """Dispatch to the configured allocator."""
    if cfg.cbba.allocator == "greedy":
        return allocate_greedy(plan, ready_task_ids, drones, busy_drone_ids, cfg)
    return allocate_cbba(plan, ready_task_ids, drones, busy_drone_ids, cfg)
