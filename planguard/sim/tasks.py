"""Task execution semantics: waypoints and completion conditions.

Each DSL task kind becomes a waypoint list plus a rule for when it is done.
Coverage work is planned over a feature's *bounding box*, so a non-rectangular
area is over-covered rather than clipped — stated as a limitation rather than
pretending to do polygon decomposition. Every map feature in the default world
is rectangular, where the bounding box is exact.

The lawnmower total path length is ``area / swath``, which is exactly the
quantity the Phase 0 validator uses to estimate sweep duration. A test pins the
two together: if they ever diverge, the validator's time check is lying.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from planguard.config import MapFeature
from planguard.dsl.schema import TaskSpec

Point = tuple[float, float]

#: Waypoints used to fly one orbit while holding station.
ORBIT_WAYPOINTS = 8


def _bbox(feature: MapFeature) -> tuple[float, float, float, float]:
    xs = [c[0] for c in feature.coords_m]
    ys = [c[1] for c in feature.coords_m]
    return min(xs), min(ys), max(xs), max(ys)


def sweep_transects(feature: MapFeature, swath_m: float) -> list[tuple[Point, Point]]:
    """Boustrophedon transects across the feature's bounding box.

    Transects run parallel to the longer side and are spaced ``swath_m`` apart
    along the shorter side, each centred in its own strip. Alternate transects
    reverse direction so consecutive legs join end to end.
    """
    x0, y0, x1, y1 = _bbox(feature)
    width, height = x1 - x0, y1 - y0
    along_x = width >= height
    long_lo, long_hi = (x0, x1) if along_x else (y0, y1)
    short_lo, short_len = (y0, height) if along_x else (x0, width)

    n = max(1, math.ceil(short_len / swath_m))
    out: list[tuple[Point, Point]] = []
    for i in range(n):
        offset = short_lo + swath_m * (i + 0.5)
        offset = min(offset, short_lo + short_len)  # keep the last strip inside the box
        a, b = (long_lo, long_hi) if i % 2 == 0 else (long_hi, long_lo)
        if along_x:
            out.append(((a, offset), (b, offset)))
        else:
            out.append(((offset, a), (offset, b)))
    return out


def share_of(
    transects: list[tuple[Point, Point]], share: int, of_shares: int
) -> list[tuple[Point, Point]]:
    """Split transects into *contiguous* bands, one per aircraft.

    Round-robin (``i % of_shares == share``) looks natural and is wrong: it
    hands each aircraft every n-th transect, and because boustrophedon
    alternates direction, every transect in one aircraft's set runs the same
    way. The aircraft then deadheads the full length of the area between each
    pair of legs. On the default world that inflated a two-aircraft ridge sweep
    from 10.0 km to 18.1 km per aircraft -- an 81% overshoot that made the
    sweep miss its budget every run, and flattened the variance Phase 4 needs.

    Contiguous bands keep each aircraft's own legs alternating, so they join
    end to end and the flown distance matches ``area / swath``, which is what
    the Phase 0 validator assumes when it estimates duration.
    """
    n = len(transects)
    lo = (n * share) // of_shares
    hi = (n * (share + 1)) // of_shares
    return transects[lo:hi]


def orbit_waypoints(centre_m: Point, radius_m: float, n: int = ORBIT_WAYPOINTS) -> list[Point]:
    return [
        (
            centre_m[0] + radius_m * math.cos(math.tau * i / n),
            centre_m[1] + radius_m * math.sin(math.tau * i / n),
        )
        for i in range(n)
    ]


def waypoints_for(
    task: TaskSpec,
    feature: MapFeature,
    swath_m: float,
    turn_radius_m: float,
    share: int,
    of_shares: int,
) -> list[Point]:
    """Waypoints for one aircraft's share of a task.

    ``share`` indexes this aircraft within the ``of_shares`` aircraft assigned
    to the task. Only coverage work is divided; dwell and rally are not, which
    mirrors the validator's duration model.
    """
    if task.kind == "sweep":
        transects = sweep_transects(feature, swath_m)
        mine = share_of(transects, share, of_shares)
        return [pt for leg in mine for pt in leg]
    if task.kind == "corridor_patrol":
        return list(feature.coords_m)
    if task.kind == "station_keep":
        return orbit_waypoints(feature.centroid_m, turn_radius_m)
    return [feature.centroid_m]  # rally


@dataclass(frozen=True)
class TaskRun:
    """One aircraft's progress through one task."""

    task_id: str
    kind: str
    waypoints_m: list[Point]
    wp_index: int = 0
    #: Dwell is counted in ticks, not accumulated in seconds. Summing dt_s
    #: drifts: ten additions of 0.1 give 0.9999999999999999, so a 1 s dwell
    #: took an extra tick to satisfy. Rule 8 schedules everything slower than
    #: the tick rate by tick index, and this is the same principle.
    on_station_ticks: int = 0
    dwell_ticks: int = 0
    done: bool = False

    @property
    def cyclic(self) -> bool:
        """Station keeping orbits until its dwell is satisfied; the rest fly once."""
        return self.kind == "station_keep"

    @property
    def waypoint_m(self) -> Point | None:
        if self.done or not self.waypoints_m:
            return None
        return self.waypoints_m[self.wp_index % len(self.waypoints_m)]

    @property
    def progress(self) -> float:
        """Fraction complete, for partial credit in mission scoring."""
        if self.done:
            return 1.0
        if self.cyclic:
            if self.dwell_ticks <= 0:
                return 1.0
            return min(1.0, self.on_station_ticks / self.dwell_ticks)
        if not self.waypoints_m:
            return 1.0
        return min(1.0, self.wp_index / len(self.waypoints_m))


def start_run(
    task: TaskSpec,
    feature: MapFeature,
    swath_m: float,
    turn_radius_m: float,
    share: int,
    of_shares: int,
    dt_s: float,
) -> TaskRun:
    dwell_s = task.dwell_s or 0.0
    return TaskRun(
        task_id=task.id,
        kind=task.kind,
        waypoints_m=waypoints_for(task, feature, swath_m, turn_radius_m, share, of_shares),
        dwell_ticks=int(round(dwell_s / dt_s)),
    )


def advance(run: TaskRun, arrived: bool, dt_s: float) -> TaskRun:
    """Advance a run by one tick.

    ``arrived`` is whether the aircraft reached the run's current waypoint this
    tick; the caller owns that test so the kinematics stay in one place.
    ``dt_s`` is accepted for symmetry with the rest of the sim API but dwell is
    tracked in ticks, so it is not used.
    """
    del dt_s
    if run.done:
        return run
    on_station = run.on_station_ticks
    if run.cyclic:
        on_station += 1
        if on_station >= run.dwell_ticks:
            return replace(run, on_station_ticks=on_station, done=True)
    if not arrived:
        return replace(run, on_station_ticks=on_station)
    index = run.wp_index + 1
    if not run.cyclic and index >= len(run.waypoints_m):
        return replace(run, wp_index=len(run.waypoints_m), on_station_ticks=on_station, done=True)
    return replace(run, wp_index=index, on_station_ticks=on_station)
