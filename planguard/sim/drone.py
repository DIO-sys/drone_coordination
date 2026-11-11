"""Point-mass aircraft kinematics.

Pure functions: every call takes a state and returns a new one, so the tick
loop owns all the mutation and rule 5's determinism is easy to keep.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from planguard.config import FleetSpec, dist_m


def wrap_pi(angle_rad: float) -> float:
    """Wrap to (-pi, pi] so heading errors never take the long way round.

    ``math.remainder`` returns exactly -pi for odd multiples of pi, so the
    half-open interval is closed explicitly. Which way an aircraft turns toward
    a waypoint dead astern is arbitrary, but it must be the *same* way every
    run (rule 5).
    """
    wrapped = math.remainder(angle_rad, math.tau)
    return math.pi if wrapped == -math.pi else wrapped


@dataclass(frozen=True)
class DroneState:
    id: str
    pos_m: tuple[float, float]
    heading_rad: float
    fuel_s: float
    link_up: bool = True
    #: Set once the aircraft runs out of endurance; it stops moving.
    grounded: bool = False

    @property
    def fuel_fraction(self) -> float:
        return max(0.0, self.fuel_s)


def capture_radius_m(fleet: FleetSpec, dt_s: float) -> float:
    """How close counts as "arrived".

    A fixed-wing aircraft cannot sit on a point, so arrival is a circle. Its
    minimum turn radius is the natural scale; the ``speed * dt`` floor stops a
    coarse timestep from letting the aircraft step over the circle every tick.
    """
    return max(fleet.turn_radius_m, 2.0 * fleet.speed_mps * dt_s)


def step(
    state: DroneState,
    waypoint_m: tuple[float, float] | None,
    fleet: FleetSpec,
    wind_mps: tuple[float, float],
    dt_s: float,
) -> DroneState:
    """Advance one tick.

    Steering is pure pursuit: the airspeed vector is turned toward the waypoint
    each tick, with no wind-triangle lead compensation, so ground tracks bow
    downwind. The loop still converges for any wind below airspeed, and the bow
    is a feature rather than a bug — it is one of the things that makes a plan's
    time margin uncertain, which Phase 4 needs.
    """
    if state.grounded:
        return state

    fuel_s = state.fuel_s - dt_s
    if fuel_s <= 0.0:
        return replace(state, fuel_s=0.0, grounded=True)

    heading = state.heading_rad
    if waypoint_m is not None:
        desired = math.atan2(waypoint_m[1] - state.pos_m[1], waypoint_m[0] - state.pos_m[0])
        error = wrap_pi(desired - heading)
        max_delta = fleet.max_turn_rate_rps * dt_s
        heading = wrap_pi(heading + max(-max_delta, min(max_delta, error)))

    vx = fleet.speed_mps * math.cos(heading) + wind_mps[0]
    vy = fleet.speed_mps * math.sin(heading) + wind_mps[1]
    pos = (state.pos_m[0] + vx * dt_s, state.pos_m[1] + vy * dt_s)
    return replace(state, pos_m=pos, heading_rad=heading, fuel_s=fuel_s)


def has_arrived(
    state: DroneState, waypoint_m: tuple[float, float], fleet: FleetSpec, dt_s: float
) -> bool:
    return dist_m(state.pos_m, waypoint_m) <= capture_radius_m(fleet, dt_s)
