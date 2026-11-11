"""Kinematics and link-availability tests.

These are the simulator's pure primitives, so they get known-value tests and
timestep-independence tests rather than end-to-end ones.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from planguard.config import FleetSpec, LinkConfig, WorldConfig, dist_m
from planguard.sim import link as link_mod
from planguard.sim.drone import DroneState, capture_radius_m, has_arrived, step, wrap_pi


@pytest.fixture
def fleet(cfg: WorldConfig) -> FleetSpec:
    return cfg.fleet


def at_origin(heading_rad: float = 0.0, fuel_s: float = 2400.0) -> DroneState:
    return DroneState(id="d0", pos_m=(0.0, 0.0), heading_rad=heading_rad, fuel_s=fuel_s)


# ---------------------------------------------------------------- geometry ---


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (0.0, 0.0),
        (math.pi / 2, math.pi / 2),
        (3 * math.pi, math.pi),
        (-3 * math.pi, math.pi),
        (math.pi, math.pi),
    ],
)
def test_wrap_pi(given: float, expected: float) -> None:
    assert wrap_pi(given) == pytest.approx(expected)


def test_wrap_pi_takes_the_short_way_round() -> None:
    """A heading error of 350 degrees is really -10 degrees."""
    assert wrap_pi(math.radians(350)) == pytest.approx(math.radians(-10))


# -------------------------------------------------------------- kinematics ---


def test_straight_and_level_covers_speed_times_dt(fleet: FleetSpec) -> None:
    s = step(at_origin(), waypoint_m=(10_000.0, 0.0), fleet=fleet, wind_mps=(0.0, 0.0), dt_s=0.1)
    assert s.pos_m == pytest.approx((fleet.speed_mps * 0.1, 0.0))
    assert s.heading_rad == pytest.approx(0.0)


def test_turn_rate_is_clamped(fleet: FleetSpec) -> None:
    """A waypoint directly behind cannot be turned onto in one tick."""
    s = step(at_origin(), waypoint_m=(-1000.0, 0.0), fleet=fleet, wind_mps=(0.0, 0.0), dt_s=0.1)
    assert abs(s.heading_rad) == pytest.approx(fleet.max_turn_rate_rps * 0.1)


def test_turn_radius_matches_speed_over_turn_rate(fleet: FleetSpec) -> None:
    assert fleet.turn_radius_m == pytest.approx(fleet.speed_mps / fleet.max_turn_rate_rps)


def test_wind_displaces_the_ground_track(fleet: FleetSpec) -> None:
    calm = step(at_origin(), (10_000.0, 0.0), fleet, (0.0, 0.0), 0.1)
    windy = step(at_origin(), (10_000.0, 0.0), fleet, (0.0, -5.0), 0.1)
    assert windy.pos_m[1] < calm.pos_m[1]
    assert windy.pos_m[1] == pytest.approx(-0.5)


def test_pure_pursuit_converges_on_the_waypoint(fleet: FleetSpec) -> None:
    waypoint = (3000.0, 2000.0)
    s = at_origin(heading_rad=math.pi)  # start pointing the wrong way
    for _ in range(3000):
        s = step(s, waypoint, fleet, (0.0, 0.0), 0.1)
        if has_arrived(s, waypoint, fleet, 0.1):
            break
    assert has_arrived(s, waypoint, fleet, 0.1), dist_m(s.pos_m, waypoint)


def test_convergence_survives_wind_below_airspeed(fleet: FleetSpec) -> None:
    waypoint = (3000.0, 2000.0)
    s = at_origin()
    for _ in range(4000):
        s = step(s, waypoint, fleet, (-8.0, 6.0), 0.1)
        if has_arrived(s, waypoint, fleet, 0.1):
            break
    assert has_arrived(s, waypoint, fleet, 0.1), dist_m(s.pos_m, waypoint)


def test_fuel_burns_down_and_grounds_the_aircraft(fleet: FleetSpec) -> None:
    s = at_origin(fuel_s=0.25)
    for _ in range(3):
        s = step(s, (10_000.0, 0.0), fleet, (0.0, 0.0), 0.1)
    assert s.grounded
    assert s.fuel_s == 0.0


def test_a_grounded_aircraft_does_not_move(fleet: FleetSpec) -> None:
    s = DroneState(id="d0", pos_m=(5.0, 5.0), heading_rad=0.0, fuel_s=0.0, grounded=True)
    assert step(s, (10_000.0, 0.0), fleet, (10.0, 10.0), 0.1) == s


def test_no_waypoint_holds_heading(fleet: FleetSpec) -> None:
    s = step(at_origin(heading_rad=1.0), None, fleet, (0.0, 0.0), 0.1)
    assert s.heading_rad == pytest.approx(1.0)


def test_capture_radius_floor_scales_with_timestep(fleet: FleetSpec) -> None:
    """A coarse timestep must not let the aircraft step over the capture circle."""
    coarse_dt = 20.0
    assert capture_radius_m(fleet, coarse_dt) >= 2.0 * fleet.speed_mps * coarse_dt
    assert capture_radius_m(fleet, 0.1) == pytest.approx(fleet.turn_radius_m)


def test_step_does_not_mutate_its_input(fleet: FleetSpec) -> None:
    s = at_origin()
    step(s, (100.0, 100.0), fleet, (0.0, 0.0), 0.1)
    assert s == at_origin()


# -------------------------------------------------------------------- link ---


def test_out_of_range_is_always_down(cfg: WorldConfig) -> None:
    rng = np.random.default_rng(0)
    for _ in range(100):
        assert not link_mod.step(True, cfg.link.range_m + 1.0, cfg.link, cfg.sim.dt_s, rng)


def test_in_range_link_is_usually_up(cfg: WorldConfig) -> None:
    rng = np.random.default_rng(0)
    up = True
    for _ in range(50):
        up = link_mod.step(up, 100.0, cfg.link, cfg.sim.dt_s, rng)
    assert up


def test_simulated_availability_matches_the_closed_form(cfg: WorldConfig) -> None:
    """Catches a per-second to per-tick conversion bug, which would otherwise
    only show up as mysteriously rare fallbacks."""
    rng = np.random.default_rng(7)
    up = True
    up_ticks = 0
    n = 400_000
    for _ in range(n):
        up = link_mod.step(up, 100.0, cfg.link, cfg.sim.dt_s, rng)
        up_ticks += up
    assert up_ticks / n == pytest.approx(link_mod.steady_state_availability(cfg.link), abs=0.01)


def test_availability_is_timestep_independent(cfg: WorldConfig) -> None:
    """The same wall-clock duration must give the same availability at 10 Hz and
    20 Hz, which is the whole reason p_per_tick is not p_per_s * dt."""
    rates = []
    for dt_s in (0.1, 0.05):
        rng = np.random.default_rng(11)
        up = True
        ticks = int(20_000 / dt_s)
        up_ticks = 0
        for _ in range(ticks):
            up = link_mod.step(up, 100.0, cfg.link, dt_s, rng)
            up_ticks += up
        rates.append(up_ticks / ticks)
    assert rates[0] == pytest.approx(rates[1], abs=0.01)


def test_link_is_deterministic_for_a_given_seed(cfg: WorldConfig) -> None:
    def run() -> list[bool]:
        rng = np.random.default_rng(3)
        up = True
        out = []
        for _ in range(500):
            up = link_mod.step(up, 100.0, cfg.link, cfg.sim.dt_s, rng)
            out.append(up)
        return out

    assert run() == run()


def test_steady_state_of_a_never_dropping_link_is_one() -> None:
    assert link_mod.steady_state_availability(LinkConfig(p_drop_per_s=0.0)) == 1.0
    assert (
        link_mod.steady_state_availability(LinkConfig(p_drop_per_s=0.0, p_recover_per_s=0.0)) == 1.0
    )


def test_a_down_link_does_recover(cfg: WorldConfig) -> None:
    rng = np.random.default_rng(1)
    up = False
    for _ in range(200):
        up = link_mod.step(up, 100.0, cfg.link, cfg.sim.dt_s, rng)
        if up:
            break
    assert up
