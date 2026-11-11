"""Per-seed scenario generation.

This is where the simulator's aleatoric variation comes from. Without it a
mission outcome would be a deterministic function of (plan, world) and Phase 4
would have nothing to predict — so the Phase 1 acceptance check that outcomes
*vary across seeds* is really a check on this module.

Map features are deliberately **not** randomised: moving them would change
whether a plan is valid, and the risk model must learn about execution risk,
not about plans silently becoming invalid.

Per rule 5 each consumer gets its own named RNG stream from one
``SeedSequence``, so adding a drone does not shift another drone's draws.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from planguard.config import WorldConfig
from planguard.sim.drone import DroneState

#: Named children of the run's SeedSequence. Order is fixed; append only.
STREAMS = ("scenario", "link")


@dataclass(frozen=True)
class Scenario:
    seed: int
    wind_mps: tuple[float, float]
    drones: dict[str, DroneState]

    @property
    def aircraft_count(self) -> int:
        return len(self.drones)


def streams(seed: int) -> dict[str, np.random.Generator]:
    children = np.random.SeedSequence(seed).spawn(len(STREAMS))
    return {name: np.random.default_rng(ss) for name, ss in zip(STREAMS, children, strict=True)}


def drone_ids(cfg: WorldConfig) -> list[str]:
    return [f"d{i}" for i in range(cfg.fleet.aircraft_count)]


def generate(cfg: WorldConfig, seed: int, rng: np.random.Generator | None = None) -> Scenario:
    """Draw a scenario for ``seed``.

    Varied: wind speed and direction, which aircraft are available, and each
    aircraft's start position and heading.
    """
    rng = rng if rng is not None else streams(seed)["scenario"]

    speed = max(0.0, float(rng.normal(cfg.wind.speed_mps, cfg.wind.speed_sigma_mps)))
    direction = float(rng.uniform(0.0, math.tau))
    wind = (speed * math.cos(direction), speed * math.sin(direction))

    ids = drone_ids(cfg)
    available = [d for d in ids if rng.random() >= cfg.scenario.p_aircraft_unavailable]
    if not available:  # never launch an empty fleet
        available = [ids[0]]

    base = cfg.base_m
    drones: dict[str, DroneState] = {}
    for did in available:
        # uniform over a disc, so sqrt on the radius rather than a square jitter
        r = cfg.scenario.start_jitter_m * math.sqrt(float(rng.random()))
        theta = float(rng.uniform(0.0, math.tau))
        drones[did] = DroneState(
            id=did,
            pos_m=(base[0] + r * math.cos(theta), base[1] + r * math.sin(theta)),
            heading_rad=float(rng.uniform(-math.pi, math.pi)),
            fuel_s=cfg.fleet.endurance_s,
            link_up=True,
        )
    return Scenario(seed=seed, wind_mps=wind, drones=drones)
