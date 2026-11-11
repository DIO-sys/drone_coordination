"""World, fleet and validator configuration.

Per rule 10: every parameter lives in a Pydantic model loaded from YAML, and
derived quantities are computed here rather than written into the YAML.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

FeatureKind = Literal["point", "area", "corridor"]

#: Which map-feature kind each task kind must target. The validator's
#: WRONG_FEATURE_KIND check is driven entirely by this table.
REQUIRED_FEATURE_KIND: dict[str, FeatureKind] = {
    "sweep": "area",
    "corridor_patrol": "corridor",
    "station_keep": "point",
    "rally": "point",
}


def dist_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


class MapFeature(BaseModel):
    """A named piece of terrain the operator can refer to by name."""

    model_config = ConfigDict(frozen=True)

    name: str
    kind: FeatureKind
    coords_m: list[tuple[float, float]]

    @model_validator(mode="after")
    def _check_arity(self) -> Self:
        need = {"point": 1, "area": 3, "corridor": 2}[self.kind]
        if len(self.coords_m) < need:
            raise ValueError(
                f"feature {self.name!r} of kind {self.kind!r} needs at least "
                f"{need} coordinate(s), got {len(self.coords_m)}"
            )
        if self.kind == "point" and len(self.coords_m) != 1:
            raise ValueError(f"point feature {self.name!r} must have exactly one coordinate")
        return self

    @property
    def centroid_m(self) -> tuple[float, float]:
        """Mean of the vertices. Exact for rectangles, an approximation otherwise."""
        n = len(self.coords_m)
        return (
            sum(c[0] for c in self.coords_m) / n,
            sum(c[1] for c in self.coords_m) / n,
        )

    @property
    def area_m2(self) -> float:
        """Shoelace area; zero for non-area features."""
        if self.kind != "area":
            return 0.0
        pts = self.coords_m
        total = 0.0
        for i in range(len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % len(pts)]
            total += x1 * y2 - x2 * y1
        return abs(total) / 2.0

    @property
    def length_m(self) -> float:
        """Polyline length; zero for non-corridor features."""
        if self.kind != "corridor":
            return 0.0
        return sum(
            dist_m(self.coords_m[i], self.coords_m[i + 1]) for i in range(len(self.coords_m) - 1)
        )


class WorldMap(BaseModel):
    model_config = ConfigDict(frozen=True)

    features: list[MapFeature]

    @model_validator(mode="after")
    def _unique_names(self) -> Self:
        names = [f.name for f in self.features]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate map feature names: {dupes}")
        return self

    def get(self, name: str) -> MapFeature | None:
        for f in sorted(self.features, key=lambda f: f.name):
            if f.name == name:
                return f
        return None

    @property
    def names(self) -> list[str]:
        return sorted(f.name for f in self.features)


class FleetSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    aircraft_count: int = Field(ge=1)
    speed_mps: float = Field(gt=0)
    sensor_swath_m: float = Field(gt=0)
    base_feature: str
    #: Airborne endurance. Without this the DSL's low_fuel fallback trigger
    #: could never fire.
    endurance_s: float = Field(default=2400.0, gt=0)
    #: A fixed-wing aircraft at 30 m/s in a 30 degree bank turns at
    #: v / (v^2 / (g tan phi)) ~= 11 deg/s, so 15 is a mild upper bound.
    max_turn_rate_dps: float = Field(default=15.0, gt=0)

    @property
    def max_concurrent_aircraft(self) -> int:
        return self.aircraft_count

    @property
    def max_turn_rate_rps(self) -> float:
        return math.radians(self.max_turn_rate_dps)

    @property
    def turn_radius_m(self) -> float:
        """Minimum turn radius implied by speed and turn rate."""
        return self.speed_mps / self.max_turn_rate_rps


class LinkConfig(BaseModel):
    """Abstract link availability.

    Deliberately *not* an RF model — see the scope section of CLAUDE.md. Beyond
    ``range_m`` the link is down; inside it, availability is a two-state Markov
    process. This is the simulator's main source of aleatoric variation, which
    Phase 4's risk surrogate needs in order to have anything to predict.
    """

    model_config = ConfigDict(frozen=True)

    range_m: float = Field(default=6000.0, gt=0)
    #: Probability per second of an in-range link dropping / recovering.
    p_drop_per_s: float = Field(default=0.02, ge=0, le=1)
    p_recover_per_s: float = Field(default=0.4, ge=0, le=1)

    def p_per_tick(self, p_per_s: float, dt_s: float) -> float:
        """Convert a per-second hazard to a per-tick probability.

        ``1 - (1 - p)**dt`` rather than ``p * dt`` so the drop rate does not
        depend on the timestep.
        """
        return float(1.0 - (1.0 - p_per_s) ** dt_s)


class WindConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    #: Mean wind speed; the scenario randomiser draws per-run values around it.
    speed_mps: float = Field(default=0.0, ge=0)
    direction_deg: float = Field(default=0.0, ge=0, lt=360)
    speed_sigma_mps: float = Field(default=2.0, ge=0)

    @property
    def vector_mps(self) -> tuple[float, float]:
        theta = math.radians(self.direction_deg)
        return (self.speed_mps * math.cos(theta), self.speed_mps * math.sin(theta))


class SimConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    #: Rule 8: fixed timestep. Anything slower is scheduled by tick index.
    dt_s: float = Field(default=0.1, gt=0)
    #: Hard stop so a malformed plan cannot hang the batch harness.
    max_duration_s: float = Field(default=3600.0, gt=0)

    def ticks(self, duration_s: float) -> int:
        return int(round(duration_s / self.dt_s))


class ValidatorConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    #: Multiplies the optimistic duration estimate before comparing against the
    #: plan's time budget. 1.0 keeps the check one-sided: the estimate assumes
    #: unlimited parallelism, so exceeding the budget proves infeasibility,
    #: while fitting inside it does not prove feasibility.
    time_estimate_slack: float = Field(default=1.0, gt=0)
    require_link_fallback: bool = True


class WorldConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    world_map: WorldMap
    fleet: FleetSpec
    sim: SimConfig = SimConfig()
    link: LinkConfig = LinkConfig()
    wind: WindConfig = WindConfig()
    validator: ValidatorConfig = ValidatorConfig()

    @model_validator(mode="after")
    def _base_exists(self) -> Self:
        base = self.world_map.get(self.fleet.base_feature)
        if base is None:
            raise ValueError(
                f"fleet.base_feature {self.fleet.base_feature!r} is not a map feature; "
                f"known features: {self.world_map.names}"
            )
        if base.kind != "point":
            raise ValueError(f"fleet base {base.name!r} must be a point feature, got {base.kind!r}")
        return self

    @property
    def base_m(self) -> tuple[float, float]:
        base = self.world_map.get(self.fleet.base_feature)
        assert base is not None  # guaranteed by _base_exists
        return base.centroid_m

    @classmethod
    def load(cls, path: str | Path) -> WorldConfig:
        data = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(data)
