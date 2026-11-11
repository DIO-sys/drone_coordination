"""Geometry known-value tests and config validation."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from planguard.config import MapFeature, WorldConfig, WorldMap, dist_m


def test_shoelace_area_of_a_rectangle() -> None:
    f = MapFeature(name="r", kind="area", coords_m=[(0, 0), (100, 0), (100, 50), (0, 50)])
    assert f.area_m2 == pytest.approx(5000.0)
    assert f.centroid_m == pytest.approx((50.0, 25.0))


def test_shoelace_is_orientation_independent() -> None:
    cw = MapFeature(name="r", kind="area", coords_m=[(0, 0), (0, 50), (100, 50), (100, 0)])
    assert cw.area_m2 == pytest.approx(5000.0)


def test_polyline_length_is_the_3_4_5_triangle() -> None:
    f = MapFeature(name="c", kind="corridor", coords_m=[(0, 0), (300, 400)])
    assert f.length_m == pytest.approx(500.0)


def test_multi_segment_polyline_length() -> None:
    f = MapFeature(name="c", kind="corridor", coords_m=[(0, 0), (100, 0), (100, 100)])
    assert f.length_m == pytest.approx(200.0)


def test_area_is_zero_for_non_area_features() -> None:
    assert MapFeature(name="p", kind="point", coords_m=[(1, 1)]).area_m2 == 0.0
    assert MapFeature(name="p", kind="point", coords_m=[(1, 1)]).length_m == 0.0


def test_dist_m() -> None:
    assert dist_m((0, 0), (3, 4)) == pytest.approx(5.0)
    assert dist_m((1, 1), (1, 1)) == 0.0


@pytest.mark.parametrize(
    ("kind", "coords"),
    [
        ("point", []),
        ("point", [(0, 0), (1, 1)]),
        ("area", [(0, 0), (1, 1)]),
        ("corridor", [(0, 0)]),
    ],
)
def test_feature_arity_is_enforced(kind: str, coords: list[tuple[float, float]]) -> None:
    with pytest.raises(ValidationError):
        MapFeature(name="x", kind=kind, coords_m=coords)  # type: ignore[arg-type]


def test_duplicate_feature_names_rejected() -> None:
    with pytest.raises(ValidationError):
        WorldMap(
            features=[
                MapFeature(name="a", kind="point", coords_m=[(0, 0)]),
                MapFeature(name="a", kind="point", coords_m=[(1, 1)]),
            ]
        )


def test_base_must_exist_and_be_a_point(cfg: WorldConfig) -> None:
    assert cfg.base_m == (500.0, 500.0)
    bad = cfg.model_dump()
    bad["fleet"]["base_feature"] = "north_ridge"
    with pytest.raises(ValidationError, match="must be a point feature"):
        WorldConfig.model_validate(bad)
    bad["fleet"]["base_feature"] = "nowhere"
    with pytest.raises(ValidationError, match="not a map feature"):
        WorldConfig.model_validate(bad)


def test_world_map_lookup_is_total(cfg: WorldConfig) -> None:
    for name in cfg.world_map.names:
        assert cfg.world_map.get(name) is not None
    assert cfg.world_map.get("west_ridge") is None


def test_known_world_durations_are_stable(cfg: WorldConfig) -> None:
    """The golden corpus comments quote these numbers; keep them honest."""
    ridge = cfg.world_map.get("north_ridge")
    assert ridge is not None
    assert ridge.area_m2 == pytest.approx(3_000_000.0)
    travel_s = dist_m(cfg.base_m, ridge.centroid_m) / cfg.fleet.speed_mps
    assert travel_s == pytest.approx(math.hypot(2500, 6500) / 30.0)
