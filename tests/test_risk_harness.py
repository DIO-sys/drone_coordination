"""Batch harness and dataset construction (Phase 3).

Two things here are load-bearing for everything downstream: the split cannot
leak, and the output cannot depend on how the work was spread across cores.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from planguard.config import ScenarioConfig, WindConfig, WorldConfig
from planguard.dsl.sample import SamplerConfig, plan_id, sample_corpus
from planguard.dsl.schema import MissionPlan, TaskSpec
from planguard.dsl.validate import makespan_lower_bound_s
from planguard.risk.harness import (
    TEST_SEED_OFFSET,
    aggregate,
    build,
    build_jobs,
    run_jobs,
    seeds_for,
    split_of,
    write_dataset,
)
from planguard.sim.drone import capture_radius_m
from planguard.sim.run import simulate

WORLD = str(Path(__file__).resolve().parent.parent / "experiments" / "configs" / "world.yaml")
N_PLANS = 12
N_SEEDS = 3


@pytest.fixture(scope="module")
def plans(cfg: WorldConfig) -> list[MissionPlan]:
    out, _ = sample_corpus(cfg, SamplerConfig(), N_PLANS, seed=1)
    return out


@pytest.fixture(scope="module")
def built(cfg: WorldConfig, plans: list[MissionPlan]) -> tuple:
    return build(plans, cfg, WORLD, n_seeds=N_SEEDS, test_fraction=0.25, workers=4)


# ------------------------------------------------------------------ splits ---


def test_split_is_stable_and_content_addressed() -> None:
    assert split_of("abc123", 0.2) == split_of("abc123", 0.2)
    # a different plan id can land either way, but the mapping is a function
    assert split_of("abc123", 0.0) == "train"
    assert split_of("abc123", 1.0) == "test"


def test_split_fraction_is_roughly_honoured() -> None:
    ids = [f"{i:016x}" for i in range(4000)]
    frac = sum(1 for i in ids if split_of(i, 0.2) == "test") / len(ids)
    assert 0.17 < frac < 0.23


def test_train_and_test_draw_disjoint_scenario_seeds() -> None:
    """Rule 6: split by plan *and* by seed. A test row must be an unseen plan
    in unseen weather, or the surrogate is scored on scenarios it has met."""
    train, test = seeds_for("train", 50), seeds_for("test", 50)
    assert not set(train) & set(test)
    assert max(train) < TEST_SEED_OFFSET <= min(test)


def test_no_plan_appears_in_both_splits(built: tuple) -> None:
    plan_rows, rows, _ = built
    by_pid: dict[str, set[str]] = {}
    for r in rows:
        by_pid.setdefault(r.pid, set()).add(r.split)
    assert all(len(v) == 1 for v in by_pid.values())
    assert len({p.plan_id for p in plan_rows}) == len(plan_rows)


def test_no_seed_appears_in_both_splits(built: tuple) -> None:
    _, rows, _ = built
    train_seeds = {r.seed for r in rows if r.split == "train"}
    test_seeds = {r.seed for r in rows if r.split == "test"}
    assert not train_seeds & test_seeds


# ----------------------------------------------------------- determinism ---


def test_row_order_does_not_depend_on_worker_count(
    cfg: WorldConfig, plans: list[MissionPlan]
) -> None:
    """``imap_unordered`` keeps the cores busy, so completion order follows
    machine load. Sorting afterwards is what makes the dataset reproducible --
    without it the same command would write two different files (rule 5), in
    the one place it would be hardest to notice."""
    jobs = build_jobs(plans[:6], n_seeds=2, test_fraction=0.25)
    serial = run_jobs(jobs, WORLD, workers=1)
    parallel = run_jobs(jobs, WORLD, workers=4)
    assert [(r.pid, r.seed) for r in serial] == [(r.pid, r.seed) for r in parallel]
    assert [r.digest for r in serial] == [r.digest for r in parallel]


def test_job_order_does_not_change_results(cfg: WorldConfig, plans: list[MissionPlan]) -> None:
    jobs = build_jobs(plans[:6], n_seeds=2, test_fraction=0.25)
    forward = run_jobs(jobs, WORLD, workers=2)
    backward = run_jobs(list(reversed(jobs)), WORLD, workers=2)
    assert [r.digest for r in forward] == [r.digest for r in backward]


# --------------------------------------------------------------- labels ---


def test_every_plan_gets_every_seed(built: tuple) -> None:
    plan_rows, rows, manifest = built
    assert len(rows) == N_PLANS * N_SEEDS
    assert all(p.n_runs == N_SEEDS for p in plan_rows)
    assert manifest["n_runs"] == len(rows)


def test_failure_rate_matches_the_runs_it_came_from(built: tuple) -> None:
    plan_rows, rows, _ = built
    by_pid: dict[str, list] = {}
    for r in rows:
        by_pid.setdefault(r.pid, []).append(r)
    for p in plan_rows:
        rs = by_pid[p.plan_id]
        assert p.failure_rate == pytest.approx(sum(1 for r in rs if not r.success) / len(rs))


def test_stored_plan_json_round_trips(built: tuple, cfg: WorldConfig) -> None:
    """Phase 4 derives its features from this column, so it has to be the
    plan -- feature engineering must not require re-simulating anything."""
    plan_rows, _, _ = built
    for p in plan_rows:
        plan = MissionPlan.model_validate_json(p.plan_json)
        assert plan_id(plan) == p.plan_id
        assert plan.time_budget_s == p.time_budget_s
        assert len(plan.tasks) == p.n_tasks


def test_makespan_lower_bound_holds_in_nominal_conditions(
    cfg: WorldConfig, plans: list[MissionPlan]
) -> None:
    """Regression test for two Phase 3 bugs, and for what the bound claims.

    ``makespan_lower_bound_s`` bounds the *nominal* mission: no wind, aircraft
    starting at base, a tight arrival tolerance. It cannot bound an individual
    simulated run, because wind speed is drawn from a normal and is therefore
    unbounded -- a tailwind makes any finite bound beatable. So the check is
    run against a clean-room config, where the property is real.

    The two bugs it caught, both of which made real runs finish earlier than
    the arithmetic said was possible:

    * ``duration_s`` charged travel to a feature's **centroid**, but an
      aircraft enters a corridor at the nearer end and an area at its edge.
    * a ``station_keep`` run counted dwell from the moment it was **assigned**
      rather than from arrival, so a 545 s watch 145 s of flying away was
      satisfied by loitering near base.
    """
    nominal = cfg.model_copy(
        update={
            "wind": WindConfig(speed_mps=0.0, direction_deg=0.0, speed_sigma_mps=0.0),
            "scenario": ScenarioConfig(p_aircraft_unavailable=0.0, start_jitter_m=0.0),
            # a tighter arrival circle, so clipping waypoints cannot explain
            # away a real violation
            "fleet": cfg.fleet.model_copy(update={"max_turn_rate_dps": 120.0}),
        }
    )
    for plan in plans:
        bound_s, binding = makespan_lower_bound_s(plan, nominal)
        result = simulate(plan, nominal, seed=0)
        if not result.success:
            continue  # a failed run says nothing about how long success takes
        slack_m = capture_radius_m(nominal.fleet, nominal.sim.dt_s) * len(plan.tasks)
        slack_s = slack_m / nominal.fleet.speed_mps + nominal.sim.dt_s
        assert result.t_end_s >= bound_s - slack_s, (
            f"{plan.name} finished at {result.t_end_s:.1f}s, below a claimed "
            f"lower bound of {bound_s:.1f}s ({binding})"
        )


def test_station_keep_dwell_starts_on_arrival_not_on_assignment(cfg: WorldConfig) -> None:
    """The second bug above, pinned directly rather than through a corpus."""
    nominal = cfg.model_copy(
        update={
            "wind": WindConfig(speed_mps=0.0, direction_deg=0.0, speed_sigma_mps=0.0),
            "scenario": ScenarioConfig(p_aircraft_unavailable=0.0, start_jitter_m=0.0),
        }
    )
    dwell_s = 300.0
    plan = MissionPlan(
        name="watch the bridge",
        time_budget_s=1200.0,
        tasks=[
            TaskSpec(id="watch", kind="station_keep", target="bridge", dwell_s=dwell_s),
        ],
    )
    feature = nominal.world_map.get("bridge")
    assert feature is not None
    travel_s = feature.nearest_dist_m(nominal.base_m) / nominal.fleet.speed_mps
    result = simulate(plan, nominal, seed=0)
    assert result.success
    # the aircraft has to get there before the clock on "watch it" can start
    assert result.t_end_s >= travel_s + dwell_s - 5.0
    assert result.t_end_s > dwell_s * 1.2


def test_aggregate_ignores_plans_with_no_runs(cfg: WorldConfig, plans: list[MissionPlan]) -> None:
    assert aggregate(plans, [], cfg) == []


# --------------------------------------------------------------- output ---


def test_dataset_round_trips_through_parquet(built: tuple, tmp_path: Path) -> None:
    plan_rows, rows, manifest = built
    write_dataset(tmp_path, plan_rows, rows, manifest)
    plans_tbl = pq.read_table(tmp_path / "plans.parquet")
    runs_tbl = pq.read_table(tmp_path / "runs.parquet")
    assert plans_tbl.num_rows == len(plan_rows)
    assert runs_tbl.num_rows == len(rows)
    assert plans_tbl.column("plan_id").to_pylist() == [p.plan_id for p in plan_rows]
    assert (tmp_path / "manifest.json").exists()


def test_manifest_records_the_split_it_used(built: tuple) -> None:
    plan_rows, _, manifest = built
    assert manifest["n_plans_train"] + manifest["n_plans_test"] == len(plan_rows)
    assert manifest["train_seeds"] == seeds_for("train", N_SEEDS)
    assert manifest["test_seeds"] == seeds_for("test", N_SEEDS)


def test_bound_matches_the_validator(built: tuple, cfg: WorldConfig) -> None:
    plan_rows, _, _ = built
    for p in plan_rows:
        plan = MissionPlan.model_validate_json(p.plan_json)
        bound_s, binding = makespan_lower_bound_s(plan, cfg)
        assert p.makespan_lower_bound_s == pytest.approx(bound_s)
        assert p.binding_bound == binding
