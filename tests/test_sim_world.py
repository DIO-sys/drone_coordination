"""Task semantics, behaviour tree, allocation and the tick loop.

Three of these are the Phase 1 acceptance checks and are marked as such:
a plan executes end to end, runs are reproducible across processes, and
outcomes vary across seeds.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

from planguard.config import (
    LinkConfig,
    ScenarioConfig,
    WindConfig,
    WorldConfig,
    dist_m,
)
from planguard.coord.allocate import allocate
from planguard.dsl.compile import AgentContext, compile_agent, render_ascii
from planguard.dsl.schema import MissionPlan, TaskSpec
from planguard.dsl.validate import duration_s, validate
from planguard.sim.drone import DroneState
from planguard.sim.logging import DRONE_SCHEMA, RunLogger
from planguard.sim.replay import replay
from planguard.sim.run import simulate
from planguard.sim.scenario import generate
from planguard.sim.tasks import (
    advance,
    orbit_waypoints,
    share_of,
    start_run,
    sweep_transects,
    waypoints_for,
)
from planguard.sim.world import World

REPO = Path(__file__).resolve().parent.parent
DEMO = json.loads((REPO / "examples" / "ridge.json").read_text())


@pytest.fixture
def plan() -> MissionPlan:
    return MissionPlan.model_validate(DEMO)


@pytest.fixture
def quiet_cfg(cfg: WorldConfig) -> WorldConfig:
    """No wind, no link drops, every aircraft available, no start jitter.

    Used to compare the simulator against the validator's duration model with
    the stochastic channels switched off.
    """
    return cfg.model_copy(
        update={
            "wind": WindConfig(speed_mps=0.0, speed_sigma_mps=0.0),
            "link": LinkConfig(range_m=cfg.link.range_m, p_drop_per_s=0.0),
            "scenario": ScenarioConfig(p_aircraft_unavailable=0.0, start_jitter_m=0.0),
        }
    )


def path_length_m(points: list[tuple[float, float]]) -> float:
    return sum(dist_m(points[i], points[i + 1]) for i in range(len(points) - 1))


# ------------------------------------------------------------ task geometry ---


def test_transect_count_covers_the_short_axis(cfg: WorldConfig) -> None:
    ridge = cfg.world_map.get("north_ridge")
    assert ridge is not None
    # 1500 m of height at a 150 m swath
    assert len(sweep_transects(ridge, cfg.fleet.sensor_swath_m)) == 10


def test_transects_run_along_the_longer_axis(cfg: WorldConfig) -> None:
    ridge = cfg.world_map.get("north_ridge")  # 2000 wide x 1500 tall
    assert ridge is not None
    (a, b) = sweep_transects(ridge, cfg.fleet.sensor_swath_m)[0]
    assert a[1] == b[1] and a[0] != b[0]


def test_transects_alternate_direction(cfg: WorldConfig) -> None:
    ridge = cfg.world_map.get("north_ridge")
    assert ridge is not None
    ts = sweep_transects(ridge, cfg.fleet.sensor_swath_m)
    assert ts[0][0][0] < ts[0][1][0]
    assert ts[1][0][0] > ts[1][1][0]


def test_share_of_is_contiguous_and_total() -> None:
    legs = [((float(i), 0.0), (float(i), 1.0)) for i in range(10)]
    shares = [share_of(legs, i, 3) for i in range(3)]
    assert [leg for s in shares for leg in s] == legs
    for s in shares:
        starts = [int(leg[0][0]) for leg in s]
        assert starts == list(range(starts[0], starts[0] + len(starts)))


def test_sweep_path_matches_the_validators_coverage_model(cfg: WorldConfig) -> None:
    """Regression for the round-robin split.

    Round-robin gave each aircraft every n-th transect, all running the same
    way, so it deadheaded the full width between legs: 18.1 km against an ideal
    of 10.0 km. The validator's duration estimate assumes ``area / swath``, so
    that 81% overshoot made the validator's time check wrong by construction.
    """
    ridge = cfg.world_map.get("north_ridge")
    assert ridge is not None
    ideal_total_m = ridge.area_m2 / cfg.fleet.sensor_swath_m
    task = TaskSpec(id="t", kind="sweep", target="north_ridge", aircraft_count=2)
    for share in (0, 1):
        wps = waypoints_for(
            task, ridge, cfg.fleet.sensor_swath_m, cfg.fleet.turn_radius_m, share, 2
        )
        flown = path_length_m(wps)
        assert flown == pytest.approx(ideal_total_m / 2, rel=0.10), flown


def test_orbit_waypoints_sit_on_the_circle() -> None:
    wps = orbit_waypoints((100.0, 200.0), 50.0, n=8)
    assert len(wps) == 8
    for w in wps:
        assert dist_m(w, (100.0, 200.0)) == pytest.approx(50.0)


def test_corridor_waypoints_are_the_polyline(cfg: WorldConfig) -> None:
    corridor = cfg.world_map.get("river_corridor")
    assert corridor is not None
    task = TaskSpec(id="t", kind="corridor_patrol", target="river_corridor")
    wps = waypoints_for(task, corridor, 150.0, 115.0, 0, 1)
    assert wps == list(corridor.coords_m)


# ---------------------------------------------------------- task progress ----


def test_non_cyclic_run_completes_after_its_last_waypoint(cfg: WorldConfig) -> None:
    valley = cfg.world_map.get("south_valley")
    assert valley is not None
    task = TaskSpec(id="t", kind="sweep", target="south_valley")
    run = start_run(task, valley, cfg.fleet.sensor_swath_m, cfg.fleet.turn_radius_m, 0, 1, 0.1)
    n = len(run.waypoints_m)
    for _ in range(n):
        assert not run.done
        run = advance(run, arrived=True, dt_s=0.1)
    assert run.done
    assert run.progress == 1.0


def test_station_keep_dwell_does_not_run_before_arrival(cfg: WorldConfig) -> None:
    """Phase 3 bug: the dwell counter ran from assignment, not from arrival,
    so a long watch over a distant point was satisfied by loitering near base.
    """
    bridge = cfg.world_map.get("bridge")
    assert bridge is not None
    task = TaskSpec(id="t", kind="station_keep", target="bridge", dwell_s=1.0)
    run = start_run(task, bridge, cfg.fleet.sensor_swath_m, cfg.fleet.turn_radius_m, 0, 1, 0.1)
    assert run.cyclic

    for _ in range(50):  # five seconds of flying, far longer than the dwell
        run = advance(run, arrived=False, dt_s=0.1)
        assert not run.done
        assert run.on_station_ticks == 0
        assert run.progress == 0.0


def test_station_keep_completes_on_dwell_not_on_waypoints(cfg: WorldConfig) -> None:
    bridge = cfg.world_map.get("bridge")
    assert bridge is not None
    task = TaskSpec(id="t", kind="station_keep", target="bridge", dwell_s=1.0)
    run = start_run(task, bridge, cfg.fleet.sensor_swath_m, cfg.fleet.turn_radius_m, 0, 1, 0.1)

    run = advance(run, arrived=True, dt_s=0.1)  # on station; the clock starts
    assert run.reached
    assert not run.done
    for _ in range(8):
        run = advance(run, arrived=False, dt_s=0.1)
        assert not run.done
    # it completes on its dwell having elapsed, not on running out of orbit
    # waypoints -- an orbit has no end
    run = advance(run, arrived=False, dt_s=0.1)
    assert run.done
    assert run.on_station_ticks == 10


def test_progress_is_monotonic(cfg: WorldConfig) -> None:
    valley = cfg.world_map.get("south_valley")
    assert valley is not None
    task = TaskSpec(id="t", kind="sweep", target="south_valley")
    run = start_run(task, valley, cfg.fleet.sensor_swath_m, cfg.fleet.turn_radius_m, 0, 1, 0.1)
    seen = [run.progress]
    for _ in range(len(run.waypoints_m)):
        run = advance(run, arrived=True, dt_s=0.1)
        seen.append(run.progress)
    assert seen == sorted(seen)


# ------------------------------------------------------------ behaviour tree --


def agent_ctx(cfg: WorldConfig, plan: MissionPlan, **kw: Any) -> AgentContext:
    state = kw.pop("state", DroneState(id="d0", pos_m=cfg.base_m, heading_rad=0.0, fuel_s=2400.0))
    return AgentContext(drone_id="d0", plan=plan, cfg=cfg, state=state, **kw)


def test_no_task_means_loiter(cfg: WorldConfig, plan: MissionPlan) -> None:
    agent = compile_agent(agent_ctx(cfg, plan))
    assert agent.tick() == "loiter"


def test_assigned_task_is_executed(cfg: WorldConfig, plan: MissionPlan) -> None:
    valley = cfg.world_map.get("south_valley")
    assert valley is not None
    task = TaskSpec(id="t", kind="sweep", target="south_valley")
    run = start_run(task, valley, 150.0, 115.0, 0, 1, 0.1)
    agent = compile_agent(agent_ctx(cfg, plan, task=task, run=run))
    assert agent.tick() == "execute"
    assert agent.ctx.waypoint_m == run.waypoint_m


def test_low_fuel_outranks_executing(cfg: WorldConfig) -> None:
    plan = MissionPlan.model_validate(
        {
            "name": "p",
            "time_budget_s": 600,
            "tasks": [{"id": "t", "kind": "sweep", "target": "south_valley"}],
            "fallbacks": [
                {"trigger": "low_fuel", "action": "return_to_rally", "rally_point": "rally_alpha"}
            ],
        }
    )
    valley = cfg.world_map.get("south_valley")
    assert valley is not None
    task = plan.tasks[0]
    run = start_run(task, valley, 150.0, 115.0, 0, 1, 0.1)
    low = DroneState(id="d0", pos_m=cfg.base_m, heading_rad=0.0, fuel_s=10.0)
    agent = compile_agent(agent_ctx(cfg, plan, task=task, run=run, state=low))
    assert agent.tick() == "rally"
    rally = cfg.world_map.get("rally_alpha")
    assert rally is not None
    assert agent.ctx.waypoint_m == rally.centroid_m


def test_link_loss_preempts_a_link_dependent_task(cfg: WorldConfig, plan: MissionPlan) -> None:
    bridge = cfg.world_map.get("bridge")
    assert bridge is not None
    task = next(t for t in plan.tasks if t.requires_link)
    run = start_run(task, bridge, 150.0, 115.0, 0, 1, 0.1)
    agent = compile_agent(agent_ctx(cfg, plan, task=task, run=run, link_down_s=30.0))
    assert agent.tick() == "rally"


def test_link_loss_does_not_preempt_a_task_that_does_not_need_a_link(
    cfg: WorldConfig, plan: MissionPlan
) -> None:
    """Regression. The demo plan's blanket link_lost rule (empty applies_to)
    used to pre-empt every task, so ``requires_link`` changed nothing. The far
    sweep band lies outside link range, so the aircraft covering it was pulled
    to the rally point for a third of every mission and the sweep never
    completed on any seed -- which flattened outcome variance to zero and would
    have made Phase 4 impossible."""
    ridge = cfg.world_map.get("north_ridge")
    assert ridge is not None
    task = next(t for t in plan.tasks if not t.requires_link)
    run = start_run(task, ridge, 150.0, 115.0, 0, 1, 0.1)
    agent = compile_agent(agent_ctx(cfg, plan, task=task, run=run, link_down_s=600.0))
    assert agent.tick() == "execute"


def test_idle_aircraft_does_not_rally_on_link_loss(cfg: WorldConfig, plan: MissionPlan) -> None:
    agent = compile_agent(agent_ctx(cfg, plan, link_down_s=600.0))
    assert agent.tick() == "loiter"


def test_continue_unlinked_falls_through_to_execution(cfg: WorldConfig) -> None:
    plan = MissionPlan.model_validate(
        {
            "name": "p",
            "time_budget_s": 600,
            "tasks": [
                {
                    "id": "t",
                    "kind": "station_keep",
                    "target": "bridge",
                    "dwell_s": 60,
                    "requires_link": True,
                }
            ],
            "fallbacks": [{"trigger": "link_lost", "action": "continue_unlinked"}],
        }
    )
    bridge = cfg.world_map.get("bridge")
    assert bridge is not None
    run = start_run(plan.tasks[0], bridge, 150.0, 115.0, 0, 1, 0.1)
    agent = compile_agent(agent_ctx(cfg, plan, task=plan.tasks[0], run=run, link_down_s=600.0))
    assert agent.tick() == "execute"


def test_hand_off_sets_its_own_mode(cfg: WorldConfig) -> None:
    plan = MissionPlan.model_validate(
        {
            "name": "p",
            "time_budget_s": 600,
            "tasks": [
                {"id": "t", "kind": "sweep", "target": "south_valley", "requires_link": True}
            ],
            "fallbacks": [{"trigger": "link_lost", "action": "hand_off"}],
        }
    )
    valley = cfg.world_map.get("south_valley")
    assert valley is not None
    run = start_run(plan.tasks[0], valley, 150.0, 115.0, 0, 1, 0.1)
    agent = compile_agent(agent_ctx(cfg, plan, task=plan.tasks[0], run=run, link_down_s=600.0))
    assert agent.tick() == "hand_off"


def test_tree_renders(cfg: WorldConfig, plan: MissionPlan) -> None:
    text = render_ascii(compile_agent(agent_ctx(cfg, plan)))
    assert "low fuel" in text and "link lost" in text and "execute task" in text


# -------------------------------------------------------------- allocation ---


def drones_at(cfg: WorldConfig, **positions: tuple[float, float]) -> dict[str, DroneState]:
    return {
        did: DroneState(id=did, pos_m=pos, heading_rad=0.0, fuel_s=2400.0)
        for did, pos in positions.items()
    }


def test_allocation_prefers_the_nearest_aircraft(cfg: WorldConfig) -> None:
    plan = MissionPlan.model_validate(
        {
            "name": "p",
            "time_budget_s": 1200,
            "tasks": [{"id": "hold", "kind": "station_keep", "target": "bridge", "dwell_s": 60}],
        }
    )
    bridge = cfg.world_map.get("bridge")
    assert bridge is not None
    drones = drones_at(cfg, d0=cfg.base_m, d1=bridge.centroid_m)
    assert allocate(plan, ["hold"], drones, [], cfg) == {"hold": ["d1"]}


def test_allocation_refuses_to_part_staff_a_task(cfg: WorldConfig, plan: MissionPlan) -> None:
    """A half-crewed sweep would silently take twice as long and corrupt the
    time-margin signal Phase 4 learns from, so it is left unassigned."""
    drones = drones_at(cfg, d0=cfg.base_m)
    got = allocate(plan, ["sweep_ridge"], drones, [], cfg)
    assert got == {}


def test_allocation_skips_busy_and_grounded_aircraft(cfg: WorldConfig) -> None:
    plan = MissionPlan.model_validate(
        {
            "name": "p",
            "time_budget_s": 1200,
            "tasks": [{"id": "hold", "kind": "station_keep", "target": "bridge", "dwell_s": 60}],
        }
    )
    drones = drones_at(cfg, d0=cfg.base_m, d1=cfg.base_m)
    drones["d1"] = DroneState(id="d1", pos_m=cfg.base_m, heading_rad=0.0, fuel_s=0.0, grounded=True)
    assert allocate(plan, ["hold"], drones, ["d0"], cfg) == {}


def test_allocation_is_deterministic(cfg: WorldConfig, plan: MissionPlan) -> None:
    drones = generate(cfg, 2).drones
    first = allocate(plan, plan.task_ids, drones, [], cfg)
    assert all(allocate(plan, plan.task_ids, drones, [], cfg) == first for _ in range(5))


def test_allocation_takes_priority_first(cfg: WorldConfig, plan: MissionPlan) -> None:
    """hold_bridge is priority 1 and needs one aircraft; with only one aircraft
    available it must win over the priority-2 sweep."""
    drones = drones_at(cfg, d0=cfg.base_m)
    assert allocate(plan, plan.task_ids, drones, [], cfg) == {"hold_bridge": ["d0"]}


# --------------------------------------------------- acceptance check 1: runs --


def test_the_demo_plan_executes_end_to_end(quiet_cfg: WorldConfig, plan: MissionPlan) -> None:
    assert validate(plan, quiet_cfg).ok
    result = simulate(plan, quiet_cfg, seed=7)
    assert result.success, result.task_status
    assert all(s == "done" for s in result.task_status.values())
    assert result.t_end_s < plan.time_budget_s


def test_simulated_sweep_duration_matches_the_validator_estimate(quiet_cfg: WorldConfig) -> None:
    """Ties Phase 0's time check to Phase 1's physics.

    The validator's estimate is a lower bound (unlimited parallelism, no
    turning, no wind), so the simulator may be slower -- but not by much, or
    the time check is not a useful bound. Wind and link drops are off here.
    """
    plan = MissionPlan.model_validate(
        {
            "name": "one sweep",
            "time_budget_s": 2000,
            "tasks": [{"id": "s", "kind": "sweep", "target": "south_valley"}],
        }
    )
    estimate_s = duration_s(plan.tasks[0], quiet_cfg)
    result = simulate(plan, quiet_cfg, seed=1)
    assert result.success
    assert estimate_s <= result.t_end_s <= estimate_s * 1.3, (estimate_s, result.t_end_s)


# ------------------------------------------- acceptance check 2: determinism --


def test_same_seed_gives_the_same_digest(cfg: WorldConfig, plan: MissionPlan) -> None:
    assert simulate(plan, cfg, 5).digest == simulate(plan, cfg, 5).digest


def test_different_seeds_give_different_digests(cfg: WorldConfig, plan: MissionPlan) -> None:
    assert simulate(plan, cfg, 5).digest != simulate(plan, cfg, 6).digest


def test_runs_are_reproducible_across_processes(plan: MissionPlan) -> None:
    """Rule 5's real test. An in-process repeat would not catch a hash-seed or
    dict-ordering dependency, which is the usual cause of a sim that drifts."""
    cmd = [
        sys.executable,
        "-m",
        "planguard.sim.run",
        "--plan",
        str(REPO / "examples" / "ridge.json"),
        "--seed",
        "11",
        "--world",
        str(REPO / "experiments" / "configs" / "world.yaml"),
        "--quiet",
    ]
    digests = {
        subprocess.run(
            cmd,
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": str(seed_env), "PATH": "/usr/bin:/bin"},
        ).stdout.strip()
        for seed_env in (0, 1)
    }
    assert len(digests) == 1, digests


# ---------------------------------------------- acceptance check 3: variance --


def test_outcomes_vary_across_seeds(cfg: WorldConfig, plan: MissionPlan) -> None:
    """The check that can invalidate the whole project.

    If mission outcome were a deterministic function of (plan, world), the
    Phase 4 risk surrogate would have nothing to predict and Phases 4 through 7
    would be built on sand. Variance comes from aircraft availability, wind and
    link outages (see sim/scenario.py).
    """
    results = [simulate(plan, cfg, seed) for seed in range(10)]
    assert len({r.success for r in results}) == 2, [r.success for r in results]
    assert len({r.completed_fraction for r in results}) > 1
    assert len({r.t_end_s for r in results}) > 1
    assert len({r.digest for r in results}) == 10


def test_tight_budgets_make_outcome_sensitive_to_execution_risk(cfg: WorldConfig) -> None:
    """Variance must not come only from how many aircraft launched.

    With a budget near the feasibility boundary, wind and link outages decide
    the outcome. Phase 3's plan sampler therefore has to sample budgets near
    that boundary, or the risk model will only ever learn "enough aircraft?".

    790 s is measured, not guessed. The demo plan's makespan lower bound is
    772 s, and the cliff is between 780 s (11/22 of full-fleet runs succeed)
    and 810 s (22/22) -- a window about 4% wide. Phase 3's sampler band is
    calibrated from this; see ARCHITECTURE section 10. The window moved here
    once ``station_keep`` stopped counting its dwell before the aircraft
    arrived, which had been making the demo plan's 600 s watch nearly free.
    """
    tight = dict(DEMO, time_budget_s=790, name="tight")
    plan = MissionPlan.model_validate(tight)
    results = [simulate(plan, cfg, seed) for seed in range(24)]
    full_fleet = [r for r in results if r.aircraft_launched >= 3]
    assert full_fleet, "no seed launched enough aircraft to test the time margin"
    assert len({r.success for r in full_fleet}) == 2, [
        (r.seed, r.aircraft_launched, r.success) for r in full_fleet
    ]


# ----------------------------------------------------------------- logging ---


def test_logging_writes_the_documented_schema(
    quiet_cfg: WorldConfig, plan: MissionPlan, tmp_path: Path
) -> None:
    """Phases 3 and 4 read only these files, so the schema is a contract.

    This path is also the one that carried an UnboundLocalError for a while,
    because nothing in the suite ran the simulator with a logger attached.
    """
    logger = RunLogger(out_dir=tmp_path)
    World(plan=plan, cfg=quiet_cfg, scenario=generate(quiet_cfg, 7), logger=logger).run()

    table = pq.read_table(tmp_path / "drones.parquet")
    assert table.schema.names == DRONE_SCHEMA.names
    assert table.num_rows > 0

    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    kinds = {e["kind"] for e in events}
    assert {"task_started", "task_completed", "mission_end"} <= kinds
    assert all("t_s" in e for e in events)


def test_replay_renders_a_logged_run(
    quiet_cfg: WorldConfig, plan: MissionPlan, tmp_path: Path
) -> None:
    logger = RunLogger(out_dir=tmp_path)
    World(plan=plan, cfg=quiet_cfg, scenario=generate(quiet_cfg, 7), logger=logger).run()
    text = replay(tmp_path, quiet_cfg)
    assert "north up" in text
    assert "time in each mode" in text
    assert "task_completed" in text
