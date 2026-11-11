"""CBBA: scoring, the rule table, consensus, and the properties that matter.

The Phase 2 acceptance check is that CBBA converges with no duplicate or
orphaned assignments. Convergence on a tidy example proves little, so the
property tests fuzz agent positions, fleet sizes and plan shapes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from planguard.config import WorldConfig
from planguard.coord.allocate import allocate, allocate_greedy
from planguard.coord.cbba import (
    Action,
    AgentState,
    allocate_cbba,
    best_insertion,
    build_bundle,
    decide,
    expand_slots,
    repair_bundle,
    run_consensus,
    score_path,
    task_value,
)
from planguard.dsl.schema import MissionPlan
from planguard.sim.drone import DroneState
from planguard.sim.scenario import generate

REPO = Path(__file__).resolve().parent.parent
DEMO = json.loads((REPO / "examples" / "ridge.json").read_text())


@pytest.fixture
def plan() -> MissionPlan:
    return MissionPlan.model_validate(DEMO)


def agents_from(cfg: WorldConfig, seed: int) -> list[AgentState]:
    return [
        AgentState(agent_id=d, pos_m=s.pos_m) for d, s in sorted(generate(cfg, seed).drones.items())
    ]


# ------------------------------------------------------------------ slots ----


def test_slots_expand_per_aircraft(plan: MissionPlan) -> None:
    slots = expand_slots(plan, plan.task_ids)
    assert sorted(slots) == ["hold_bridge#0", "sweep_ridge#0", "sweep_ridge#1"]


def test_slots_only_cover_ready_tasks(plan: MissionPlan) -> None:
    assert sorted(expand_slots(plan, ["hold_bridge"])) == ["hold_bridge#0"]


def test_task_value_descends_with_priority_number(plan: MissionPlan) -> None:
    high = next(t for t in plan.tasks if t.priority == 1)
    low = next(t for t in plan.tasks if t.priority == 2)
    assert task_value(high) > task_value(low)


# ---------------------------------------------------------------- scoring ----


def test_empty_path_scores_zero(cfg: WorldConfig, plan: MissionPlan) -> None:
    agent = AgentState(agent_id="d0", pos_m=cfg.base_m)
    assert score_path(agent, [], expand_slots(plan, plan.task_ids), plan, cfg) == 0.0


def test_closer_aircraft_bids_higher(cfg: WorldConfig, plan: MissionPlan) -> None:
    """The whole point of bidding: distance has to show up in the score."""
    slots = expand_slots(plan, plan.task_ids)
    bridge = cfg.world_map.get("bridge")
    assert bridge is not None
    near = AgentState(agent_id="near", pos_m=bridge.centroid_m)
    far = AgentState(agent_id="far", pos_m=cfg.base_m)
    key = "hold_bridge#0"
    assert (
        best_insertion(near, key, slots, plan, cfg)[0]
        > (best_insertion(far, key, slots, plan, cfg)[0])
    )


def test_marginal_gains_diminish(cfg: WorldConfig, plan: MissionPlan) -> None:
    """CBBA's convergence proof needs diminishing marginal gains. Time
    discounting supplies it: inserting a task can only delay the ones after it,
    so the same task is worth less when added to a longer bundle."""
    slots = expand_slots(plan, plan.task_ids)
    agent = AgentState(agent_id="d0", pos_m=cfg.base_m)
    first, _ = best_insertion(agent, "hold_bridge#0", slots, plan, cfg)
    agent.path = ["sweep_ridge#0"]
    later, _ = best_insertion(agent, "hold_bridge#0", slots, plan, cfg)
    assert later < first


def test_bundle_fills_to_the_configured_limit(cfg: WorldConfig, plan: MissionPlan) -> None:
    slots = expand_slots(plan, plan.task_ids)
    agent = AgentState(agent_id="d0", pos_m=cfg.base_m)
    build_bundle(agent, slots, plan, cfg)
    # only two tasks exist, and one aircraft may not take two slots of one task
    assert len(agent.bundle) == 2
    assert {slots[k].task_id for k in agent.bundle} == {"sweep_ridge", "hold_bridge"}


def test_one_aircraft_never_fills_two_slots_of_the_same_task(
    cfg: WorldConfig, plan: MissionPlan
) -> None:
    """One aircraft cannot be two aircraft."""
    slots = expand_slots(plan, ["sweep_ridge"])
    agent = AgentState(agent_id="d0", pos_m=cfg.base_m)
    build_bundle(agent, slots, plan, cfg)
    assert len(agent.bundle) == 1


# ------------------------------------------------------------- rule table ----


def state(
    agent_id: str,
    winner: str | None,
    bid: float,
    stamps: dict[str, int] | None = None,
) -> AgentState:
    return AgentState(
        agent_id=agent_id,
        pos_m=(0.0, 0.0),
        winners={"j": winner},
        bids={"j": bid},
        timestamps=stamps or {},
    )


def test_sender_claiming_with_a_better_bid_wins(cfg: WorldConfig) -> None:
    i = state("i", "i", 1.0)
    k = state("k", "k", 2.0)
    assert decide(i, k, "j", 0.0) is Action.UPDATE


def test_sender_claiming_with_a_worse_bid_is_ignored(cfg: WorldConfig) -> None:
    i = state("i", "i", 3.0)
    k = state("k", "k", 2.0)
    assert decide(i, k, "j", 0.0) is Action.LEAVE


def test_sender_agreeing_i_hold_it_changes_nothing() -> None:
    i = state("i", "i", 1.0)
    k = state("k", "i", 1.0)
    assert decide(i, k, "j", 0.0) is Action.LEAVE


def test_sender_says_i_hold_it_but_i_think_sender_does_resets() -> None:
    """Neither view can be right, so the slot goes back to open rather than
    both sides keeping a belief."""
    i = state("i", "k", 1.0)
    k = state("k", "i", 1.0)
    assert decide(i, k, "j", 0.0) is Action.RESET


def test_sender_releasing_a_slot_i_credited_to_it_updates() -> None:
    i = state("i", "k", 1.0)
    k = state("k", None, 0.0)
    assert decide(i, k, "j", 0.0) is Action.UPDATE


def test_third_party_claim_needs_fresher_news_to_displace_me() -> None:
    i = state("i", "i", 1.0, {"m": 5})
    stale = state("k", "m", 9.0, {"m": 3})
    assert decide(i, stale, "j", 0.0) is Action.LEAVE
    fresh = state("k", "m", 9.0, {"m": 7})
    assert decide(i, fresh, "j", 0.0) is Action.UPDATE


def test_third_party_claim_with_fresh_news_but_lower_bid_leaves_mine() -> None:
    i = state("i", "i", 10.0, {"m": 1})
    k = state("k", "m", 2.0, {"m": 9})
    assert decide(i, k, "j", 0.0) is Action.LEAVE


def test_unclaimed_on_both_sides_is_a_no_op() -> None:
    assert decide(state("i", None, 0.0), state("k", None, 0.0), "j", 0.0) is Action.LEAVE


# ---------------------------------------------------------------- repair -----


def test_losing_a_bundle_entry_releases_everything_after_it(
    cfg: WorldConfig, plan: MissionPlan
) -> None:
    """Later bids were computed assuming the earlier entries were held, so they
    are stale once an earlier one is lost."""
    slots = expand_slots(plan, plan.task_ids)
    agent = AgentState(agent_id="d0", pos_m=cfg.base_m)
    build_bundle(agent, slots, plan, cfg)
    assert len(agent.bundle) == 2
    first, second = agent.bundle
    agent.winners[first] = "someone_else"

    assert repair_bundle(agent, slots)
    assert agent.bundle == []
    assert agent.path == []
    assert agent.winner(second) is None
    assert agent.bid(second) == 0.0


def test_repair_is_a_no_op_when_nothing_was_lost(cfg: WorldConfig, plan: MissionPlan) -> None:
    slots = expand_slots(plan, plan.task_ids)
    agent = AgentState(agent_id="d0", pos_m=cfg.base_m)
    build_bundle(agent, slots, plan, cfg)
    assert not repair_bundle(agent, slots)


# ------------------------------------------- acceptance: conflict freedom ----


def test_consensus_converges_without_conflicts(cfg: WorldConfig, plan: MissionPlan) -> None:
    agents = agents_from(cfg, 2)
    slots = expand_slots(plan, plan.task_ids)
    result = run_consensus(agents, slots, plan, cfg)
    assert result.converged
    assert result.rounds < cfg.cbba.max_rounds
    # every slot claimed by exactly one agent, and agents agree
    claims = [k for a in agents for k in a.won_keys()]
    assert len(claims) == len(set(claims))
    for agent in agents:
        for key in agent.won_keys():
            for other in agents:
                assert other.winner(key) in (agent.agent_id, None)


def test_consensus_is_deterministic(cfg: WorldConfig, plan: MissionPlan) -> None:
    slots = expand_slots(plan, plan.task_ids)
    runs = [run_consensus(agents_from(cfg, 2), slots, plan, cfg).winners for _ in range(5)]
    assert all(r == runs[0] for r in runs)


def test_allocation_refuses_to_part_staff(cfg: WorldConfig, plan: MissionPlan) -> None:
    """Only one aircraft, and the sweep needs two: the sweep must be left in the
    pool rather than flown short-handed."""
    drones = {"d0": DroneState(id="d0", pos_m=cfg.base_m, heading_rad=0.0, fuel_s=2400.0)}
    got = allocate_cbba(plan, plan.task_ids, drones, [], cfg)
    assert "sweep_ridge" not in got
    assert got == {"hold_bridge": ["d0"]}


def test_a_bundle_may_hold_several_tasks_but_only_the_first_is_committed(
    cfg: WorldConfig,
) -> None:
    """One aircraft, two single-aircraft tasks: CBBA correctly bundles both, and
    the executive commits only the first. Committing the whole bundle would have
    the tick loop create two runs for one aircraft and keep only the last."""
    plan = MissionPlan.model_validate(
        {
            "name": "two jobs one aircraft",
            "time_budget_s": 100_000,
            "tasks": [
                {"id": "a", "kind": "corridor_patrol", "target": "river_corridor"},
                {"id": "b", "kind": "corridor_patrol", "target": "highway_corridor"},
            ],
        }
    )
    drones = {"d0": DroneState(id="d0", pos_m=cfg.base_m, heading_rad=0.0, fuel_s=2400.0)}
    agents = [AgentState(agent_id="d0", pos_m=cfg.base_m)]
    slots = expand_slots(plan, plan.task_ids)
    run_consensus(agents, slots, plan, cfg)
    assert len(agents[0].bundle) == 2, "CBBA should want both tasks"

    committed = allocate_cbba(plan, plan.task_ids, drones, [], cfg)
    assert len(committed) == 1
    assert next(iter(committed.values())) == ["d0"]


def test_no_aircraft_means_no_assignment(cfg: WorldConfig, plan: MissionPlan) -> None:
    assert allocate_cbba(plan, plan.task_ids, {}, [], cfg) == {}


def test_busy_and_grounded_aircraft_do_not_bid(cfg: WorldConfig, plan: MissionPlan) -> None:
    drones = {
        "d0": DroneState(id="d0", pos_m=cfg.base_m, heading_rad=0.0, fuel_s=2400.0),
        "d1": DroneState(id="d1", pos_m=cfg.base_m, heading_rad=0.0, fuel_s=0.0, grounded=True),
    }
    assert allocate_cbba(plan, plan.task_ids, drones, ["d0"], cfg) == {}


def test_dispatch_honours_the_configured_allocator(cfg: WorldConfig, plan: MissionPlan) -> None:
    drones = generate(cfg, 2).drones
    greedy_cfg = cfg.model_copy(
        update={"cbba": cfg.cbba.model_copy(update={"allocator": "greedy"})}
    )
    assert allocate(plan, plan.task_ids, drones, [], greedy_cfg) == allocate_greedy(
        plan, plan.task_ids, drones, [], greedy_cfg
    )
    assert allocate(plan, plan.task_ids, drones, [], cfg) == allocate_cbba(
        plan, plan.task_ids, drones, [], cfg
    )


# ----------------------------------------------------------- property test ---

TARGETS: dict[str, list[str]] = {
    "sweep": ["north_ridge", "south_valley", "east_sector"],
    "station_keep": ["bridge", "relay_hill"],
    "corridor_patrol": ["river_corridor", "highway_corridor"],
    "rally": ["rally_alpha", "rally_bravo"],
}

task_st = st.builds(
    lambda kind, i, ac, prio: {
        "id": f"t{i}",
        "kind": kind,
        "target": TARGETS[kind][i % len(TARGETS[kind])],
        "aircraft_count": ac,
        "priority": prio,
        **({"dwell_s": 120.0} if kind == "station_keep" else {}),
    },
    kind=st.sampled_from(sorted(TARGETS)),
    i=st.integers(min_value=0, max_value=5),
    ac=st.integers(min_value=1, max_value=3),
    prio=st.integers(min_value=1, max_value=3),
)


@settings(
    max_examples=120, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    raw_tasks=st.lists(task_st, min_size=1, max_size=4),
    positions=st.lists(
        st.tuples(
            st.floats(min_value=0.0, max_value=7000.0),
            st.floats(min_value=0.0, max_value=7000.0),
        ),
        min_size=1,
        max_size=5,
    ),
)
def test_cbba_always_converges_conflict_free(
    raw_tasks: list[dict[str, Any]], positions: list[tuple[float, float]], cfg: WorldConfig
) -> None:
    """The Phase 2 acceptance check, fuzzed.

    For any fleet placement and any plan shape: consensus terminates, no slot is
    claimed twice, and no committed task is part-staffed.
    """
    seen: set[str] = set()
    tasks = []
    for t in raw_tasks:
        if t["id"] in seen:
            continue
        seen.add(t["id"])
        tasks.append(t)
    plan = MissionPlan.model_validate({"name": "fuzz", "time_budget_s": 100_000, "tasks": tasks})
    drones = {
        f"d{i}": DroneState(id=f"d{i}", pos_m=pos, heading_rad=0.0, fuel_s=2400.0)
        for i, pos in enumerate(positions)
    }
    agents = [AgentState(agent_id=d, pos_m=s.pos_m) for d, s in sorted(drones.items())]
    slots = expand_slots(plan, plan.task_ids)
    result = run_consensus(agents, slots, plan, cfg)

    assert result.converged, f"did not settle in {cfg.cbba.max_rounds} rounds"
    claims = [k for a in agents for k in a.won_keys()]
    assert len(claims) == len(set(claims)), "a slot was claimed twice"
    for agent in agents:
        held = [slots[k].task_id for k in agent.won_keys()]
        assert len(held) == len(set(held)), "one aircraft took two slots of a task"
        assert len(agent.bundle) <= cfg.cbba.max_bundle

    committed = allocate_cbba(plan, plan.task_ids, drones, [], cfg)
    for task_id, ids in committed.items():
        task = plan.task(task_id)
        assert task is not None
        assert len(ids) == task.aircraft_count, "committed a part-staffed task"
        assert len(set(ids)) == len(ids)
    # An aircraft may hold several tasks in its *bundle* -- that is what CBBA is
    # for -- but the executive flies one at a time, so it may appear in the
    # committed assignment at most once.
    assigned = [d for ids in committed.values() for d in ids]
    assert len(assigned) == len(set(assigned)), "an aircraft was committed twice"
