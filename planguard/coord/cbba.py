"""Consensus-Based Bundle Algorithm (Choi, Brunet & How, 2009).

CBBA is a decentralised auction. Each aircraft privately builds a *bundle* of
tasks it wants, then aircraft exchange their winning-bid vectors and resolve
conflicts with a fixed rule table. Nobody allocates centrally: each aircraft
reasons only from bids it has heard. Under diminishing marginal gains the
algorithm provably converges to a conflict-free assignment.

Four things about this implementation are worth stating plainly.

**Multi-aircraft tasks become slots.** CBBA assigns one task to one agent, but
a plan can ask for two aircraft on a sweep. Such a task is expanded into
``aircraft_count`` independent slots (``sweep_ridge#0``, ``sweep_ridge#1``) and
consensus runs over slots. After convergence a task is committed only if *all*
of its slots were won, preserving the Phase 1 rule that a half-crewed sweep is
worse than an unstaffed one. An agent may never hold two slots of the same
task -- one aircraft cannot be two aircraft.

**The message exchange is simulated, not distributed.** ``run_consensus``
drives rounds of simultaneous pairwise exchange in-process. The algorithm is
faithful -- each agent updates only from vectors it receives, never from global
state -- but this is a single-process simulation of a decentralised protocol,
not a deployment of one.

**Diminishing marginal gains comes from time discounting.** A path's score is
the sum over its tasks of ``value * discount ** arrival_time_s``. Inserting a
task can only delay those after it, so adding to a longer bundle is worth less,
which is the property CBBA's convergence proof needs.

**Exchange is simultaneous within a round.** Every agent resolves against
snapshots taken before any update in that round, so the outcome cannot depend
on the order agents happen to be iterated in (rule 5).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

from planguard.config import WorldConfig, dist_m
from planguard.dsl.schema import MissionPlan, TaskSpec
from planguard.dsl.validate import per_aircraft_work_s
from planguard.sim.drone import DroneState

Point = tuple[float, float]


@dataclass(frozen=True)
class Slot:
    """One aircraft's worth of a task."""

    task_id: str
    index: int

    @property
    def key(self) -> str:
        return f"{self.task_id}#{self.index}"


def expand_slots(plan: MissionPlan, ready_task_ids: Sequence[str]) -> dict[str, Slot]:
    """One slot per aircraft a ready task asks for, keyed deterministically."""
    ready = set(ready_task_ids)
    out: dict[str, Slot] = {}
    for task in sorted(plan.tasks, key=lambda t: t.id):
        if task.id not in ready:
            continue
        for i in range(task.aircraft_count):
            slot = Slot(task.id, i)
            out[slot.key] = slot
    return out


def task_value(task: TaskSpec) -> float:
    """Priority 1 is most important, so value descends with the number."""
    return float(4 - task.priority)


class Action(Enum):
    LEAVE = "leave"
    UPDATE = "update"
    RESET = "reset"


@dataclass
class AgentState:
    """One aircraft's private view. It holds nothing global."""

    agent_id: str
    pos_m: Point
    #: slot keys in the order they were added; releasing one releases all later
    bundle: list[str] = field(default_factory=list)
    #: slot keys in route order, which is what the score is computed over
    path: list[str] = field(default_factory=list)
    #: slot key -> agent believed to hold it
    winners: dict[str, str | None] = field(default_factory=dict)
    #: slot key -> that agent's bid
    bids: dict[str, float] = field(default_factory=dict)
    #: agent id -> round this agent last had news of them
    timestamps: dict[str, int] = field(default_factory=dict)

    def winner(self, key: str) -> str | None:
        return self.winners.get(key)

    def bid(self, key: str) -> float:
        return self.bids.get(key, 0.0)

    def stamp(self, agent_id: str) -> int:
        return self.timestamps.get(agent_id, 0)

    def won_keys(self) -> list[str]:
        return sorted(k for k, v in self.winners.items() if v == self.agent_id)


# --------------------------------------------------------------- scoring ----


def score_path(
    state: AgentState,
    path: Sequence[str],
    slots: Mapping[str, Slot],
    plan: MissionPlan,
    cfg: WorldConfig,
) -> float:
    """Time-discounted reward for flying ``path`` in order from the agent's
    current position."""
    t_s = 0.0
    pos = state.pos_m
    total = 0.0
    for key in path:
        slot = slots[key]
        task = plan.task(slot.task_id)
        feature = cfg.world_map.get(task.target) if task else None
        if task is None or feature is None:  # unreachable for a validated plan
            continue
        # entry and exit, not the centroid in both places. A corridor patrol
        # starts at the nearer end and finishes at the far one; charging
        # travel to the centroid and then pretending the aircraft is parked
        # there understated the trip out and misplaced the aircraft by up to
        # 1.5 km for every task later in the bundle.
        t_s += dist_m(pos, feature.entry_m(pos)) / cfg.fleet.speed_mps
        total += task_value(task) * cfg.cbba.discount_per_s**t_s
        t_s += per_aircraft_work_s(task, cfg)
        pos = feature.exit_m(pos)
    return total


def best_insertion(
    state: AgentState,
    key: str,
    slots: Mapping[str, Slot],
    plan: MissionPlan,
    cfg: WorldConfig,
) -> tuple[float, int]:
    """Marginal gain of adding ``key``, and where in the path to put it."""
    base = score_path(state, state.path, slots, plan, cfg)
    best_gain = float("-inf")
    best_at = 0
    for at in range(len(state.path) + 1):
        trial = [*state.path[:at], key, *state.path[at:]]
        gain = score_path(state, trial, slots, plan, cfg) - base
        if gain > best_gain:
            best_gain, best_at = gain, at
    return best_gain, best_at


# ------------------------------------------------- phase 1: build a bundle ---


def build_bundle(
    state: AgentState, slots: Mapping[str, Slot], plan: MissionPlan, cfg: WorldConfig
) -> bool:
    """Greedily add the best outbidding task until the bundle is full.

    Returns whether anything changed. Ties break on the slot key so the result
    does not depend on dict ordering.
    """
    changed = False
    eps = cfg.cbba.bid_epsilon
    while len(state.bundle) < cfg.cbba.max_bundle:
        mine = {slots[k].task_id for k in state.bundle}
        best: tuple[float, str, int] | None = None
        for key in sorted(slots):
            if key in state.bundle:
                continue
            # one aircraft cannot fill two slots of the same task
            if slots[key].task_id in mine:
                continue
            gain, at = best_insertion(state, key, slots, plan, cfg)
            if gain <= state.bid(key) + eps:
                continue
            if best is None or gain > best[0]:
                best = (gain, key, at)
        if best is None:
            break
        gain, key, at = best
        state.path.insert(at, key)
        state.bundle.append(key)
        state.bids[key] = gain
        state.winners[key] = state.agent_id
        changed = True
    return changed


# --------------------------------------------- phase 2: resolve conflicts ----


def claim_beats(y_k: float, z_k: str | None, y_i: float, z_i: str | None, eps: float) -> bool:
    """Does the sender's claim displace the receiver's?

    Higher bid wins. **On an exact tie the lower agent id wins**, which the
    published rule table does not specify because it assumes bids are distinct.
    They are not: two aircraft at the same position produce bit-identical bids,
    and then neither can outbid the other. Both sides keep the slot, nothing
    changes, and consensus reports success while holding a duplicate
    assignment -- exactly the failure the Phase 2 acceptance check exists to
    catch. Found by the property test with two aircraft at (0, 0).

    The tie-break must be *antisymmetric* to work: when d0 and d1 tie, d0 has
    to win from both points of view, or the slot ping-pongs forever.
    """
    if y_k > y_i + eps:
        return True
    if y_i > y_k + eps:
        return False
    if z_k is None or z_i is None:
        return False
    return z_k < z_i


def decide(receiver: AgentState, sender: AgentState, key: str, eps: float) -> Action:
    """The CBBA rule table (Choi et al. 2009, Table 1).

    ``i`` is the receiver, ``k`` the sender, ``j`` the slot. ``m``/``n`` are any
    other agents. Each branch below is one row of the published table; the
    structure is kept verbose deliberately, because a clever refactor of this
    table is how subtle allocation bugs get in. Bid comparisons go through
    ``claim_beats`` so exact ties resolve consistently.
    """
    i, k = receiver.agent_id, sender.agent_id
    z_i, z_k = receiver.winner(key), sender.winner(key)
    y_i, y_k = receiver.bid(key), sender.bid(key)

    def newer(agent: str) -> bool:
        return sender.stamp(agent) > receiver.stamp(agent)

    def older(agent: str) -> bool:
        return receiver.stamp(agent) > sender.stamp(agent)

    def beats() -> bool:
        return claim_beats(y_k, z_k, y_i, z_i, eps)

    if z_k == k:
        if z_i == i:
            return Action.UPDATE if beats() else Action.LEAVE
        if z_i == k:
            return Action.UPDATE
        if z_i is None:
            return Action.UPDATE
        return Action.UPDATE if (newer(z_i) or beats()) else Action.LEAVE

    if z_k == i:
        if z_i == i:
            return Action.LEAVE
        if z_i == k:
            return Action.RESET
        if z_i is None:
            return Action.LEAVE
        return Action.RESET if newer(z_i) else Action.LEAVE

    if z_k is None:
        if z_i == i:
            return Action.LEAVE
        if z_i == k:
            return Action.UPDATE
        if z_i is None:
            return Action.LEAVE
        return Action.UPDATE if newer(z_i) else Action.LEAVE

    # z_k is some third agent m
    m = z_k
    if z_i == i:
        return Action.UPDATE if (newer(m) and beats()) else Action.LEAVE
    if z_i == k:
        return Action.UPDATE if newer(m) else Action.RESET
    if z_i == m:
        return Action.UPDATE if newer(m) else Action.LEAVE
    if z_i is None:
        return Action.UPDATE if newer(m) else Action.LEAVE
    n = z_i
    if newer(m) and (newer(n) or beats()):
        return Action.UPDATE
    if newer(n) and older(m):
        return Action.RESET
    return Action.LEAVE


def resolve(receiver: AgentState, sender: AgentState, round_index: int, eps: float) -> bool:
    """Apply the rule table for every slot the sender has an opinion on, then
    advance the receiver's clocks."""
    changed = False
    keys = sorted(set(receiver.winners) | set(sender.winners))
    for key in keys:
        action = decide(receiver, sender, key, eps)
        if action is Action.UPDATE:
            if receiver.winner(key) != sender.winner(key) or receiver.bid(key) != sender.bid(key):
                receiver.winners[key] = sender.winner(key)
                receiver.bids[key] = sender.bid(key)
                changed = True
        elif action is Action.RESET and (
            receiver.winner(key) is not None or receiver.bid(key) != 0.0
        ):
            receiver.winners[key] = None
            receiver.bids[key] = 0.0
            changed = True

    # direct contact is the freshest possible news about the sender
    receiver.timestamps[sender.agent_id] = round_index
    for agent, stamp in sender.timestamps.items():
        if agent != receiver.agent_id and stamp > receiver.stamp(agent):
            receiver.timestamps[agent] = stamp
    return changed


def repair_bundle(state: AgentState, slots: Mapping[str, Slot]) -> bool:
    """Drop the first lost slot and everything added after it.

    Later entries' bids were computed assuming the earlier ones were held, so
    they are no longer valid -- this cascade is what makes CBBA's bids
    consistent, and skipping it is the classic way to get stale bids that never
    settle.
    """
    for idx, key in enumerate(state.bundle):
        if state.winner(key) == state.agent_id:
            continue
        for lost in state.bundle[idx:]:
            if state.winner(lost) == state.agent_id:
                state.winners[lost] = None
                state.bids[lost] = 0.0
            if lost in state.path:
                state.path.remove(lost)
        del state.bundle[idx:]
        return True
    return False


# ------------------------------------------------------------- consensus -----


@dataclass(frozen=True)
class ConsensusResult:
    winners: dict[str, str]
    rounds: int
    converged: bool


def run_consensus(
    agents: list[AgentState], slots: Mapping[str, Slot], plan: MissionPlan, cfg: WorldConfig
) -> ConsensusResult:
    """Alternate bundle building and conflict resolution until nothing changes."""
    eps = cfg.cbba.bid_epsilon
    rounds = 0
    converged = False
    for round_index in range(1, cfg.cbba.max_rounds + 1):
        rounds = round_index
        changed = False
        for agent in agents:
            changed |= build_bundle(agent, slots, plan, cfg)

        # snapshot first, so every agent sees the same pre-round state
        snapshots = {
            a.agent_id: AgentState(
                agent_id=a.agent_id,
                pos_m=a.pos_m,
                winners=dict(a.winners),
                bids=dict(a.bids),
                timestamps=dict(a.timestamps),
            )
            for a in agents
        }
        for receiver in agents:
            for sender_id in sorted(snapshots):
                if sender_id == receiver.agent_id:
                    continue
                changed |= resolve(receiver, snapshots[sender_id], round_index, eps)
        for agent in agents:
            changed |= repair_bundle(agent, slots)

        if not changed:
            converged = True
            break

    winners = {key: agent.agent_id for agent in agents for key in agent.won_keys()}
    return ConsensusResult(winners=winners, rounds=rounds, converged=converged)


def allocate_cbba(
    plan: MissionPlan,
    ready_task_ids: Sequence[str],
    drones: Mapping[str, DroneState],
    busy_drone_ids: Sequence[str],
    cfg: WorldConfig,
) -> dict[str, list[str]]:
    """CBBA allocation with the same signature as the greedy placeholder.

    A task is committed only if every one of its slots was won, so a partly
    staffed task is left in the pool rather than flown short-handed.
    """
    busy = set(busy_drone_ids)
    available = [d for d in sorted(drones) if d not in busy and not drones[d].grounded]
    slots = expand_slots(plan, ready_task_ids)
    if not available or not slots:
        return {}

    agents = [AgentState(agent_id=d, pos_m=drones[d].pos_m) for d in available]
    run_consensus(agents, slots, plan, cfg)
    return commit(agents, slots, plan)


def commit(
    agents: Sequence[AgentState], slots: Mapping[str, Slot], plan: MissionPlan
) -> dict[str, list[str]]:
    """Turn settled bundles into an assignment the executive can fly.

    **Only each aircraft's first task in path order is committed.** CBBA plans a
    whole bundle, but the simulator flies one task per aircraft at a time, and
    committing a bundle of three would have had the tick loop create three runs
    for one aircraft and silently keep the last. Found by the property test:
    one aircraft and two single-aircraft tasks produced a correct bundle of two
    that the executive could not execute.

    The rest of the bundle is not wasted. It still shapes the bids -- an agent's
    bid for its first task already accounts for what it intends to fly next,
    which is where CBBA's sequencing value lives -- and consensus simply reruns
    next tick once aircraft free up. This is CBBA as a receding-horizon planner
    with incremental commitment, which is also the honest way to handle tasks
    that become ready over time, something CBBA's static guarantees do not
    cover.

    A task is committed only when all of its slots are filled, so a
    part-staffed sweep waits rather than flying short-handed.
    """
    first_choice: dict[str, str] = {}
    for agent in agents:
        for key in agent.path:
            if agent.winner(key) == agent.agent_id:
                first_choice[agent.agent_id] = key
                break

    by_task: dict[str, list[str]] = {}
    for agent_id, key in sorted(first_choice.items()):
        by_task.setdefault(slots[key].task_id, []).append(agent_id)

    out: dict[str, list[str]] = {}
    for task_id, ids in sorted(by_task.items()):
        task = plan.task(task_id)
        if task is not None and len(ids) == task.aircraft_count:
            out[task_id] = sorted(ids)
    return out
