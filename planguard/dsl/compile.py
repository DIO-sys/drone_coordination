"""Compile a MissionPlan into a per-aircraft py_trees behaviour tree.

The split of responsibility matters: **the tree decides mode and target, the
simulator does physics.** A tick ticks the tree, reads the mode and waypoint it
wrote into the agent's context, and then integrates kinematics. Nothing in the
tree moves an aircraft or draws a random number, which is what keeps rule 5
(determinism) cheap to hold.

Both composites use ``memory=False`` so the whole tree is re-evaluated from the
root every tick. A fallback must be able to pre-empt a task mid-leg; with
memory the aircraft would finish its current waypoint first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import py_trees

from planguard.config import WorldConfig
from planguard.dsl.schema import FallbackAction, FallbackRule, MissionPlan, TaskSpec

if TYPE_CHECKING:
    from planguard.sim.drone import DroneState
    from planguard.sim.tasks import TaskRun

Point = tuple[float, float]

#: Modes the tree can select. The simulator reads these, never writes them.
Mode = str


@dataclass
class AgentContext:
    """Everything one aircraft's tree may read, and the two fields it writes.

    Per the project's central rule the tree sees only this object, so it cannot
    reach into world state it should not know about.
    """

    drone_id: str
    plan: MissionPlan
    cfg: WorldConfig
    state: DroneState
    task: TaskSpec | None = None
    run: TaskRun | None = None
    link_down_s: float = 0.0
    # written by the tree each tick
    mode: Mode = "loiter"
    waypoint_m: Point | None = None
    fallback: FallbackRule | None = None

    def matching_fallback(self, trigger: str) -> FallbackRule | None:
        """First fallback in plan order that fires on ``trigger`` and covers the
        current task. ``continue_unlinked`` is not a behaviour change, so it is
        not reported as a match and the tree falls through to execution.

        A ``link_lost`` rule only pre-empts a task that actually declares
        ``requires_link``. Without this, a blanket rule (empty ``applies_to``)
        pre-empts everything and ``requires_link`` has no effect on behaviour at
        all -- a decorative field. It also showed up as a hard bug: the demo
        plan's far sweep band lies outside link range, so the aircraft covering
        it was pulled to the rally point for a third of every mission and the
        sweep never finished on any seed.

        An idle aircraft has no task to pre-empt, so ``link_lost`` does not
        fire for it either. ``low_fuel`` is not gated this way: running out of
        fuel matters whatever the aircraft is doing.
        """
        if trigger == "link_lost" and (self.task is None or not self.task.requires_link):
            return None
        task_id = self.task.id if self.task else None
        for rule in self.plan.fallbacks:
            if rule.trigger != trigger or rule.action == "continue_unlinked":
                continue
            if task_id is None or rule.covers(task_id):
                return rule
        return None


class _Condition(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, ctx: AgentContext, trigger: str) -> None:
        super().__init__(name=name)
        self._ctx = ctx
        self._trigger = trigger

    def update(self) -> py_trees.common.Status:
        ctx = self._ctx
        if self._trigger == "low_fuel":
            fired = ctx.state.fuel_s <= ctx.cfg.behaviour.low_fuel_reserve_s
        else:
            fired = ctx.link_down_s >= ctx.cfg.behaviour.link_lost_grace_s
        if not fired:
            return py_trees.common.Status.FAILURE
        rule = ctx.matching_fallback(self._trigger)
        if rule is None:
            return py_trees.common.Status.FAILURE
        ctx.fallback = rule
        return py_trees.common.Status.SUCCESS


class _ApplyFallback(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, ctx: AgentContext) -> None:
        super().__init__(name=name)
        self._ctx = ctx

    def update(self) -> py_trees.common.Status:
        ctx = self._ctx
        rule = ctx.fallback
        if rule is None:  # pragma: no cover - the condition sets it
            return py_trees.common.Status.FAILURE
        action: FallbackAction = rule.action
        if action == "return_to_rally" and rule.rally_point is not None:
            feature = ctx.cfg.world_map.get(rule.rally_point)
            if feature is not None:
                ctx.mode, ctx.waypoint_m = "rally", feature.centroid_m
                return py_trees.common.Status.RUNNING
        if action == "hand_off":
            ctx.mode, ctx.waypoint_m = "hand_off", ctx.cfg.base_m
            return py_trees.common.Status.RUNNING
        ctx.mode, ctx.waypoint_m = "loiter", None
        return py_trees.common.Status.RUNNING


class _HasTask(py_trees.behaviour.Behaviour):
    def __init__(self, ctx: AgentContext) -> None:
        super().__init__(name="has assigned task")
        self._ctx = ctx

    def update(self) -> py_trees.common.Status:
        run = self._ctx.run
        if run is None or run.done or run.waypoint_m is None:
            return py_trees.common.Status.FAILURE
        return py_trees.common.Status.SUCCESS


class _ExecuteTask(py_trees.behaviour.Behaviour):
    def __init__(self, ctx: AgentContext) -> None:
        super().__init__(name="fly task")
        self._ctx = ctx

    def update(self) -> py_trees.common.Status:
        ctx = self._ctx
        assert ctx.run is not None
        ctx.mode, ctx.waypoint_m = "execute", ctx.run.waypoint_m
        return py_trees.common.Status.RUNNING


class _Loiter(py_trees.behaviour.Behaviour):
    def __init__(self, ctx: AgentContext) -> None:
        super().__init__(name="loiter")
        self._ctx = ctx

    def update(self) -> py_trees.common.Status:
        self._ctx.mode, self._ctx.waypoint_m = "loiter", None
        return py_trees.common.Status.RUNNING


@dataclass
class CompiledAgent:
    ctx: AgentContext
    tree: py_trees.trees.BehaviourTree
    _ticks: int = field(default=0, repr=False)

    def tick(self) -> Mode:
        self.ctx.mode, self.ctx.waypoint_m, self.ctx.fallback = "loiter", None, None
        self.tree.tick()
        self._ticks += 1
        return self.ctx.mode


def compile_agent(ctx: AgentContext) -> CompiledAgent:
    """Build one aircraft's tree.

    Branch order is the priority order: running out of fuel outranks losing the
    link, which outranks executing the task.
    """
    root = py_trees.composites.Selector(name=f"root:{ctx.drone_id}", memory=False)

    low_fuel = py_trees.composites.Sequence(name="low fuel", memory=False)
    low_fuel.add_children(
        [_Condition("fuel below reserve", ctx, "low_fuel"), _ApplyFallback("apply", ctx)]
    )

    link_lost = py_trees.composites.Sequence(name="link lost", memory=False)
    link_lost.add_children(
        [_Condition("link down past grace", ctx, "link_lost"), _ApplyFallback("apply", ctx)]
    )

    execute = py_trees.composites.Sequence(name="execute task", memory=False)
    execute.add_children([_HasTask(ctx), _ExecuteTask(ctx)])

    root.add_children([low_fuel, link_lost, execute, _Loiter(ctx)])
    return CompiledAgent(ctx=ctx, tree=py_trees.trees.BehaviourTree(root))


def render_ascii(agent: CompiledAgent) -> str:
    """Text rendering of the tree, for the debug viewer and the dashboard."""
    return str(py_trees.display.ascii_tree(agent.tree.root))
