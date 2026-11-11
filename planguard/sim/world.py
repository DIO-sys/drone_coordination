"""The tick loop.

Fixed timestep (rule 8). Each tick, in this order:

1. link availability per aircraft, from its range to base
2. mark tasks whose dependencies are satisfied as ready
3. allocate idle aircraft to ready tasks (``coord.allocate``; CBBA in Phase 2)
4. tick each aircraft's behaviour tree, which writes a mode and a waypoint
5. integrate kinematics toward that waypoint
6. advance task progress if the aircraft arrived
7. settle task completion and handoffs, then log

Steps 4 and 5 are deliberately separate: the tree decides, the simulator moves.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Literal

from planguard.config import WorldConfig, dist_m
from planguard.coord.allocate import allocate
from planguard.dsl.compile import AgentContext, CompiledAgent, compile_agent
from planguard.dsl.schema import MissionPlan
from planguard.sim import link as link_mod
from planguard.sim.drone import DroneState, has_arrived, step
from planguard.sim.logging import RunLogger
from planguard.sim.scenario import Scenario, streams
from planguard.sim.tasks import TaskRun, advance, start_run

TaskStatus = Literal["pending", "active", "done", "failed"]


@dataclass(frozen=True)
class RunResult:
    seed: int
    plan_name: str
    success: bool
    t_end_s: float
    ticks: int
    task_status: dict[str, TaskStatus]
    task_progress: dict[str, float]
    aircraft_launched: int
    aircraft_grounded: int
    drone_seconds_no_link: float
    handoffs: int
    wind_speed_mps: float
    #: Digest of the whole trajectory, for the cross-process determinism check.
    digest: str

    @property
    def completed_fraction(self) -> float:
        if not self.task_status:
            return 0.0
        done = sum(1 for s in self.task_status.values() if s == "done")
        return done / len(self.task_status)


@dataclass
class World:
    plan: MissionPlan
    cfg: WorldConfig
    scenario: Scenario
    logger: RunLogger | None = None

    drones: dict[str, DroneState] = field(init=False)
    agents: dict[str, CompiledAgent] = field(init=False)
    runs: dict[str, TaskRun] = field(init=False)
    assignment: dict[str, list[str]] = field(init=False)
    status: dict[str, TaskStatus] = field(init=False)
    progress: dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        self.drones = dict(self.scenario.drones)
        self.runs = {}
        self.assignment = {}
        self.status = {t.id: "pending" for t in self.plan.tasks}
        self.progress = {t.id: 0.0 for t in self.plan.tasks}
        self._handoffs: dict[str, int] = {t.id: 0 for t in self.plan.tasks}
        self._link_down_s: dict[str, float] = dict.fromkeys(self.drones, 0.0)
        self._no_link_s = 0.0
        self._link_rng = streams(self.scenario.seed)["link"]
        self._hash = hashlib.sha256()
        self.agents = {
            did: compile_agent(
                AgentContext(drone_id=did, plan=self.plan, cfg=self.cfg, state=state)
            )
            for did, state in sorted(self.drones.items())
        }

    # ------------------------------------------------------------- helpers ---

    def _task_of(self, drone_id: str) -> str | None:
        for task_id, ids in sorted(self.assignment.items()):
            if drone_id in ids:
                return task_id
        return None

    def _ready_task_ids(self) -> list[str]:
        out = []
        for task in self.plan.tasks:
            if self.status[task.id] != "pending":
                continue
            if all(self.status.get(d) == "done" for d in task.depends_on):
                out.append(task.id)
        return sorted(out)

    def _release(self, task_id: str, t_s: float) -> None:
        """Hand a task back to the pool, failing it if it has bounced too often."""
        for did in self.assignment.pop(task_id, []):
            self.runs.pop(did, None)
        self._handoffs[task_id] += 1
        if self._handoffs[task_id] > self.cfg.behaviour.max_handoffs:
            self.status[task_id] = "failed"
            if self.logger:
                self.logger.event(t_s, "task_failed", task_id=task_id, reason="handoff_limit")
        else:
            self.status[task_id] = "pending"
            if self.logger:
                self.logger.event(t_s, "task_handoff", task_id=task_id)

    # ---------------------------------------------------------------- tick ---

    def tick(self, tick_index: int) -> None:
        dt_s = self.cfg.sim.dt_s
        t_s = tick_index * dt_s

        # 1. link availability
        for did in sorted(self.drones):
            state = self.drones[did]
            up = link_mod.step(
                state.link_up,
                dist_m(state.pos_m, self.cfg.base_m),
                self.cfg.link,
                dt_s,
                self._link_rng,
            )
            if up != state.link_up and self.logger:
                self.logger.event(t_s, "link_up" if up else "link_down", drone_id=did)
            self.drones[did] = replace(state, link_up=up)
            if up:
                self._link_down_s[did] = 0.0
            else:
                self._link_down_s[did] += dt_s
                self._no_link_s += dt_s

        # 2 & 3. allocate ready tasks to idle aircraft
        ready = self._ready_task_ids()
        if ready:
            busy = [d for d, r in self.runs.items() if not r.done]
            fresh = allocate(self.plan, ready, self.drones, busy, self.cfg)
            for task_id, ids in sorted(fresh.items()):
                task = self.plan.task(task_id)
                feature = self.cfg.world_map.get(task.target) if task else None
                if task is None or feature is None:  # unreachable for a validated plan
                    continue
                self.assignment[task_id] = ids
                self.status[task_id] = "active"
                for share, did in enumerate(ids):
                    self.runs[did] = start_run(
                        task,
                        feature,
                        self.cfg.fleet.sensor_swath_m,
                        self.cfg.fleet.turn_radius_m,
                        share,
                        len(ids),
                        dt_s,
                    )
                if self.logger:
                    self.logger.event(t_s, "task_started", task_id=task_id, drones=ids)

        # 4 - 6. decide, move, progress
        for did in sorted(self.drones):
            current_task_id = self._task_of(did)
            agent = self.agents[did]
            agent.ctx.state = self.drones[did]
            agent.ctx.run = self.runs.get(did)
            agent.ctx.task = self.plan.task(current_task_id) if current_task_id else None
            agent.ctx.link_down_s = self._link_down_s[did]
            mode = agent.tick()

            before = self.drones[did]
            moved = step(before, agent.ctx.waypoint_m, self.cfg.fleet, self.scenario.wind_mps, dt_s)
            self.drones[did] = moved
            if moved.grounded and not before.grounded and self.logger:
                self.logger.event(t_s, "aircraft_grounded", drone_id=did)

            run = self.runs.get(did)
            if mode == "execute" and run is not None and run.waypoint_m is not None:
                arrived = has_arrived(moved, run.waypoint_m, self.cfg.fleet, dt_s)
                self.runs[did] = advance(run, arrived, dt_s)
            elif mode == "hand_off" and current_task_id is not None:
                self._release(current_task_id, t_s)

            if self.logger:
                self.logger.drone(
                    tick=tick_index,
                    t_s=round(t_s, 4),
                    drone_id=did,
                    x_m=moved.pos_m[0],
                    y_m=moved.pos_m[1],
                    heading_rad=moved.heading_rad,
                    fuel_s=moved.fuel_s,
                    link_up=moved.link_up,
                    grounded=moved.grounded,
                    mode=mode,
                    task_id=current_task_id or "",
                )
            self._hash.update(
                f"{tick_index}|{did}|{moved.pos_m[0]!r}|{moved.pos_m[1]!r}"
                f"|{moved.heading_rad!r}|{moved.fuel_s!r}|{moved.link_up}|{mode}".encode()
            )

        # 7. settle completion
        for task_id, ids in sorted(self.assignment.items()):
            if self.status[task_id] != "active":
                continue
            task_runs = [self.runs[d] for d in ids if d in self.runs]
            if task_runs:
                self.progress[task_id] = sum(r.progress for r in task_runs) / len(task_runs)
            if task_runs and all(r.done for r in task_runs):
                self.status[task_id] = "done"
                self.progress[task_id] = 1.0
                for d in ids:
                    self.runs.pop(d, None)
                if self.logger:
                    self.logger.event(t_s, "task_completed", task_id=task_id)

    # ----------------------------------------------------------------- run ---

    def _finished(self) -> bool:
        if all(s in ("done", "failed") for s in self.status.values()):
            return True
        return all(d.grounded for d in self.drones.values())

    def run(self) -> RunResult:
        duration_s = min(self.plan.time_budget_s, self.cfg.sim.max_duration_s)
        total_ticks = self.cfg.sim.ticks(duration_s)
        ticks = 0
        for tick_index in range(total_ticks):
            self.tick(tick_index)
            ticks = tick_index + 1
            if self._finished():
                break

        t_end_s = ticks * self.cfg.sim.dt_s
        for task_id, st in sorted(self.status.items()):
            if st in ("pending", "active"):
                self.status[task_id] = "failed"
                if self.logger:
                    # debrief reads only the logs, so a task that simply ran out
                    # of time has to leave a record of its own
                    self.logger.event(t_end_s, "task_failed", task_id=task_id, reason="timeout")

        # Success is all-or-nothing: every task completed and no aircraft lost.
        # An earlier version only required priority-1 tasks, which reported
        # success on a run whose main sweep had failed. Priority orders
        # allocation; it does not excuse a task. ``completed_fraction`` and
        # ``task_progress`` carry the graded signal Phase 4 regresses on.
        grounded = sum(1 for d in self.drones.values() if d.grounded)
        success = all(s == "done" for s in self.status.values()) and grounded == 0

        if self.logger:
            self.logger.event(
                t_end_s,
                "mission_end",
                success=success,
                task_status=dict(sorted(self.status.items())),
                task_progress={k: round(v, 4) for k, v in sorted(self.progress.items())},
            )
            self.logger.write()

        wind = self.scenario.wind_mps
        return RunResult(
            seed=self.scenario.seed,
            plan_name=self.plan.name,
            success=success,
            t_end_s=t_end_s,
            ticks=ticks,
            task_status=dict(self.status),
            task_progress={k: round(v, 4) for k, v in self.progress.items()},
            aircraft_launched=len(self.drones),
            aircraft_grounded=grounded,
            drone_seconds_no_link=round(self._no_link_s, 2),
            handoffs=sum(self._handoffs.values()),
            wind_speed_mps=round((wind[0] ** 2 + wind[1] ** 2) ** 0.5, 3),
            digest=self._hash.hexdigest(),
        )
