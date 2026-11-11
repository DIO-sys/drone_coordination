"""The static validator — the spine of the project.

Nine checks. The first four establish that the plan is *schedulable*; the
capacity and time checks need a schedule, so when a blocking error is found
they are reported as skipped rather than passing vacuously.

Nothing here is expressible as a JSON Schema constraint, which is why the
headline metric is validator-pass rate rather than schema validity.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from planguard.config import REQUIRED_FEATURE_KIND, WorldConfig, dist_m
from planguard.dsl.errors import BLOCKING_CODES, ValidationError, ValidationReport
from planguard.dsl.schema import MissionPlan, TaskSpec

CHECK_NAMES = (
    "unique_ids",
    "feature_refs",
    "required_fields",
    "dependencies",
    "fleet_capacity",
    "time_feasibility",
    "link_fallbacks",
    "link_reachability",
    "fallback_targets",
)


@dataclass(frozen=True)
class ScheduledTask:
    task_id: str
    start_s: float
    end_s: float
    aircraft_count: int


# --------------------------------------------------------------------------
# duration model
# --------------------------------------------------------------------------


def work_s(task: TaskSpec, cfg: WorldConfig) -> float:
    """Sensor work for a task, before dividing across aircraft."""
    feature = cfg.world_map.get(task.target)
    if feature is None:
        return 0.0
    speed = cfg.fleet.speed_mps
    if task.kind == "sweep":
        return feature.area_m2 / (speed * cfg.fleet.sensor_swath_m)
    if task.kind == "corridor_patrol":
        return feature.length_m / speed
    if task.kind == "station_keep":
        return task.dwell_s or 0.0
    return 0.0  # rally


def per_aircraft_work_s(task: TaskSpec, cfg: WorldConfig) -> float:
    """Sensor work for one aircraft's share of a task.

    Dwell does not shrink with more aircraft; coverage work does. Shared with
    CBBA's scoring so the allocator and the validator cannot drift apart on
    how long a task takes.
    """
    work = work_s(task, cfg)
    if task.kind in ("sweep", "corridor_patrol"):
        work /= task.aircraft_count
    return work


def duration_s(task: TaskSpec, cfg: WorldConfig) -> float:
    """Travel from base plus one aircraft's share of the sensor work."""
    feature = cfg.world_map.get(task.target)
    if feature is None:
        return 0.0
    travel_s = dist_m(cfg.base_m, feature.centroid_m) / cfg.fleet.speed_mps
    return travel_s + per_aircraft_work_s(task, cfg)


def estimate_schedule(plan: MissionPlan, cfg: WorldConfig) -> list[ScheduledTask]:
    """As-soon-as-possible schedule assuming *unlimited* aircraft.

    This makes the time check one-sided and therefore sound: if even an
    unlimited-parallelism schedule overruns the budget, the plan is genuinely
    infeasible. Fitting inside the budget here proves nothing, which is why
    Phase 1 simulation exists.

    Requires an acyclic plan with resolvable targets; callers must run the
    blocking checks first.
    """
    remaining = {t.id: set(t.depends_on) for t in plan.tasks}
    end_of: dict[str, float] = {}
    out: list[ScheduledTask] = []

    while remaining:
        ready = sorted(tid for tid, deps in remaining.items() if not deps - end_of.keys())
        if not ready:  # pragma: no cover - guarded by check_dependencies
            raise ValueError("unschedulable plan: run the dependency check first")
        for tid in ready:
            task = plan.task(tid)
            assert task is not None
            dep_end = max((end_of[d] for d in sorted(task.depends_on)), default=0.0)
            start = max(dep_end, task.start_after_s or 0.0)
            end = start + duration_s(task, cfg)
            end_of[tid] = end
            out.append(ScheduledTask(tid, start, end, task.aircraft_count))
            del remaining[tid]
    return sorted(out, key=lambda s: (s.start_s, s.task_id))


def peak_concurrency(schedule: list[ScheduledTask]) -> tuple[int, float]:
    """Maximum simultaneous aircraft demand, and when it occurs.

    Ends are processed before starts at the same instant, so a task handing off
    to its successor does not register a false peak.
    """
    events: list[tuple[float, int, int]] = []
    for s in schedule:
        events.append((s.start_s, 1, s.aircraft_count))
        events.append((s.end_s, 0, -s.aircraft_count))
    events.sort()
    current = peak = 0
    at_s = 0.0
    for t, _order, delta in events:
        current += delta
        if current > peak:
            peak, at_s = current, t
    return peak, at_s


# --------------------------------------------------------------------------
# the eight checks
# --------------------------------------------------------------------------


def check_unique_ids(plan: MissionPlan, cfg: WorldConfig) -> list[ValidationError]:
    seen: set[str] = set()
    errs: list[ValidationError] = []
    for i, task in enumerate(plan.tasks):
        if task.id in seen:
            errs.append(
                ValidationError(
                    code="DUPLICATE_TASK_ID",
                    path=f"tasks[{i}].id",
                    message=f"task id {task.id!r} is used more than once",
                    hint="give every task a unique id; dependencies reference tasks by id",
                )
            )
        seen.add(task.id)
    return errs


def check_feature_refs(plan: MissionPlan, cfg: WorldConfig) -> list[ValidationError]:
    errs: list[ValidationError] = []
    for i, task in enumerate(plan.tasks):
        feature = cfg.world_map.get(task.target)
        if feature is None:
            errs.append(
                ValidationError(
                    code="UNKNOWN_MAP_FEATURE",
                    path=f"tasks[{i}].target",
                    message=f"{task.target!r} is not a known map feature",
                    hint=f"use one of: {', '.join(cfg.world_map.names)}",
                )
            )
            continue
        want = REQUIRED_FEATURE_KIND[task.kind]
        if feature.kind != want:
            candidates = [f.name for f in cfg.world_map.features if f.kind == want]
            errs.append(
                ValidationError(
                    code="WRONG_FEATURE_KIND",
                    path=f"tasks[{i}].target",
                    message=(
                        f"task kind {task.kind!r} needs a {want!r} feature but "
                        f"{task.target!r} is a {feature.kind!r} feature"
                    ),
                    hint=f"{want} features available: {', '.join(sorted(candidates)) or 'none'}",
                )
            )
    for j, rule in enumerate(plan.fallbacks):
        if rule.rally_point is None:
            continue
        feature = cfg.world_map.get(rule.rally_point)
        if feature is None:
            errs.append(
                ValidationError(
                    code="UNKNOWN_MAP_FEATURE",
                    path=f"fallbacks[{j}].rally_point",
                    message=f"{rule.rally_point!r} is not a known map feature",
                    hint=f"use one of: {', '.join(cfg.world_map.names)}",
                )
            )
        elif feature.kind != "point":
            errs.append(
                ValidationError(
                    code="WRONG_FEATURE_KIND",
                    path=f"fallbacks[{j}].rally_point",
                    message=f"a rally point must be a point feature, {rule.rally_point!r} is "
                    f"a {feature.kind!r} feature",
                    hint="reference a point feature as the rally point",
                )
            )
    return errs


def check_required_fields(plan: MissionPlan, cfg: WorldConfig) -> list[ValidationError]:
    """Conditional requirements a JSON Schema grammar will not enforce."""
    errs: list[ValidationError] = []
    for i, task in enumerate(plan.tasks):
        if task.kind == "station_keep" and task.dwell_s is None:
            errs.append(
                ValidationError(
                    code="MISSING_REQUIRED_FIELD",
                    path=f"tasks[{i}].dwell_s",
                    message="station_keep requires dwell_s, the time to hold on station",
                    hint="set dwell_s to a positive number of seconds",
                )
            )
    for j, rule in enumerate(plan.fallbacks):
        if rule.action == "return_to_rally" and rule.rally_point is None:
            errs.append(
                ValidationError(
                    code="MISSING_RALLY_POINT",
                    path=f"fallbacks[{j}].rally_point",
                    message="the return_to_rally action requires a rally_point",
                    hint="name a point feature for the aircraft to return to",
                )
            )
    return errs


def check_dependencies(plan: MissionPlan, cfg: WorldConfig) -> list[ValidationError]:
    errs: list[ValidationError] = []
    known = set(plan.task_ids)
    for i, task in enumerate(plan.tasks):
        for dep in task.depends_on:
            if dep == task.id:
                errs.append(
                    ValidationError(
                        code="SELF_DEPENDENCY",
                        path=f"tasks[{i}].depends_on",
                        message=f"task {task.id!r} depends on itself",
                        hint="remove the self-reference",
                    )
                )
            elif dep not in known:
                errs.append(
                    ValidationError(
                        code="UNKNOWN_TASK_REF",
                        path=f"tasks[{i}].depends_on",
                        message=f"{dep!r} is not a task in this plan",
                        hint=f"tasks in this plan: {', '.join(sorted(known))}",
                    )
                )
    if any(e.code in ("SELF_DEPENDENCY", "UNKNOWN_TASK_REF") for e in errs):
        return errs

    # Kahn's algorithm; whatever is left when nothing is ready forms the cycle.
    remaining = {t.id: set(t.depends_on) for t in plan.tasks}
    done: set[str] = set()
    while True:
        ready = sorted(tid for tid, deps in remaining.items() if not deps - done)
        if not ready:
            break
        for tid in ready:
            done.add(tid)
            del remaining[tid]
    if remaining:
        cycle = sorted(remaining)
        errs.append(
            ValidationError(
                code="DEPENDENCY_CYCLE",
                path="tasks[].depends_on",
                message=f"depends_on forms a cycle among tasks: {', '.join(cycle)}",
                hint="dependencies must form a DAG; break the cycle",
            )
        )
    return errs


def check_fleet_capacity(plan: MissionPlan, cfg: WorldConfig) -> list[ValidationError]:
    """Only a *single* task demanding more aircraft than exist is infeasible.

    An earlier version also rejected plans whose peak concurrent demand exceeded
    the fleet, computed from the unlimited-parallelism schedule. That was a
    false positive, and it fired on the first five-task plan written for the
    allocator comparison: five independent tasks with a four-aircraft fleet and
    a 1500 s budget against a 439 s critical path -- obviously flyable, four
    aircraft start, the fifth task follows.

    The reason no such check can be sound: ``start_after_s`` is a *lower* bound
    ("cannot start before") and no task carries a deadline, so nothing ever
    forces two tasks to overlap. The executive can always sequence them.
    Concurrent demand therefore can never prove infeasibility.

    False positives are worse than misses here. The Phase 7 repair loop would
    be told to fix a correct plan, and the only available fix is to invent
    ``depends_on`` edges that serialise work for no reason. The real signal that
    was being approximated -- "this fleet cannot get through this much work in
    time" -- is now a sound lower bound in ``check_time_feasibility``.
    """
    fleet = cfg.fleet.aircraft_count
    return [
        ValidationError(
            code="FLEET_OVERSUBSCRIBED",
            path=f"tasks[{i}].aircraft_count",
            message=(
                f"task {task.id!r} asks for {task.aircraft_count} aircraft but the "
                f"fleet has {fleet}"
            ),
            hint=f"reduce aircraft_count to at most {fleet}",
        )
        for i, task in enumerate(plan.tasks)
        if task.aircraft_count > fleet
    ]


def makespan_lower_bound_s(plan: MissionPlan, cfg: WorldConfig) -> tuple[float, str]:
    """A sound lower bound on how long the plan must take, and which term binds.

    Two independent bounds, both valid whatever the executive does:

    * **critical path** -- the dependency chain, costed with unlimited aircraft.
    * **work conservation** -- total sensor work divided by the fleet size. No
      schedule can beat it, because the work has to be done by somebody. Travel
      is deliberately excluded: an aircraft flying two tasks in sequence does
      not pay travel-from-base twice, so including it would overestimate and
      reintroduce false positives.

    The larger of the two is still a lower bound, so the check stays one-sided:
    exceeding it proves infeasibility, fitting inside it proves nothing.
    """
    schedule = estimate_schedule(plan, cfg)
    critical_s = max((s.end_s for s in schedule), default=0.0)
    total_work_s = sum(work_s(task, cfg) for task in plan.tasks)
    fleet_s = total_work_s / cfg.fleet.aircraft_count
    return (
        (critical_s, "critical path")
        if critical_s >= fleet_s
        else (fleet_s, "total work spread across the fleet")
    )


def check_time_feasibility(plan: MissionPlan, cfg: WorldConfig) -> list[ValidationError]:
    errs: list[ValidationError] = []
    for i, task in enumerate(plan.tasks):
        if task.start_after_s is not None and task.start_after_s >= plan.time_budget_s:
            errs.append(
                ValidationError(
                    code="START_AFTER_BEYOND_BUDGET",
                    path=f"tasks[{i}].start_after_s",
                    message=(
                        f"task {task.id!r} cannot start before t={task.start_after_s:.0f}s, "
                        f"which is at or past the {plan.time_budget_s:.0f}s budget"
                    ),
                    hint="lower start_after_s or raise time_budget_s",
                )
            )
    bound_s, binding = makespan_lower_bound_s(plan, cfg)
    needed = bound_s * cfg.validator.time_estimate_slack
    if needed > plan.time_budget_s:
        errs.append(
            ValidationError(
                code="TIME_BUDGET_EXCEEDED",
                path="time_budget_s",
                message=(
                    f"the plan needs at least {needed:.0f}s ({binding}) but the budget is "
                    f"{plan.time_budget_s:.0f}s"
                ),
                hint="raise time_budget_s, drop a task, or add aircraft to the slow tasks",
            )
        )
    return errs


def check_link_fallbacks(plan: MissionPlan, cfg: WorldConfig) -> list[ValidationError]:
    if not cfg.validator.require_link_fallback:
        return []
    errs: list[ValidationError] = []
    link_rules = [r for r in plan.fallbacks if r.trigger == "link_lost"]
    for i, task in enumerate(plan.tasks):
        if not task.requires_link:
            continue
        if not any(r.covers(task.id) for r in link_rules):
            errs.append(
                ValidationError(
                    code="MISSING_LINK_FALLBACK",
                    path=f"tasks[{i}].requires_link",
                    message=(
                        f"task {task.id!r} requires a link but no link_lost fallback covers it"
                    ),
                    hint=(
                        "add a fallback with trigger 'link_lost' whose applies_to includes "
                        f"{task.id!r}, or leave applies_to empty to cover every task"
                    ),
                )
            )
    return errs


def check_link_reachability(plan: MissionPlan, cfg: WorldConfig) -> list[ValidationError]:
    """A task that requires a link must be inside link range for its *whole* extent.

    Found by running the simulator in Phase 1: the demo plan asked two aircraft
    to sweep an area wholly outside link range while declaring requires_link,
    so the link_lost fallback fired on arrival every single run and the sweep
    could never progress. That is statically detectable, so it should never have
    reached the simulator.

    The worst-case coordinate is used, not the centroid, because the aircraft
    has to cover the entire feature. This assumes the Phase 1 link model:
    aircraft-to-base only, no relay. If relaying is ever modelled, the DSL's
    max_two_hops constraint is where that would be expressed.
    """
    errs: list[ValidationError] = []
    for i, task in enumerate(plan.tasks):
        if not task.requires_link:
            continue
        feature = cfg.world_map.get(task.target)
        if feature is None:  # already reported by check_feature_refs
            continue
        farthest_m = max(dist_m(cfg.base_m, c) for c in feature.coords_m)
        if farthest_m > cfg.link.range_m:
            errs.append(
                ValidationError(
                    code="LINK_RANGE_UNREACHABLE",
                    path=f"tasks[{i}].requires_link",
                    message=(
                        f"task {task.id!r} requires a link but {task.target!r} reaches "
                        f"{farthest_m:.0f} m from base, beyond the {cfg.link.range_m:.0f} m "
                        f"link range"
                    ),
                    hint=(
                        "clear requires_link, or retarget the task to a feature within "
                        f"{cfg.link.range_m:.0f} m of base"
                    ),
                )
            )
    return errs


def check_fallback_targets(plan: MissionPlan, cfg: WorldConfig) -> list[ValidationError]:
    errs: list[ValidationError] = []
    known = set(plan.task_ids)
    for j, rule in enumerate(plan.fallbacks):
        for ref in rule.applies_to:
            if ref not in known:
                errs.append(
                    ValidationError(
                        code="UNKNOWN_FALLBACK_TARGET",
                        path=f"fallbacks[{j}].applies_to",
                        message=f"{ref!r} is not a task in this plan",
                        hint=f"tasks in this plan: {', '.join(sorted(known))}",
                    )
                )
    return errs


CHECKS = (
    check_unique_ids,
    check_feature_refs,
    check_required_fields,
    check_dependencies,
    check_fleet_capacity,
    check_time_feasibility,
    check_link_fallbacks,
    check_link_reachability,
    check_fallback_targets,
)
#: Checks that need a schedulable plan, and so are skipped after a blocking error.
#: ``fleet_capacity`` left this set when its concurrency half was removed -- it
#: no longer builds a schedule.
NEEDS_SCHEDULE = frozenset({"time_feasibility"})


def validate(plan: MissionPlan, cfg: WorldConfig) -> ValidationReport:
    errors: list[ValidationError] = []
    skipped: list[str] = []
    blocked = False
    for name, check in zip(CHECK_NAMES, CHECKS, strict=True):
        if blocked and name in NEEDS_SCHEDULE:
            skipped.append(name)
            continue
        found = check(plan, cfg)
        errors.extend(found)
        if any(e.code in BLOCKING_CODES for e in found):
            blocked = True
    return ValidationReport(errors=errors, skipped_checks=skipped)


def _main(argv: list[str]) -> int:
    if not argv:
        print("usage: python -m planguard.dsl.validate <plan.json|cases.yaml> [--world path]")
        return 2
    target = Path(argv[0])
    world_path = (
        Path(argv[argv.index("--world") + 1])
        if "--world" in argv
        else Path("experiments/configs/world.yaml")
    )
    cfg = WorldConfig.load(world_path)

    if target.suffix in (".yaml", ".yml"):
        from planguard.dsl.golden import run_corpus

        return run_corpus(target, cfg)

    plan = MissionPlan.model_validate(json.loads(target.read_text()))
    report = validate(plan, cfg)
    print(report.as_feedback())
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
