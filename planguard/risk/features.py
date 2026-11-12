"""Plan-time features for the risk surrogate.

**Rule 2 is the whole contract of this module.** Every feature must be
computable from the plan and the world config *before the plan runs*. No
realised outage timings, no wind that was actually drawn, no aircraft that
actually launched, no simulation outcome of any kind. The surrogate exists to
screen candidate plans inside the Phase 7 agent loop, and a feature the agent
cannot have at that moment is not a feature, it is leakage -- and leakage would
show up as a flattering AUC and a model that is useless in the loop it was
built for.

Two consequences worth stating:

* Features come from ``plan_json`` in ``plans.parquet``, never from
  ``runs.parquet``. That is why Phase 3 stores the plan next to the label.
* Anything constant across the corpus (the fleet size, the wind *mean*, the
  link parameters) carries no information and is left out. They are part of
  the world, not of the plan being scored.

One feature was built and then removed: ``binding_is_critical_path``, which
said whether the makespan bound came from the dependency chain or from total
work divided by the fleet. Across the sampled corpus it is **always** the
critical path -- with at most four tasks and four aircraft the work term never
binds -- so it carried no information and was dropped rather than left for SHAP
to report as zero. It should come back if the sampler ever draws plans with
many more tasks than aircraft. ``test_risk_features.py`` fails on any feature
that is constant across the corpus, which is how this was found.

The headline Phase 4 comparison is against a "path length over budget"
heuristic, which is one of these features on its own: ``budget_ratio``. Beating
it has to mean beating it, so it is computed here by the same code the model
sees.
"""

from __future__ import annotations

from typing import get_args

from planguard.config import WorldConfig, dist_m
from planguard.dsl.schema import MissionPlan, TaskKind
from planguard.dsl.validate import (
    estimate_schedule,
    makespan_lower_bound_s,
    per_aircraft_work_s,
    work_s,
)

TASK_KINDS: tuple[TaskKind, ...] = tuple(sorted(get_args(TaskKind)))


def _dependency_depth(plan: MissionPlan) -> int:
    """Longest chain in the dependency graph, in edges.

    Acyclicity is the validator's job; this assumes a validated plan.
    """
    depth: dict[str, int] = {}
    for task in sorted(plan.tasks, key=lambda t: t.id):
        depth[task.id] = 0
    changed = True
    while changed:  # at most |V| passes on a DAG
        changed = False
        for task in sorted(plan.tasks, key=lambda t: t.id):
            for dep in sorted(task.depends_on):
                if dep in depth and depth[dep] + 1 > depth[task.id]:
                    depth[task.id] = depth[dep] + 1
                    changed = True
    return max(depth.values(), default=0)


def features_for(plan: MissionPlan, cfg: WorldConfig) -> dict[str, float]:
    """One plan's feature row. Keys are stable; see ``FEATURE_NAMES``."""
    fleet = cfg.fleet
    budget_s = plan.time_budget_s
    bound_s, _binding = makespan_lower_bound_s(plan, cfg)
    schedule = estimate_schedule(plan, cfg)
    critical_s = max((s.end_s for s in schedule), default=0.0)
    total_work = sum(work_s(t, cfg) for t in plan.tasks)

    dists = []
    for task in plan.tasks:
        feature = cfg.world_map.get(task.target)
        if feature is not None:
            dists.append(feature.nearest_dist_m(cfg.base_m))
    max_dist_m = max(dists, default=0.0)

    link_tasks = [t for t in plan.tasks if t.requires_link]
    link_dists = []
    for task in link_tasks:
        feature = cfg.world_map.get(task.target)
        if feature is not None:
            link_dists.append(dist_m(cfg.base_m, feature.centroid_m))

    triggers = {fb.trigger for fb in plan.fallbacks}
    covered = sum(1 for t in plan.tasks if any(fb.covers(t.id) for fb in plan.fallbacks))

    row: dict[str, float] = {
        # --- shape of the plan -------------------------------------------
        "n_tasks": float(len(plan.tasks)),
        "n_dependency_edges": float(sum(len(t.depends_on) for t in plan.tasks)),
        "dependency_depth": float(_dependency_depth(plan)),
        "n_fallbacks": float(len(plan.fallbacks)),
        "fallback_coverage": covered / len(plan.tasks) if plan.tasks else 0.0,
        "has_link_lost_fallback": float("link_lost" in triggers),
        "has_low_fuel_fallback": float("low_fuel" in triggers),
        # --- fleet demand ------------------------------------------------
        # the dominant term in the Phase 3 corpus: a quarter of all failures
        # are plans asking for more aircraft than happened to launch
        "max_aircraft_count": float(max((t.aircraft_count for t in plan.tasks), default=0)),
        "total_aircraft_demand": float(sum(t.aircraft_count for t in plan.tasks)),
        "aircraft_demand_ratio": (
            max((t.aircraft_count for t in plan.tasks), default=0) / fleet.aircraft_count
        ),
        # --- time --------------------------------------------------------
        "time_budget_s": budget_s,
        "makespan_lower_bound_s": bound_s,
        # this one *is* the baseline heuristic, on its own
        "budget_ratio": budget_s / bound_s if bound_s > 0 else 0.0,
        "budget_slack_s": budget_s - bound_s,
        "critical_path_s": critical_s,
        "fleet_work_s": total_work / fleet.aircraft_count,
        "total_work_s": total_work,
        "max_task_work_s": max((per_aircraft_work_s(t, cfg) for t in plan.tasks), default=0.0),
        "max_start_after_s": max((t.start_after_s or 0.0 for t in plan.tasks), default=0.0),
        # --- geometry ----------------------------------------------------
        "max_dist_from_base_m": max_dist_m,
        "mean_dist_from_base_m": sum(dists) / len(dists) if dists else 0.0,
        "total_travel_s": sum(dists) / fleet.speed_mps,
        # --- link exposure -----------------------------------------------
        "n_requires_link": float(len(link_tasks)),
        "frac_requires_link": len(link_tasks) / len(plan.tasks) if plan.tasks else 0.0,
        # how close the most distant link-dependent task sits to the range
        # limit; the validator rejects outright breaches, so what is left is
        # the margin, which is where the Markov link actually bites
        "link_margin_m": (cfg.link.range_m - max(link_dists) if link_dists else cfg.link.range_m),
        # --- priority ----------------------------------------------------
        "min_priority": float(min((t.priority for t in plan.tasks), default=0)),
        "mean_priority": (
            sum(t.priority for t in plan.tasks) / len(plan.tasks) if plan.tasks else 0.0
        ),
    }
    for kind in TASK_KINDS:
        row[f"n_{kind}"] = float(sum(1 for t in plan.tasks if t.kind == kind))
    return row


#: Stable column order for the feature matrix. Spelled out rather than taken
#: from a dict's iteration order, because a silently reordered matrix mistrains
#: a model without raising anything. ``test_risk_features.py`` asserts this
#: tuple and ``features_for`` agree exactly, so it cannot drift unnoticed.
FEATURE_NAMES: tuple[str, ...] = (
    "aircraft_demand_ratio",
    "budget_ratio",
    "budget_slack_s",
    "critical_path_s",
    "dependency_depth",
    "fallback_coverage",
    "fleet_work_s",
    "frac_requires_link",
    "has_link_lost_fallback",
    "has_low_fuel_fallback",
    "link_margin_m",
    "makespan_lower_bound_s",
    "max_aircraft_count",
    "max_dist_from_base_m",
    "max_start_after_s",
    "max_task_work_s",
    "mean_dist_from_base_m",
    "mean_priority",
    "min_priority",
    "n_corridor_patrol",
    "n_dependency_edges",
    "n_fallbacks",
    "n_rally",
    "n_requires_link",
    "n_station_keep",
    "n_sweep",
    "n_tasks",
    "time_budget_s",
    "total_aircraft_demand",
    "total_travel_s",
    "total_work_s",
)


def feature_vector(plan: MissionPlan, cfg: WorldConfig) -> list[float]:
    """One plan's features in ``FEATURE_NAMES`` order."""
    row = features_for(plan, cfg)
    return [row[name] for name in FEATURE_NAMES]


def baseline_score(row: dict[str, float]) -> float:
    """The heuristic Phase 4 must beat: path length over budget.

    Higher means riskier, so it is the reciprocal of the budget ratio. A plan
    with exactly its arithmetic minimum scores 1.0; twice the budget it needs
    scores 0.5.
    """
    ratio = row.get("budget_ratio", 0.0)
    return 1.0 / ratio if ratio > 0 else 1.0
