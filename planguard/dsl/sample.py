"""Random plan sampler — the source of the Phase 3 risk dataset.

Two requirements shape every decision here.

**The plans must be valid.** A risk label for a plan the validator would reject
is a label for something that could never be flown. The sampler therefore aims
at validity by construction (correct feature kinds, acyclic dependencies, a
``link_lost`` fallback whenever a task needs a link) and then submits every
candidate to ``validate`` anyway. The validator is the arbiter, not the
sampler, and the fraction surviving is reported rather than assumed.

**The budgets must sit near the feasibility boundary.** This is the one thing
that decides whether Phase 4 has anything to learn. With a loose budget every
run succeeds unless the fleet fails to launch, so the only feature that
predicts anything is aircraft count and the risk model is a fleet-size lookup
table. Budgets are therefore drawn as a multiple of ``makespan_lower_bound_s``
-- the validator's own sound lower bound -- which normalises for plan size:
a multiplier of 1.3 means "30% above the arithmetic minimum" whether the plan
is one rally or four sweeps.

Per rule 5 the generator is passed in explicitly; nothing here touches global
RNG state.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from planguard.config import REQUIRED_FEATURE_KIND, WorldConfig
from planguard.dsl.schema import FallbackRule, MissionPlan, TaskSpec
from planguard.dsl.validate import makespan_lower_bound_s, validate

#: Which task kinds can target each feature kind. The inverse of
#: REQUIRED_FEATURE_KIND, derived rather than retyped so the two cannot drift.
KINDS_FOR_FEATURE: dict[str, tuple[str, ...]] = {}
for _kind, _feature_kind in sorted(REQUIRED_FEATURE_KIND.items()):
    KINDS_FOR_FEATURE.setdefault(_feature_kind, ())
    KINDS_FOR_FEATURE[_feature_kind] += (_kind,)


class SamplerConfig(BaseModel):
    """Per rule 10, YAML into Pydantic; derived values computed here."""

    model_config = ConfigDict(frozen=True)

    #: Number of tasks per plan, inclusive.
    min_tasks: int = Field(default=1, ge=1)
    max_tasks: int = Field(default=4, ge=1)
    #: Time budget as a multiple of the validator's makespan lower bound.
    #: The lower end must stay above 1.0 or every draw is rejected by the
    #: time-feasibility check, which is pure waste rather than a hard case.
    #: The band is calibrated, not guessed: at 1.05-2.2 the mean failure rate
    #: was flat at ~0.1 across the whole range and only 38% of plans had a
    #: label strictly between 0 and 1. At 1.01-1.80 it falls 0.40 -> 0.20
    #: monotonically and 55% of plans do. See ARCHITECTURE section 10.
    min_budget_ratio: float = Field(default=1.01, gt=1.0)
    max_budget_ratio: float = Field(default=1.80, gt=1.0)
    #: Chance a task asks for more than one aircraft, and the cap if it does.
    p_multi_aircraft: float = Field(default=0.35, ge=0, le=1)
    max_aircraft_per_task: int = Field(default=3, ge=1)
    #: Chance a task requires a link.
    p_requires_link: float = Field(default=0.4, ge=0, le=1)
    #: Chance an eligible (later) task depends on an earlier one.
    p_depends: float = Field(default=0.25, ge=0, le=1)
    #: Chance a task carries a start_after_s, drawn as a fraction of the bound.
    p_start_after: float = Field(default=0.15, ge=0, le=1)
    #: Dwell range for station_keep, seconds.
    min_dwell_s: float = Field(default=120.0, gt=0)
    max_dwell_s: float = Field(default=900.0, gt=0)
    #: Chance of a low_fuel fallback on top of the required link_lost one.
    p_low_fuel_fallback: float = Field(default=0.5, ge=0, le=1)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.max_tasks < self.min_tasks:
            raise ValueError("max_tasks must be >= min_tasks")
        if self.max_budget_ratio < self.min_budget_ratio:
            raise ValueError("max_budget_ratio must be >= min_budget_ratio")
        if self.max_dwell_s < self.min_dwell_s:
            raise ValueError("max_dwell_s must be >= min_dwell_s")
        return self

    @classmethod
    def load(cls, path: str | Path) -> SamplerConfig:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()) or {})


def canonical_json(plan: MissionPlan) -> str:
    """The plan's content, with the name stripped.

    The name is cosmetic -- two plans differing only by name are the same plan,
    and rule 6 says the same plan must not land in both train and test.
    """
    data = plan.model_dump(mode="json", exclude_none=True)
    data.pop("name", None)
    data.pop("rationale", None)
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def plan_id(plan: MissionPlan) -> str:
    """Content-addressed id.

    Deliberately a hash of the content and not a counter: the id, and therefore
    the train/test split derived from it, stays stable if the sampler is
    re-tuned and the corpus regenerated, so a model trained on one corpus can
    be scored on another without silent leakage.
    """
    return hashlib.sha256(canonical_json(plan).encode()).hexdigest()[:16]


@dataclass
class SampleStats:
    """Yield accounting. Reported, not assumed."""

    drawn: int = 0
    accepted: int = 0
    rejected_invalid: int = 0
    rejected_duplicate: int = 0
    rejected_over_max_duration: int = 0
    rejection_codes: dict[str, int] = field(default_factory=dict)

    @property
    def yield_rate(self) -> float:
        return self.accepted / self.drawn if self.drawn else 0.0


def _draw_tasks(
    cfg: WorldConfig, scfg: SamplerConfig, rng: np.random.Generator
) -> list[TaskSpec] | None:
    """Draw task bodies. Dependencies and budget are added by the caller."""
    eligible = sorted(
        f.name
        for f in cfg.world_map.features
        if f.kind in KINDS_FOR_FEATURE and f.name != cfg.fleet.base_feature
    )
    if not eligible:
        return None
    n = int(rng.integers(scfg.min_tasks, scfg.max_tasks + 1))
    n = min(n, len(eligible))
    # sampled without replacement: two tasks on the same feature would be a
    # duplicate-work plan the executive has no way to express
    targets = [eligible[i] for i in sorted(rng.choice(len(eligible), size=n, replace=False))]

    tasks: list[TaskSpec] = []
    for i, target in enumerate(targets):
        feature = cfg.world_map.get(target)
        assert feature is not None
        choices = KINDS_FOR_FEATURE[feature.kind]
        kind = choices[int(rng.integers(len(choices)))]

        count = 1
        if kind in ("sweep", "corridor_patrol") and rng.random() < scfg.p_multi_aircraft:
            cap = min(scfg.max_aircraft_per_task, cfg.fleet.aircraft_count)
            count = int(rng.integers(2, cap + 1)) if cap >= 2 else 1

        dwell = None
        if kind == "station_keep":
            dwell = float(round(rng.uniform(scfg.min_dwell_s, scfg.max_dwell_s)))

        tasks.append(
            TaskSpec(
                id=f"t{i}_{kind}",
                kind=kind,  # type: ignore[arg-type]
                target=target,
                aircraft_count=count,
                priority=int(rng.integers(1, 4)),
                dwell_s=dwell,
                requires_link=bool(rng.random() < scfg.p_requires_link),
            )
        )
    return tasks


def _add_dependencies(
    tasks: list[TaskSpec], scfg: SamplerConfig, rng: np.random.Generator
) -> list[TaskSpec]:
    """Only backwards edges, so the graph is acyclic by construction.

    The validator's cycle check still runs -- but a sampler that produced
    cycles would spend its time generating plans the validator rejects for a
    reason that has nothing to do with execution risk.
    """
    out: list[TaskSpec] = []
    for i, task in enumerate(tasks):
        deps = [tasks[j].id for j in range(i) if rng.random() < scfg.p_depends]
        out.append(task.model_copy(update={"depends_on": deps}))
    return out


def _fallbacks(
    tasks: list[TaskSpec], cfg: WorldConfig, scfg: SamplerConfig, rng: np.random.Generator
) -> list[FallbackRule]:
    rules: list[FallbackRule] = []
    rally_points = sorted(
        f.name
        for f in cfg.world_map.features
        if f.kind == "point" and f.name != cfg.fleet.base_feature
    )
    rally = rally_points[int(rng.integers(len(rally_points)))] if rally_points else None

    if any(t.requires_link for t in tasks):
        # the link_fallbacks check requires one; which action is a free choice
        if rng.random() < 0.5 or rally is None:
            rules.append(FallbackRule(trigger="link_lost", action="continue_unlinked"))
        else:
            rules.append(
                FallbackRule(trigger="link_lost", action="return_to_rally", rally_point=rally)
            )
    if rng.random() < scfg.p_low_fuel_fallback and rally is not None:
        rules.append(FallbackRule(trigger="low_fuel", action="return_to_rally", rally_point=rally))
    return rules


def draw_plan(
    cfg: WorldConfig, scfg: SamplerConfig, rng: np.random.Generator, index: int
) -> MissionPlan | None:
    """One candidate plan. Validity is the caller's problem."""
    tasks = _draw_tasks(cfg, scfg, rng)
    if tasks is None:
        return None
    tasks = _add_dependencies(tasks, scfg, rng)
    fallbacks = _fallbacks(tasks, cfg, scfg, rng)

    # A provisional budget large enough that the lower bound can be computed
    # without the time check interfering, then the real budget from that bound.
    draft = MissionPlan(
        name=f"sampled_{index}",
        time_budget_s=cfg.sim.max_duration_s * 10,
        tasks=tasks,
        fallbacks=fallbacks,
    )
    bound_s, _binding = makespan_lower_bound_s(draft, cfg)
    if bound_s <= 0.0:
        return None

    # start_after_s is drawn before the budget so it is inside the bound the
    # budget is scaled from; drawn after, it could push the critical path past
    # a budget already chosen and reject the plan for the wrong reason.
    if rng.random() < scfg.p_start_after:
        i = int(rng.integers(len(tasks)))
        delay = float(round(rng.uniform(0.0, 0.5 * bound_s)))
        tasks = list(tasks)
        tasks[i] = tasks[i].model_copy(update={"start_after_s": delay})
        draft = draft.model_copy(update={"tasks": tasks})
        bound_s, _binding = makespan_lower_bound_s(draft, cfg)

    ratio = float(rng.uniform(scfg.min_budget_ratio, scfg.max_budget_ratio))
    budget_s = float(round(bound_s * ratio))
    return draft.model_copy(update={"time_budget_s": budget_s})


def sample_corpus(
    cfg: WorldConfig,
    scfg: SamplerConfig,
    n_plans: int,
    seed: int,
    max_attempts_per_plan: int = 40,
) -> tuple[list[MissionPlan], SampleStats]:
    """``n_plans`` distinct validator-passing plans, plus the yield accounting."""
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    stats = SampleStats()
    seen: set[str] = set()
    out: list[MissionPlan] = []

    budget = n_plans * max_attempts_per_plan
    while len(out) < n_plans and stats.drawn < budget:
        stats.drawn += 1
        plan = draw_plan(cfg, scfg, rng, len(out))
        if plan is None:
            stats.rejected_invalid += 1
            continue
        if plan.time_budget_s > cfg.sim.max_duration_s:
            # the simulator would silently clamp the run, so the label would
            # not be a label for this plan's budget
            stats.rejected_over_max_duration += 1
            continue
        report = validate(plan, cfg)
        if not report.ok:
            stats.rejected_invalid += 1
            for err in report.errors:
                stats.rejection_codes[err.code] = stats.rejection_codes.get(err.code, 0) + 1
            continue
        key = canonical_json(plan)
        if key in seen:
            stats.rejected_duplicate += 1
            continue
        seen.add(key)
        stats.accepted += 1
        out.append(plan.model_copy(update={"name": f"sampled_{len(out)}"}))
    return out, stats
