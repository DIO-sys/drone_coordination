"""Plan sampler (Phase 3).

The sampler feeds the risk dataset, so its contract is narrow and worth
pinning: every plan it returns validates, no plan appears twice, and the same
seed gives the same corpus.
"""

from __future__ import annotations

import numpy as np
import pytest

from planguard.config import WorldConfig
from planguard.dsl.sample import (
    KINDS_FOR_FEATURE,
    SamplerConfig,
    canonical_json,
    draw_plan,
    plan_id,
    sample_corpus,
)
from planguard.dsl.validate import makespan_lower_bound_s, validate

N = 60


@pytest.fixture(scope="module")
def scfg() -> SamplerConfig:
    return SamplerConfig()


@pytest.fixture(scope="module")
def corpus(cfg: WorldConfig, scfg: SamplerConfig) -> list:
    plans, _ = sample_corpus(cfg, scfg, N, seed=11)
    return plans


def test_kinds_for_feature_is_derived_not_retyped() -> None:
    # every task kind must be reachable, or the sampler would quietly never
    # produce one and the dataset would have a hole in it
    reachable = {k for kinds in KINDS_FOR_FEATURE.values() for k in kinds}
    assert reachable == {"sweep", "station_keep", "corridor_patrol", "rally"}


def test_corpus_is_the_requested_size(corpus: list) -> None:
    assert len(corpus) == N


def test_every_sampled_plan_validates(corpus: list, cfg: WorldConfig) -> None:
    """The point of the sampler. A risk label for an invalid plan is a label
    for something that could never be flown."""
    for plan in corpus:
        report = validate(plan, cfg)
        assert report.ok, f"{plan.name}:\n{report.as_feedback()}"


def test_no_duplicate_plans(corpus: list) -> None:
    """Rule 6: the same plan must not land in both train and test, which it
    cannot do if it only appears once."""
    keys = [canonical_json(p) for p in corpus]
    assert len(set(keys)) == len(keys)
    assert len({plan_id(p) for p in corpus}) == len(keys)


def test_plan_id_ignores_cosmetic_fields(cfg: WorldConfig, scfg: SamplerConfig) -> None:
    rng = np.random.default_rng(3)
    plan = draw_plan(cfg, scfg, rng, 0)
    assert plan is not None
    renamed = plan.model_copy(update={"name": "something else", "rationale": "because"})
    assert plan_id(renamed) == plan_id(plan)


def test_same_seed_same_corpus(cfg: WorldConfig, scfg: SamplerConfig) -> None:
    a, sa = sample_corpus(cfg, scfg, 20, seed=5)
    b, sb = sample_corpus(cfg, scfg, 20, seed=5)
    assert [canonical_json(p) for p in a] == [canonical_json(p) for p in b]
    assert sa.drawn == sb.drawn


def test_different_seed_different_corpus(cfg: WorldConfig, scfg: SamplerConfig) -> None:
    a, _ = sample_corpus(cfg, scfg, 20, seed=5)
    b, _ = sample_corpus(cfg, scfg, 20, seed=6)
    assert [canonical_json(p) for p in a] != [canonical_json(p) for p in b]


def test_budget_ratios_land_in_the_configured_band(corpus: list, cfg: WorldConfig) -> None:
    """Budgets are drawn relative to the validator's lower bound, not in
    absolute seconds, so the band means the same thing for a one-rally plan and
    a four-sweep plan."""
    scfg = SamplerConfig()
    for plan in corpus:
        bound_s, _ = makespan_lower_bound_s(plan, cfg)
        ratio = plan.time_budget_s / bound_s
        # rounding the budget to whole seconds can nudge a ratio a hair outside
        assert scfg.min_budget_ratio - 0.02 <= ratio <= scfg.max_budget_ratio + 0.02


def test_budgets_stay_inside_the_simulator_horizon(corpus: list, cfg: WorldConfig) -> None:
    """A budget past max_duration_s would be silently clamped by the simulator,
    so the label would not describe the plan that was sampled."""
    for plan in corpus:
        assert plan.time_budget_s <= cfg.sim.max_duration_s


def test_dependencies_are_acyclic_by_construction(corpus: list) -> None:
    for plan in corpus:
        order = {t.id: i for i, t in enumerate(plan.tasks)}
        for task in plan.tasks:
            for dep in task.depends_on:
                assert order[dep] < order[task.id]


def test_corpus_spans_the_interesting_shapes(corpus: list) -> None:
    """A sampler that only ever drew one-task, one-aircraft, no-link plans
    would validate cleanly and teach the risk model nothing."""
    assert {len(p.tasks) for p in corpus} > {1}
    assert any(t.aircraft_count > 1 for p in corpus for t in p.tasks)
    assert any(t.requires_link for p in corpus for t in p.tasks)
    assert any(t.depends_on for p in corpus for t in p.tasks)
    assert any(t.start_after_s is not None for p in corpus for t in p.tasks)
    assert {t.kind for p in corpus for t in p.tasks} == {
        "sweep",
        "station_keep",
        "corridor_patrol",
        "rally",
    }


def test_yield_is_accounted_for(cfg: WorldConfig, scfg: SamplerConfig) -> None:
    _, stats = sample_corpus(cfg, scfg, 30, seed=2)
    assert stats.drawn == stats.accepted + stats.rejected_invalid + stats.rejected_duplicate + (
        stats.rejected_over_max_duration
    )
    assert 0.0 < stats.yield_rate <= 1.0
