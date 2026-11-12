"""Plan-time features (Phase 4).

Rule 2 is the contract: every feature must be computable before the plan runs.
A test suite cannot prove the absence of leakage, but it can pin the three
things that would let leakage in unnoticed -- a feature that needs a run, a
column order that drifts, and a feature that silently depends on the seed.
"""

from __future__ import annotations

import pytest

from planguard.config import WorldConfig
from planguard.dsl.sample import SamplerConfig, sample_corpus
from planguard.dsl.schema import MissionPlan, TaskSpec
from planguard.risk import features as feat
from planguard.risk.features import (
    FEATURE_NAMES,
    baseline_score,
    feature_vector,
    features_for,
)


@pytest.fixture(scope="module")
def corpus(cfg: WorldConfig) -> list[MissionPlan]:
    plans, _ = sample_corpus(cfg, SamplerConfig(), 40, seed=23)
    return plans


def test_feature_names_match_the_extractor(cfg: WorldConfig, corpus: list[MissionPlan]) -> None:
    """FEATURE_NAMES is spelled out, so something has to hold it to account.

    A reordered or renamed feature does not raise -- it mistrains the model and
    makes SHAP attribute the wrong name to the wrong effect.
    """
    for plan in corpus:
        assert set(features_for(plan, cfg)) == set(FEATURE_NAMES)
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
    assert list(FEATURE_NAMES) == sorted(FEATURE_NAMES)


def test_feature_vector_follows_the_declared_order(
    cfg: WorldConfig, corpus: list[MissionPlan]
) -> None:
    for plan in corpus[:5]:
        row = features_for(plan, cfg)
        assert feature_vector(plan, cfg) == [row[n] for n in FEATURE_NAMES]


def test_every_feature_is_finite(cfg: WorldConfig, corpus: list[MissionPlan]) -> None:
    """XGBoost tolerates NaN; a silent inf from a division does more damage."""
    for plan in corpus:
        for name, value in features_for(plan, cfg).items():
            assert value == value, f"{name} is NaN"
            assert abs(value) != float("inf"), f"{name} is infinite"


def test_features_do_not_touch_the_simulator() -> None:
    """Rule 2, enforced structurally rather than by reading the code.

    If this module has no reference to the simulator or to ``runs.parquet``,
    it cannot accidentally featurise an outcome. Checked against the module's
    resolved imports rather than its source text, so prose about what the
    module must *not* do cannot fail its own test.
    """
    for name, value in vars(feat).items():
        if name.startswith("__"):
            continue
        module = getattr(value, "__module__", None) or getattr(value, "__name__", "")
        assert "planguard.sim" not in str(module), f"{name} comes from the simulator"
        assert not str(module).startswith("planguard.risk.harness"), (
            f"{name} comes from the harness, which holds realised outcomes"
        )


def test_features_depend_only_on_the_plan_and_the_config(
    cfg: WorldConfig, corpus: list[MissionPlan]
) -> None:
    """Called twice, identical. Nothing here may draw from an RNG."""
    for plan in corpus[:10]:
        assert features_for(plan, cfg) == features_for(plan, cfg)


def test_renaming_a_plan_does_not_change_its_features(
    cfg: WorldConfig, corpus: list[MissionPlan]
) -> None:
    plan = corpus[0]
    renamed = plan.model_copy(update={"name": "zzz", "rationale": "because"})
    assert features_for(renamed, cfg) == features_for(plan, cfg)


def test_budget_ratio_is_the_baseline_heuristic(
    cfg: WorldConfig, corpus: list[MissionPlan]
) -> None:
    """The Phase 4 acceptance check is "beats path-length-over-budget", so the
    baseline has to be computed by the same code the model is fed."""
    for plan in corpus[:10]:
        row = features_for(plan, cfg)
        assert baseline_score(row) == pytest.approx(1.0 / row["budget_ratio"])
        # a tighter budget must score riskier
        looser = plan.model_copy(update={"time_budget_s": plan.time_budget_s * 2})
        assert baseline_score(features_for(looser, cfg)) < baseline_score(row)


def test_dependency_depth_counts_edges_not_nodes(cfg: WorldConfig) -> None:
    chain = MissionPlan(
        name="chain",
        time_budget_s=5000.0,
        tasks=[
            TaskSpec(id="a", kind="rally", target="rally_alpha"),
            TaskSpec(id="b", kind="rally", target="rally_bravo", depends_on=["a"]),
            TaskSpec(id="c", kind="station_keep", target="bridge", dwell_s=60.0, depends_on=["b"]),
        ],
    )
    assert features_for(chain, cfg)["dependency_depth"] == 2.0
    assert features_for(chain, cfg)["n_dependency_edges"] == 2.0

    flat = chain.model_copy(
        update={"tasks": [t.model_copy(update={"depends_on": []}) for t in chain.tasks]}
    )
    assert features_for(flat, cfg)["dependency_depth"] == 0.0


def test_fleet_demand_features_catch_the_dominant_risk_term(cfg: WorldConfig) -> None:
    """A quarter of Phase 3's failures are plans asking for more aircraft than
    launched, so the feature set has to express that at all."""
    greedy = MissionPlan(
        name="greedy",
        time_budget_s=5000.0,
        tasks=[TaskSpec(id="s", kind="sweep", target="north_ridge", aircraft_count=3)],
    )
    modest = greedy.model_copy(
        update={"tasks": [greedy.tasks[0].model_copy(update={"aircraft_count": 1})]}
    )
    gf, mf = features_for(greedy, cfg), features_for(modest, cfg)
    assert gf["max_aircraft_count"] == 3.0
    assert gf["aircraft_demand_ratio"] == pytest.approx(3 / cfg.fleet.aircraft_count)
    assert mf["aircraft_demand_ratio"] < gf["aircraft_demand_ratio"]


def test_link_margin_is_a_margin_not_a_breach(cfg: WorldConfig) -> None:
    """The validator already rejects tasks outside link range, so what reaches
    the model is the margin -- which is where the Markov link actually bites."""
    near = MissionPlan(
        name="near",
        time_budget_s=5000.0,
        tasks=[
            TaskSpec(
                id="w", kind="station_keep", target="rally_alpha", dwell_s=60.0, requires_link=True
            )
        ],
    )
    far = near.model_copy(update={"tasks": [near.tasks[0].model_copy(update={"target": "bridge"})]})
    assert features_for(near, cfg)["link_margin_m"] > features_for(far, cfg)["link_margin_m"]
    assert features_for(near, cfg)["link_margin_m"] > 0


def test_no_link_tasks_means_full_margin(cfg: WorldConfig) -> None:
    plan = MissionPlan(
        name="no link",
        time_budget_s=5000.0,
        tasks=[TaskSpec(id="s", kind="sweep", target="south_valley")],
    )
    row = features_for(plan, cfg)
    assert row["n_requires_link"] == 0.0
    assert row["frac_requires_link"] == 0.0
    assert row["link_margin_m"] == cfg.link.range_m


def test_task_kind_counts_cover_every_kind(cfg: WorldConfig, corpus: list[MissionPlan]) -> None:
    for plan in corpus:
        row = features_for(plan, cfg)
        counted = sum(row[f"n_{k}"] for k in ("sweep", "station_keep", "corridor_patrol", "rally"))
        assert counted == row["n_tasks"]


def test_the_corpus_actually_varies_in_every_feature(
    cfg: WorldConfig, corpus: list[MissionPlan]
) -> None:
    """A feature that is constant across the corpus is dead weight, and SHAP
    will report it as such -- better to know now.

    This is how ``binding_is_critical_path`` was caught and removed: the
    work-conservation term in the makespan bound never binds for plans with at
    most four tasks and a fleet of four.
    """
    rows = [features_for(p, cfg) for p in corpus]
    constant = [n for n in FEATURE_NAMES if len({r[n] for r in rows}) == 1]
    assert not constant, f"constant across 40 sampled plans: {constant}"
