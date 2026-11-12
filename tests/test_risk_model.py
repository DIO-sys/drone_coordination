"""Phase 4: the risk surrogate.

Most of these run on synthetic data so the suite works on a fresh clone --
``data/risk`` is 40,000 simulated missions and is not in the repository. The
tests that need the real dataset skip when it is absent, and say so.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from planguard.config import WorldConfig
from planguard.risk.explain import explain_plan, global_importance, shap_values
from planguard.risk.model import (
    BASELINE_FEATURE,
    CalibrationBin,
    ScreeningCost,
    calibration_curve,
    expected_calibration_error,
    fit,
    fit_baseline,
    isotonic_recalibrate,
    load_dataset,
    measure_screening_cost,
    predict_clipped,
    score,
)

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "risk"
needs_data = pytest.mark.skipif(
    not (DATA / "plans.parquet").exists(),
    reason="run `python -m planguard.risk.build` first; data/risk is not in the repo",
)


@pytest.fixture
def synthetic() -> tuple[np.ndarray, np.ndarray]:
    """A learnable signal in feature 0, noise elsewhere."""
    rng = np.random.default_rng(0)
    x = rng.random((400, 6))
    y = np.clip(0.9 - 0.8 * x[:, 0] + rng.normal(0, 0.05, 400), 0.0, 1.0)
    return x, y


# ------------------------------------------------------------ calibration ---


def test_calibration_curve_bins_are_disjoint_and_cover_the_predictions() -> None:
    y_true = np.linspace(0, 1, 100)
    y_pred = np.linspace(0, 1, 100)
    bins = calibration_curve(y_true, y_pred, bins=10)
    assert sum(b.n for b in bins) == 100, "a prediction fell outside every bin"


def test_calibration_curve_includes_the_endpoint() -> None:
    """A prediction of exactly 1.0 must land in the last bin, not vanish."""
    bins = calibration_curve(np.array([1.0]), np.array([1.0]), bins=10)
    assert sum(b.n for b in bins) == 1


def test_perfect_calibration_scores_zero_error() -> None:
    y = np.repeat(np.linspace(0.05, 0.95, 10), 20)
    bins = calibration_curve(y, y, bins=10)
    assert expected_calibration_error(bins) == pytest.approx(0.0, abs=1e-9)


def test_expected_calibration_error_is_weighted_by_bin_size() -> None:
    """A large well-calibrated bin must outweigh a tiny badly-calibrated one,
    or ECE would be dominated by whichever bin happened to be nearly empty."""
    big = CalibrationBin(0.0, 0.1, n=990, mean_predicted=0.05, mean_observed=0.05)
    small = CalibrationBin(0.9, 1.0, n=10, mean_predicted=0.95, mean_observed=0.15)
    ece = expected_calibration_error([big, small])
    assert ece == pytest.approx(10 * 0.8 / 1000)
    assert ece < 0.05


def test_systematic_overconfidence_is_detected() -> None:
    y_true = np.full(200, 0.2)
    y_pred = np.full(200, 0.8)
    bins = calibration_curve(y_true, y_pred, bins=10)
    assert expected_calibration_error(bins) == pytest.approx(0.6)


# ----------------------------------------------------------------- model ---


def test_model_learns_a_learnable_signal(synthetic: tuple[np.ndarray, np.ndarray]) -> None:
    x, y = synthetic
    model = fit(x[:300], y[:300])
    pred = predict_clipped(model, x[300:])
    assert np.abs(pred - y[300:]).mean() < 0.1


def test_predictions_are_clipped_to_probabilities(
    synthetic: tuple[np.ndarray, np.ndarray],
) -> None:
    """Squared-error regression on a rate can leave [0, 1], and a risk of
    -0.04 would break both Brier and the calibration bins."""
    x, y = synthetic
    pred = predict_clipped(fit(x, y), x)
    assert pred.min() >= 0.0
    assert pred.max() <= 1.0


def test_score_handles_a_single_class_without_raising() -> None:
    """AUC is undefined when every plan has the same binarised label; the
    report must degrade to NaN rather than crash a training run."""
    y_true = np.zeros(50)
    s = score(y_true, np.full(50, 0.1))
    assert s.auc != s.auc  # NaN
    assert s.mae == pytest.approx(0.1)


def test_isotonic_recalibration_preserves_ranking(
    synthetic: tuple[np.ndarray, np.ndarray],
) -> None:
    """Calibration and discrimination are separate properties. A monotone map
    can only change magnitudes, so AUC must be untouched -- if this ever fails
    the recalibration is doing something it should not."""
    from planguard.risk.model import Dataset

    x, y = synthetic
    data = Dataset(
        x_train=x[:300],
        y_train=y[:300],
        x_test=x[300:],
        y_test=y[300:],
        feature_names=tuple(f"f{i}" for i in range(6)),
    )
    model = fit(data.x_train, data.y_train)
    before = score(data.y_test, predict_clipped(model, data.x_test))
    _, y_iso = isotonic_recalibrate(model, data)
    after = score(data.y_test, y_iso)
    if before.auc == before.auc:  # not NaN
        assert after.auc == pytest.approx(before.auc, abs=0.01)


def test_screening_cost_uses_the_realistic_per_plan_number() -> None:
    """The batched figure is real and useless: the Phase 7 graph screens one
    plan at a time and must featurise it first. The headline speedup has to be
    the end-to-end one or it overstates by two orders of magnitude."""
    cost = ScreeningCost(
        n=100,
        batch_predict_s=0.0001,
        single_predict_s=0.00009,
        featurise_s=0.00006,
        simulate_s_per_run=1.0,
        seeds_per_plan=20,
    )
    assert cost.per_plan_s == pytest.approx(0.00015)
    assert cost.speedup == pytest.approx(20 / 0.00015, rel=1e-6)
    assert cost.batched_speedup > cost.speedup * 100


def test_measure_screening_cost_counts_featurisation(
    synthetic: tuple[np.ndarray, np.ndarray],
) -> None:
    x, y = synthetic
    model = fit(x, y)
    calls = {"n": 0}

    def featurise() -> int:
        calls["n"] += 1
        return sum(range(500))

    cost = measure_screening_cost(model, x, 1.0, 20, featurise=featurise, repeats=2)
    assert calls["n"] > 0
    assert cost.featurise_s > 0
    assert cost.per_plan_s > cost.single_predict_s


# ------------------------------------------------------ the real dataset ---


@needs_data
def test_dataset_splits_do_not_share_a_plan(cfg: WorldConfig) -> None:
    """Rule 6, asserted on the file that will actually train the model."""
    data = load_dataset(DATA, cfg)
    assert not set(data.train_plan_ids) & set(data.test_plan_ids)
    assert len(data.y_train) > len(data.y_test)


@needs_data
def test_dataset_is_loaded_deterministically(cfg: WorldConfig) -> None:
    a = load_dataset(DATA, cfg)
    b = load_dataset(DATA, cfg)
    assert a.train_plan_ids == b.train_plan_ids
    assert np.array_equal(a.x_train, b.x_train)


@needs_data
def test_full_model_beats_the_heuristic(cfg: WorldConfig) -> None:
    """Phase 4's acceptance check.

    The baseline is the same model family with the same hyperparameters fitted
    on ``budget_ratio`` alone, not a raw unscaled ratio -- beating a strawman
    would prove nothing about the feature set.
    """
    data = load_dataset(DATA, cfg)
    model = fit(data.x_train, data.y_train)
    baseline = fit_baseline(data)

    y_model = predict_clipped(model, data.x_test)
    y_base = predict_clipped(baseline, data.x_test[:, [data.baseline_index]])
    s_model, s_base = score(data.y_test, y_model), score(data.y_test, y_base)

    assert s_model.mae < s_base.mae
    assert s_model.auc > s_base.auc
    assert s_model.spearman > s_base.spearman


@needs_data
def test_calibration_is_reported_and_reasonable(cfg: WorldConfig) -> None:
    """Rule: the acceptance check asks for a calibration curve, not just AUC.
    A model can rank perfectly and still be unusable for thresholding."""
    data = load_dataset(DATA, cfg)
    model = fit(data.x_train, data.y_train)
    y_pred = predict_clipped(model, data.x_test)
    bins = calibration_curve(data.y_test, y_pred, bins=10)
    assert len(bins) >= 5, "predictions are bunched into too few bins to calibrate"
    assert sum(b.n for b in bins) == len(data.y_test)
    assert expected_calibration_error(bins) < 0.15


@needs_data
def test_shap_attributions_sum_towards_the_prediction(cfg: WorldConfig) -> None:
    """SHAP values plus the base value reconstruct the model output; if that
    identity fails, the explanations shown to an operator are not the model's
    reasons."""
    data = load_dataset(DATA, cfg)
    model = fit(data.x_train, data.y_train)
    sample = data.x_test[:50]
    values = shap_values(model, sample)
    raw = model.predict(sample)
    import shap as shap_lib

    base = shap_lib.TreeExplainer(model).expected_value
    assert np.allclose(values.sum(axis=1) + base, raw, atol=1e-3)


@needs_data
def test_shap_names_line_up_with_the_feature_matrix(cfg: WorldConfig) -> None:
    """A feature-name offset would attribute the wrong effect to the wrong
    name, and nothing would raise."""
    data = load_dataset(DATA, cfg)
    model = fit(data.x_train, data.y_train)
    values = shap_values(model, data.x_test[:50])
    assert values.shape[1] == len(data.feature_names)
    importance = global_importance(values, data.feature_names)
    assert {n for n, _ in importance} == set(data.feature_names)
    top = explain_plan(values, data.x_test[0], 0, top=3, names=data.feature_names)
    assert all(a.feature in data.feature_names for a in top)


@needs_data
def test_budget_ratio_is_not_the_only_thing_the_model_uses(cfg: WorldConfig) -> None:
    """If the heuristic feature dominated everything, the surrogate would be a
    time margin in a trenchcoat and Phase 4 would be a negative result."""
    data = load_dataset(DATA, cfg)
    model = fit(data.x_train, data.y_train)
    importance = dict(global_importance(shap_values(model, data.x_test), data.feature_names))
    total = sum(importance.values())
    assert importance[BASELINE_FEATURE] / total < 0.5
