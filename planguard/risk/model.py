"""Phase 4: the XGBoost failure-risk surrogate.

What it is for: screening. Inside the Phase 7 agent loop, candidate plans have
to be ranked before any of them is worth simulating, and a simulated mission
costs ~1.2 CPU-seconds while a prediction costs microseconds. The surrogate
only earns its place if it is **both** faster than simulating and better than
the obvious heuristic.

Three things this module is careful about, because each is an easy way to
report a flattering number.

**The baseline is the strongest fair version of the heuristic, not a
strawman.** CLAUDE.md asks the model to beat "path length over budget". The
temptation is to compare a tuned, calibrated XGBoost against a raw unscaled
ratio, which would win on error metrics for reasons that have nothing to do
with the feature set. So the baseline here is *the same model family, the same
hyperparameters, fitted on the single feature* ``budget_ratio``. The
comparison then answers the only interesting question: does everything else in
the feature set add anything beyond the time margin?

**Calibration is reported, not just AUC** (rule: Phase 4's acceptance check
says so explicitly). AUC measures ranking. Screening does use ranking, but the
Phase 7 graph also wants to *threshold* -- "simulate anything above 0.3 risk"
-- and for that a model can rank perfectly and still be badly miscalibrated.

**The split is the one Phase 3 built.** Train and test plans are disjoint
(rule 6) and drawn from disjoint scenario-seed blocks, so a test row is an
unseen plan in unseen weather. Nothing here re-splits.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from xgboost import XGBRegressor

from planguard.config import WorldConfig
from planguard.dsl.schema import MissionPlan
from planguard.risk.features import FEATURE_NAMES, feature_vector

#: Fixed so a rerun reproduces the model. XGBoost is deterministic given the
#: same data, parameters and thread count; ``n_jobs`` is pinned for the same
#: reason (rule 5 in spirit -- a model that changes between runs cannot be
#: compared against anything).
PARAMS: dict[str, object] = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
    "random_state": 0,
    "n_jobs": 4,
    "objective": "reg:squarederror",
}

#: The single feature the baseline heuristic is allowed to use.
BASELINE_FEATURE = "budget_ratio"


@dataclass
class Dataset:
    """Features and labels, already split by Phase 3."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_names: tuple[str, ...] = FEATURE_NAMES
    train_plan_ids: list[str] = field(default_factory=list)
    test_plan_ids: list[str] = field(default_factory=list)

    @property
    def baseline_index(self) -> int:
        return self.feature_names.index(BASELINE_FEATURE)


def load_dataset(path: str | Path, cfg: WorldConfig) -> Dataset:
    """Read ``plans.parquet`` and featurise every plan from its stored JSON.

    Features come from ``plan_json``, never from the outcome columns. That is
    what rule 2 requires, and it is why Phase 3 stored the plan next to the
    label: feature engineering can be redone without re-simulating 40,000
    missions.
    """
    rows = pq.read_table(Path(path) / "plans.parquet").to_pylist()
    rows.sort(key=lambda r: str(r["plan_id"]))  # determinism

    split: dict[str, list[tuple[str, list[float], float]]] = {"train": [], "test": []}
    for row in rows:
        plan = MissionPlan.model_validate_json(str(row["plan_json"]))
        split[str(row["split"])].append(
            (str(row["plan_id"]), feature_vector(plan, cfg), float(row["failure_rate"]))
        )

    def arrays(key: str) -> tuple[list[str], np.ndarray, np.ndarray]:
        ids = [r[0] for r in split[key]]
        x = np.array([r[1] for r in split[key]], dtype=np.float64)
        y = np.array([r[2] for r in split[key]], dtype=np.float64)
        return ids, x, y

    train_ids, x_train, y_train = arrays("train")
    test_ids, x_test, y_test = arrays("test")

    overlap = set(train_ids) & set(test_ids)
    assert not overlap, f"rule 6 violated: {len(overlap)} plans in both splits"

    return Dataset(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        train_plan_ids=train_ids,
        test_plan_ids=test_ids,
    )


def fit(x: np.ndarray, y: np.ndarray) -> XGBRegressor:
    model = XGBRegressor(**PARAMS)
    model.fit(x, y)
    return model


def fit_baseline(data: Dataset) -> XGBRegressor:
    """The heuristic, given every advantage except the other features.

    Same family, same hyperparameters, one feature. If the full model cannot
    beat this, the honest conclusion is that the risk surrogate is a time
    margin in a trenchcoat, and Phase 4 is a negative result.
    """
    i = data.baseline_index
    return fit(data.x_train[:, [i]], data.y_train)


def predict_clipped(model: XGBRegressor, x: np.ndarray) -> np.ndarray:
    """Squared-error regression on a rate can leave [0, 1]; a risk of -0.04 is
    not a probability and would break both Brier and the calibration bins."""
    return np.asarray(np.clip(model.predict(x), 0.0, 1.0), dtype=np.float64)


@dataclass
class CalibrationBin:
    lo: float
    hi: float
    n: int
    mean_predicted: float
    mean_observed: float


def calibration_curve(
    y_true: np.ndarray, y_pred: np.ndarray, bins: int = 10
) -> list[CalibrationBin]:
    """Predicted risk against observed failure rate, in equal-width bins.

    Reported as a table so it needs no plotting library, and so the numbers
    are readable in a terminal and in ARCHITECTURE section 10.
    """
    out: list[CalibrationBin] = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (y_pred >= lo) & (y_pred < hi if hi < 1.0 else y_pred <= 1.0)
        if not mask.any():
            continue
        out.append(
            CalibrationBin(
                lo=float(lo),
                hi=float(hi),
                n=int(mask.sum()),
                mean_predicted=float(y_pred[mask].mean()),
                mean_observed=float(y_true[mask].mean()),
            )
        )
    return out


def expected_calibration_error(bins: list[CalibrationBin]) -> float:
    """Weighted mean gap between predicted and observed risk.

    The single number that says whether a *threshold* on this model's output
    means what it says, which is what the Phase 7 graph needs.
    """
    total = sum(b.n for b in bins)
    if not total:
        return 0.0
    return sum(b.n * abs(b.mean_predicted - b.mean_observed) for b in bins) / total


@dataclass
class Scores:
    mae: float
    rmse: float
    brier: float
    auc: float
    ece: float
    spearman: float


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    rho = spearmanr(a, b).statistic
    return float(rho) if rho == rho else 0.0


def score(y_true: np.ndarray, y_pred: np.ndarray, bins: int = 10) -> Scores:
    """Error, ranking and calibration together.

    ``y_true`` is a *rate* over 20 seeds, not a 0/1 outcome, so AUC is computed
    against a binarised label (did this plan fail in the majority of its
    seeds). Reporting AUC alone against a thresholded rate would hide how far
    off the magnitudes are, which is exactly what calibration is for.
    """
    binary = (y_true > 0.5).astype(int)
    auc = float(roc_auc_score(binary, y_pred)) if 0 < binary.sum() < len(binary) else float("nan")
    bins_ = calibration_curve(y_true, y_pred, bins)
    return Scores(
        mae=float(np.abs(y_true - y_pred).mean()),
        rmse=float(np.sqrt(((y_true - y_pred) ** 2).mean())),
        brier=float(brier_score_loss(binary, y_pred)),
        auc=auc,
        ece=expected_calibration_error(bins_),
        spearman=_spearman(y_true, y_pred),
    )


def isotonic_recalibrate(
    model: XGBRegressor, data: Dataset
) -> tuple[IsotonicRegression, np.ndarray]:
    """Fit a monotone recalibration map on the **training** split only.

    Fitting it on test would be using the answers. The map can only reorder
    magnitudes, not ranking, so AUC is unchanged by construction -- which is
    itself a useful check that calibration and discrimination are separate
    properties.
    """
    train_pred = predict_clipped(model, data.x_train)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(train_pred, data.y_train)
    return iso, np.clip(iso.predict(predict_clipped(model, data.x_test)), 0.0, 1.0)


@dataclass
class ScreeningCost:
    """What screening one plan actually costs, end to end.

    Reported three ways because the flattering version is very easy to quote.
    A batch of 386 predictions runs in well under a millisecond, which divides
    out to ~1 us per plan -- but the Phase 7 graph screens candidates one at a
    time, and it must *featurise* each one first, which means building a
    schedule estimate. The batched figure is real and useless; the one that
    matters is ``per_plan_s``.
    """

    n: int
    batch_predict_s: float
    single_predict_s: float
    featurise_s: float
    simulate_s_per_run: float
    seeds_per_plan: int

    @property
    def batch_predict_s_per_plan(self) -> float:
        return self.batch_predict_s / self.n if self.n else 0.0

    @property
    def per_plan_s(self) -> float:
        """Featurise one plan, then predict on it. The honest unit cost."""
        return self.featurise_s + self.single_predict_s

    @property
    def simulate_s_per_plan(self) -> float:
        """CPU-seconds, not wall-clock: the comparison is of work done, and
        both sides parallelise."""
        return self.simulate_s_per_run * self.seeds_per_plan

    @property
    def speedup(self) -> float:
        """Against the realistic per-plan cost, not the batched one."""
        return self.simulate_s_per_plan / self.per_plan_s if self.per_plan_s > 0 else float("inf")

    @property
    def batched_speedup(self) -> float:
        """The flattering number, kept only so the gap between the two is
        visible rather than quietly chosen."""
        per = self.batch_predict_s_per_plan
        return self.simulate_s_per_plan / per if per > 0 else float("inf")


def measure_screening_cost(
    model: XGBRegressor,
    x: np.ndarray,
    simulate_s_per_run: float,
    seeds_per_plan: int,
    featurise: Callable[[], object] | None = None,
    repeats: int = 5,
) -> ScreeningCost:
    """Time batched prediction, single-plan prediction and featurisation.

    ``simulate_s_per_run`` is measured, not assumed -- it comes from the Phase
    3 manifest's own throughput, so the speedup is against this machine's real
    simulator rather than a remembered figure.
    """
    predict_clipped(model, x[:1])  # warm up

    batch = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        predict_clipped(model, x)
        batch = min(batch, time.perf_counter() - t0)

    one = x[:1]
    single = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(50):
            predict_clipped(model, one)
        single = min(single, (time.perf_counter() - t0) / 50)

    featurise_s = 0.0
    if featurise is not None:
        featurise()  # warm up
        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            for _ in range(20):
                featurise()
            best = min(best, (time.perf_counter() - t0) / 20)
        featurise_s = best

    return ScreeningCost(
        n=len(x),
        batch_predict_s=batch,
        single_predict_s=single,
        featurise_s=featurise_s,
        simulate_s_per_run=simulate_s_per_run,
        seeds_per_plan=seeds_per_plan,
    )
