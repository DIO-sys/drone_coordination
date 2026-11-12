"""SHAP attributions for the risk surrogate.

Two audiences, which is the point of using SHAP rather than gain importance:

* **the operator**, who gets "this plan is risky *because* it commits three of
  four aircraft to one task", not a bare 0.62;
* **the Phase 7 repair loop**, which can aim a revision at the feature doing
  the damage instead of resampling blindly.

Gain importance would answer neither. It is a global, model-internal quantity
with no per-plan meaning, and it is biased towards high-cardinality features.

A caution that belongs next to the numbers: SHAP explains **the model**, not
the simulator. If the model has learned a correlation that does not hold
causally -- and the Phase 3 corpus has an obvious candidate, since
``max_aircraft_count`` is both genuinely risky and correlated with everything
else about a demanding plan -- SHAP will faithfully report the correlation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shap
from xgboost import XGBRegressor

from planguard.risk.features import FEATURE_NAMES


@dataclass(frozen=True)
class Attribution:
    feature: str
    value: float
    shap_value: float

    @property
    def direction(self) -> str:
        return "raises risk" if self.shap_value > 0 else "lowers risk"


def shap_values(model: XGBRegressor, x: np.ndarray) -> np.ndarray:
    explainer = shap.TreeExplainer(model)
    return np.asarray(explainer.shap_values(x), dtype=np.float64)


def global_importance(
    values: np.ndarray, names: tuple[str, ...] = FEATURE_NAMES
) -> list[tuple[str, float]]:
    """Mean |SHAP| per feature, most important first."""
    mean_abs = np.abs(values).mean(axis=0)
    return sorted(
        ((names[i], float(mean_abs[i])) for i in range(len(names))),
        key=lambda kv: -kv[1],
    )


def explain_plan(
    values: np.ndarray,
    row: np.ndarray,
    index: int,
    top: int = 5,
    names: tuple[str, ...] = FEATURE_NAMES,
) -> list[Attribution]:
    """The ``top`` features driving one plan's prediction, largest effect first."""
    contributions = values[index]
    order = np.argsort(-np.abs(contributions))[:top]
    return [
        Attribution(
            feature=names[i],
            value=float(row[i]),
            shap_value=float(contributions[i]),
        )
        for i in order
    ]


def render_explanation(attributions: list[Attribution]) -> str:
    """One plan's explanation as text an operator could read."""
    lines = []
    for a in attributions:
        lines.append(f"  {a.feature:28s} = {a.value:10.3f}  {a.shap_value:+.4f}  {a.direction}")
    return "\n".join(lines)
