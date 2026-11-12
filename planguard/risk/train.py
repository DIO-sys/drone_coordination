"""CLI: train the risk surrogate and report E2.

    uv run python -m planguard.risk.train --data data/risk

Phase 4's acceptance check has three parts and this prints all three:

1. **beats "path length over budget" on held-out plans** -- against the
   strongest fair version of that heuristic, the same model fitted on
   ``budget_ratio`` alone;
2. **a calibration curve, not just AUC** -- as a table, plus the expected
   calibration error, before and after isotonic recalibration;
3. **screening speedup measured** -- predictions per second against the
   simulator's own measured throughput.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from planguard.config import WorldConfig
from planguard.dsl.schema import MissionPlan
from planguard.risk.explain import explain_plan, global_importance, render_explanation, shap_values
from planguard.risk.features import feature_vector
from planguard.risk.model import (
    BASELINE_FEATURE,
    PARAMS,
    Scores,
    calibration_curve,
    fit,
    fit_baseline,
    isotonic_recalibrate,
    load_dataset,
    measure_screening_cost,
    predict_clipped,
    score,
)


def _table(rows: list[tuple[str, Scores]]) -> str:
    head = (
        f"{'model':26s} {'MAE':>7s} {'RMSE':>7s} {'Brier':>7s} {'AUC':>7s} {'ECE':>7s} {'rho':>7s}"
    )
    out = [head, "-" * len(head)]
    for name, s in rows:
        out.append(
            f"{name:26s} {s.mae:7.4f} {s.rmse:7.4f} {s.brier:7.4f} "
            f"{s.auc:7.4f} {s.ece:7.4f} {s.spearman:7.4f}"
        )
    return "\n".join(out)


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="planguard.risk.train")
    ap.add_argument("--data", type=Path, default=Path("data/risk"))
    ap.add_argument("--world", type=Path, default=Path("experiments/configs/world.yaml"))
    ap.add_argument("--out", type=Path, default=Path("models/risk"))
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)

    cfg = WorldConfig.load(args.world)
    data = load_dataset(args.data, cfg)
    manifest_path = args.data / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    print(
        f"dataset: {len(data.y_train)} train plans / {len(data.y_test)} test plans, "
        f"{len(data.feature_names)} features"
    )
    print(
        f"labels:  train mean failure rate {data.y_train.mean():.3f}, test {data.y_test.mean():.3f}"
    )

    model = fit(data.x_train, data.y_train)
    baseline = fit_baseline(data)

    y_model = predict_clipped(model, data.x_test)
    y_base = predict_clipped(baseline, data.x_test[:, [data.baseline_index]])
    _, y_iso = isotonic_recalibrate(model, data)

    s_base, s_model, s_iso = (
        score(data.y_test, y_base, args.bins),
        score(data.y_test, y_model, args.bins),
        score(data.y_test, y_iso, args.bins),
    )

    print("\nE2 -- held-out plans, unseen scenarios")
    print(
        _table(
            [
                (f"baseline ({BASELINE_FEATURE})", s_base),
                ("full feature set", s_model),
                ("full + isotonic", s_iso),
            ]
        )
    )
    improvement = (s_base.mae - s_model.mae) / s_base.mae if s_base.mae else 0.0
    verdict = "BEATS" if s_model.mae < s_base.mae else "DOES NOT BEAT"
    print(f"\n  {verdict} the heuristic: MAE {improvement:+.1%} vs baseline")

    print("\ncalibration (full model, before recalibration)")
    print(f"  {'predicted':>18s}  {'n':>5s}  {'mean pred':>9s}  {'observed':>9s}  {'gap':>7s}")
    for b in calibration_curve(data.y_test, y_model, args.bins):
        gap = b.mean_predicted - b.mean_observed
        print(
            f"  [{b.lo:.2f}, {b.hi:.2f})      {b.n:5d}  {b.mean_predicted:9.3f}  "
            f"{b.mean_observed:9.3f}  {gap:+7.3f}"
        )
    print(f"  ECE before {s_model.ece:.4f}  ->  after isotonic {s_iso.ece:.4f}")
    print(f"  AUC is unchanged by recalibration: {s_model.auc:.4f} -> {s_iso.auc:.4f}")

    sim_s = 1.0 / float(manifest.get("runs_per_s", 13.6)) * int(manifest.get("workers", 1))
    seeds = int(manifest.get("n_seeds_per_plan", 20))
    # featurisation is timed on a real plan, because it is the dominant cost
    # of screening and leaving it out is how a 20-million-fold speedup gets
    # quoted for something that actually runs at a few thousand plans a second
    sample_plan = MissionPlan.model_validate_json(
        str(pq.read_table(args.data / "plans.parquet").to_pylist()[0]["plan_json"])
    )
    cost = measure_screening_cost(
        model, data.x_test, sim_s, seeds, featurise=lambda: feature_vector(sample_plan, cfg)
    )
    print("\nscreening cost")
    print(f"  featurise one plan           {cost.featurise_s * 1e6:9.1f} us")
    print(f"  predict on one plan          {cost.single_predict_s * 1e6:9.1f} us")
    print(f"  -> per plan, end to end      {cost.per_plan_s * 1e6:9.1f} us")
    print(f"  simulate one plan ({seeds} seeds) {cost.simulate_s_per_plan:9.2f} s  (CPU-seconds)")
    print(f"  speedup                      {cost.speedup:9,.0f}x")
    print(
        f"  (batched prediction alone would read {cost.batched_speedup:,.0f}x, "
        "which no caller experiences)"
    )

    values = shap_values(model, data.x_test)
    print("\nSHAP, mean |contribution| across held-out plans")
    for name, importance in global_importance(values, data.feature_names)[:10]:
        print(f"  {name:28s} {importance:.4f}")

    worst = int(np.argmax(data.y_test))
    print(
        f"\nexample: plan {data.test_plan_ids[worst]} (observed failure rate "
        f"{data.y_test[worst]:.2f}, predicted {y_model[worst]:.2f})"
    )
    print(
        render_explanation(
            explain_plan(values, data.x_test[worst], worst, names=data.feature_names)
        )
    )

    if args.no_write:
        print("\n--no-write: nothing saved", file=sys.stderr)
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_model(args.out / "model.json")
    report = {
        "params": PARAMS,
        "n_train": len(data.y_train),
        "n_test": len(data.y_test),
        "features": list(data.feature_names),
        "baseline_feature": BASELINE_FEATURE,
        "scores": {
            "baseline": vars(s_base),
            "model": vars(s_model),
            "model_isotonic": vars(s_iso),
        },
        "mae_improvement_vs_baseline": improvement,
        "calibration": [vars(b) for b in calibration_curve(data.y_test, y_model, args.bins)],
        "screening": {
            "n": cost.n,
            "featurise_s": cost.featurise_s,
            "single_predict_s": cost.single_predict_s,
            "per_plan_s": cost.per_plan_s,
            "simulate_s_per_plan": cost.simulate_s_per_plan,
            "speedup": cost.speedup,
            "batched_speedup": cost.batched_speedup,
        },
        "shap_global": dict(global_importance(values, data.feature_names)),
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {args.out}/model.json and report.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
