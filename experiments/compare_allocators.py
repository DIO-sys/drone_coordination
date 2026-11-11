"""CBBA versus the greedy placeholder on identical scenarios.

Keeping the Phase 1 allocator rather than deleting it is what makes "CBBA was
worth implementing" a measurement instead of an assertion. Both allocators see
the same seeds, the same wind, the same aircraft availability.

Because the scenarios are identical, the comparison is **paired**: the same seed
is run under both allocators and the per-seed difference is what gets a
confidence interval. That matters more than it sounds. Scenario-to-scenario
variance here is large (one to four aircraft launch), so an unpaired comparison
of two means at n=30 cannot see a 5% effect at all -- the first run of this
script showed success rates differing by exactly one run in thirty, in opposite
directions at two budgets, which is noise being read as a finding.

A bootstrap CI on the paired difference is used rather than a t-test: success is
a Bernoulli outcome and makespan is bounded and skewed by the budget cap, so
neither is close to normal.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from planguard.config import WorldConfig
from planguard.dsl.schema import MissionPlan
from planguard.sim.run import simulate


def paired_ci(
    diffs: list[float], rng: np.random.Generator, resamples: int = 10_000
) -> tuple[float, float, float]:
    """Mean paired difference and a 95% bootstrap CI."""
    arr = np.asarray(diffs, dtype=float)
    means = rng.choice(arr, size=(resamples, arr.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(arr.mean()), float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser(prog="experiments.compare_allocators")
    ap.add_argument("--plan", type=Path, default=Path("examples/ridge.json"))
    ap.add_argument("--world", type=Path, default=Path("experiments/configs/world.yaml"))
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--budget", type=float, default=None, help="override time_budget_s")
    args = ap.parse_args()

    base_cfg = WorldConfig.load(args.world)
    raw = json.loads(args.plan.read_text())
    if args.budget is not None:
        raw["time_budget_s"] = args.budget
    plan = MissionPlan.model_validate(raw)

    print(f"plan={plan.name!r} budget={plan.time_budget_s:.0f}s seeds={args.seeds}")
    print(f"{'allocator':10s} {'success':>8s} {'completed':>10s} {'makespan':>9s} {'s/run':>7s}")

    runs = {}
    for allocator in ("greedy", "cbba"):
        cfg = base_cfg.model_copy(
            update={"cbba": base_cfg.cbba.model_copy(update={"allocator": allocator})}
        )
        t0 = time.perf_counter()
        results = [simulate(plan, cfg, s) for s in range(args.seeds)]
        elapsed = time.perf_counter() - t0
        runs[allocator] = results
        print(
            f"{allocator:10s} {sum(r.success for r in results) / len(results):8.2f} "
            f"{statistics.mean(r.completed_fraction for r in results):10.3f} "
            f"{statistics.mean(r.t_end_s for r in results):8.0f}s "
            f"{elapsed / len(results):7.2f}"
        )

    rng = np.random.default_rng(0)
    print("\npaired difference, cbba minus greedy (95% bootstrap CI):")
    for label, get in (
        ("success rate", lambda r: float(r.success)),
        ("completed fraction", lambda r: r.completed_fraction),
        ("makespan (s)", lambda r: r.t_end_s),
    ):
        diffs = [get(c) - get(g) for c, g in zip(runs["cbba"], runs["greedy"], strict=True)]
        mean, lo, hi = paired_ci(diffs, rng)
        verdict = "no detectable difference" if lo <= 0.0 <= hi else "difference detected"
        print(f"  {label:20s} {mean:+9.3f}  [{lo:+.3f}, {hi:+.3f}]  {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
