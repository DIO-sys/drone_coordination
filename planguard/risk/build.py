"""CLI: sample plans, simulate them across seeds, write the risk dataset.

    uv run python -m planguard.risk.build --plans 2000 --seeds 20 --out data/risk

The pilot flag exists because the one thing that can silently ruin this dataset
is a budget distribution that sits too far above the feasibility boundary --
every run succeeds, the label is all zeros, and Phase 4 learns a fleet-size
lookup table. ``--pilot`` runs a small corpus and prints the label distribution
by budget ratio, which is the check to run before spending hours on the full
sweep.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from planguard.config import WorldConfig
from planguard.dsl.sample import SamplerConfig, sample_corpus
from planguard.risk.harness import PlanRow, build, write_dataset


def _histogram(plan_rows: list[PlanRow], bins: int = 6) -> str:
    """Failure rate against budget ratio. The diagnostic that matters."""
    rows = sorted(plan_rows, key=lambda r: r.budget_ratio)
    if not rows:
        return "no plans"
    lo, hi = rows[0].budget_ratio, rows[-1].budget_ratio
    width = (hi - lo) / bins if hi > lo else 1.0
    out = ["budget_ratio       plans   mean failure rate   all-fail   all-succeed"]
    for b in range(bins):
        a, z = lo + b * width, lo + (b + 1) * width
        sel = [
            r for r in rows if a <= r.budget_ratio < z or (b == bins - 1 and r.budget_ratio == hi)
        ]
        if not sel:
            continue
        fr = [r.failure_rate for r in sel]
        out.append(
            f"{a:5.2f} - {z:5.2f}  {len(sel):7d}   {sum(fr) / len(fr):17.3f}"
            f"   {sum(1 for f in fr if f == 1.0):8d}   {sum(1 for f in fr if f == 0.0):11d}"
        )
    spread = sum(1 for r in rows if 0.0 < r.failure_rate < 1.0)
    out.append(
        f"\nlabel spread: {spread}/{len(rows)} plans have a failure rate strictly between 0 and 1"
    )
    return "\n".join(out)


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="planguard.risk.build")
    ap.add_argument("--plans", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=20, help="scenario seeds per plan")
    ap.add_argument("--test-fraction", type=float, default=0.2)
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--world", type=Path, default=Path("experiments/configs/world.yaml"))
    ap.add_argument("--sampler", type=Path, default=Path("experiments/configs/sampler.yaml"))
    ap.add_argument("--out", type=Path, default=Path("data/risk"))
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument(
        "--pilot", action="store_true", help="small corpus, print diagnostics, no write"
    )
    args = ap.parse_args(argv)

    cfg = WorldConfig.load(args.world)
    scfg = SamplerConfig.load(args.sampler) if args.sampler.exists() else SamplerConfig()

    n_plans = min(args.plans, 120) if args.pilot else args.plans
    n_seeds = min(args.seeds, 8) if args.pilot else args.seeds

    plans, stats = sample_corpus(cfg, scfg, n_plans, args.sample_seed)
    print(
        f"sampled {len(plans)} plans from {stats.drawn} draws "
        f"(yield {stats.yield_rate:.1%}; {stats.rejected_invalid} invalid, "
        f"{stats.rejected_duplicate} duplicate)",
        file=sys.stderr,
    )
    if stats.rejection_codes:
        top = sorted(stats.rejection_codes.items(), key=lambda kv: (-kv[1], kv[0]))
        print("  rejected by: " + ", ".join(f"{c}={n}" for c, n in top), file=sys.stderr)
    if not plans:
        print("no valid plans sampled", file=sys.stderr)
        return 1

    plan_rows, rows, manifest = build(
        plans,
        cfg,
        str(args.world),
        n_seeds=n_seeds,
        test_fraction=args.test_fraction,
        workers=args.workers,
    )
    manifest["sampler"] = scfg.model_dump()
    manifest["sample_yield_rate"] = stats.yield_rate
    manifest["sample_draws"] = stats.drawn

    print(
        f"ran {manifest['n_runs']} missions in {manifest['elapsed_s']}s "
        f"({manifest['runs_per_s']}/s on {manifest['workers']} workers)",
        file=sys.stderr,
    )
    print(_histogram(plan_rows))

    if args.pilot:
        print("\npilot: nothing written", file=sys.stderr)
        return 0

    write_dataset(args.out, plan_rows, rows, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"wrote {args.out}/plans.parquet, runs.parquet, manifest.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
