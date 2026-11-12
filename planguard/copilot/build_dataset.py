"""CLI: build the Phase 5 NL dataset and report both filter yields.

    uv run python -m planguard.copilot.build_dataset --pairs 4000 --out data/nl

Reports, because Phase 5's acceptance check is the yield rate and the held-out
set's independence, not the file existing:

* **validator yield** -- share of raw sampler draws the static validator passed
* **simulation yield** -- share of those that also flew above threshold
* **drift report** -- how far ``evals/heldout.yaml`` sits from this generator's
  own output, which is the only thing that makes rule 4's separate reporting
  mean anything
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from planguard.config import WorldConfig
from planguard.copilot.dataset_gen import back_translate, generate_pairs, split_pairs, write_jsonl
from planguard.copilot.heldout import drift_report, load_heldout, plan_shape_report
from planguard.dsl.sample import SamplerConfig
from planguard.dsl.schema import MissionPlan


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="planguard.copilot.build_dataset")
    ap.add_argument("--pairs", type=int, default=4000)
    ap.add_argument("--verify-seeds", type=int, default=5)
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="minimum simulated success rate for a plan to become a target",
    )
    ap.add_argument("--test-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--world", type=Path, default=Path("experiments/configs/world.yaml"))
    ap.add_argument("--sampler", type=Path, default=Path("experiments/configs/sampler.yaml"))
    ap.add_argument("--heldout", type=Path, default=Path("evals/heldout.yaml"))
    ap.add_argument("--out", type=Path, default=Path("data/nl"))
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = ap.parse_args(argv)

    cfg = WorldConfig.load(args.world)
    scfg = SamplerConfig.load(args.sampler) if args.sampler.exists() else SamplerConfig()

    pairs, stats = generate_pairs(
        cfg,
        args.pairs,
        seed=args.seed,
        scfg=scfg,
        verify_seeds=args.verify_seeds,
        threshold=args.threshold,
    )
    if not pairs:
        print("no pairs survived the filters", file=sys.stderr)
        return 1

    split = split_pairs(pairs, args.test_fraction)

    print("verifier filter")
    print(f"  raw sampler draws          {stats.sampler_draws}")
    print(f"  passed the validator       {stats.sampled}  ({stats.validator_yield:.1%})")
    print(f"  rejected by snapping       {stats.rejected_by_snapping}")
    print(f"  rejected by simulation     {stats.rejected_by_simulation}")
    print(f"  duplicate instructions     {stats.duplicate_instructions}")
    print(f"  accepted                   {stats.accepted}")
    print(f"  simulation yield           {stats.simulation_yield:.1%}")
    print(f"  overall yield              {stats.overall_yield:.1%}")
    if stats.rejection_codes:
        top = sorted(stats.rejection_codes.items(), key=lambda kv: (-kv[1], kv[0]))
        print("  validator rejections: " + ", ".join(f"{c}={n}" for c, n in top))
    print(f"\nin-distribution split: {len(split['train'])} train / {len(split['test'])} test")

    report: dict[str, float] = {}
    if args.heldout.exists():
        heldout = load_heldout(args.heldout)
        rng = np.random.default_rng(args.seed + 1)
        # the comparison is about language, so it runs on back-translations of
        # the accepted plans rather than re-simulating anything
        generated = [
            back_translate(MissionPlan.model_validate_json(p.plan_json), rng) for p in pairs[:1000]
        ]
        report = drift_report([i.instruction for i in heldout], generated)
        print(f"\nheld-out set: {len(heldout)} hand-written items")
        print("  drift from this generator (rule 4 made measurable)")
        for key in (
            "ngram_echo_n4",
            "vocab_shared",
            "vocab_unique_to_heldout",
            "heldout_mean_words",
            "generated_mean_words",
            "heldout_type_token_ratio",
            "generated_type_token_ratio_matched",
        ):
            print(f"    {key:38s} {report[key]:8.3f}")
        print("  plan shapes covered")
        for key, value in sorted(plan_shape_report(heldout).items()):
            print(f"    {key:38s} {value:8.2f}")

    if args.dry_run:
        print("\ndry run: nothing written", file=sys.stderr)
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in sorted(split.items()):
        write_jsonl(args.out / f"{name}.jsonl", rows)
    manifest = {
        "n_pairs": len(pairs),
        "n_train": len(split["train"]),
        "n_test": len(split["test"]),
        "verify_seeds": args.verify_seeds,
        "success_threshold": args.threshold,
        "validator_yield": stats.validator_yield,
        "simulation_yield": stats.simulation_yield,
        "overall_yield": stats.overall_yield,
        "sampler_draws": stats.sampler_draws,
        "rejected_by_simulation": stats.rejected_by_simulation,
        "rejected_by_snapping": stats.rejected_by_snapping,
        "duplicate_instructions": stats.duplicate_instructions,
        "validator_rejection_codes": stats.rejection_codes,
        "heldout_drift": report,
        "sampler": scfg.model_dump(),
        "world_config": str(args.world),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {args.out}/train.jsonl, test.jsonl, manifest.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
