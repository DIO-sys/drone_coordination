"""CLI entry point: validate a plan, then simulate it for a seed.

A plan is always validated before it is run. Simulating an invalid plan would
produce a risk label for something that could never be flown, which would
poison the Phase 3 dataset.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from planguard.config import WorldConfig
from planguard.dsl.schema import MissionPlan
from planguard.dsl.validate import validate
from planguard.sim.logging import RunLogger
from planguard.sim.scenario import generate
from planguard.sim.world import RunResult, World


def simulate(
    plan: MissionPlan, cfg: WorldConfig, seed: int, out_dir: Path | None = None
) -> RunResult:
    logger = RunLogger(out_dir=out_dir) if out_dir is not None else None
    world = World(plan=plan, cfg=cfg, scenario=generate(cfg, seed), logger=logger)
    return world.run()


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="planguard.sim.run")
    ap.add_argument("--plan", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--world", type=Path, default=Path("experiments/configs/world.yaml"))
    ap.add_argument("--out", type=Path, default=None, help="write drones.parquet / events.jsonl")
    ap.add_argument("--quiet", action="store_true", help="print only the digest")
    args = ap.parse_args(argv)

    cfg = WorldConfig.load(args.world)
    plan = MissionPlan.model_validate(json.loads(args.plan.read_text()))
    report = validate(plan, cfg)
    if not report.ok:
        print(report.as_feedback(), file=sys.stderr)
        return 1

    result = simulate(plan, cfg, args.seed, args.out)
    if args.quiet:
        print(result.digest)
    else:
        print(json.dumps(dataclasses.asdict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
