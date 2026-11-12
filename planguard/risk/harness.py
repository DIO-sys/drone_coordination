"""Batch simulation harness — turns sampled plans into risk labels.

Phase 3's deliverable. The shape of the dataset is fixed by two rules.

**Rule 6: split by plan and by seed.** The same plan must never appear in both
train and test, and neither must the same scenario. So the split is decided
first, from a stable hash of the plan's content, and each side then draws from
its **own disjoint block of scenario seeds**. A test row is therefore an unseen
plan flown in unseen weather, which is the question actually being asked of the
Phase 4 surrogate. The cost is that any difference in difficulty between the
two seed blocks shows up as train/test skew; with 20 seeds per plan over
thousands of plans that averages out, and it is the honest direction to err in.

**Rule 2: features must be plan-time observable.** This module therefore
stores the *plan* next to the label and no simulation internals beyond the
outcome itself. Phase 4 derives its features from the stored plan JSON, which
means feature engineering can be redone without re-running a single mission --
the expensive half of the dataset is the 40,000 simulated missions, and they
should never have to be paid for twice.

Multiprocessing is not an optimisation here. At 555 ms per run, 2,000 plans x
20 seeds is 6.2 hours on one core; the work is embarrassingly parallel and each
run is independent by construction, because determinism comes from the seed
and not from any shared state.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from planguard.config import WorldConfig
from planguard.dsl.sample import canonical_json, plan_id
from planguard.dsl.schema import MissionPlan
from planguard.dsl.validate import makespan_lower_bound_s
from planguard.sim.scenario import generate
from planguard.sim.world import World

#: Scenario seeds for test-split plans start here. Far enough from the train
#: block that no corpus size can make the two overlap.
TEST_SEED_OFFSET = 1_000_000


@dataclass(frozen=True)
class Job:
    """One (plan, scenario) pair. Must be picklable: it crosses a process."""

    pid: str
    plan_json: str
    seed: int
    split: str


@dataclass(frozen=True)
class PlanRow:
    """One row per plan: the label, the plan itself, and plan-time context.

    A dataclass rather than a dict so the Parquet column order is fixed by the
    source and two builds cannot produce schema-incompatible files.
    """

    plan_id: str
    split: str
    n_runs: int
    #: The label Phase 4 regresses on.
    failure_rate: float
    mean_completed_fraction: float
    #: Plan-time context, all derivable from the plan alone (rule 2).
    time_budget_s: float
    makespan_lower_bound_s: float
    budget_ratio: float
    binding_bound: str
    n_tasks: int
    #: Kept so Phase 4 can derive features without re-simulating anything.
    plan_json: str
    plan_canonical_sha256: str
    #: Outcome aggregates. Diagnostics only -- never features, per rule 2.
    mean_aircraft_launched: float
    mean_drone_seconds_no_link: float


@dataclass(frozen=True)
class RunRow:
    pid: str
    seed: int
    split: str
    success: bool
    completed_fraction: float
    t_end_s: float
    aircraft_launched: int
    aircraft_grounded: int
    drone_seconds_no_link: float
    handoffs: int
    wind_speed_mps: float
    digest: str


def split_of(pid: str, test_fraction: float) -> str:
    """Stable content-addressed split assignment.

    Salted so it cannot correlate with ``plan_id``, which is the same hash
    without the salt and is used as a key elsewhere.
    """
    h = hashlib.sha256(f"split:{pid}".encode()).digest()
    # 32 bits is plenty of resolution and keeps the arithmetic exact
    frac = int.from_bytes(h[:4], "big") / 2**32
    return "test" if frac < test_fraction else "train"


def seeds_for(split: str, n_seeds: int) -> list[int]:
    base = TEST_SEED_OFFSET if split == "test" else 0
    return [base + i for i in range(n_seeds)]


# --------------------------------------------------------------------------
# worker side
# --------------------------------------------------------------------------

_CFG: WorldConfig | None = None


def _init_worker(world_path: str) -> None:
    """Parse the world config once per process rather than once per run."""
    global _CFG
    _CFG = WorldConfig.load(world_path)


def run_job(job: Job) -> RunRow:
    """Simulate one (plan, seed) pair. No logging: 40,000 runs of Parquet is
    gigabytes, and the aggregate outcome is all the label needs."""
    assert _CFG is not None, "worker not initialised"
    cfg = _CFG
    plan = MissionPlan.model_validate_json(job.plan_json)
    world = World(plan=plan, cfg=cfg, scenario=generate(cfg, job.seed), logger=None)
    r = world.run()
    return RunRow(
        pid=job.pid,
        seed=job.seed,
        split=job.split,
        success=r.success,
        completed_fraction=r.completed_fraction,
        t_end_s=r.t_end_s,
        aircraft_launched=r.aircraft_launched,
        aircraft_grounded=r.aircraft_grounded,
        drone_seconds_no_link=r.drone_seconds_no_link,
        handoffs=r.handoffs,
        wind_speed_mps=r.wind_speed_mps,
        digest=r.digest,
    )


# --------------------------------------------------------------------------
# driver side
# --------------------------------------------------------------------------


def build_jobs(plans: list[MissionPlan], n_seeds: int, test_fraction: float) -> list[Job]:
    jobs: list[Job] = []
    for plan in plans:
        pid = plan_id(plan)
        split = split_of(pid, test_fraction)
        plan_json = plan.model_dump_json(exclude_none=True)
        for seed in seeds_for(split, n_seeds):
            jobs.append(Job(pid=pid, plan_json=plan_json, seed=seed, split=split))
    return jobs


def run_jobs(
    jobs: list[Job], world_path: str, workers: int | None = None, chunksize: int = 8
) -> list[RunRow]:
    """Run every job and return the rows in a **deterministic** order.

    ``imap_unordered`` is used because it is the cheapest way to keep every
    core busy with uneven run lengths, and the result is then sorted by
    (plan id, seed). Returning rows in completion order would make the dataset
    depend on machine load -- the file would differ between two runs of the
    same command, which is rule 5 broken in the one place it would be hardest
    to notice.
    """
    workers = workers or min(len(jobs), os.cpu_count() or 1)
    if workers <= 1:
        _init_worker(world_path)
        rows = [run_job(j) for j in jobs]
    else:
        ctx = mp.get_context("spawn")  # fork + numpy RNG state is asking for trouble
        with ctx.Pool(workers, initializer=_init_worker, initargs=(world_path,)) as pool:
            rows = list(pool.imap_unordered(run_job, jobs, chunksize=chunksize))
    return sorted(rows, key=lambda r: (r.pid, r.seed))


def aggregate(plans: list[MissionPlan], rows: list[RunRow], cfg: WorldConfig) -> list[PlanRow]:
    """One row per plan: the label, plus the plan itself for Phase 4."""
    by_pid: dict[str, list[RunRow]] = {}
    for row in rows:
        by_pid.setdefault(row.pid, []).append(row)

    out: list[PlanRow] = []
    for plan in sorted(plans, key=plan_id):
        pid = plan_id(plan)
        rs = by_pid.get(pid)
        if not rs:
            continue
        n = len(rs)
        bound_s, binding = makespan_lower_bound_s(plan, cfg)
        out.append(
            PlanRow(
                plan_id=pid,
                split=rs[0].split,
                n_runs=n,
                failure_rate=sum(1 for r in rs if not r.success) / n,
                mean_completed_fraction=sum(r.completed_fraction for r in rs) / n,
                time_budget_s=plan.time_budget_s,
                makespan_lower_bound_s=bound_s,
                budget_ratio=plan.time_budget_s / bound_s if bound_s > 0 else float("inf"),
                binding_bound=binding,
                n_tasks=len(plan.tasks),
                plan_json=plan.model_dump_json(exclude_none=True),
                plan_canonical_sha256=hashlib.sha256(canonical_json(plan).encode()).hexdigest(),
                mean_aircraft_launched=sum(r.aircraft_launched for r in rs) / n,
                mean_drone_seconds_no_link=sum(r.drone_seconds_no_link for r in rs) / n,
            )
        )
    return out


def write_dataset(
    out_dir: Path,
    plan_rows: list[PlanRow],
    rows: list[RunRow],
    manifest: dict[str, object],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([asdict(p) for p in plan_rows]), out_dir / "plans.parquet")
    pq.write_table(pa.Table.from_pylist([asdict(r) for r in rows]), out_dir / "runs.parquet")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def build(
    plans: list[MissionPlan],
    cfg: WorldConfig,
    world_path: str,
    n_seeds: int,
    test_fraction: float,
    workers: int | None = None,
) -> tuple[list[PlanRow], list[RunRow], dict[str, object]]:
    jobs = build_jobs(plans, n_seeds, test_fraction)
    t0 = time.perf_counter()
    rows = run_jobs(jobs, world_path, workers=workers)
    elapsed_s = time.perf_counter() - t0
    plan_rows = aggregate(plans, rows, cfg)

    n_test = sum(1 for p in plan_rows if p.split == "test")
    manifest: dict[str, object] = {
        "n_plans": len(plan_rows),
        "n_plans_train": len(plan_rows) - n_test,
        "n_plans_test": n_test,
        "n_runs": len(rows),
        "n_seeds_per_plan": n_seeds,
        "test_fraction_requested": test_fraction,
        "test_fraction_actual": n_test / len(plan_rows) if plan_rows else 0.0,
        "train_seeds": seeds_for("train", n_seeds),
        "test_seeds": seeds_for("test", n_seeds),
        "workers": workers or os.cpu_count(),
        "elapsed_s": round(elapsed_s, 1),
        "runs_per_s": round(len(rows) / elapsed_s, 1) if elapsed_s > 0 else None,
        "mean_failure_rate": (
            sum(p.failure_rate for p in plan_rows) / len(plan_rows) if plan_rows else 0.0
        ),
        "world_config": world_path,
    }
    return plan_rows, rows, manifest
