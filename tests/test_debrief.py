"""Mission metrics reconstructed from logs.

Phase 3 builds its dataset from stored runs and Phase 4 trains on it, so
anything the debrief cannot recover from a log does not exist as far as the
risk model is concerned. These tests are really about log completeness.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from planguard.config import WorldConfig
from planguard.debrief.metrics import score
from planguard.dsl.schema import MissionPlan
from planguard.sim.logging import RunLogger
from planguard.sim.scenario import generate
from planguard.sim.world import World

REPO = Path(__file__).resolve().parent.parent
DEMO = json.loads((REPO / "examples" / "ridge.json").read_text())


def run_with_logs(cfg: WorldConfig, seed: int, out: Path, **plan_overrides: object):
    plan = MissionPlan.model_validate(dict(DEMO, **plan_overrides))
    logger = RunLogger(out_dir=out)
    result = World(plan=plan, cfg=cfg, scenario=generate(cfg, seed), logger=logger).run()
    return result, score(out, cfg.sim.dt_s)


def test_metrics_from_logs_match_the_live_result(cfg: WorldConfig, tmp_path: Path) -> None:
    """If these diverge, the log is missing something the risk model will need."""
    result, metrics = run_with_logs(cfg, 7, tmp_path)
    assert metrics.success == result.success
    assert metrics.makespan_s == pytest.approx(result.t_end_s)
    assert metrics.aircraft_launched == result.aircraft_launched
    assert metrics.aircraft_grounded == result.aircraft_grounded
    assert metrics.drone_seconds_no_link == pytest.approx(result.drone_seconds_no_link, abs=0.2)
    assert metrics.completed_fraction == pytest.approx(result.completed_fraction)
    assert metrics.task_progress == result.task_progress


def test_a_timed_out_task_leaves_a_record(cfg: WorldConfig, tmp_path: Path) -> None:
    """A task that simply runs out of time used to end the run marked failed in
    memory with nothing in the log to say so, which made the debrief disagree
    with the simulator."""
    result, metrics = run_with_logs(cfg, 7, tmp_path, time_budget_s=200.0, name="too short")
    assert not result.success
    assert metrics.failed_task_ids == sorted(
        k for k, v in result.task_status.items() if v == "failed"
    )
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    timeouts = [e for e in events if e["kind"] == "task_failed"]
    assert timeouts and all(e["reason"] == "timeout" for e in timeouts)


def test_mode_seconds_cover_every_aircraft_for_the_whole_run(
    cfg: WorldConfig, tmp_path: Path
) -> None:
    result, metrics = run_with_logs(cfg, 7, tmp_path)
    assert set(metrics.mode_seconds) == {f"d{i}" for i in range(result.aircraft_launched)}
    for per_mode in metrics.mode_seconds.values():
        assert sum(per_mode.values()) == pytest.approx(result.t_end_s, abs=0.2)


def test_scoring_a_directory_with_no_run_raises(tmp_path: Path) -> None:
    """Better a hard failure than silently scoring an empty mission as a
    success, which is what a defaulting metric would do."""
    with pytest.raises(FileNotFoundError):
        score(tmp_path, 0.1)
