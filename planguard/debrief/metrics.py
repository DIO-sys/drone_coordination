"""Mission scoring, computed from the logs and nothing else.

Per the conventions in CLAUDE.md the debrief and the risk model read only from
``drones.parquet`` and ``events.jsonl``, never from live simulator objects.
That is not fussiness: Phase 3 generates its dataset from stored runs, and
Phase 4 trains on it, so anything the debrief cannot reconstruct from a log
does not exist as far as the risk model is concerned.

A test asserts these numbers match ``RunResult`` for the same run. When they
disagree, the log is incomplete -- which is how the timeout ``task_failed``
event came to be written at all.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


@dataclass(frozen=True)
class MissionMetrics:
    success: bool
    makespan_s: float
    aircraft_launched: int
    aircraft_grounded: int
    drone_seconds_no_link: float
    completed_task_ids: list[str]
    failed_task_ids: list[str]
    task_progress: dict[str, float]
    #: aircraft id -> mode -> seconds, for the replay viewer and the dashboard
    mode_seconds: dict[str, dict[str, float]]

    @property
    def completed_fraction(self) -> float:
        total = len(self.completed_task_ids) + len(self.failed_task_ids)
        return len(self.completed_task_ids) / total if total else 0.0


def load_events(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def score(run_dir: Path, dt_s: float) -> MissionMetrics:
    rows = pq.read_table(run_dir / "drones.parquet").to_pylist()
    events = load_events(run_dir)

    end: dict[str, Any] = next((e for e in reversed(events) if e["kind"] == "mission_end"), {})
    status: dict[str, str] = end.get("task_status", {})
    progress: dict[str, float] = end.get("task_progress", {})

    last_tick = max((r["tick"] for r in rows), default=-1)
    final = {r["drone_id"]: r for r in rows if r["tick"] == last_tick}

    no_link_ticks = sum(1 for r in rows if not r["link_up"])
    mode_counts = Counter((r["drone_id"], r["mode"]) for r in rows)
    mode_seconds: dict[str, dict[str, float]] = {}
    for (did, mode), n in mode_counts.items():
        mode_seconds.setdefault(did, {})[mode] = round(n * dt_s, 3)

    return MissionMetrics(
        success=bool(end.get("success", False)),
        # the last logged row is at (ticks - 1) * dt, so the run ends one tick later
        makespan_s=round((last_tick + 1) * dt_s, 3),
        aircraft_launched=len({r["drone_id"] for r in rows}),
        aircraft_grounded=sum(1 for r in final.values() if r["grounded"]),
        drone_seconds_no_link=round(no_link_ticks * dt_s, 2),
        completed_task_ids=sorted(k for k, v in status.items() if v == "done"),
        failed_task_ids=sorted(k for k, v in status.items() if v == "failed"),
        task_progress=dict(sorted(progress.items())),
        mode_seconds={k: dict(sorted(v.items())) for k, v in sorted(mode_seconds.items())},
    )
