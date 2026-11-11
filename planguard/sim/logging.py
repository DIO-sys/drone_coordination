"""Run logging.

Per the conventions in CLAUDE.md the risk model and the debrief read *only*
from these files, never from live simulator objects, so the schema here is the
contract for Phases 3 and 4.

Logging is optional: the Phase 3 batch harness runs tens of thousands of
simulations and only needs the summary, so ``World`` accepts ``logger=None``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

DRONE_SCHEMA = pa.schema(
    [
        ("tick", pa.int32()),
        ("t_s", pa.float64()),
        ("drone_id", pa.string()),
        ("x_m", pa.float64()),
        ("y_m", pa.float64()),
        ("heading_rad", pa.float64()),
        ("fuel_s", pa.float64()),
        ("link_up", pa.bool_()),
        ("grounded", pa.bool_()),
        ("mode", pa.string()),
        ("task_id", pa.string()),
    ]
)


@dataclass
class RunLogger:
    out_dir: Path
    drone_rows: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    def drone(self, **row: Any) -> None:
        self.drone_rows.append(row)

    def event(self, t_s: float, kind: str, **fields: Any) -> None:
        self.events.append({"t_s": round(t_s, 3), "kind": kind, **fields})

    def write(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(self.drone_rows, schema=DRONE_SCHEMA)
        pq.write_table(table, self.out_dir / "drones.parquet")
        with (self.out_dir / "events.jsonl").open("w") as fh:
            for ev in self.events:
                fh.write(json.dumps(ev, sort_keys=True) + "\n")
