"""Thin text replay viewer.

Deliberately dependency-free ASCII rather than a plotting library: this exists
to answer "what did the aircraft actually do?" during development, and a
five-aircraft partition cannot be read out of a Parquet file by eye. The real
map lands in the Phase 8 dashboard.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from planguard.config import WorldConfig

GRID_W, GRID_H = 78, 26


def _load(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = pq.read_table(run_dir / "drones.parquet").to_pylist()
    events_path = run_dir / "events.jsonl"
    events = (
        [json.loads(line) for line in events_path.read_text().splitlines()]
        if events_path.exists()
        else []
    )
    return rows, events


def render_map(rows: list[dict[str, Any]], cfg: WorldConfig) -> str:
    """Aircraft tracks over the world, as a character grid."""
    xs = [c[0] for f in cfg.world_map.features for c in f.coords_m]
    ys = [c[1] for f in cfg.world_map.features for c in f.coords_m]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    span_x = max(x1 - x0, 1.0)
    span_y = max(y1 - y0, 1.0)

    def cell(x: float, y: float) -> tuple[int, int]:
        col = min(GRID_W - 1, max(0, int((x - x0) / span_x * (GRID_W - 1))))
        # row 0 is the top of the printed grid, so north is up
        row = min(GRID_H - 1, max(0, int((y1 - y) / span_y * (GRID_H - 1))))
        return row, col

    grid = [[" "] * GRID_W for _ in range(GRID_H)]
    for feature in cfg.world_map.features:
        mark = {"point": "+", "area": ".", "corridor": "-"}[feature.kind]
        for coord in feature.coords_m:
            gr, gc = cell(*coord)
            grid[gr][gc] = mark

    ids = sorted({row["drone_id"] for row in rows})
    for row in rows:
        gr, gc = cell(row["x_m"], row["y_m"])
        if grid[gr][gc] in " .+-":
            grid[gr][gc] = str(ids.index(row["drone_id"]))
    for did in ids:
        last = [row for row in rows if row["drone_id"] == did][-1]
        gr, gc = cell(last["x_m"], last["y_m"])
        grid[gr][gc] = chr(ord("A") + ids.index(did))

    header = f"  {x0:.0f}..{x1:.0f} m east, {y0:.0f}..{y1:.0f} m north (north up)"
    legend = "  features: + point  . area  - corridor   aircraft: digit=track UPPER=final"
    body = "\n".join("  |" + "".join(line) + "|" for line in grid)
    return f"{header}\n{legend}\n{body}"


def render_timeline(events: list[dict[str, Any]]) -> str:
    out = ["  events:"]
    for ev in events:
        if ev["kind"] in ("link_up", "link_down"):
            continue
        detail = {k: v for k, v in ev.items() if k not in ("t_s", "kind")}
        out.append(f"    {ev['t_s']:8.1f}s  {ev['kind']:18s} {detail if detail else ''}")
    link = Counter(e["kind"] for e in events if e["kind"] in ("link_up", "link_down"))
    if link:
        out.append(f"    (plus {link['link_down']} link drops / {link['link_up']} recoveries)")
    return "\n".join(out)


def render_modes(rows: list[dict[str, Any]], dt_s: float) -> str:
    counts = Counter((r["drone_id"], r["mode"]) for r in rows)
    out = ["  time in each mode:"]
    for (did, mode), n in sorted(counts.items()):
        out.append(f"    {did}  {mode:10s} {n * dt_s:7.1f}s")
    return "\n".join(out)


def replay(run_dir: Path, cfg: WorldConfig) -> str:
    rows, events = _load(run_dir)
    return "\n\n".join(
        [render_map(rows, cfg), render_modes(rows, cfg.sim.dt_s), render_timeline(events)]
    )


def _main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="planguard.sim.replay")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--world", type=Path, default=Path("experiments/configs/world.yaml"))
    args = ap.parse_args(argv)
    print(replay(args.run_dir, WorldConfig.load(args.world)))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
