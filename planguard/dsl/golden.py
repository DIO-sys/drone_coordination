"""Golden-corpus runner for the validator (Phase 0 acceptance check).

Each case names the exact set of error codes the validator must produce. A case
expecting ``[]`` must validate cleanly; a case tagged ``schema_error: true``
must be rejected by Pydantic before the validator ever sees it, which is how
the corpus documents the boundary between schema validity and semantic
validity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError as PydanticValidationError

from planguard.config import WorldConfig
from planguard.dsl.schema import MissionPlan
from planguard.dsl.validate import validate


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(Path(path).read_text())
    cases: list[dict[str, Any]] = data["cases"]
    return cases


def run_case(case: dict[str, Any], cfg: WorldConfig) -> tuple[bool, str]:
    """Returns (passed, detail)."""
    expect_schema_error = bool(case.get("schema_error", False))
    try:
        plan = MissionPlan.model_validate(case["plan"])
    except PydanticValidationError as exc:
        if expect_schema_error:
            return True, "rejected by schema as expected"
        first = exc.errors()[0]
        return False, f"unexpected schema error at {first['loc']}: {first['msg']}"
    if expect_schema_error:
        return False, "expected a schema error but the plan parsed cleanly"

    report = validate(plan, cfg)
    want = sorted(case.get("expect", []))
    got = sorted(report.codes)
    if want == got:
        return True, "ok" if not got else f"produced {got}"
    missing = [c for c in want if c not in got]
    extra = [c for c in got if c not in want]
    bits = []
    if missing:
        bits.append(f"missing {missing}")
    if extra:
        bits.append(f"unexpected {extra}")
    return False, "; ".join(bits)


def run_corpus(path: str | Path, cfg: WorldConfig) -> int:
    cases = load_cases(path)
    failures: list[tuple[str, str]] = []
    for case in cases:
        passed, detail = run_case(case, cfg)
        if not passed:
            failures.append((case["name"], detail))
    valid = sum(1 for c in cases if not c.get("expect") and not c.get("schema_error"))
    print(f"{len(cases)} cases ({valid} valid, {len(cases) - valid} rejected)")
    for name, detail in failures:
        print(f"  FAIL {name}: {detail}")
    if failures:
        print(f"{len(failures)} failing")
        return 1
    print("all cases classified correctly")
    return 0
