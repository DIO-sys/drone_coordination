"""The validator's output contract.

Per rule 12, every rejection is machine-readable: a stable ``code``, a JSON
path to the offending field, and a ``hint``. The Phase 7 repair loop feeds
these straight back to the model, so the wording is part of the contract and
is covered by tests.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

ErrorCode = Literal[
    "DUPLICATE_TASK_ID",
    "UNKNOWN_MAP_FEATURE",
    "WRONG_FEATURE_KIND",
    "MISSING_REQUIRED_FIELD",
    "MISSING_RALLY_POINT",
    "UNKNOWN_TASK_REF",
    "SELF_DEPENDENCY",
    "DEPENDENCY_CYCLE",
    "FLEET_OVERSUBSCRIBED",
    "PEAK_CONCURRENCY_EXCEEDED",
    "TIME_BUDGET_EXCEEDED",
    "START_AFTER_BEYOND_BUDGET",
    "MISSING_LINK_FALLBACK",
    "LINK_RANGE_UNREACHABLE",
    "UNKNOWN_FALLBACK_TARGET",
]

#: Codes that make the plan unschedulable, so the capacity and time checks
#: cannot run and are reported as skipped rather than passing vacuously.
BLOCKING_CODES: frozenset[str] = frozenset(
    {
        "UNKNOWN_MAP_FEATURE",
        "WRONG_FEATURE_KIND",
        "UNKNOWN_TASK_REF",
        "SELF_DEPENDENCY",
        "DEPENDENCY_CYCLE",
    }
)


class ValidationError(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: ErrorCode
    path: str
    message: str
    hint: str

    def __str__(self) -> str:
        return f"[{self.code}] {self.path}: {self.message} ({self.hint})"


class ValidationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    errors: list[ValidationError] = []
    skipped_checks: list[str] = []

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def codes(self) -> list[str]:
        return [e.code for e in self.errors]

    def as_feedback(self) -> str:
        """Exactly what the repair node sends back to the model."""
        if self.ok:
            return "The plan passed every static check."
        lines = [f"The plan failed {len(self.errors)} static check(s):"]
        lines += [f"  - {e}" for e in self.errors]
        lines.append("Fix every item above and return the corrected plan.")
        return "\n".join(lines)
