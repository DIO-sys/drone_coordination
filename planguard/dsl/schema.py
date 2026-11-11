"""The mission DSL — and the fine-tune target.

This schema is what the model emits. Two properties matter:

* ``extra="forbid"`` everywhere, so a hallucinated field is a *schema* error
  rather than silently ignored data.
* The semantics a JSON Schema cannot express — which feature kind a task kind
  must target, when ``dwell_s`` is required, fleet arithmetic, dependency
  acyclicity — live in ``validate.py``. That split is the whole point of the
  project: constrained decoding gets the first part for free and can never get
  the second.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

TaskKind = Literal["sweep", "station_keep", "corridor_patrol", "rally"]
FallbackTrigger = Literal["link_lost", "low_fuel", "task_timeout"]
FallbackAction = Literal["return_to_rally", "hand_off", "loiter", "continue_unlinked"]
MissionConstraint = Literal["stay_within_map", "max_two_hops", "single_pass"]


class TaskSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    kind: TaskKind
    #: Name of a map feature. The required feature *kind* depends on this
    #: task's kind; see config.REQUIRED_FEATURE_KIND.
    target: str = Field(min_length=1)
    aircraft_count: int = Field(default=1, ge=1)
    priority: int = Field(default=2, ge=1, le=3)
    #: Required for station_keep, meaningless otherwise.
    dwell_s: float | None = Field(default=None, gt=0)
    depends_on: list[str] = Field(default_factory=list)
    requires_link: bool = False
    start_after_s: float | None = Field(default=None, ge=0)


class FallbackRule(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    trigger: FallbackTrigger
    action: FallbackAction
    #: Task ids this rule covers. Empty means every task in the plan.
    applies_to: list[str] = Field(default_factory=list)
    #: Required when action is return_to_rally.
    rally_point: str | None = None

    def covers(self, task_id: str) -> bool:
        return not self.applies_to or task_id in self.applies_to


class MissionPlan(BaseModel):
    """Team-level intent. Per rule 7 it never names individual aircraft —
    CBBA assigns tail numbers at execution time."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    time_budget_s: float = Field(gt=0)
    tasks: list[TaskSpec] = Field(min_length=1)
    fallbacks: list[FallbackRule] = Field(default_factory=list)
    constraints: list[MissionConstraint] = Field(default_factory=list)
    rationale: str | None = None

    def task(self, task_id: str) -> TaskSpec | None:
        for t in self.tasks:
            if t.id == task_id:
                return t
        return None

    @property
    def task_ids(self) -> list[str]:
        return [t.id for t in self.tasks]
