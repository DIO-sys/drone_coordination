"""Validator tests: the golden corpus, the error contract, and the schedule model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from planguard.config import WorldConfig
from planguard.dsl.errors import BLOCKING_CODES, ErrorCode
from planguard.dsl.golden import load_cases, run_case
from planguard.dsl.schema import MissionPlan
from planguard.dsl.validate import (
    CHECK_NAMES,
    NEEDS_SCHEDULE,
    ScheduledTask,
    estimate_schedule,
    peak_concurrency,
    validate,
)

REPO = Path(__file__).resolve().parent.parent
CASES = load_cases(REPO / "tests" / "golden" / "cases.yaml")
ALL_CODES: tuple[str, ...] = tuple(ErrorCode.__args__)  # type: ignore[attr-defined]


# ----------------------------------------------------------- golden corpus ---


@pytest.mark.parametrize("case", CASES, ids=lambda c: str(c["name"]))
def test_golden_case(case: dict[str, Any], cfg: WorldConfig) -> None:
    passed, detail = run_case(case, cfg)
    assert passed, detail


def test_corpus_is_big_enough_and_balanced() -> None:
    """Phase 0 acceptance: ~40 hand-written good and bad plans."""
    valid = [c for c in CASES if not c.get("expect") and not c.get("schema_error")]
    assert len(CASES) >= 40
    assert len(valid) >= 10


def test_corpus_exercises_every_error_code() -> None:
    """A code nobody tests is a code the repair loop will meet for the first time
    in production."""
    covered = {code for case in CASES for code in case.get("expect", [])}
    assert set(ALL_CODES) - covered == set()


def test_corpus_case_names_are_unique() -> None:
    names = [c["name"] for c in CASES]
    assert len(names) == len(set(names))


# ----------------------------------------------------------- error contract ---


FAILING_CASES = [c for c in CASES if c.get("expect")]


@pytest.mark.parametrize("case", FAILING_CASES, ids=lambda c: str(c["name"]))
def test_every_error_carries_a_usable_repair_message(
    case: dict[str, Any], cfg: WorldConfig
) -> None:
    """Rule 12: a stable code, a JSON path to the offending field, and a hint."""
    report = validate(MissionPlan.model_validate(case["plan"]), cfg)
    for err in report.errors:
        assert err.code in ALL_CODES
        assert err.path and not err.path.startswith("."), err
        assert err.path.split("[")[0] in {"tasks", "fallbacks", "time_budget_s"}, err
        # Messages and hints are prompt text in Phase 7, so they must be
        # substantial and consistently styled (no trailing full stop).
        assert len(err.message.strip()) >= 20, err
        assert not err.message.rstrip().endswith("."), err
        assert len(err.hint.strip()) >= 10, err
        assert not err.hint.rstrip().endswith("."), err


def test_feedback_mentions_every_error(cfg: WorldConfig) -> None:
    plan = MissionPlan.model_validate(
        next(c for c in CASES if c["name"] == "several_independent_errors")["plan"]
    )
    report = validate(plan, cfg)
    feedback = report.as_feedback()
    assert "failed 3 static check(s)" in feedback
    for code in report.codes:
        assert code in feedback
    assert "return the corrected plan" in feedback


def test_clean_report_says_so(cfg: WorldConfig) -> None:
    plan = MissionPlan.model_validate(json.loads((REPO / "examples" / "ridge.json").read_text()))
    report = validate(plan, cfg)
    assert report.ok, report.as_feedback()
    assert report.codes == []
    assert "passed every static check" in report.as_feedback()


def test_unknown_map_feature_hint_lists_real_features(cfg: WorldConfig) -> None:
    plan = MissionPlan.model_validate(
        {
            "name": "x",
            "time_budget_s": 600,
            "tasks": [{"id": "t", "kind": "sweep", "target": "west_ridge"}],
        }
    )
    err = validate(plan, cfg).errors[0]
    assert err.code == "UNKNOWN_MAP_FEATURE"
    assert "north_ridge" in err.hint


def test_wrong_feature_kind_hint_lists_only_the_right_kind(cfg: WorldConfig) -> None:
    plan = MissionPlan.model_validate(
        {
            "name": "x",
            "time_budget_s": 600,
            "tasks": [{"id": "t", "kind": "sweep", "target": "bridge"}],
        }
    )
    err = validate(plan, cfg).errors[0]
    assert err.code == "WRONG_FEATURE_KIND"
    assert "north_ridge" in err.hint
    assert "bridge" not in err.hint


# -------------------------------------------------------- blocking / skips ---


def test_blocking_error_skips_the_schedule_checks(cfg: WorldConfig) -> None:
    """A cycle hides a 10 s budget against ~800 s of work. The time check must be
    reported as skipped, never as passed."""
    plan = MissionPlan.model_validate(
        next(c for c in CASES if c["name"] == "cycle_blocks_the_schedule_checks")["plan"]
    )
    report = validate(plan, cfg)
    assert report.codes == ["DEPENDENCY_CYCLE"]
    assert report.skipped_checks == ["fleet_capacity", "time_feasibility"]


def test_clean_plan_skips_nothing(cfg: WorldConfig) -> None:
    plan = MissionPlan.model_validate(json.loads((REPO / "examples" / "ridge.json").read_text()))
    assert validate(plan, cfg).skipped_checks == []


def test_skippable_checks_are_declared_consistently() -> None:
    assert set(CHECK_NAMES) >= NEEDS_SCHEDULE
    assert len(CHECK_NAMES) == 9


def test_every_blocking_code_is_a_real_code() -> None:
    assert set(ALL_CODES) >= BLOCKING_CODES


# ------------------------------------------------------------- schedule ------


def test_peak_concurrency_processes_ends_before_starts() -> None:
    """A handoff at the same instant is not a peak of two."""
    handoff = [ScheduledTask("a", 0.0, 10.0, 1), ScheduledTask("b", 10.0, 20.0, 1)]
    assert peak_concurrency(handoff)[0] == 1
    overlap = [ScheduledTask("a", 0.0, 10.0, 1), ScheduledTask("b", 9.9, 20.0, 1)]
    assert peak_concurrency(overlap)[0] == 2


def test_peak_concurrency_sums_aircraft_counts() -> None:
    tasks = [ScheduledTask("a", 0.0, 10.0, 2), ScheduledTask("b", 1.0, 5.0, 3)]
    peak, at_s = peak_concurrency(tasks)
    assert peak == 5
    assert at_s == pytest.approx(1.0)


def test_peak_concurrency_of_nothing_is_zero() -> None:
    assert peak_concurrency([]) == (0, 0.0)


def test_schedule_respects_dependencies(cfg: WorldConfig) -> None:
    plan = MissionPlan.model_validate(
        next(c for c in CASES if c["name"] == "sequential_dependency")["plan"]
    )
    schedule = {s.task_id: s for s in estimate_schedule(plan, cfg)}
    assert schedule["sweep_east"].start_s == pytest.approx(schedule["sweep_valley"].end_s)


def test_schedule_honours_start_after(cfg: WorldConfig) -> None:
    plan = MissionPlan.model_validate(
        next(c for c in CASES if c["name"] == "start_after_within_budget")["plan"]
    )
    assert estimate_schedule(plan, cfg)[0].start_s == pytest.approx(100.0)


def test_schedule_order_is_deterministic(cfg: WorldConfig) -> None:
    """Rule 5: sorted iteration everywhere, so repeated calls are identical."""
    plan = MissionPlan.model_validate(
        next(c for c in CASES if c["name"] == "peak_concurrency_exceeded")["plan"]
    )
    runs = [estimate_schedule(plan, cfg) for _ in range(5)]
    assert all(r == runs[0] for r in runs)


def test_more_aircraft_shortens_coverage_but_not_dwell(cfg: WorldConfig) -> None:
    def end(kind: str, target: str, ac: int, dwell: float | None) -> float:
        task: dict[str, Any] = {"id": "t", "kind": kind, "target": target, "aircraft_count": ac}
        if dwell is not None:
            task["dwell_s"] = dwell
        plan = MissionPlan.model_validate({"name": "x", "time_budget_s": 100000, "tasks": [task]})
        return estimate_schedule(plan, cfg)[0].end_s

    assert end("sweep", "north_ridge", 2, None) < end("sweep", "north_ridge", 1, None)
    assert end("station_keep", "bridge", 2, 600.0) == pytest.approx(
        end("station_keep", "bridge", 1, 600.0)
    )


# --------------------------------------------------------- property test -----

FEATURES = ["base", "bridge", "rally_alpha", "north_ridge", "south_valley", "river_corridor"]
TASK_IDS = ["t0", "t1", "t2", "t3"]

task_st = st.fixed_dictionaries(
    {
        "id": st.sampled_from(TASK_IDS),
        "kind": st.sampled_from(["sweep", "station_keep", "corridor_patrol", "rally"]),
        "target": st.sampled_from([*FEATURES, "does_not_exist"]),
        "aircraft_count": st.integers(min_value=1, max_value=6),
        "dwell_s": st.one_of(st.none(), st.floats(min_value=1.0, max_value=900.0)),
        "depends_on": st.lists(st.sampled_from([*TASK_IDS, "ghost"]), max_size=3),
        "requires_link": st.booleans(),
        "start_after_s": st.one_of(st.none(), st.floats(min_value=0.0, max_value=2000.0)),
    }
)
plan_st = st.fixed_dictionaries(
    {
        "name": st.just("fuzz"),
        "time_budget_s": st.floats(min_value=1.0, max_value=5000.0),
        "tasks": st.lists(task_st, min_size=1, max_size=4),
        "fallbacks": st.lists(
            st.fixed_dictionaries(
                {
                    "trigger": st.sampled_from(["link_lost", "low_fuel", "task_timeout"]),
                    "action": st.sampled_from(
                        ["return_to_rally", "hand_off", "loiter", "continue_unlinked"]
                    ),
                    "applies_to": st.lists(st.sampled_from([*TASK_IDS, "ghost"]), max_size=2),
                    "rally_point": st.one_of(st.none(), st.sampled_from([*FEATURES, "nope"])),
                }
            ),
            max_size=2,
        ),
    }
)


@settings(max_examples=400, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(raw=plan_st)
def test_validator_never_raises_on_any_schema_valid_plan(
    raw: dict[str, Any], cfg: WorldConfig
) -> None:
    """The validator is the one component the agent loop calls on untrusted model
    output, so it must *reject* bad plans, never crash on them.

    This is the test that would have caught a dangling depends_on reaching the
    scheduler and raising instead of being reported.
    """
    plan = MissionPlan.model_validate(raw)
    report = validate(plan, cfg)
    assert isinstance(report.ok, bool)
    for err in report.errors:
        assert err.code in ALL_CODES
        assert err.hint
    if not report.ok and any(c in BLOCKING_CODES for c in report.codes):
        assert set(report.skipped_checks) <= NEEDS_SCHEDULE
