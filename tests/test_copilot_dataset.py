"""Phase 5: NL dataset generation and the hand-written held-out set.

The contract worth pinning: every target is verified, every instruction can be
inverted back to its plan, and the held-out set is genuinely not the generator's
output wearing a hat.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from planguard.config import WorldConfig
from planguard.copilot.dataset_gen import (
    GRID_S,
    back_translate,
    generate_pairs,
    humanise_duration,
    humanise_feature,
    snap_plan,
    split_pairs,
    write_jsonl,
)
from planguard.copilot.heldout import (
    REGISTERS,
    drift_report,
    load_heldout,
    plan_shape_report,
)
from planguard.dsl.sample import SamplerConfig, sample_corpus
from planguard.dsl.schema import MissionPlan
from planguard.dsl.validate import validate

REPO = Path(__file__).resolve().parent.parent
HELDOUT = REPO / "evals" / "heldout.yaml"


@pytest.fixture(scope="module")
def heldout() -> list:
    return load_heldout(HELDOUT)


@pytest.fixture(scope="module")
def generated(cfg: WorldConfig) -> list[str]:
    plans, _ = sample_corpus(cfg, SamplerConfig(), 300, seed=77)
    rng = np.random.default_rng(5)
    return [back_translate(p, rng) for p in plans]


# ------------------------------------------------------- back-translation ---


def test_humanise_feature_does_not_leak_the_identifier() -> None:
    """If the instruction contained ``north_ridge`` verbatim the task would
    collapse into string copying and the Phase 6 table would flatter itself."""
    assert humanise_feature("north_ridge") == "the north ridge"
    assert "_" not in humanise_feature("highway_corridor")
    assert humanise_feature("rally_alpha") == "rally alpha"


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (60, "1 minute"),
        (600, "10 minutes"),
        (1800, "30 minutes"),
        (30, "30 seconds"),
        (90, "a minute and a half"),
        (450, "7 and a half minutes"),
    ],
)
def test_humanise_duration_is_sayable(seconds: int, expected: str) -> None:
    assert humanise_duration(seconds) == expected


def test_every_grid_duration_renders_distinctly() -> None:
    """The rendering must be injective, or two different budgets produce the
    same instruction and the pair becomes unscoreable on that field."""
    seen: dict[str, int] = {}
    for s in range(GRID_S, 3600 + GRID_S, GRID_S):
        text = humanise_duration(s)
        assert text not in seen, f"{s}s and {seen.get(text)}s both render as {text!r}"
        seen[text] = s


def test_snap_plan_puts_every_duration_on_the_grid(cfg: WorldConfig) -> None:
    plans, _ = sample_corpus(cfg, SamplerConfig(), 60, seed=31)
    snapped = [p for p in (snap_plan(raw, cfg) for raw in plans) if p is not None]
    assert snapped, "snapping rejected the entire corpus"
    for plan in snapped:
        assert plan.time_budget_s % GRID_S == 0
        for task in plan.tasks:
            if task.dwell_s is not None:
                assert task.dwell_s % GRID_S == 0
            if task.start_after_s:
                assert task.start_after_s % GRID_S == 0


def test_snap_plan_never_returns_an_invalid_plan(cfg: WorldConfig) -> None:
    """Snapping dwell up adds work, so the result is re-validated rather than
    assumed still feasible."""
    plans, _ = sample_corpus(cfg, SamplerConfig(), 60, seed=32)
    for raw in plans:
        snapped = snap_plan(raw, cfg)
        if snapped is not None:
            assert validate(snapped, cfg).ok


def test_snapping_only_ever_loosens_the_budget(cfg: WorldConfig) -> None:
    plans, _ = sample_corpus(cfg, SamplerConfig(), 40, seed=33)
    for raw in plans:
        snapped = snap_plan(raw, cfg)
        if snapped is not None:
            assert snapped.time_budget_s >= raw.time_budget_s


def test_instruction_always_states_the_budget(cfg: WorldConfig) -> None:
    """An instruction that omits the budget leaves ``time_budget_s``
    unrecoverable from the model's only input."""
    plans, _ = sample_corpus(cfg, SamplerConfig(), 40, seed=34)
    rng = np.random.default_rng(1)
    for raw in plans:
        plan = snap_plan(raw, cfg)
        if plan is None:
            continue
        text = back_translate(plan, rng)
        assert humanise_duration(plan.time_budget_s) in text


def test_instruction_mentions_every_task(cfg: WorldConfig) -> None:
    plans, _ = sample_corpus(cfg, SamplerConfig(), 40, seed=35)
    rng = np.random.default_rng(2)
    for raw in plans:
        plan = snap_plan(raw, cfg)
        if plan is None:
            continue
        text = back_translate(plan, rng).lower()
        for task in plan.tasks:
            assert humanise_feature(task.target).lower() in text

        for task in plan.tasks:
            if task.aircraft_count > 1:
                assert str(task.aircraft_count) in text or any(
                    w in text for w in ("one", "two", "three", "four")
                )
            if task.dwell_s:
                assert humanise_duration(task.dwell_s) in text


def test_back_translation_is_deterministic_given_a_generator(cfg: WorldConfig) -> None:
    plans, _ = sample_corpus(cfg, SamplerConfig(), 5, seed=36)
    a = [back_translate(p, np.random.default_rng(7)) for p in plans]
    b = [back_translate(p, np.random.default_rng(7)) for p in plans]
    assert a == b


def test_instructions_have_no_double_spaces(generated: list[str]) -> None:
    for text in generated:
        assert "  " not in text, repr(text)
        assert " ." not in text and " ;" not in text, repr(text)


def test_sentences_start_capitalised(generated: list[str]) -> None:
    for text in generated:
        for sentence in (s.strip() for s in text.split(". ")):
            if sentence and sentence[0].isalpha():
                assert sentence[0].isupper(), repr(sentence)


# -------------------------------------------------------------- pipeline ---


def test_generated_pairs_are_verified(cfg: WorldConfig) -> None:
    """Every target passes the validator *and* flew above threshold. That is
    the whole reason the dataset is built plan-first."""
    pairs, stats = generate_pairs(cfg, 8, seed=12, verify_seeds=3, threshold=0.66)
    assert pairs
    for pair in pairs:
        plan = MissionPlan.model_validate_json(pair.plan_json)
        assert validate(plan, cfg).ok
        assert pair.success_rate >= 0.66
    assert stats.accepted == len(pairs)


def test_yield_rates_are_reported_and_consistent(cfg: WorldConfig) -> None:
    """Phase 5's acceptance check is that the filter yield is reported, so the
    arithmetic behind it has to hold."""
    _, stats = generate_pairs(cfg, 8, seed=13, verify_seeds=3, threshold=0.66)
    assert 0.0 < stats.validator_yield <= 1.0
    assert 0.0 < stats.simulation_yield <= 1.0
    assert stats.overall_yield <= stats.validator_yield
    assert stats.sampler_draws >= stats.sampled >= stats.accepted


def test_no_two_pairs_share_an_instruction(cfg: WorldConfig) -> None:
    """A duplicate instruction with a different target plan is a contradictory
    training example."""
    pairs, _ = generate_pairs(cfg, 20, seed=14, verify_seeds=2, threshold=0.5)
    texts = [p.instruction for p in pairs]
    assert len(set(texts)) == len(texts)


def test_split_keeps_plans_on_one_side(cfg: WorldConfig) -> None:
    pairs, _ = generate_pairs(cfg, 20, seed=15, verify_seeds=2, threshold=0.5)
    split = split_pairs(pairs, test_fraction=0.25)
    train_ids = {p.plan_id for p in split["train"]}
    test_ids = {p.plan_id for p in split["test"]}
    assert not train_ids & test_ids
    assert len(split["train"]) + len(split["test"]) == len(pairs)


def test_jsonl_round_trips(cfg: WorldConfig, tmp_path: Path) -> None:
    pairs, _ = generate_pairs(cfg, 6, seed=16, verify_seeds=2, threshold=0.5)
    path = tmp_path / "pairs.jsonl"
    write_jsonl(path, pairs)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == len(pairs)
    assert [r["plan_id"] for r in rows] == sorted(r["plan_id"] for r in rows)
    for row in rows:
        assert validate(MissionPlan.model_validate_json(row["plan_json"]), cfg).ok


# --------------------------------------------------------------- held-out ---


def test_heldout_set_is_the_documented_size(heldout: list) -> None:
    """CLAUDE.md asks for 150-300 items."""
    assert 150 <= len(heldout) <= 300


def test_every_heldout_reference_plan_validates(heldout: list, cfg: WorldConfig) -> None:
    """A held-out item whose reference plan the validator rejects is an item
    no model could ever score on."""
    for item in heldout:
        report = validate(item.plan, cfg)
        assert report.ok, f"{item.item_id}:\n{report.as_feedback()}"


def test_heldout_ids_are_unique(heldout: list) -> None:
    ids = [i.item_id for i in heldout]
    assert len(set(ids)) == len(ids)


def test_heldout_registers_are_known_and_spread(heldout: list) -> None:
    counts = {r: sum(1 for i in heldout if i.register == r) for r in REGISTERS}
    for item in heldout:
        assert item.register in REGISTERS, item.item_id
    # no register may be a token presence
    for register, n in counts.items():
        assert n >= 15, f"only {n} items in register {register!r}"


def test_heldout_covers_the_plan_shapes_the_generator_produces(heldout: list) -> None:
    """Varied prose over uniformly trivial plans would test nothing."""
    shape = plan_shape_report(heldout)
    assert shape["max_tasks"] >= 4
    assert shape["frac_multi_task"] >= 0.3
    assert shape["frac_with_dependency"] >= 0.1
    assert shape["frac_with_link"] >= 0.1
    assert shape["frac_multi_aircraft"] >= 0.2
    assert shape["frac_with_start_after"] >= 0.05
    for kind in ("sweep", "station_keep", "corridor_patrol", "rally"):
        assert shape[f"tasks_{kind}"] >= 10


def test_heldout_does_not_echo_the_templates(heldout: list, generated: list[str]) -> None:
    """Rule 4 made measurable.

    The same author wrote the phrase banks and these instructions, so
    "independent of the templates" needs a number rather than a promise. If
    the hand-written set were quietly reproducing the generator's phrasing,
    reporting the two separately would be theatre.
    """
    report = drift_report([i.instruction for i in heldout], generated)
    assert report["ngram_echo_n4"] < 0.15, report["ngram_echo_n4"]
    assert report["vocab_unique_to_heldout"] > 0.4, report["vocab_unique_to_heldout"]


def test_no_heldout_instruction_contains_a_schema_identifier(heldout: list) -> None:
    """The instruction is prose. A model should not be able to grep it for the
    answer."""
    for item in heldout:
        text = item.instruction.lower()
        assert "_" not in text.replace("\n", " ") or "rally_" not in text, item.item_id
        for kind in ("corridor_patrol", "station_keep", "requires_link", "time_budget_s"):
            assert kind not in text, f"{item.item_id} leaks {kind!r}"


def test_heldout_instructions_vary_in_length(heldout: list) -> None:
    """A set written to one shape would test one shape.

    Asserted as a spread rather than a count of distinct values, because the
    count saturates: 150 items can only land on so many word-lengths, and the
    first version of this test failed on 40-distinct-of-150 purely because the
    threshold was picked before the data existed.
    """
    import statistics

    lengths = [len(i.instruction.split()) for i in heldout]
    mean = statistics.mean(lengths)
    assert min(lengths) <= 8, f"nothing terse: shortest is {min(lengths)} words"
    assert max(lengths) >= 60, f"nothing long: longest is {max(lengths)} words"
    # coefficient of variation: spread relative to the mean, so it does not
    # depend on how wordy the set happens to be overall
    assert statistics.stdev(lengths) / mean > 0.5, "lengths cluster too tightly"
