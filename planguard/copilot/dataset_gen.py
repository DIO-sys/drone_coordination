"""Phase 5: natural-language training pairs, generated plan-first.

The direction matters and is the reason this works at all. The obvious way to
build an instruction-to-plan dataset is to write instructions and then author
plans for them -- and then every target is only as good as whoever wrote it.
Here it runs backwards:

1. **sample a plan** from the DSL (``planguard.dsl.sample``),
2. **keep it only if the validator passes it** and it **simulates above a
   success threshold** across seeds,
3. **back-translate** it into the instruction an operator might have given.

So every training target in the file is a plan that has been statically
verified *and* flown. There is no hand-authored target to be wrong.

**The weakness, stated plainly:** back-translated instructions read as
model-ese. They are generated from phrase banks, so they carry the generator's
habits -- consistent clause order, a bounded vocabulary, no ellipsis or
mid-sentence correction. A model fine-tuned on them and scored on them will
look better than it is. That is what ``evals/heldout.yaml`` is for: 150
instructions written by hand, in operator voice, never passed through these
templates, and reported as a separate column (rule 4). Numbers from this
generator and numbers from that file are not comparable and must never be
averaged together.

Rule 11: nothing here imports an LLM dependency. Back-translation is phrase
banks and a seeded ``Generator``, so the whole Phase 5 dataset builds on a
machine with no GPU and no ``llm`` extra.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from planguard.config import WorldConfig
from planguard.dsl.sample import SamplerConfig, canonical_json, plan_id, sample_corpus
from planguard.dsl.schema import FallbackRule, MissionPlan, TaskSpec
from planguard.sim.run import simulate

# --------------------------------------------------------------------------
# phrase banks
# --------------------------------------------------------------------------
# Several surface forms per concept, so the generator does not emit one
# sentence shape 40,000 times. This is variety, not naturalness -- see the
# module docstring.

_SWEEP = (
    "sweep {target}",
    "run a coverage sweep of {target}",
    "get full sensor coverage of {target}",
    "search {target}",
    "survey {target}",
)
_CORRIDOR = (
    "patrol {target}",
    "run a patrol along {target}",
    "fly the length of {target}",
    "cover {target} end to end",
)
_STATION = (
    "hold station over {target}",
    "keep eyes on {target}",
    "watch {target}",
    "maintain overwatch on {target}",
)
_RALLY = (
    "stage at {target}",
    "position at {target}",
    "send someone to {target}",
)
_CLAUSE: dict[str, tuple[str, ...]] = {
    "sweep": _SWEEP,
    "corridor_patrol": _CORRIDOR,
    "station_keep": _STATION,
    "rally": _RALLY,
}

_OPENER = (
    "",
    "I need you to ",
    "Tasking: ",
    "Plan this: ",
    "For this next window, ",
)
_AIRCRAFT = (
    " with {n} aircraft",
    " using {n} aircraft",
    " ({n} aircraft)",
    " and put {n} aircraft on it",
)
_DWELL = (
    " for {d}",
    " and stay for {d}",
    ", holding {d}",
)
_LINK = (
    "keep a link the whole time",
    "that one needs comms up",
    "it has to stay linked",
)
_BUDGET = (
    "The whole thing has to fit in {b}.",
    "You have {b}.",
    "Budget is {b}.",
    "Wrap it all up inside {b}.",
)
_AFTER = (
    " after {dep} is done",
    " once {dep} finishes",
    " but not until {dep} is complete",
)
_NOT_BEFORE = (
    " and do not start it before {t}",
    ", earliest start {t}",
    " starting no sooner than {t}",
)
_FALLBACK_LINK = {
    "continue_unlinked": (
        "If the link drops, carry on without it.",
        "Lost comms is acceptable -- keep working.",
    ),
    "return_to_rally": (
        "If comms go down, pull back to {rally}.",
        "On link loss, recover to {rally}.",
    ),
    "loiter": ("If the link drops, hold where you are.",),
    "hand_off": ("If the link drops, pass the task to someone else.",),
}
_FALLBACK_FUEL = {
    "return_to_rally": (
        "Anyone getting low on fuel goes to {rally}.",
        "Bingo fuel, recover to {rally}.",
    ),
    "loiter": ("Low fuel, just hold.",),
    "hand_off": ("Low fuel, hand the task off.",),
    "continue_unlinked": ("Low on fuel is not a reason to stop.",),
}
#: Sentence joins only. " and " was in here and produced clauses like "keep
#: eyes on the bridge, holding 7 minutes and stage at rally alpha", where the
#: task boundary is genuinely ambiguous to a reader, let alone a model.
_JOIN = (". Then ", ", then ", ". Also ", ". ")


def humanise_feature(name: str) -> str:
    """``north_ridge`` -> ``the north ridge``.

    Deliberate: the model has to map prose back to the exact feature name the
    schema requires. If instructions contained the identifier verbatim the
    task would collapse into string copying and the eval would flatter it.
    """
    words = name.replace("_", " ")
    if name.startswith("rally_"):
        return words  # "rally alpha" is already how anyone would say it
    return f"the {words}"


#: Every duration in a Phase 5 plan is snapped to a multiple of this before
#: back-translation. See ``snap_plan``.
GRID_S = 30


def humanise_duration(seconds: float) -> str:
    """Render a grid-aligned duration the way someone would say it.

    Must be **exactly invertible**: the instruction is the only thing the model
    sees, so if it cannot name the budget precisely then per-field exact match
    is unachievable by construction and the Phase 6 table measures nothing.
    "3.9 minutes" is both unsayable and unrecoverable; "7 and a half minutes"
    is one number.
    """
    s = int(round(seconds))
    if s % 60 == 0:
        mins = s // 60
        return "1 minute" if mins == 1 else f"{mins} minutes"
    if s % 30 == 0:
        half = s // 60
        if half == 0:
            return "30 seconds"
        return "a minute and a half" if half == 1 else f"{half} and a half minutes"
    return f"{s} seconds"  # off-grid: only reachable if snapping was skipped


_NUMBER_WORD = {1: "one", 2: "two", 3: "three", 4: "four"}


def humanise_count(n: int, rng: np.random.Generator) -> str:
    if n in _NUMBER_WORD and rng.random() < 0.5:
        return _NUMBER_WORD[n]
    return str(n)


def _pick(options: tuple[str, ...], rng: np.random.Generator) -> str:
    return options[int(rng.integers(len(options)))]


def _task_clause(task: TaskSpec, plan: MissionPlan, rng: np.random.Generator) -> str:
    clause = _pick(_CLAUSE[task.kind], rng).format(target=humanise_feature(task.target))
    if task.aircraft_count > 1:
        clause += _pick(_AIRCRAFT, rng).format(n=humanise_count(task.aircraft_count, rng))
    if task.kind == "station_keep" and task.dwell_s:
        clause += _pick(_DWELL, rng).format(d=humanise_duration(task.dwell_s))
    if task.depends_on:
        dep = plan.task(sorted(task.depends_on)[0])
        if dep is not None:
            clause += _pick(_AFTER, rng).format(dep=humanise_feature(dep.target))
    if task.start_after_s:
        clause += _pick(_NOT_BEFORE, rng).format(t=humanise_duration(task.start_after_s))
    if task.requires_link:
        clause += "; " + _pick(_LINK, rng)
    return clause


def _fallback_sentence(rule: FallbackRule, rng: np.random.Generator) -> str | None:
    bank = _FALLBACK_LINK if rule.trigger == "link_lost" else _FALLBACK_FUEL
    options = bank.get(rule.action)
    if not options:
        return None
    rally = humanise_feature(rule.rally_point) if rule.rally_point else "base"
    return _pick(options, rng).format(rally=rally)


def _capitalise_sentences(text: str) -> str:
    """Capitalise after every sentence break a join may have introduced."""
    out = list(text)
    start = True
    for i, ch in enumerate(out):
        if start and ch.isalpha():
            out[i] = ch.upper()
            start = False
        elif ch == ".":
            start = True
    return "".join(out)


def back_translate(plan: MissionPlan, rng: np.random.Generator) -> str:
    """The instruction an operator might have given to get ``plan``."""
    clauses = [_task_clause(t, plan, rng) for t in plan.tasks]
    body = clauses[0]
    for clause in clauses[1:]:
        body += _pick(_JOIN, rng) + clause
    text = _pick(_OPENER, rng) + body
    if not text.endswith("."):
        text += "."
    parts = [_capitalise_sentences(text)]

    # The budget is stated every time, deliberately. An instruction that omits
    # it leaves ``time_budget_s`` unrecoverable from the only input the model
    # gets, so the pair would be unscoreable on that field no matter how good
    # the model is -- and a silently unscoreable field inflates nothing and
    # measures nothing. Real tasking often does omit it; that is a gap between
    # this dataset and reality, and it is in ARCHITECTURE section 9 as one.
    parts.append(_pick(_BUDGET, rng).format(b=humanise_duration(plan.time_budget_s)))
    for rule in plan.fallbacks:
        sentence = _fallback_sentence(rule, rng)
        if sentence is not None:
            parts.append(sentence)
    return " ".join(parts)


# --------------------------------------------------------------------------
# the pipeline
# --------------------------------------------------------------------------


def snap_plan(plan: MissionPlan, cfg: WorldConfig) -> MissionPlan | None:
    """Round every duration in ``plan`` onto ``GRID_S``, or give up.

    The sampler draws a budget as ``round(bound * ratio)``, which lands on
    arbitrary values like 234 s. Phase 3 does not care -- a risk label is a
    number either way -- but an instruction has to *say* the budget, and no
    operator says "3.9 minutes". Worse, a model cannot recover 234 from any
    natural phrasing, so every such pair would be unscoreable on the budget
    field however good the model was.

    Dwell is snapped to the nearest grid point and the budget **upwards**, so
    the plan can only become more feasible, never less. It is re-validated
    anyway: snapping dwell up adds work, and nothing here is allowed to assume
    the result still passes.
    """
    from planguard.dsl.validate import makespan_lower_bound_s, validate

    def grid(value: float, up: bool = False) -> float:
        n = value / GRID_S
        import math

        return float(GRID_S * (math.ceil(n) if up else max(1, round(n))))

    tasks = []
    for task in plan.tasks:
        update: dict[str, float | None] = {}
        if task.dwell_s is not None:
            update["dwell_s"] = grid(task.dwell_s)
        if task.start_after_s:
            update["start_after_s"] = grid(task.start_after_s)
        tasks.append(task.model_copy(update=update) if update else task)

    snapped = plan.model_copy(update={"tasks": tasks})
    bound_s, _ = makespan_lower_bound_s(snapped, cfg)
    budget_s = max(grid(plan.time_budget_s, up=True), grid(bound_s * 1.02, up=True))
    snapped = snapped.model_copy(update={"time_budget_s": budget_s})

    if budget_s > cfg.sim.max_duration_s or not validate(snapped, cfg).ok:
        return None
    return snapped


@dataclass(frozen=True)
class Pair:
    """One training example. ``plan_json`` is the target the model must emit."""

    plan_id: str
    instruction: str
    plan_json: str
    #: Fraction of verifier seeds in which the plan flew to success. Kept so a
    #: later run can re-filter at a different threshold without re-simulating.
    success_rate: float
    n_verify_seeds: int


@dataclass
class GenStats:
    """Yield accounting for both filter stages, which Phase 5 must report."""

    sampled: int = 0
    rejected_by_snapping: int = 0
    rejected_by_simulation: int = 0
    accepted: int = 0
    duplicate_instructions: int = 0
    sampler_draws: int = 0
    sampler_rejected_invalid: int = 0
    rejection_codes: dict[str, int] = field(default_factory=dict)

    @property
    def validator_yield(self) -> float:
        """Share of raw draws the static validator let through."""
        if not self.sampler_draws:
            return 0.0
        return self.sampled / self.sampler_draws

    @property
    def simulation_yield(self) -> float:
        """Share of *valid* plans that also flew well enough to keep."""
        considered = self.sampled - self.rejected_by_snapping
        if considered <= 0:
            return 0.0
        return (considered - self.rejected_by_simulation) / considered

    @property
    def overall_yield(self) -> float:
        if not self.sampler_draws:
            return 0.0
        return self.accepted / self.sampler_draws


def verify(plan: MissionPlan, cfg: WorldConfig, seeds: int, threshold: float) -> tuple[bool, float]:
    """Fly ``plan`` across ``seeds`` and report whether it clears ``threshold``.

    The second filter, and the expensive one. A plan can be perfectly valid and
    still be a bad target: it fits the budget arithmetically, needs three of
    four aircraft, and fails whenever one is unavailable. Training on it would
    teach the model to write plans that pass the validator and lose missions,
    which is precisely the failure mode this project exists to measure.
    """
    successes = sum(1 for s in range(seeds) if simulate(plan, cfg, s).success)
    rate = successes / seeds if seeds else 0.0
    return rate >= threshold, rate


def generate_pairs(
    cfg: WorldConfig,
    n_pairs: int,
    seed: int,
    scfg: SamplerConfig | None = None,
    verify_seeds: int = 5,
    threshold: float = 0.8,
    oversample: float = 2.5,
) -> tuple[list[Pair], GenStats]:
    """Sample, verify, back-translate. Returns pairs plus the yield accounting."""
    scfg = scfg or SamplerConfig()
    stats = GenStats()

    # one sampler call, oversampled, so the validator yield is measured on a
    # single well-defined draw rather than accumulated across retries
    want = max(n_pairs, int(n_pairs * oversample))
    plans, sstats = sample_corpus(cfg, scfg, want, seed)
    stats.sampler_draws = sstats.drawn
    stats.sampler_rejected_invalid = sstats.rejected_invalid
    stats.rejection_codes = dict(sstats.rejection_codes)
    stats.sampled = len(plans)

    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(2)[1])
    seen: set[str] = set()
    pairs: list[Pair] = []
    for raw in plans:
        if len(pairs) >= n_pairs:
            break
        plan = snap_plan(raw, cfg)
        if plan is None:
            stats.rejected_by_snapping += 1
            continue
        ok, rate = verify(plan, cfg, verify_seeds, threshold)
        if not ok:
            stats.rejected_by_simulation += 1
            continue
        instruction = back_translate(plan, rng)
        if instruction in seen:
            # the phrase banks are finite; a duplicate instruction with a
            # different target plan is a contradictory training pair
            stats.duplicate_instructions += 1
            continue
        seen.add(instruction)
        pairs.append(
            Pair(
                plan_id=plan_id(plan),
                instruction=instruction,
                plan_json=plan.model_dump_json(exclude_none=True),
                success_rate=rate,
                n_verify_seeds=verify_seeds,
            )
        )
    stats.accepted = len(pairs)
    return pairs, stats


def write_jsonl(path: Path, pairs: list[Pair]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for pair in sorted(pairs, key=lambda p: p.plan_id):
            fh.write(json.dumps(asdict(pair), sort_keys=True) + "\n")


def split_pairs(pairs: list[Pair], test_fraction: float = 0.1) -> dict[str, list[Pair]]:
    """In-distribution train/test split, by plan content (rule 6).

    This is *not* the held-out set. It shares the generator's phrasing habits
    with the training data, so it measures fit, not generalisation to an
    operator. ``evals/heldout.yaml`` is the one that measures that.
    """
    from planguard.risk.harness import split_of

    out: dict[str, list[Pair]] = {"train": [], "test": []}
    for pair in sorted(pairs, key=lambda p: p.plan_id):
        out[split_of(pair.plan_id, test_fraction)].append(pair)
    return out


def canonical_target(pair: Pair) -> str:
    """The pair's plan in the canonical form used for exact-match scoring."""
    return canonical_json(MissionPlan.model_validate_json(pair.plan_json))
