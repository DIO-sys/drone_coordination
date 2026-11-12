# PlanGuard

Verifier-in-the-loop mission planning for drone teams. An operator states intent in natural language; a fine-tuned small model emits a typed mission plan; a static validator and a simulator verify it; an XGBoost surrogate predicts failure risk cheaply enough to screen candidates; a LangGraph loop repairs failures from specific machine-generated errors; a human approves before anything executes.

Full design: the plan at `~/.claude/plans/read-through-this-understand-parsed-dragon.md`.

## Why this project exists

An LLM that authors mission plans for aircraft is only acceptable if every plan is machine-verified before a human ever sees it. This project builds the verification loop, not just the generator.

The claim that organizes the whole design:

> Schema-constrained decoding takes any competent base model to ~100% *schema-valid* JSON. It cannot produce referential integrity, correct aircraft arithmetic, or an acyclic dependency graph — those require understanding the domain.

So the headline metrics are **validator-pass rate** and **simulated mission success**, never JSON validity. Fine-tuning has something real to win here, and the evaluation is built to let it lose: if a constrained-decoding base model closes the gap, that is a negative result and gets published as one.

## Scope

- Missions are surveillance coverage, search, station-keeping and relay positioning. No weapons, targeting, or strike logic anywhere in the codebase.
- Link availability is modeled abstractly (range-gated two-state Markov process). RF propagation fidelity is explicitly out of scope; `sim/link.py` stays behind a narrow interface so a real channel model could be swapped in later.
- This is a research simulation, not flight software.

## Layers

1. `planguard/dsl` — the mission DSL: Pydantic schema (the fine-tune target), static validator, py_trees compiler, plan sampler
2. `planguard/sim` — world, map features, kinematics, task semantics, stochastic link and wind, scenario randomization, logging, replay
3. `planguard/coord` — aircraft allocation. `cbba.py` is the real allocator; `allocate.py` dispatches and keeps the greedy placeholder so the two can be compared on identical seeds
4. `planguard/risk` — plan-time features, XGBoost failure-risk surrogate, SHAP, calibration
5. `planguard/copilot` — dataset generation, QLoRA fine-tune, constrained decoding, LangChain tools, LangGraph approval graph
6. `planguard/debrief` — mission scoring, experiment figures
7. `planguard/api` + `dashboard/` — FastAPI WebSocket stream, plan diff, tree view, risk panel, live map

## Non-negotiable rules

1. **The LLM never touches the execution loop.** It proposes a plan; the validator, the simulator and a human gate it. No plan executes without an approval event logged in `events.jsonl`, and a test must prove it.
2. **Risk features must be plan-time observable.** No realized outage timings, no simulation outcomes, nothing the agent would not have before the plan runs.
3. **Never evaluate the fine-tune on schema validity alone.** Validator-pass rate, simulated mission success and repair iterations are the metrics. Schema validity is reported but is never the claim.
4. **The held-out evaluation set is hand-written**, never template-generated or back-translated. Report it separately from the in-distribution test split.
5. **Determinism.** Same seed + same config = identical run, bit for bit. `numpy.random.Generator` passed explicitly, per-entity streams via `SeedSequence.spawn`, sorted iteration over every collection. Never the global RNG.
6. **Split by scenario seed and by plan.** Never a random row shuffle — the same plan must not appear in both train and test of the risk model.
7. **CBBA owns aircraft assignment.** The plan expresses team-level intent; it never names tail numbers.
8. **Fixed timestep** `dt_s = 0.1`. Anything slower is scheduled by tick index. Nothing uses wall-clock time inside the sim.
9. **Units in names.** `dist_m`, `speed_mps`, `t_s`, `budget_s`, `area_m2`.
10. **Config is YAML into Pydantic.** Derived values are computed in validators, never written into YAML. No magic numbers in code.
11. **LLM deps are an optional `llm` dependency group.** `uv sync` without it must still run the sim, CBBA, the validator and the risk model. Nothing outside `planguard/copilot/` may import from it, so a reviewer with no GPU can reproduce everything but Phase 6.
12. **Every validator rejection is machine-readable.** A `ValidationError` carries a stable `code`, a JSON path to the offending field, and a `hint`. The repair loop is only as good as these messages, so they are part of the contract and are tested.

## Commands

```bash
uv sync                                   # add --extra llm for phase 6
uv run pytest
uv run ruff check . && uv run mypy planguard
uv run python -m planguard.dsl.validate tests/golden/cases.yaml      # phase 0 acceptance
uv run python -m planguard.dsl.validate examples/ridge.json
uv run python -m planguard.sim.run --plan examples/ridge.json --seed 7 --out runs/demo
uv run python -m planguard.sim.replay runs/demo                     # text map + timeline
```

## Build order

Work one phase at a time. Each phase ends with passing tests and its acceptance check.

| Phase | Deliverable | Acceptance check |
|---|---|---|
| **0** | DSL schema + static validator + golden corpus | **Done.** 9 checks implemented; 50 hand-written good/bad plans classified correctly; every rejection carries a code, a JSON path and a hint |
| **1** | Sim: world, map features, kinematics, task semantics, stochastic link + wind, scenario randomization, logging, replay viewer | **Done.** Seeded run reproducible bit-for-bit across two processes; the demo plan compiles to py_trees and executes to completion; outcome varies across seeds, and varies with *execution risk* rather than only aircraft count when the budget is tight |
| **2** | CBBA allocation + end-to-end execution + mission metrics | **Done.** Consensus converges conflict-free under fuzzed fleet placements and plan shapes; mission metrics reconstructed from logs alone and asserted equal to the live result |
| **3** | Plan sampler + batch harness -> risk dataset | **Done.** 2,000 plans x 20 seeds = 40,000 missions in 49 min on 16 cores; sampler yield 88.7%; budget band calibrated so 1,034/2,000 plans carry a label strictly between 0 and 1 |
| **4** | XGBoost risk surrogate + SHAP + calibration | **Done.** MAE 0.137 vs 0.261 for the same model on `budget_ratio` alone; AUC 0.927; ECE 0.038 with the calibration table reported; screening 149,767x faster than simulating |
| **5** | NL dataset: plan-first sampling, verifier filter, back-translation, hand-written held-out set | **Done.** Validator yield 84-89%, simulation yield 92%; 150 hand-written held-out items, all validating; 2.7% 4-gram echo against the generator |
| **6** | QLoRA fine-tune + constrained decoding + five-arm evaluation | Headline table: base zero-shot / base few-shot / base+constrained / fine-tuned / fine-tuned+constrained, scored on schema-valid %, **validator-pass %**, per-field exact match, task-level F1, **simulated mission success**, repair iterations, prompt tokens, p50/p95 latency. Needs a GPU |
| **7** | LangChain tools + LangGraph graph with approval interrupt + repair loop | End-to-end generate -> validate -> risk screen -> simulate -> repair (bounded) -> approve -> commit; approval-gate test from rule 1 passes |
| **8** | FastAPI + dashboard | Plan diff, behavior-tree render, risk panel with SHAP, live map, approve button. Timeboxed |
| **9** | `RESULTS.md` + demo GIF | Headline table, E3 ablation plot, method in a paragraph, nulls stated plainly, limitations section |

**Why this order:** the validator before everything, because the dataset filter, the repair loop and the headline metric all depend on it. The sim before the risk model, because the risk model trains on simulated outcomes. The risk model before the fine-tune, because the agent graph uses it to screen. The fine-tune before the graph, because the graph orchestrates it.

## Experiments

- **E1** five-arm generation quality (Phase 6) — the headline.
- **E2** surrogate calibration, SHAP, screening speedup (Phase 4).
- **E3** verification-stage ablation (Phase 7): none / schema only / schema+validator / schema+validator+risk+sim, scored on end-to-end approved-plan mission success. **This is what makes the project a systems result rather than an LLM demo.**
- **E4** repair convergence (Phase 7): iterations to validity, fraction unrecoverable, which validator errors the model can and cannot self-repair.

## Do not build

- **RAG / vector index over logs.** A deterministic query over `events.jsonl` is better, faster and more honest.
- **An RF channel model.** Out of scope by decision; `RESULTS.md` says so plainly. Do not claim comms fidelity anywhere.
- **Collision avoidance or airspace deconfliction.** Named as a limitation, not simulated.
- **Multi-agent LLM negotiation.** Violates rule 1.

## Documentation (update these at the end of every phase)

Three documents are kept current as the project grows. Updating them is part of a phase's
definition of done, not a separate chore — a phase is not finished until they reflect it.

| File | Audience | What goes in it |
|---|---|---|
| `README.md` | someone finding the repo | what it is, the claim, quickstart, the status table (mark the phase done), current test counts |
| `ARCHITECTURE.md` | someone changing the code | module responsibilities, data flow, and **every non-obvious decision with its rationale**, including bugs that shaped the design |
| `EXPLANATION.md` | the author, revisiting this later | plain-language guide. Assumes no knowledge of behaviour trees, CBBA, gradient boosting or LLM fine-tuning. Explains *why* each piece exists, not just what it does |

Rules for keeping them useful:

- **Record what went wrong, not just what the code does.** Every bug that changed a design
  decision gets a sentence saying what broke and what it would have corrupted downstream.
  These are the most valuable lines in all three documents.
- **Never let a number drift.** Test counts, golden-case counts, check counts, timing
  figures: re-measure, do not retype from memory.
- **Limitations go in `ARCHITECTURE.md` §9 and stay there** until they are fixed or they
  reach `RESULTS.md`. Do not quietly drop one.
- `CLAUDE.md` (this file) stays the working rules and the phase plan. It is not a fourth
  explanation.

## Conventions

- Every module gets `tests/test_<module>.py`. The validator gets a golden corpus. The sim gets determinism tests. Coordination gets property tests.
- Logs: one row per tick per drone (`drones.parquet`), one row per event (`events.jsonl`), one row per run (`runs.parquet`). Risk and debrief read only from these.
- Keep functions pure where possible; side effects (I/O, RNG draws) at the edges.
- Commit at the end of each phase with a message naming the phase.
