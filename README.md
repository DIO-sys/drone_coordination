# PlanGuard

**Verifier-in-the-loop mission planning for drone teams.**

An operator states intent in plain language. A fine-tuned language model turns it into a
typed mission plan. That plan then has to survive a gauntlet before any human sees it: a
static validator, a learned risk model, and a simulator that actually flies it. Anything
that fails comes back as a specific machine-readable error, and the model repairs it.
Only then does a human approve. The model never touches the execution loop.

> *"Sweep the north ridge with two aircraft, keep one on station over the bridge, and if
> anything loses link send it back to the rally point."*

## The claim

Schema-constrained decoding takes any competent base model to ~100% **schema-valid**
JSON. It cannot produce referential integrity, correct aircraft arithmetic, or an acyclic
dependency graph — those require understanding the domain.

So the headline metrics here are **validator-pass rate** and **simulated mission
success**, never JSON validity. A fine-tune that only fixes syntax has done nothing;
constrained decoding already did that for free. The evaluation is built to let the
fine-tune lose, and a negative result gets published as one.

## Status

| Phase | Deliverable | State |
|---|---|---|
| 0 | Mission DSL + 9-check static validator + golden corpus | **done** |
| 1 | Simulator, behaviour trees, scenario randomisation, replay | **done** |
| 2 | CBBA task allocation + mission metrics from logs | **done** |
| 3 | Plan sampler + batch harness -> risk dataset | **done** |
| 4 | XGBoost risk surrogate + SHAP + calibration | **done** |
| 5 | Natural-language dataset (plan-first, verifier-filtered) + 150 hand-written held-out items | **done** |
| 6 | QLoRA fine-tune + constrained decoding + five-arm evaluation | next (needs a GPU) |
| 7 | LangChain tools + LangGraph approval graph + repair loop | |
| 8 | FastAPI + dashboard | |
| 9 | `RESULTS.md` + demo GIF | |

Current: **289 tests**, 50 golden validator cases (15 valid, 35 rejected), 150 hand-written
held-out NL items, ruff and mypy --strict clean.

## Quickstart

```bash
uv sync
uv run pytest

# validate the whole golden corpus (phase 0 acceptance check)
uv run python -m planguard.dsl.validate tests/golden/cases.yaml

# validate one plan, then fly it
uv run python -m planguard.dsl.validate examples/ridge.json
uv run python -m planguard.sim.run --plan examples/ridge.json --seed 7 --out runs/demo

# text map, time in each mode, event timeline
uv run python -m planguard.sim.replay runs/demo

# does CBBA actually beat the greedy placeholder? paired, same seeds
uv run python -m experiments.compare_allocators --plan examples/contested.json --seeds 150

# phase 3: is the sampler's budget band near the feasibility boundary?
# run this before the full build -- it is the check that decides whether the
# risk dataset has any signal in it at all
uv run python -m planguard.risk.build --pilot

# the full risk dataset: 2,000 plans x 20 seeds = 40,000 missions (~50 min on 16 cores)
uv run python -m planguard.risk.build --plans 2000 --seeds 20 --out data/risk

# phase 4: train the surrogate and print E2 -- the heuristic comparison, the
# calibration table and the measured screening speedup
uv run python -m planguard.risk.train --data data/risk

# phase 5: the NL dataset, with both verifier-filter yields and the drift
# report against the hand-written held-out set
uv run python -m planguard.copilot.build_dataset --pairs 4000 --out data/nl
```

## Headline numbers so far

The risk surrogate beats "path length over budget" on held-out plans in unseen weather,
where the baseline is the same model fitted on that one feature rather than a strawman:

| model | MAE | AUC | ECE |
|---|---|---|---|
| baseline (`budget_ratio` alone) | 0.2613 | 0.7471 | 0.0636 |
| **full feature set** | **0.1372** | **0.9274** | 0.0378 |

Screening costs 157 us per plan against 23.5 CPU-seconds to simulate it — a **149,767x**
speedup. Full numbers, including what each one replaced and why, are in `ARCHITECTURE.md`
section 10.

A plan is always validated before it is simulated. Simulating an invalid plan would
produce a risk label for something that could never be flown.

## Layout

```
planguard/
  dsl/        mission DSL: schema (the fine-tune target), validator, py_trees compiler
  sim/        world, kinematics, task semantics, link model, scenarios, logging, replay
  coord/      aircraft allocation: CBBA, plus the greedy placeholder for comparison
  risk/       plan-time features, XGBoost failure-risk surrogate, SHAP   [phase 4]
  copilot/    dataset gen, QLoRA fine-tune, constrained decoding, LangGraph  [phases 5-7]
  debrief/    mission scoring, experiment figures
  api/        FastAPI + WebSocket                                        [phase 8]
experiments/  world config, feasibility and evaluation harnesses
examples/     hand-written plans (ridge.json, contested.json)
tests/golden/ the validator's golden corpus
```

## Documentation

- **[EXPLANATION.md](EXPLANATION.md)** — plain-language guide. Start here if any of
  "CBBA", "behaviour tree", "QLoRA" or "constrained decoding" is unfamiliar.
- **[ARCHITECTURE.md](ARCHITECTURE.md)** — how it is built and why, module by module,
  including every design decision that is not obvious.
- **[CLAUDE.md](CLAUDE.md)** — the working rules and the phase plan.

## Scope

Surveillance coverage, search, station-keeping and rally. **No weapons, targeting or
strike logic anywhere in the codebase.** Link availability is modelled abstractly, not as
RF propagation. This is a research simulation, not flight software.
