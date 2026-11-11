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
| 2 | CBBA task allocation | in progress |
| 3 | Plan sampler + batch harness -> risk dataset | |
| 4 | XGBoost risk surrogate + SHAP + calibration | |
| 5 | Natural-language dataset (plan-first, verifier-filtered) | |
| 6 | QLoRA fine-tune + constrained decoding + five-arm evaluation | |
| 7 | LangChain tools + LangGraph approval graph + repair loop | |
| 8 | FastAPI + dashboard | |
| 9 | `RESULTS.md` + demo GIF | |

Current: **167 tests**, 49 golden validator cases, ruff and mypy --strict clean.

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
```

A plan is always validated before it is simulated. Simulating an invalid plan would
produce a risk label for something that could never be flown.

## Layout

```
planguard/
  dsl/        mission DSL: schema (the fine-tune target), validator, py_trees compiler
  sim/        world, kinematics, task semantics, link model, scenarios, logging, replay
  coord/      aircraft allocation (CBBA; a greedy placeholder until phase 2)
  risk/       plan-time features, XGBoost failure-risk surrogate, SHAP   [phase 4]
  copilot/    dataset gen, QLoRA fine-tune, constrained decoding, LangGraph  [phases 5-7]
  debrief/    mission scoring, experiment figures
  api/        FastAPI + WebSocket                                        [phase 8]
experiments/  world config, feasibility and evaluation harnesses
examples/     hand-written plans
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
