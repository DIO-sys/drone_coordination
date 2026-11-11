# PlanGuard Architecture

How the system is built, and *why* each non-obvious decision was made. Kept current as
each phase lands; the status markers say what exists today.

## 1. The pipeline

```
operator intent (natural language)
   │
   ▼
[copilot] fine-tuned model + constrained decoding          ── phase 6
   │  emits JSON
   ▼
[dsl.schema] Pydantic parse ................................. schema validity
   │  parsed
   ▼
[dsl.validate] 9 static checks ............................... SEMANTIC validity
   │  errors carry {code, JSON path, hint}
   ├──────────── invalid ──► [copilot.agent] repair node ──┐   ── phase 7
   │                              (bounded retries) ◄──────┘
   │  valid
   ▼
[risk] XGBoost surrogate: p(failure) in microseconds ......... screen K candidates
   │                                                           ── phase 4
   ▼
[sim] fly the survivors across seeds ......................... ground truth
   │
   ▼
[copilot.agent] human approval interrupt ..................... nothing executes
   │  approved                                                 without this
   ▼
[sim] execute + [api] live map                               ── phase 8
```

Two properties hold at every stage:

1. **Nothing executes unverified or unapproved.** The approval gate is tested, not
   asserted.
2. **The verification signal is machine-generated, never LLM-judged.** The repair loop is
   driven by validator error codes and simulator outcomes. No model grades another model.

## 2. The mission DSL (`planguard/dsl/`) — phase 0, done

### Why a JSON DSL and not behaviour-tree XML

The model emits a typed JSON document which is then *compiled* to a py_trees behaviour
tree. BT XML was the obvious alternative and was rejected:

- A Pydantic model yields a JSON Schema, which `outlines`/`xgrammar` can constrain
  decoding against directly. Grammar-constraining XML is painful.
- XML is verbose, so it costs tokens, and it has many ways to be wrong that nobody cares
  about (attribute order, whitespace).
- The "absent from pretraining data" argument that justifies fine-tuning at all is
  *stronger* for a bespoke DSL, not weaker.

The compiler is where a plan becomes executable, and where the dashboard gets a real tree
to render.

### Schema (`dsl/schema.py`)

```python
MissionPlan: name, time_budget_s, tasks[], fallbacks[], constraints[], rationale?
TaskSpec:    id, kind, target, aircraft_count, priority, dwell_s?,
             depends_on[], requires_link, start_after_s?
FallbackRule: trigger, action, applies_to[], rally_point?
```

`kind` is one of `sweep | station_keep | corridor_patrol | rally`. `target` is a single
field naming a map feature — *not* separate `area`/`point`/`corridor` fields. Fewer fields
means fewer ways for the model to be wrong, and it pushes the interesting question ("is
that the right *kind* of feature?") into the validator where it belongs.

`extra="forbid"` everywhere, so a hallucinated field is a schema error rather than
silently ignored data.

### The validator (`dsl/validate.py`) — the spine

Built first, because the Phase 5 dataset filter, the Phase 7 repair loop and the headline
metric all depend on it. **Nine checks, 15 error codes**, none expressible as a JSON
Schema constraint:

| Check | Catches |
|---|---|
`unique_ids` | duplicate task ids |
`feature_refs` | unknown map features; a task kind targeting the wrong feature *kind* |
`required_fields` | `station_keep` without `dwell_s`; `return_to_rally` without a rally point |
`dependencies` | dangling refs, self-dependency, cycles |
`fleet_capacity` | a task asking for more aircraft than exist; **peak concurrent demand** |
`time_feasibility` | work that cannot fit the budget; a start time past the budget |
`link_fallbacks` | a link-dependent task with no `link_lost` recovery |
`link_reachability` | a link-dependent task whose target lies outside link range |
`fallback_targets` | `applies_to` naming a task that does not exist |

Three design points:

**The time check is deliberately one-sided.** The scheduler assumes *unlimited* aircraft,
so overrunning the budget proves infeasibility, while fitting inside it proves nothing.
Claiming more would be a validator that lies; that is what the simulator is for.

**Blocking errors skip the schedule checks.** Unresolvable references and cycles make a
plan unschedulable, so `fleet_capacity` and `time_feasibility` are reported as **skipped**
rather than passing vacuously. A golden case pins this by hiding a 10 s budget behind a
dependency cycle — without the skip logic that plan would appear to pass its time check.

**`link_reachability` exists because of a bug.** The first hand-written demo plan asked
two aircraft to sweep an area wholly outside link range while declaring `requires_link`,
so the fallback fired on arrival on every seed. A human wrote it, a human reviewed it, and
the simulator caught it. It is statically detectable, so it should never have reached the
simulator.

### The error contract (`dsl/errors.py`)

Every rejection carries `{code, path, message, hint}`. `ValidationReport.as_feedback()` is
*literally* what the Phase 7 repair node sends back to the model, so the wording is part
of the contract and is covered by tests (minimum lengths, consistent style, hints that
list real alternatives). The golden corpus covers all 15 codes: a code nobody tests is a
code the repair loop meets for the first time in production.

### The compiler (`dsl/compile.py`)

One behaviour tree per aircraft:

```
Selector (memory=False)
├── Sequence: low fuel     → condition fires, apply fallback
├── Sequence: link lost    → condition fires, apply fallback
├── Sequence: execute task → has a run, fly to its waypoint
└── Action:   loiter
```

- **Branch order is priority order.** Running out of fuel outranks losing the link, which
  outranks executing the task.
- **`memory=False` on both composites** so the whole tree re-evaluates from the root every
  tick. With memory, a fallback could not pre-empt a task mid-leg — the aircraft would
  finish its current waypoint first.
- **The tree decides; the simulator moves.** The tree writes a `mode` and a `waypoint_m`
  into its `AgentContext` and nothing else. It never moves an aircraft and never draws a
  random number, which is what keeps determinism cheap.
- **`AgentContext` is the only thing a tree can see**, so it cannot reach into world state
  it should not know about.
- **A `link_lost` rule only pre-empts a task that declares `requires_link`.** Without this
  a blanket rule (empty `applies_to`) pre-empts everything and `requires_link` has no
  behavioural effect at all. This was a real bug: it flattened outcome variance to exactly
  zero, which would have made Phase 4 impossible.

## 3. The simulator (`planguard/sim/`) — phase 1, done

Fixed timestep, `dt_s = 0.1`. Each tick, in order:

1. **link** availability per aircraft, from its range to base
2. **ready** tasks — those whose dependencies are satisfied
3. **allocate** idle aircraft to ready tasks (`coord/`)
4. **tick** each aircraft's behaviour tree → a mode and a waypoint
5. **integrate** kinematics toward that waypoint
6. **advance** task progress if the aircraft arrived
7. **settle** completion and handoffs, then log

### Kinematics (`sim/drone.py`)

Point mass, pure functions. Pure-pursuit steering with the turn rate clamped; wind added
as a ground-track offset; fuel burn that grounds the aircraft at zero.

There is **no wind-triangle lead compensation**, so ground tracks bow downwind. This is
deliberate: the loop still converges for any wind below airspeed, and the resulting
uncertainty in a plan's time margin is a signal Phase 4 needs, not a defect.

Arrival is a circle (`capture_radius_m`), because a fixed-wing cannot sit on a point. Its
scale is the minimum turn radius, with a `2 · speed · dt` floor so a coarse timestep cannot
step over the circle every tick.

### Link model (`sim/link.py`)

Range-gated two-state Markov availability. **Not an RF model**, and the project must never
claim otherwise. It exists for two reasons: `link_lost` fallbacks need something to fire
on, and the Phase 4 surrogate needs genuine aleatoric variation. The interface is
deliberately narrow (`step(up, dist_m, ...)`) so a real channel model could replace it
without touching anything else.

Per-second hazards convert to per-tick probabilities as `1 - (1-p)^dt`, not `p · dt`, so
the drop rate does not depend on the timestep. `steady_state_availability` is computed from
hazard rates `λ = -ln(1-p)`, giving `λ_rec / (λ_rec + λ_drop)`. Using the probabilities
directly is the steady state of a chain stepped at exactly 1 s — a different model, off by
about a percentage point here. Left wrong, that would never have thrown an error; it would
have shown up as fallbacks firing ~20% less often than the config implies, quietly biasing
every downstream risk label.

### Task semantics (`sim/tasks.py`)

| Kind | Waypoints | Done when |
|---|---|---|
`sweep` | boustrophedon lawnmower over the feature's bounding box | last waypoint reached |
`corridor_patrol` | the polyline vertices | last vertex reached |
`station_keep` | an orbit at turn radius around the point | dwell satisfied |
`rally` | the point | arrival |

Coverage is planned over the **bounding box**, so a non-rectangular area is over-covered
rather than clipped — stated as a limitation rather than pretending to do polygon
decomposition. Every feature in the default world is rectangular, where the box is exact.

**Multi-aircraft sweeps split into contiguous bands, not round-robin.** Round-robin
(`i % n == share`) looks natural and is wrong: because boustrophedon alternates direction,
every transect in one aircraft's set runs the same way, so the aircraft deadheads the full
width of the area between legs. On the default world that inflated a two-aircraft sweep
from 10.0 km to 18.1 km per aircraft — an 81% overshoot that made the sweep miss its budget
on every seed. Contiguous bands keep each aircraft's own legs alternating, so they join end
to end and the flown distance matches `area / swath`, which is exactly what the validator
assumes when it estimates duration. **A test pins the two together**; if they diverge, the
validator's time check is lying.

**Dwell is counted in ticks, not accumulated in seconds.** Summing `dt_s` drifts: ten
additions of 0.1 give 0.9999999999999999, so a 1 s dwell took an extra tick.

### Scenario randomisation (`sim/scenario.py`)

Per seed: wind speed and direction, which aircraft are available, and each aircraft's start
position and heading. **This module is the reason Phase 4 is possible at all** — without
aleatoric variation, mission outcome is a deterministic function of (plan, world) and there
is no risk to predict.

Map features are deliberately **not** randomised. Moving them would change whether a plan
is valid, and the risk model must learn about execution risk, not about plans silently
becoming invalid.

Each consumer gets its own named stream from one `SeedSequence`, so adding a drone does not
shift another drone's draws.

### Determinism (rule 5)

Three mechanisms, because reproducibility is load-bearing for every later phase:

1. `numpy.random.Generator` passed explicitly, never the global RNG; per-entity streams via
   `SeedSequence.spawn`.
2. Sorted iteration over every node, link and message collection. Unsorted `dict`/`set`
   iteration is the usual cause of a sim that drifts.
3. A running SHA-256 over every aircraft's per-tick state, exposed as `RunResult.digest`.
   The test runs two **subprocesses** with different `PYTHONHASHSEED` and compares digests;
   an in-process repeat would not catch an ordering dependency.

### Mission scoring

`success` is all-or-nothing: every task completed and no aircraft lost. An earlier version
only required priority-1 tasks and reported success on a run whose main sweep had failed.
Priority orders *allocation*; it does not excuse a task. `completed_fraction` and
`task_progress` carry the graded signal that Phase 4 regresses on.

### Logs (`sim/logging.py`) — the Phase 3/4 contract

`drones.parquet` (one row per tick per aircraft), `events.jsonl` (one row per event). The
risk model and the debrief read **only** from these files, never from live simulator
objects. Logging is optional (`logger=None`) because the Phase 3 harness runs tens of
thousands of simulations and needs only summaries.

## 4. Allocation (`planguard/coord/`) — phase 2, in progress

Per rule 7 a plan expresses **team-level intent** and never names aircraft, so allocation
happens here and only here. Phase 1 shipped a greedy nearest-aircraft placeholder; Phase 2
replaces it with CBBA behind the same `allocate` signature, so swapping it in does not
touch the tick loop.

The placeholder refuses to **part-staff** a task: a half-crewed sweep would silently take
twice as long and corrupt the time-margin signal Phase 4 learns from.

## 5. Risk surrogate (`planguard/risk/`) — phase 4

Simulating every candidate plan is expensive. The surrogate featurises a plan (~20 tabular
features: aircraft count, node count, total commanded path length, max range from base,
link-dependent task count, time budget versus estimated duration, fallback coverage, …)
plus scenario parameters, and predicts failure probability in microseconds. That lets the
agent rank and screen candidates before paying for simulation.

**Features must be plan-time observable** (rule 2): no realised outage timings, no
simulation outcomes, nothing the agent would not have before the plan runs.

Labels come from the Phase 3 harness: each (plan, scenario) pair run over K seeds, label =
failure rate. A regressor on the rate and a classifier on per-run outcome, split by plan
*and* by seed so the same plan never appears in both train and test.

SHAP feeds back in both directions: to the operator as an explanation, and to the model as
targeted repair guidance.

## 6. The copilot (`planguard/copilot/`) — phases 5-7

**Training pairs are generated plan-first, then back-translated.** Sample a structurally
valid plan from the DSL, keep it only if it passes the validator *and* simulates above a
success threshold, then write the natural-language instruction from it. Every training
target is verified by construction — rejection sampling against a real verifier.

The known weakness is that back-translated instructions read as model-ese rather than
operator-ese. That is handled by a **hand-written** held-out set, reported separately. This
is a hard requirement (rule 4), not a nice-to-have: a template-generated test set is
in-distribution and its numbers mean nothing.

The five evaluation arms are base zero-shot, base few-shot, base + constrained decoding,
fine-tuned, and fine-tuned + constrained decoding — scored on schema validity (reported,
never the claim), **validator-pass rate**, per-field exact match, task-level F1,
**simulated mission success of the generated plan**, repair iterations, prompt tokens, and
latency.

The LangGraph graph adds the repair loop and a human-approval interrupt. Per rule 1 the LLM
runs at the ground station only, never inside an aircraft's tick, and no plan reaches the
task pool without a logged approval event.

## 7. Configuration

All parameters live in Pydantic models in `config.py`, loaded from
`experiments/configs/world.yaml`. Derived quantities (turn radius, tick counts, per-tick
hazards, link thresholds) are **computed**, never written into YAML — rule 6: any constant
in a config file that could have been computed is a bug.

Blocks: `WorldMap`, `FleetSpec`, `SimConfig`, `LinkConfig`, `WindConfig`,
`BehaviourConfig`, `ScenarioConfig`, `ValidatorConfig`.

## 8. Performance

555 ms per 1200 s mission with 4 aircraft (~39k aircraft-ticks/s). Phase 3's 40,000 runs is
**6.2 hours single-core**, so the batch harness needs multiprocessing from the start.

## 9. Known limitations

Stated here and repeated in `RESULTS.md`; never hidden.

- **No RF fidelity.** Link availability is a range-gated Markov process.
- **Bounding-box coverage.** Non-rectangular areas are over-covered.
- **No wind-triangle compensation.** Tracks bow downwind.
- **No collision avoidance or airspace deconfliction.**
- **Back-translated training instructions**, so in-distribution scores overstate real
  operator performance — hence the hand-written held-out set.
- **Reference plans are authored by the same person building the system.** The hand-written
  set mitigates but does not eliminate the circularity.
- NL-to-config copilots are a crowded space. The differentiator is the verifier-in-the-loop
  architecture and the E3 ablation, which must be foregrounded.
