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
metric all depend on it. **Nine checks, 14 error codes**, none expressible as a JSON
Schema constraint:

| Check | Catches |
|---|---|
`unique_ids` | duplicate task ids |
`feature_refs` | unknown map features; a task kind targeting the wrong feature *kind* |
`required_fields` | `station_keep` without `dwell_s`; `return_to_rally` without a rally point |
`dependencies` | dangling refs, self-dependency, cycles |
`fleet_capacity` | a task asking for more aircraft than exist |
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

## 4. Allocation (`planguard/coord/`) — phase 2, done

Per rule 7 a plan expresses **team-level intent** and never names aircraft, so allocation
happens here and only here. `allocate.py` dispatches on `cfg.cbba.allocator` between CBBA
and the Phase 1 greedy placeholder; the tick loop does not know which ran.

Both allocators refuse to **part-staff** a task: a half-crewed sweep would silently take
twice as long and corrupt the time-margin signal Phase 4 learns from.

### CBBA (`coord/cbba.py`)

Consensus-Based Bundle Algorithm (Choi, Brunet & How, 2009). Each aircraft builds a bundle
of tasks it wants by marginal score, then aircraft exchange winning-bid vectors and resolve
conflicts with a fixed rule table. Each agent reasons only from vectors it has received —
`AgentState` holds nothing global.

**Multi-aircraft tasks become slots.** CBBA assigns one task to one agent, so a task asking
for two aircraft expands into `sweep_ridge#0` and `sweep_ridge#1` and consensus runs over
slots. An agent may never hold two slots of the same task — one aircraft cannot be two
aircraft.

**Diminishing marginal gains comes from time discounting.** A path scores
`Σ value · discount^arrival_time_s`. Inserting a task can only delay those after it, so
adding to a longer bundle is worth less — which is the property CBBA's convergence proof
requires. Without DMG the algorithm is not guaranteed to terminate at all.

**Exchange is simultaneous within a round.** Every agent resolves against snapshots taken
before any update in that round, so the outcome cannot depend on agent iteration order.

**Releasing a bundle entry cascades.** When an agent loses a slot, it drops that slot *and
everything added after it*. Later entries' bids were computed assuming the earlier ones
were held, so they are stale. Skipping the cascade is the classic route to bids that never
settle.

**The message exchange is simulated, not distributed.** `run_consensus` drives rounds
in-process. The algorithm is faithful, but this is a single-process simulation of a
decentralised protocol, not a deployment of one.

#### The published rule table has a tie-breaking gap

Choi et al.'s Table 1 compares bids with strict inequality and does not say what happens
when two bids are exactly equal, because it assumes bids are in general position. They are
not: **two aircraft at the same position produce bit-identical bids.** Then neither can
outbid the other, both sides keep the slot, nothing changes, and consensus reports
convergence while holding a duplicate assignment — precisely the failure the Phase 2
acceptance check exists to catch.

`claim_beats` adds an **antisymmetric** tie-break: higher bid wins, and on an exact tie the
lower agent id wins. Antisymmetry is the requirement — if d0 and d1 tie, d0 must win from
*both* points of view, or the slot ping-pongs forever. Found by the property test with two
aircraft at (0, 0).

#### Only the first task of each bundle is committed

CBBA plans a whole bundle; the simulator flies one task per aircraft at a time. Committing a
bundle of three would have had the tick loop create three runs for one aircraft and silently
keep the last. Found by the property test: one aircraft and two single-aircraft tasks
produced a correct bundle of two that the executive could not execute.

So `commit()` takes each aircraft's first task in path order. The rest of the bundle is not
wasted — it still shapes the bids, since an agent's bid for its first task already accounts
for what it intends to fly next, which is where CBBA's sequencing value lives — and
consensus simply reruns next tick as aircraft free up. This is CBBA as a receding-horizon
planner with incremental commitment, which is also the honest way to handle tasks that
become ready over time, something CBBA's static guarantees do not cover.

### Was CBBA worth implementing?

The greedy allocator was kept rather than deleted so this is a measurement, not an
assertion. `experiments/compare_allocators.py` runs both on identical seeds and reports the
**paired** per-seed difference with a bootstrap confidence interval.

Pairing is not a nicety here. Scenario variance is large (one to four aircraft launch), so
an unpaired comparison of two means at n=30 cannot resolve a 5% effect — the first run of
that script showed success rates differing by exactly one run in thirty, in opposite
directions at two budgets, which is noise presented as a finding. A bootstrap CI is used
rather than a t-test because success is Bernoulli and makespan is bounded and skewed by the
budget cap.

Results are in §10.

## 4a. The concurrency check that had to be removed

Writing the five-task plan for the allocator comparison immediately tripped
`PEAK_CONCURRENCY_EXCEEDED`: five independent tasks, a fleet of four. The plan was
obviously flyable — a 439 s critical path against a 1500 s budget, four aircraft start and
the fifth task follows. **The validator was wrong.**

The reason no such check can be sound: `start_after_s` is a *lower* bound ("cannot start
before") and no task carries a deadline, so nothing in the DSL ever forces two tasks to
overlap. The executive can always sequence them, and concurrent demand therefore cannot
prove infeasibility.

False positives are worse than misses in this system. The Phase 7 repair loop would be
handed a correct plan and told to fix it, and the only available fix is inventing
`depends_on` edges that serialise work for no reason — the model would learn to produce
worse plans to satisfy a broken check.

The real signal being approximated — "this fleet cannot get through this much work in
time" — is now a sound lower bound in `check_time_feasibility`:

```
makespan ≥ max( critical path with unlimited aircraft,
                total sensor work / fleet size )
```

Both terms are valid lower bounds whatever the executive does, so the larger of the two is
still one-sided. Travel is deliberately excluded from the work term: an aircraft flying two
tasks in sequence does not pay travel-from-base twice, so including it would overestimate
and reintroduce the false positives.

Net effect: 9 checks still, 14 error codes instead of 15, and the validator no longer
rejects feasible plans.

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

## 7a. Debrief (`planguard/debrief/`)

`metrics.py` reconstructs mission outcome from `drones.parquet` and `events.jsonl` and
nothing else. A test asserts those numbers equal the live `RunResult` for the same run.

That test immediately found an incomplete log: a task that simply ran out of time was
marked failed in memory with nothing written to say so, so the debrief disagreed with the
simulator. Timeouts now emit a `task_failed` event, and `mission_end` carries the final
per-task status and progress. Phase 3 builds its dataset from stored runs, so anything the
debrief cannot reconstruct does not exist as far as the risk model is concerned.

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
- **CBBA's message exchange is simulated in-process**, so nothing here demonstrates
  robustness to dropped or delayed consensus messages. The algorithm is faithful; the
  distribution is not.
- **Only the first task of each CBBA bundle is committed**, so the deeper sequencing value
  CBBA can express is reflected only through its effect on bids.

## 10. Measured results so far

Re-measure before editing these; do not retype from memory.

**Tests:** 200 passing. 50 golden validator cases: 15 valid, 28 rejected by the validator,
7 rejected by the schema before the validator runs. 9 validator checks, 14 error codes.

**Simulator:** 555 ms per 1200 s / 4-aircraft mission. Steady-state link availability
0.96196 at the default hazard rates, matched by the simulator to three decimals.

**Feasibility boundary** (`examples/contested.json`, 5 tasks, fleet of 4, n=80 seeds,
restricted to runs where at least 3 aircraft launched):

| budget | success | mean completed |
|---|---|---|
| 650 s | 0.84 | 0.919 |
| 700 s | 0.97 | 0.986 |
| 750 s | 1.00 | 1.000 |
| 800 s and above | 1.00 | 1.000 |

The cliff is between 650 s and 750 s. Phase 3 must sample budgets in that band: above it
every run succeeds and the risk model has no signal to learn.

**CBBA vs greedy** (`experiments/compare_allocators.py`, same plan, budget 1500 s, n=150
paired seeds, 95% bootstrap CI on the per-seed difference):

| metric | greedy | cbba | paired difference (cbba − greedy) | verdict |
|---|---|---|---|---|
| success rate | 0.98 | 0.99 | +0.007 [+0.000, +0.020] | no detectable difference |
| completed fraction | 0.996 | 0.997 | +0.001 [+0.000, +0.004] | no detectable difference |
| makespan | 599 s | 579 s | −20.0 s [−29.1, −11.0] | **difference detected** |

On `examples/ridge.json` the two allocators are **byte-identical**: two tasks and
distance-dominated bids, so CBBA converges on the same nearest-aircraft answer.

Honest reading: **CBBA buys no measurable success-rate improvement on these plans.** With a
budget this loose, success is decided by how many aircraft launched, not by who flies what.
The real and repeatable effect is a **~3% shorter makespan** — CBBA sequences each
aircraft's tasks instead of committing greedily task-by-task. That margin should matter more
at tight budgets, where it converts into success; demonstrating that is Phase 3 work, since
it needs the feasibility-boundary sweep above run against both allocators.
