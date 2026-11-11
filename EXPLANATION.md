# PlanGuard, explained

A plain-language guide to what this project is, what every piece does, and why it is
built this way. No prior exposure to behaviour trees, CBBA, gradient boosting or LLM
fine-tuning assumed. Read this before `ARCHITECTURE.md`.

---

## 1. The problem in one paragraph

Suppose a drone operator could just *say* what they want — "sweep the north ridge with two
aircraft, keep one on station over the bridge, and if anything loses link send it back to
the rally point" — and a computer turned that into a mission the aircraft actually fly.
Language models are good enough at that translation to be tempting. They are also
confidently wrong often enough that nobody sane would let one fly aircraft unsupervised.

So the interesting engineering question is not "can a model write a mission plan?" It is
**"what has to sit between the model and the aircraft for the result to be trustworthy?"**

PlanGuard is an answer to that second question. The model is the smallest and least
interesting part of it.

## 2. The core insight

There are two very different kinds of "wrong" a generated plan can be.

**Syntactically wrong** — malformed JSON, a misspelled field name, a string where a number
belongs. This is a *solved problem*. A technique called **constrained decoding** (more in
§8) forces a model to emit only text matching a schema, so you get well-formed output
essentially 100% of the time. No fine-tuning needed.

**Semantically wrong** — perfectly well-formed, and nonsense:

- "Sweep `west_ridge`" when no such place exists on the map.
- "Sweep the bridge" — a bridge is a point; you cannot sweep a point.
- Task A waits for task B, and task B waits for task A. Neither ever starts.
- Five aircraft doing simultaneous jobs, from a fleet of four.
- Forty minutes of flying inside a twenty-minute budget.
- A task that needs a radio link, aimed at a place outside radio range.

**No schema can catch any of these.** They require knowing what the words mean, what is on
the map, how many aircraft there are, and how fast they fly. That gap — between
*well-formed* and *actually sensible* — is where this entire project lives.

Which gives the one sentence everything else follows from:

> Constrained decoding gets you schema-valid JSON for free. It cannot get you referential
> integrity, correct arithmetic, or an acyclic dependency graph.

And therefore the one measurement that matters: not "was the JSON valid?" but **"did the
plan survive the validator, and did it actually work when flown?"**

## 3. The shape of the whole system

```
  operator speaks
        │
        ▼
   ┌─────────┐   plan as JSON
   │  model  │ ─────────────────┐
   └─────────┘                  ▼
        ▲              ┌──────────────────┐
        │   "task A    │    validator     │  9 checks a schema can't do
        │    and B     │  (the gatekeeper)│
        │    deadlock" └──────────────────┘
        │                       │ passes
        └───── repair ◄─┐       ▼
                        │  ┌──────────────────┐
                        │  │  risk model      │  "this plan is 30% likely
                        │  │  (the fast guess)│   to fail — here's why"
                        │  └──────────────────┘
                        │       │ looks survivable
                        │       ▼
                        │  ┌──────────────────┐
                        └──│   simulator      │  actually fly it, many times
                           │  (the ground     │
                           │   truth)         │
                           └──────────────────┘
                                   │ worked
                                   ▼
                              human approves
                                   │
                                   ▼
                               execution
```

Read the arrows: a plan has to get past **three** independent judges, and the only one
that can approve it is a person. The feedback flowing back up to the model is always a
*concrete machine-generated error* — "`depends_on` forms a cycle among tasks: a, b" — never
another model's opinion. Nothing here is an LLM grading an LLM.

## 4. The mission DSL — the thing the model writes

DSL means "domain-specific language": a small, strict vocabulary for one job. Ours is JSON:

```json
{
  "name": "ridge sweep with bridge overwatch",
  "time_budget_s": 1200,
  "tasks": [
    {"id": "sweep_ridge", "kind": "sweep", "target": "north_ridge", "aircraft_count": 2},
    {"id": "hold_bridge", "kind": "station_keep", "target": "bridge",
     "dwell_s": 600, "priority": 1, "requires_link": true}
  ],
  "fallbacks": [
    {"trigger": "link_lost", "action": "return_to_rally", "rally_point": "rally_alpha"}
  ]
}
```

That is the whole language. Four task kinds (`sweep`, `station_keep`, `corridor_patrol`,
`rally`), three fallback triggers, four fallback actions.

Three decisions worth understanding:

**The plan never names an aircraft.** It says "two aircraft", not "d0 and d1". Deciding
*which* aircraft is a separate problem, solved later by CBBA (§6). This separation is why
the same plan works whether four aircraft launched or three.

**One `target` field, not three.** An earlier design had separate `area`, `point` and
`corridor` fields. Collapsing them to one field means fewer ways for the model to be wrong,
*and* it moves the real question — "is that the right kind of place for this kind of task?"
— into the validator, where it belongs.

**`fallbacks` are first-class.** What to do when things go wrong is part of the plan, not an
afterthought. The validator refuses a plan where a link-dependent task has no plan for
losing its link.

## 5. The validator — the gatekeeper, and the heart of the project

Nine checks. For each, note that it needs knowledge a schema does not have:

| Check | The question it asks | Knowledge it needs |
|---|---|---|
unique ids | two tasks called the same thing? | — |
feature refs | does that place exist, and is it the right *kind*? | the map |
required fields | `station_keep` without a dwell time? | task semantics |
dependencies | dangling reference, self-reference, deadlock? | graph theory |
fleet capacity | one task needing more aircraft than exist? | the fleet |
time feasibility | can this fit the budget at all? | speeds, distances, a schedule |
link fallbacks | link-dependent task with no recovery plan? | task semantics |
link reachability | needs a link, but outside radio range? | the map and the radio |
fallback targets | recovery rule naming a task that doesn't exist? | — |

Three things about it are worth more than the list:

**It had to lose a check to stay honest.** An earlier version also rejected plans whose
*concurrent* demand exceeded the fleet — five jobs, four aircraft. That looked sensible and
was wrong: nothing in the DSL ever forces two tasks to happen at the same time, so four
aircraft can simply do four jobs and then the fifth. The check rejected a perfectly flyable
plan. That is worse than missing a bad one, because in Phase 7 the model would be handed a
correct plan, told to fix it, and the only available fix is to invent dependencies that
serialise the work for no reason — it would learn to write worse plans to please a broken
check. The sound version of the same intuition is now folded into the time check: total
sensor work divided by fleet size is a lower bound on how long the mission takes, whatever
order things happen in.

**It is honest about what it does not know.** The time check assumes *unlimited* aircraft.
So if the plan overruns even then, it is definitely infeasible — but fitting inside the
budget proves nothing, because the real fleet is finite and aircraft have to queue. The
check is therefore **one-sided on purpose**: it only ever says "definitely impossible",
never "definitely fine". Pretending otherwise would be a validator that lies, and the
simulator exists to answer the harder question.

**It refuses to guess when it is confused.** Some errors make a plan *unschedulable* — if
task A depends on a task that does not exist, you cannot work out any timeline at all. In
that case the capacity and time checks are reported as **skipped**, not passed. There is a
test for exactly this: a plan that hides a 10-second budget behind a dependency cycle. A
naive validator would report "time check: passed" on it. Silence about something you cannot
evaluate beats a reassuring lie.

**Every rejection is written for a machine to act on.** Not "invalid plan", but:

```
[WRONG_FEATURE_KIND] tasks[0].target: task kind 'sweep' needs an 'area' feature
  but 'bridge' is a 'point' feature
  (area features available: east_sector, north_ridge, south_valley)
```

A stable code, the exact JSON path, and a hint listing *real alternatives*. That text is
literally what gets sent back to the model to repair itself in Phase 7, which is why the
test suite checks the messages for length and style. The validator's error messages are
part of the product, not logging.

### The ninth check is the project's thesis in miniature

The ninth check did not come from a design document. It came from writing the demo plan by
hand, reviewing it, and then running it.

That plan asked two aircraft to sweep `north_ridge` while declaring `requires_link: true`.
`north_ridge` sits 8,051 m from base at its far corner. The radio range was 6,000 m. So on
*every single seed*, both aircraft flew out, lost the link on arrival, triggered the
"return to rally" fallback, and flew home without sweeping anything.

A human wrote that plan. A human reviewed it. It is statically detectable in about ten
lines of code. If a careful person writes an invisibly broken plan on their first attempt,
the argument that a language model needs a verifier in front of it stops being theoretical.

## 6. Flying the plan — the simulator

The simulator advances in fixed 0.1-second ticks. Each tick: update radio links → see which
tasks are unblocked → assign idle aircraft → let each aircraft decide what to do → move it
→ update progress → write logs.

### Behaviour trees — how an aircraft decides

A **behaviour tree** is a way of writing "what should I do right now?" as a tree you
re-read from the top every time, instead of as a tangle of if-statements or a state machine.
Each aircraft has one:

```
Try, in order:
├── Am I low on fuel?        → if yes, do the fuel fallback (go home)
├── Have I lost the link?    → if yes, do the link fallback (go to rally)
├── Do I have a task?        → if yes, fly to its next waypoint
└── Otherwise                → loiter
```

Top to bottom is priority. Running out of fuel beats losing the link, which beats doing
your job. Two properties matter:

**It is re-read from the root every tick** (`memory=False`). A tree *with* memory would
remember it was mid-task and finish the current leg before noticing the fuel problem. For
an aircraft, "finish what you were doing first" is the wrong default.

**The tree decides; the simulator moves.** The tree's entire output is a *mode* ("execute",
"rally", "loiter") and a *waypoint*. It never moves an aircraft and never draws a random
number. That separation is what makes runs reproducible, and it means you can read an
aircraft's decision-making without reading any physics.

### Why `requires_link` nearly broke the project

A subtle bug lived here. A fallback rule with an empty `applies_to` list means "applies to
all tasks". So a blanket "on link loss, return to rally" rule was pre-empting *every* task
— including tasks explicitly marked `requires_link: false`, i.e. tasks that had declared
they are fine without a radio.

The effect: the aircraft covering the far half of the ridge got yanked to the rally point on
every mission, so the sweep never finished, **on any seed**. Outcome variance was exactly
zero.

Zero variance sounds like a tidy, deterministic simulator. It is actually fatal, for a
reason that only shows up two phases later: the Phase 4 risk model's whole job is predicting
*how likely a plan is to fail*. If every run of every plan fails identically, there is
nothing to predict and nothing to learn. The fix was one condition — a `link_lost` rule only
pre-empts tasks that actually declare `requires_link` — which also makes the field mean
something instead of being decorative.

### Why the simulator is deliberately random

Three things change from run to run: the wind, which aircraft are available for duty, and
where each one starts. This is on purpose. A simulator where the same plan always produces
the same outcome cannot teach a risk model anything.

The map, though, is **fixed**. Randomising terrain would change whether a plan is even
*valid*, and the risk model needs to learn about execution risk, not about plans silently
becoming nonsense.

### Reproducible *and* random

Those sound contradictory; they are not. Given a seed number, a run is identical every
time, down to the last decimal — but different seeds give genuinely different runs. Three
mechanisms hold that:

1. Random number generators are passed around explicitly, never global, and each aircraft
   gets its own independent stream. Adding a fifth aircraft does not change what the first
   four do.
2. Everything iterates in sorted order. Looping over an unsorted Python `set` is the classic
   cause of a simulator that drifts.
3. Every run produces a SHA-256 fingerprint of its entire trajectory. The test launches two
   *separate operating-system processes* and compares fingerprints. Running it twice inside
   one process would not catch an ordering bug.

### Two bugs here worth your time

**The 81% lawnmower.** To cover an area you fly parallel stripes, alternating direction so
each pass starts where the last one ended — like mowing a lawn. Two aircraft should split
the stripes. The obvious split is "you take the odd ones, I take the even ones". That is
wrong, and quietly so: because the stripes alternate direction, every *odd* stripe runs
left-to-right, so an aircraft finishing one at the right edge has to fly 2 km back to the
left edge to start its next. Measured: **18.1 km flown against an ideal of 10.0 km.**

Splitting into *contiguous blocks* instead — "you take the north half, I take the south" —
keeps the alternation intact and the legs join up. Now there is a test asserting the flown
distance matches `area ÷ swath`, because the validator's time estimate assumes exactly that.
Those two numbers disagreeing meant **the validator's time check was wrong by construction**,
and no test anywhere would have noticed.

**The drifting stopwatch.** To hold station for 600 seconds, the obvious approach is to add
0.1 every tick and stop at 600. But 0.1 is not exactly representable in binary: ten
additions give 0.9999999999999999, not 1.0. Every dwell took one tick too long. Now dwell is
counted in whole ticks — consistent with the rule that anything slower than the tick rate is
scheduled *by tick number*.

Neither bug would ever have raised an error. Both would have silently biased every number
produced downstream.

### Reading the mission back from the logs (the debrief)

The simulator knows how the mission went while it is running. But Phase 3 needs to score
*forty thousand* missions that have already finished, from files on disk. So
`planguard/debrief/metrics.py` reconstructs the outcome from the two log files alone —
one row per aircraft per tick, one row per event — and a test asserts its numbers equal
the live result for the same run.

Writing that test immediately found a hole. A task that simply ran out of time was marked
failed in memory, and *nothing was written to the log to say so*: the simulator and the
debrief disagreed. The rule it establishes is the one Phase 3 depends on — anything the
logs cannot reconstruct does not exist, as far as the risk model is concerned.

## 7. Assigning aircraft to tasks — CBBA (Phase 2, done)

The plan says "two aircraft sweep the ridge". Something must decide *which* two.

Phase 1 used the obvious greedy rule: whichever idle aircraft are nearest. It works, and it
is still in the codebase — not as dead weight, but as the control group. If CBBA cannot beat
it on identical seeds, that is worth knowing and worth saying.

**CBBA** — Consensus-Based Bundle Algorithm — is an auction:

1. **Bundle building.** Each aircraft privately works out how much it "wants" each task —
   a score based mainly on how far it would have to fly — and builds a small bundle (up to
   three) of the tasks it wants most, in the order it would fly them.
2. **Consensus.** Aircraft exchange their best-bid-per-task vectors. Where two want the same
   task, a fixed rule table decides the winner from the bid values and who heard what when.
   Losers drop that task and rebuild.

Repeat until nobody changes their mind. The appeal is that it is **decentralised**: no
central allocator, each aircraft reasons only from what it has heard, and the algorithm
provably converges to a conflict-free assignment. That is the property you want when
aircraft are spread out and may not all hear each other.

### Three things that had to be worked out, not copied

**A task wanting two aircraft doesn't fit the algorithm.** CBBA assigns one task to one
agent. So `sweep_ridge` with `aircraft_count: 2` is expanded into two *slots*,
`sweep_ridge#0` and `sweep_ridge#1`, and the auction runs over slots. One extra rule: an
aircraft may never win two slots of the same task, because one aircraft cannot be two
aircraft.

**The score has to shrink as the bundle grows.** CBBA's convergence proof requires
*diminishing marginal gains*: the fourth task you add must be worth no more than the third.
This is enforced by discounting each task's value by how late you would arrive
(`value × 0.999^arrival_seconds`). Inserting a task can only push the later ones further
out, so a longer bundle really does make each addition worth less. Without this the
algorithm is not merely suboptimal — it may never stop trading bids.

**The published rule table has a hole, and a property test found it.** The paper compares
bids with strict `>` and never says what happens on an exact tie, because it assumes bids
are never exactly equal. They are: put two aircraft at the same coordinates and they compute
*bit-identical* bids. Then neither can outbid the other, both keep the slot, and the
algorithm reports "converged, conflict-free" while holding a duplicate assignment —
precisely the thing it exists to prevent. The fix is a tie-break that is **antisymmetric**:
higher bid wins, and on an exact tie the lower aircraft id wins. Antisymmetry is the whole
point — if d0 and d1 tie, d0 must win when d0 evaluates it *and* when d1 evaluates it, or
the slot ping-pongs between them forever.

A fourth, smaller one: CBBA hands back a bundle of three tasks per aircraft, but the
simulator flies one task per aircraft at a time. Committing the whole bundle had the tick
loop create three flights for one aircraft and silently keep the last. So only each
aircraft's *first* task is committed, and consensus simply runs again when aircraft free up.
The rest of the bundle still does work — an aircraft's bid for its first task already
accounts for what it plans to fly next, which is where the sequencing value lives.

### Was it worth it? (the honest answer)

Both allocators were run on 150 identical scenario seeds and the per-seed difference
measured with a bootstrap confidence interval:

| metric | greedy | cbba | difference |
|---|---|---|---|
| success rate | 0.98 | 0.99 | +0.007, CI includes zero — **no real difference** |
| makespan | 599 s | 579 s | −20 s, CI [−29, −11] — **real** |

And on the two-task demo plan the two allocators produce **byte-identical** output.

So: **CBBA buys no measurable improvement in mission success on these plans.** With a
generous time budget, what decides success is how many aircraft managed to launch, not who
flies what. The effect that is real and repeatable is a **~3% shorter makespan**, because
CBBA plans each aircraft's sequence instead of committing one task at a time.

This is the sort of result worth reporting rather than burying. The comparison was built
*before* knowing the answer, with pairing and confidence intervals, specifically so a null
result would be legible as a null result — the first version of this comparison used 30
seeds and showed CBBA "better" at one budget and "worse" at another, differing by a single
run out of thirty in each direction. That is noise, and it is exactly what an unpaired
comparison at small n looks like when it is about to be mistaken for a finding.

Where CBBA should pay off is at budgets near the feasibility cliff, where 20 seconds of
makespan is the difference between finishing and not. Testing that needs the Phase 3 sweep.

One standing caveat: the message exchange here is *simulated in-process*. The algorithm is
faithful, but nothing in this project yet shows it surviving dropped or delayed consensus
messages — and CBBA's guarantees also assume a fixed task list, which this project does not
have (tasks become available as dependencies clear). Characterising where those guarantees
break is more interesting than implementing the happy path, and is named as a limitation
rather than glossed.

## 8. The machine-learning pieces

### XGBoost — the fast guess (Phase 4)

Flying a plan in simulation takes about half a second. Flying it twenty times, to see how
often it fails, takes ten. If the model proposes a hundred candidate plans, you cannot
simulate them all.

So: train a model to *guess* the simulator's answer instantly. Describe a plan with ~20
plain numbers — aircraft count, total distance to be flown, furthest point from base, how
many tasks need a radio, budget versus estimated duration, and so on — and predict the
probability it fails.

**XGBoost** is gradient-boosted decision trees: build a shallow tree, see what it got wrong,
build another tree that fixes those mistakes, repeat hundreds of times. For a few dozen
numeric features it reliably beats a neural network, trains in seconds, and runs in
microseconds. That last part is the point: it screens a hundred candidates in about a
millisecond, versus ten seconds of simulation. Roughly five orders of magnitude, which is
what makes an iterative repair loop affordable.

Two things that will be done carefully:

**Calibration, not just accuracy.** It is not enough to rank plans correctly. If the model
says "30% chance of failure", roughly 30% of such plans should actually fail — otherwise the
number cannot be shown to an operator. Calibration gets reported alongside accuracy.

**Only plan-time information.** The features may contain nothing the system would not know
*before* flying: no actual wind that occurred, no actual radio outages, no outcome. Obvious
when stated, easy to violate by accident, and it invalidates everything if you do.

**SHAP** then explains individual predictions — "this plan is risky mainly because of its
distance from base and its number of link-dependent tasks". That serves the operator *and*
tells the repair loop what to change.

### QLoRA fine-tuning — teaching the model our language (Phase 6)

A general model has never seen our DSL; we invented it. It will guess at field names and
mis-structure things.

**Fine-tuning** is further training on examples of the thing you actually want. Full
retraining of a multi-billion-parameter model needs a data-centre. **LoRA** (Low-Rank
Adaptation) avoids that: freeze the original model and train a small set of extra weights
alongside it — often under 1% of the total. **QLoRA** adds quantisation, squeezing the frozen
model to 4 bits per weight, so the whole thing fits on one consumer GPU.

The data is generated **backwards**, which is the clever bit. Rather than writing
instructions and hand-authoring plans for them, we:

1. randomly generate a structurally valid plan,
2. **throw it away unless it passes the validator and actually flies successfully**,
3. then write the natural-language instruction that would have asked for it.

Step 2 is rejection sampling against a real verifier, so every training example is
*provably* good. No hand-labelling, no "looks fine to me".

The catch, stated plainly: instructions written by a model read like a model, not like a
person. So the test set is **hand-written by a human**, and scored separately. A
template-generated test set is in-distribution, and its numbers mean nothing.

### Constrained decoding — free correctness (Phase 6)

A language model picks the next token from a probability distribution over the vocabulary.
Constrained decoding simply sets the probability of every token that would break the schema
to zero. If the grammar says the next character must be `"` or `}`, nothing else can be
chosen. Malformed output becomes *impossible* rather than unlikely.

This is why the evaluation has **five arms** rather than two:

| arm | what it tests |
|---|---|
base, zero-shot | the naive starting point |
base, few-shot | base model shown a few examples first |
base + constrained | **the honest baseline** |
fine-tuned | does training help? |
fine-tuned + constrained | both together |

A two-arm comparison — base versus fine-tuned, scored on JSON validity — would show a huge
improvement and would be **meaningless**, because arm 3 gets the same improvement for free
with no training at all. Anyone who knows this field spots that in seconds. So the
fine-tune has to prove itself on *semantic* measures: validator-pass rate, whether the plan
matches what was asked, and whether it actually flies.

And it genuinely might not. If arm 3 closes the gap, that gets published as the result.

### LangChain and LangGraph — wiring the loop (Phase 7)

**LangChain** supplies "tools": functions the model may call — look up the map, check the
fleet, validate this plan, estimate its risk, simulate it.

**LangGraph** expresses the *process* as a graph with state, loops and pause points:

```
generate → check schema → check validator → [failed? → repair, try again, max 3 times]
         → screen risk → simulate → [failed? → repair] → PAUSE FOR HUMAN → execute
```

That "PAUSE FOR HUMAN" is an `interrupt`: the graph genuinely stops and waits. It is the
most important node in the system and the easiest to fake, so there is a test asserting that
no plan can reach execution without a logged approval event.

## 9. How the pieces end up justifying each other

Worth noticing, because it was not the first design:

- The **validator** gives the fine-tune a meaningful target (validator-pass rate) instead of
  a trivial one (JSON validity).
- The **simulator** gives the training data a quality filter that needs no human labelling.
- The **risk model** makes the repair loop affordable, by replacing 10 seconds of simulation
  with a millisecond of inference.
- The **repair loop** needs the validator's error messages to be specific, which is why they
  are written as prompts rather than logs.
- The **approval gate** is what makes the whole thing defensible rather than reckless.

Remove any one and something else loses its justification. That interlock is the actual
contribution; the individual components are all standard.

## 10. What "done" looks like

Type the sentence from the top of this document into a dashboard, and watch the system
generate a plan, fail a *specific* check, repair itself from that exact error, show its risk
estimate with an explanation, wait for approval, and then fly it on a live map.

Plus a two-page `RESULTS.md` with the five-arm table, the calibration curve, and — the one
experiment that makes this a systems result rather than an LLM demo — an **ablation of the
verification stages**: how good are the approved plans with no verification, with schema
checking only, with the validator added, and with risk screening and simulation added? That
number is the project's actual claim, quantified.

With the nulls and the limitations stated plainly. Phase 1 alone produced four findings that
contradicted what the code looked like it did. That is the standard the headline table has
to be held to.
