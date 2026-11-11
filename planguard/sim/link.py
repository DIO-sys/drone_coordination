"""Abstract link availability — a two-state Markov process gated on range.

This is **not** an RF model, and the project must never claim otherwise (see
the scope section of CLAUDE.md). It exists for two reasons: the DSL's
``link_lost`` fallback needs something to fire on, and Phase 4's risk surrogate
needs genuine aleatoric variation to predict.

The interface is deliberately narrow — ``step`` plus a distance — so a real
channel model could replace it without touching the rest of the simulator.
"""

from __future__ import annotations

import math

import numpy as np

from planguard.config import LinkConfig


def step(up: bool, dist_m: float, link: LinkConfig, dt_s: float, rng: np.random.Generator) -> bool:
    """One tick of link state for a single aircraft-to-base link.

    Beyond ``range_m`` the link is unconditionally down: range dominates, and a
    stochastic recovery out of range would be indefensible even in an abstract
    model.
    """
    if dist_m > link.range_m:
        return False
    if up:
        return not bool(rng.random() < link.p_per_tick(link.p_drop_per_s, dt_s))
    return bool(rng.random() < link.p_per_tick(link.p_recover_per_s, dt_s))


def _hazard_per_s(p_per_s: float) -> float:
    """Continuous-time hazard rate behind a per-second transition probability.

    ``p = 1 - exp(-lambda)``, so ``lambda = -ln(1 - p)``.
    """
    if p_per_s <= 0.0:
        return 0.0
    if p_per_s >= 1.0:
        return math.inf
    return -math.log1p(-p_per_s)


def steady_state_availability(link: LinkConfig) -> float:
    """Long-run fraction of time an in-range link is up.

    This is ``lambda_recover / (lambda_recover + lambda_drop)`` in the *hazard
    rates*, not in the per-second probabilities. Using the probabilities
    directly gives the steady state of a chain stepped at exactly 1 s, which is
    a different model and differs here by about one percentage point.

    The discrete chain at timestep ``dt`` differs from this limit by O(dt): at
    dt = 0.1 s it sits ~0.001 low. Tests assert the simulated rate converges on
    this value, which is what catches a per-tick conversion bug — the kind of
    error that would otherwise only surface as link fallbacks firing at the
    wrong rate.
    """
    lam_drop = _hazard_per_s(link.p_drop_per_s)
    lam_recover = _hazard_per_s(link.p_recover_per_s)
    if lam_drop == 0.0 or math.isinf(lam_recover):
        return 1.0
    if math.isinf(lam_drop) or lam_recover == 0.0:
        return 0.0
    return lam_recover / (lam_recover + lam_drop)
