import numpy as np
import math
from dataclasses import dataclass
from typing import Tuple

# ─────────────────────────────────────────────
# MODEL PARAMETERS (as per Busso 2003, scaled to km instead of TRIMP)
# ─────────────────────────────────────────────
@dataclass
class BussoParams:
    p0:   float = 0       # baseline performance (AU) -> AU: arbitrary units
    k1:   float = 0.31    # fitness gain factor (the magnitude of fitness gained by a unit of training)
    k3:   float = 0.0035  # fatigue sensitivity multiplier (drives dynamic k2)
    tau1: float = 30.8    # fitness decay constant (days)
    tau2: float = 16.8    # fatigue decay constant (days)
    tau3: float = 2.3     # fatigue sensitivity decay constant (days)

# ─────────────────────────────────────────────
# BUSSO VDR (Variable Dose-Response) MODEL
# As training accumulates, the body becomes more susceptible to fatigue.
# ─────────────────────────────────────────────
def simulate_busso(
    loads: np.ndarray,
    params: BussoParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate the Busso Variable Dose-Response model.

    Parameters
    ----------
    loads  : daily training loads, shape (n_days,)
    params : BussoParams

    Returns
    -------
    perf    : predicted performance p(t)
    fitness : fitness impulse g(t)
    fatigue : fatigue impulse h(t)
    k2      : dynamic fatigue factor k2(t)
    """
    n = len(loads)
    g    = np.zeros(n)   # fitness
    h    = np.zeros(n)   # fatigue
    k2   = np.zeros(n)   # dynamic fatigue factor
    perf = np.zeros(n)

    d1 = math.exp(-1.0 / params.tau1)  # fitness decay: retain ~97.8% per day
    d2 = math.exp(-1.0 / params.tau2)  # fatigue decay: retain ~94.2% per day
    d3 = math.exp(-1.0 / params.tau3)  # fatigue sensitivity decay

    k2_prev = 0.0  # fatigue sensitivity starts at 0 (ref: Busso 2003)
    g_prev  = 0.0
    h_prev  = 0.0

    for t in range(n):
        w = loads[t]

        # Performance: athlete's readiness before today's workout,
        # computed from previous-day state variables.
        perf[t] = params.p0 + params.k1 * g_prev - k2_prev * h_prev

        # Update state variables
        g[t]  = g_prev  * d1 + w
        h[t]  = h_prev  * d2 + w
        k2[t] = k2_prev * d3 + params.k3 * w  # dynamic fatigue sensitivity

        g_prev  = g[t]
        h_prev  = h[t]
        k2_prev = k2[t]

    return perf, g, h, k2


# ─────────────────────────────────────────────
# SHARED CONFIGURATION
# ─────────────────────────────────────────────
params_busso  = BussoParams()
n_days        = 112      # 16-week training block leading up to race day

RAMP_RATE     = 0.10     # max week-on-week volume increase (10%)
FIRST_WEEK_KM = 40.0     # week-1 anchor total volume (km)
LONG_RUN_KM   = 32.0     # minimum distance of the longest run in the plan
MAX_RUN_KM    = 32.0     # per-run hard cap (km)
MARATHON_KM   = 42.2     # race distance (final day)


def busso_objective(loads: np.ndarray) -> float:
    """Negative race-day performance (minimising this maximises performance)."""
    perf, _, _, _ = simulate_busso(loads, params_busso)
    return -perf[-1]
