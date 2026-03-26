import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple
import math
from scipy.optimize import differential_evolution

# ─────────────────────────────────────────────
# MODEL PARAMETERS (as per Busso 2003)
# ─────────────────────────────────────────────
@dataclass
class BussoParams:
    p0:   float = 0         # baseline performance (AU) -> AU: arbitrary units
    k1:   float = 0.031     # fitness gain factor (fixed, the magnitude of fitness gained by a unit of training, depends on the athlete)
    k3:   float = 0.000035  # fatigue sensitivity multiplier (drives dynamic k2: the magnitude of fatigue incurred by a unit of training, depends on the athlete)
    tau1: float = 30.8      # fitness decay constant (days)
    tau2: float = 16.8      # fatigue decay constant (days)
    tau3: float = 2.3       # fatigue sensitivity decay constant (days)

# ─────────────────────────────────────────────
# BUSSO VDR (Variable Dose-Response) MODEL: as training accumulates, the body becomes more susceptible to fatigue
# ─────────────────────────────────────────────
def simulate_busso(loads: np.ndarray,
                   params: BussoParams
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

    d1 = math.exp(-1.0 / params.tau1)
    d2 = math.exp(-1.0 / params.tau2)
    d3 = math.exp(-1.0 / params.tau3)

    k2_prev = 0.0 # Fatigue sensitivity starts at 0 (ref: Busso 2003)
    g_prev  = 0.0
    h_prev  = 0.0

    for t in range(n):
        w = loads[t]

        # Performance
        # calculates the athlete's readiness to perform right now, before they do today's workout
        # uses variables from the *previous* day
        perf[t] = params.p0 + params.k1 * g_prev - k2_prev * h_prev

        # Update state variables
        g[t]  = g_prev  * d1 + w
        h[t]  = h_prev  * d2 + w
        k2[t] = k2_prev * d3 + params.k3 * w   # dynamic fatigue sensitivity driven by load

        g_prev  = g[t]
        h_prev  = h[t]
        k2_prev = k2[t]

    return perf, g, h, k2


params_busso = BussoParams()
n_days = 112  # 16 weeks training block leading up to race day

def busso_objective(loads: np.ndarray) -> float:
    """Negative race-day performance (minimizing this maximizes performance)."""
    perf, _, _, _ = simulate_busso(loads, params_busso)
    return -perf[-1]

# ─────────────────────────────────────────────
# Differential Evolution
# ─────────────────────────────────────────────

########  Defining the constraints  ########

def apply_all_constraints(loads: np.ndarray) -> np.ndarray:
    """Forces the optimizer's guess to comply with human training rules."""
    loads = np.maximum(0, loads)  # Enforce non-negative loads

    # At least one rest day (0 km) per week
    for w in range(len(loads) // 7):
        week_slice = slice(w*7, (w+1)*7)
        min_day_idx = np.argmin(loads[week_slice]) # Find the day scheduled to min volume
        loads[w*7 + min_day_idx] = 0.0             # and force it to 0

    # First week volume anchored to 40 km (realistic for intermediate runner)
    first_week_sum = loads[:7].sum()
    if first_week_sum > 0:
        loads[:7] *= (40.0 / first_week_sum)
    else:                                          # Fallback if the whole week was zeroed out
        loads[:7] = 40.0 / 6.0
        loads[np.argmin(loads[:7])] = 0.0

    # max 10% ramp rate per week to prevent injury
    for w in range(1, len(loads) // 7):
        prev_sum = loads[(w-1)*7 : w*7].sum()
        curr_sum = loads[w*7 : (w+1)*7].sum()
        if curr_sum > prev_sum * 1.10:
            scale = (prev_sum * 1.10) / curr_sum if curr_sum > 0 else 0
            loads[w*7 : (w+1)*7] *= scale

    # Specific Taper Caps (Weeks 14, 15, 16)
    wk13_sum = loads[12 * 7: 13 * 7].sum()

    # Week 14: 20% lower than Week 13
    wk14_slice = slice(13 * 7, 14 * 7)
    if loads[wk14_slice].sum() > (wk13_sum * 0.80):
        loads[wk14_slice] *= (wk13_sum * 0.80) / loads[wk14_slice].sum()

    # Week 15: 40% lower than Week 13
    wk15_slice = slice(14 * 7, 15 * 7)
    if loads[wk15_slice].sum() > (wk13_sum * 0.60):
        loads[wk15_slice] *= (wk13_sum * 0.60) / loads[wk15_slice].sum()

    # Week 16: Race Week
    wk16_slice = slice(15 * 7, 16 * 7 - 1)  # excluding race day
    if loads[wk16_slice].sum() > (wk13_sum * 0.35):
        loads[wk16_slice] *= (wk13_sum * 0.35) / loads[wk16_slice].sum()

    # At least one 32 km run (to prepare for the marathon specifically)
    # We find the day the optimizer already assigned the highest load, and boost it to 32 km
    # if it hasn't organically reached that distance.
    max_day_idx = np.argmax(loads)
    if loads[max_day_idx] < 32.0:
        loads[max_day_idx] = 32.0

    # No tiny runs (under 5 km) before the taper weeks
    is_not_taper = np.arange(len(loads)) < (n_days - 21)
    small_run_mask = (loads > 0) & (loads < 5.0) & is_not_taper
    snap_to_zero = small_run_mask & (loads < 2.5)
    loads[snap_to_zero] = 0.0
    snap_to_five = small_run_mask & (loads >= 2.5)
    loads[snap_to_five] = 5.0

    # No run above 32 km
    loads = np.minimum(loads, 32.0)

    # THE MARATHON
    # force the final day to the race distance: does not affect final performance
    # implemented here, after the 32 km cap applied during the training block
    loads[-1] = 42.2

    return loads

########  The DE algorithm  ########

def de_objective(raw_loads: np.ndarray) -> float:
    """ The DE algorithm passes in an array. We repair it to meet constraints,
        run the simulation, and return the negative performance. """
    repaired_loads = apply_all_constraints(raw_loads)
    perf, _, _, _ = simulate_busso(repaired_loads, params_busso)
    return -perf[-1]

# Set bounds for the optimizer: "search space" for each day
bounds = [(0, 32) for _ in range(n_days)]

result = differential_evolution(
    de_objective,
    bounds,
    strategy='best1bin',
    maxiter=1000,      # Maximum generations
    popsize=15,        # Multiplier for population size (15 * 112 days = 1680 individuals)
    tol=0.01,          # Convergence tolerance
    disp=True          # Print progress to console
)

# The optimizer returns the raw parameters. We must repair them one last time
# to get the final, usable training schedule.
optimal_loads = apply_all_constraints(result.x)
final_perf, final_g, final_h, final_k2 = simulate_busso(optimal_loads, params_busso)

rounded_loads = np.rint(optimal_loads) # rounding the final loads for readability of the training schedule
rounded_loads[-1] = 42.2

print(f"Final Race-Day Performance: {final_perf[-1]:.2f} AU")

# ─────────────────────────────────────────────
# Visualize DE Results
# ─────────────────────────────────────────────

def print_detailed_summary(loads):
    print("\n" + "=" * 65)
    print(f"{'WEEK':<5} | {'TOTAL':<8} | {'DAILY BREAKDOWN (km)':<40}")
    print("-" * 65)

    for w in range(len(loads) // 7):
        week_data = loads[w * 7: (w + 1) * 7]
        total_km = week_data.sum()

        # Format the 7 days into a clean string
        day_str = "  ".join([f"{d:4.1f}" for d in week_data])

        print(f"W{w + 1:>2}   | {total_km:>6.1f} | {day_str}")

    print("=" * 65)
    print(f"TOTAL PLAN DISTANCE: {loads.sum():.1f} km")

def plot_optimization_results(loads, perf, g, h):
    days = np.arange(len(loads))
    n_weeks = len(loads) // 7
    weekly_totals = [loads[w * 7:(w + 1) * 7].sum() for w in range(n_weeks)]

    # 1. DAILY LOADS
    plt.figure(figsize=(12, 5))
    plt.bar(days, loads, color='royalblue', alpha=0.6, label='Daily Load (km)')
    plt.ylabel('Distance (km)')
    plt.title('Optimized 16-Week Training Plan (Daily Loads)')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2. WEEKLY VOLUME
    plt.figure(figsize=(12, 5))
    plt.bar(range(1, n_weeks + 1), weekly_totals, color='orange', edgecolor='black', alpha=0.8)
    plt.plot(range(1, n_weeks + 1), weekly_totals, color='darkred', marker='o', linestyle='--', linewidth=1)
    plt.title('Total Weekly Training Volume')
    plt.xlabel('Week Number')
    plt.ylabel('Total Distance (km)')
    plt.xticks(range(1, n_weeks + 1))
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # 3. PERFORMANCE DYNAMICS
    plt.figure(figsize=(12, 5))
    plt.plot(days, g, color='green', linewidth=2, label='Fitness (g)')
    plt.plot(days, h, color='red', linewidth=2, label='Fatigue (h)')
    plt.plot(days, perf, color='black', linewidth=3, linestyle='--', label='Performance (p)')
    plt.ylabel('AU / Units')
    plt.xlabel('Days')
    plt.title('Busso Model: Performance as a function of Fitness vs. Fatigue')
    plt.legend(loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

print_detailed_summary(rounded_loads)
plot_optimization_results(optimal_loads, final_perf, final_g, final_h)