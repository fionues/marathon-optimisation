import sympy as sp
import numpy as np
import numdifftools as nd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass
from typing import List, Tuple
import math, random

# ─────────────────────────────────────────────
# 1.  MODEL PARAMETERS
# ─────────────────────────────────────────────
@dataclass
class BussoParams:
    p0:   float = 0    # baseline performance (AU) -> AU: arbitrary units
    k1:   float = 1     # fitness gain factor (fixed, the magnitude of fitness gained by a unit of training) TODO: re-check number - depends on the athlete
    k3:   float = 0.05  # fatigue sensitivity multiplier (drives dynamic k2: the magnitude of fatigue incurred by a unit of training) TODO: re-check number - depends on the athlete
    tau1: float = 45.0    # fitness decay constant (days)
    tau2: float = 15.0    # fatigue decay constant (days)
    tau3: float = 5    # fatigue sensitivity decay constant (days)

# ─────────────────────────────────────────────
# 2.  BUSSO VDR (Variable Dose-Response) MODEL: as training accumulates, the body becomes more susceptible to *fatigue*
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

    d1 = math.exp(-1.0 / params.tau1) # fitness decay factor: every day, the athlete retains 97.8% of their fitness from the day before
    d2 = math.exp(-1.0 / params.tau2) # fatigue decay factor: the athlete retains 93.5% of yesterday's fatigue
    d3 = math.exp(-1.0 / params.tau3) # every day, the athlete retains 97.4% of their compounding fatigue sensitivity from the day before TODO check

    k2_prev = 0.0 # Fatigue sensitivity starts at 0 TODO check
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
n_days = 112  # training horizon leading up to race day

def busso_objective(loads: np.ndarray) -> float:
    """Negative race-day performance (minimizing this maximizes performance)."""
    perf = simulate_busso(loads, params_busso)
    return -perf[-1]


# ##########################   Scipy Simulated Annealing  ###########################
from scipy.optimize import dual_annealing

def busso_objective_penalized(loads, weekly_cap=15.0, ramp_rate=0.10, penalty_weight=1e4):
    # TODO what is the correct way to apply penalties? should it be done before the model runs or after. for evoluionary algorithm its before.
    perf, _, _, _ = simulate_busso(loads, params_busso)
    perf = -perf[-1]
    
    penalty = 0.0
    for w in range(n_days // 7):
        curr_week = loads[w*7 : (w+1)*7].sum()
        
        # Constraint 1: weekly volume cap
        penalty += max(0, curr_week - weekly_cap) ** 2
        
        # Constraint 2: weekly ramp rate (skip first week, no previous to compare)
        if w > 0:
            prev_week = loads[(w-1)*7 : w*7].sum()
            penalty += max(0, curr_week - (1 + ramp_rate) * prev_week) ** 2

    return perf + penalty_weight * penalty

res_da = dual_annealing(
    busso_objective_penalized,
    bounds=[(0, 3)] * n_days, # upper bound of 3 is derived from weekly cap of 15, because if the athlete only trains 5 days a week, he can't exceed 15/5 = 3 on those days. So 3 is a reasonable upper bound for daily load.
    #TODO check that the bounds are the same as in the other alog. bounds=[(0, 32) for _ in range(n_days)], # allow higher upper bound to give the optimizer more freedom, but the penalty will discourage it from going too high
    maxiter=1000,
    seed=42
)

print(f'Dual Annealing performance: {-res_da.fun:.4f}')

optimal_loads = res_da.x
final_perf, final_g, final_h, final_k2 = simulate_busso(optimal_loads, params_busso)

# ─────────────────────────────────────────────
# 5. VISUALIZE Results
# ─────────────────────────────────────────────

def plot_performance_dynamics(loads, perf, g, h, k2):
    days = np.arange(len(loads))
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    ax1.bar(days, loads, color='royalblue', alpha=0.6, label='Daily Load (km)')
    ax1.set_ylabel('Distance (km)')
    ax1.set_title('Optimized 16-Week Training Plan (Dual Annealing)')
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.legend()

    ax2.plot(days, g, color='green', linewidth=2, label='Fitness (g) - Gain')
    ax2.plot(days, h, color='red', linewidth=2, label='Fatigue (h) - Drain')
    ax2.set_ylabel('Impulse Units')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Subplot 3: Race Readiness (p) & Fatigue Sensitivity (k2)
    ax3.plot(days, perf, color='black', linewidth=2.5, label='Performance (p)')
    ax3.set_ylabel('Performance (AU)', color='black')

    # Create a second y-axis for k2 since its scale is much smaller (e.g., 0.0005)
    ax3b = ax3.twinx()
    ax3b.plot(days, k2, color='purple', linestyle='--', alpha=0.7, label='Fatigue Sensitivity (k2)')
    ax3b.set_ylabel('k2 Factor', color='purple')

    ax3.set_xlabel('Days')
    ax3.legend(loc='upper left')
    ax3b.legend(loc='upper right')
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_weekly_volume(loads):
    n_weeks = len(loads) // 7
    weekly_totals = [loads[w * 7:(w + 1) * 7].sum() for w in range(n_weeks)]

    plt.figure(figsize=(10, 5))
    plt.bar(range(1, n_weeks + 1), weekly_totals, color='orange', edgecolor='black', alpha=0.8)

    plt.plot(range(1, n_weeks + 1), weekly_totals, color='darkred', marker='o', linestyle='--', linewidth=1)

    plt.title('Total Weekly Training Volume')
    plt.xlabel('Week Number')
    plt.ylabel('Total Distance (km)')
    plt.xticks(range(1, n_weeks + 1))
    plt.grid(axis='y', linestyle=':', alpha=0.6)

    # Annotate the max week
    max_vol = max(weekly_totals)
    max_week = weekly_totals.index(max_vol) + 1
    plt.annotate(f'Peak: {max_vol:.1f}km', xy=(max_week, max_vol), xytext=(max_week, max_vol + 5),
                 arrowprops=dict(facecolor='black', shrink=0.05), ha='center')

    plt.tight_layout()
    plt.show()

def print_weekly_summary(loads):
    print("\n" + "=" * 45)
    print(f"{'WEEK':<6} | {'TOTAL KM':<10} | {'LONGEST RUN':<12} | {'REST'}")
    print("-" * 45)

    for w in range(len(loads) // 7):
        week_data = loads[w * 7: (w + 1) * 7]
        total_km = week_data.sum()
        max_run = week_data.max()
        has_rest = 0.0 in week_data

        print(f"{w + 1:<6} | {total_km:<10.1f} | {max_run:<12.1f} | {'Yes' if has_rest else 'No'}")
    print("=" * 45)

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

print_weekly_summary(optimal_loads)
print_detailed_summary(optimal_loads)

plot_performance_dynamics(optimal_loads, final_perf, final_g, final_h, final_k2)
plot_weekly_volume(optimal_loads)