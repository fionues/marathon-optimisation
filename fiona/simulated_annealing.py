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
# @dataclass
# class BussoParams:
#     p0:   float = 0    # baseline performance (AU) -> AU: arbitrary units
#     k1:   float = 1     # fitness gain factor (fixed, the magnitude of fitness gained by a unit of training) TODO: re-check number - depends on the athlete
#     k3:   float = 0.05  # fatigue sensitivity multiplier (drives dynamic k2: the magnitude of fatigue incurred by a unit of training) TODO: re-check number - depends on the athlete
#     tau1: float = 45.0    # fitness decay constant (days)
#     tau2: float = 15.0    # fatigue decay constant (days)
#     tau3: float = 5    # fatigue sensitivity decay constant (days)
@dataclass
class BussoParams:
    p0:   float = 0    # baseline performance (AU) -> AU: arbitrary units
    k1:   float = 0.031     # fitness gain factor (fixed, the magnitude of fitness gained by a unit of training) TODO: re-check number - depends on the athlete
    k3:   float = 0.000035  # fatigue sensitivity multiplier (drives dynamic k2: the magnitude of fatigue incurred by a unit of training) TODO: re-check number - depends on the athlete
    tau1: float = 30.8    # fitness decay constant (days)
    tau2: float = 16.8    # fatigue decay constant (days)
    tau3: float = 2.3    # fatigue sensitivity decay constant (days)

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

# ── Shared constants ──────────────────────────────────────────────────────────
PENALTY_WEIGHT = 1e4   # multiplier that converts constraint violations into cost
RAMP_RATE      = 0.10  # max 10 % week-on-week volume increase
FIRST_WEEK_KM  = 40.0  # anchor for week 1 total volume
LONG_RUN_KM    = 32.0  # minimum distance of the longest run in the plan
MAX_RUN_KM     = 32.0  # hard cap per single run
MARATHON_KM    = 42.2  # race-day distance (last day)


# ─────────────────────────────────────────────────────────────────────────────
# APPROACH A — HARD CONSTRAINTS  (repair / projection)
# Each candidate solution is projected onto the feasible region before the
# model is evaluated.  The optimizer never sees an infeasible point.
# ─────────────────────────────────────────────────────────────────────────────
def apply_all_constraints(loads: np.ndarray) -> np.ndarray:
    """Projects a candidate load vector onto the feasible region."""
    loads = np.maximum(0, loads)                       # non-negative loads

    # At least one rest day (0 km) per week
    for w in range(len(loads) // 7):
        min_day_idx = np.argmin(loads[w*7:(w+1)*7])
        loads[w*7 + min_day_idx] = 0.0

    # First week anchored to FIRST_WEEK_KM
    first_week_sum = loads[:7].sum()
    if first_week_sum > 0:
        loads[:7] *= (FIRST_WEEK_KM / first_week_sum)
    else:
        loads[:7] = FIRST_WEEK_KM / 6.0
        loads[np.argmin(loads[:7])] = 0.0

    # Max 10 % ramp rate per week
    for w in range(1, len(loads) // 7):
        prev_sum = loads[(w-1)*7 : w*7].sum()
        curr_sum = loads[w*7 : (w+1)*7].sum()
        if curr_sum > prev_sum * (1 + RAMP_RATE):
            scale = (prev_sum * (1 + RAMP_RATE)) / curr_sum if curr_sum > 0 else 0
            loads[w*7 : (w+1)*7] *= scale

    # At least one run of exactly LONG_RUN_KM (boost the busiest day if needed)
    max_day_idx = np.argmax(loads)
    if loads[max_day_idx] < LONG_RUN_KM:
        loads[max_day_idx] = LONG_RUN_KM

    # No single run above MAX_RUN_KM
    loads = np.minimum(loads, MAX_RUN_KM)

    # No load in the open interval (0, 5): snap to 0 if < 2.5, else to 5
    loads = np.where((loads > 0) & (loads < 5), np.where(loads < 2.5, 0.0, 5.0), loads)

    # Race day
    loads[-1] = MARATHON_KM

    return loads


def busso_objective_hard(loads: np.ndarray) -> float:
    """Repair → simulate → return negative race-day performance."""
    loads = apply_all_constraints(loads.copy())
    perf, _, _, _ = simulate_busso(loads, params_busso)
    return -perf[-1]


# ─────────────────────────────────────────────────────────────────────────────
# APPROACH B — PENALTY CONSTRAINTS
# Infeasible solutions are allowed but penalised.  The same five rules are
# mirrored here as quadratic penalties so the comparison is apples-to-apples.
# ─────────────────────────────────────────────────────────────────────────────
def busso_objective_penalty(loads: np.ndarray) -> float:
    """Simulate on raw loads, then add penalty for every constraint violated."""
    perf, _, _, _ = simulate_busso(loads, params_busso)
    objective = -perf[-1]

    penalty = 0.0
    n_weeks = len(loads) // 7

    # Penalty 1: non-negative loads
    penalty += np.sum(np.minimum(loads, 0) ** 2)

    # Penalty 2: at least one rest day per week
    #   → penalise the minimum daily load in each week for being > 0
    for w in range(n_weeks):
        week = loads[w*7:(w+1)*7]
        penalty += min(week) ** 2

    # Penalty 3: first week anchored to FIRST_WEEK_KM
    first_week_sum = loads[:7].sum()
    penalty += (first_week_sum - FIRST_WEEK_KM) ** 2

    # Penalty 4: max 10 % ramp rate per week
    for w in range(1, n_weeks):
        prev_sum = loads[(w-1)*7 : w*7].sum()
        curr_sum = loads[w*7 : (w+1)*7].sum()
        penalty += max(0, curr_sum - prev_sum * (1 + RAMP_RATE)) ** 2

    # Penalty 5a: at least one run >= LONG_RUN_KM
    #   → penalise the gap if the longest run falls short
    penalty += max(0, LONG_RUN_KM - loads.max()) ** 2

    # Penalty 5b: no run above MAX_RUN_KM
    penalty += np.sum(np.maximum(0, loads - MAX_RUN_KM) ** 2)

    # Penalty 6: race day = MARATHON_KM
    penalty += (loads[-1] - MARATHON_KM) ** 2

    # Penalty 7: tapering — last 2 weeks must reduce volume relative to peak
    #   week n-2 must be ≤ 80 % of the peak non-taper week; week n-1 ≤ 60 %
    if n_weeks >= 3:
        peak_vol = max(loads[w*7:(w+1)*7].sum() for w in range(n_weeks - 2))
        for taper_idx, taper_cap in enumerate([0.80, 0.60]):
            w = n_weeks - 2 + taper_idx
            week_sum = loads[w*7 : (w+1)*7].sum()
            penalty += max(0, week_sum - taper_cap * peak_vol) ** 2

    # Penalty 8: no load in the open interval (0, 5) — must be 0 or ≥ 5
    #   penalise by the squared distance to the nearest feasible value
    in_gap = (loads > 0) & (loads < 5)
    penalty += np.sum(np.minimum(loads, 5 - loads)[in_gap] ** 2)

    return objective + PENALTY_WEIGHT * penalty


# ─────────────────────────────────────────────────────────────────────────────
# RUN BOTH OPTIMISERS
# ─────────────────────────────────────────────────────────────────────────────
# print("Running Dual Annealing — hard constraints …")
# res_hard = dual_annealing(
#     busso_objective_hard,
#     bounds=[(0, MAX_RUN_KM)] * n_days,
#     maxiter=1000,
#     seed=42,
# )
# res_hard.x = apply_all_constraints(res_hard.x)   # guarantee feasibility on output

print("Running Dual Annealing — penalty constraints …")

# Track convergence: record best objective value at each iteration
_convergence_history: List[float] = []
_best_so_far = [np.inf]

def _sa_callback(x, f, context):
    if f < _best_so_far[0]:
        _best_so_far[0] = f
    _convergence_history.append(_best_so_far[0])

# there is no other stop mechanism than maxiter? So we needed to plot the convergence curve with a lot of iterations, to find that 100 is enough
res_penalty = dual_annealing(
    busso_objective_penalty,
    bounds=[(0, MAX_RUN_KM)] * n_days,
    maxiter=100,
    seed=42,
    callback=_sa_callback,
)

# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
# loads_hard    = res_hard.x
loads_penalty = res_penalty.x

# perf_hard,    g_hard,    h_hard,    k2_hard    = simulate_busso(loads_hard,    params_busso)
perf_penalty, g_penalty, h_penalty, k2_penalty = simulate_busso(loads_penalty, params_busso)

def constraint_report(label: str, loads: np.ndarray) -> None:
    """Print a quick feasibility check for a given load vector."""
    n_weeks = len(loads) // 7
    rest_ok    = all((loads[w*7:(w+1)*7] == 0).any()      for w in range(n_weeks))
    ramp_ok    = all(
        loads[w*7:(w+1)*7].sum() <= loads[(w-1)*7:w*7].sum() * (1 + RAMP_RATE) + 1e-6
        for w in range(1, n_weeks)
    )
    long_ok    = loads.max() >= LONG_RUN_KM - 1e-6
    cap_ok     = (loads[:-1] <= MAX_RUN_KM + 1e-6).all()   # exclude race day
    week1_ok   = abs(loads[:7].sum() - FIRST_WEEK_KM) < 1.0

    print(f"\n  {label}")
    # print(f"    Race-day performance : {perf_hard[-1] if 'Hard' in label else perf_penalty[-1]:.4f} AU")
    print(f"    Total plan distance  : {loads.sum():.1f} km")
    print(f"    Rest day every week  : {'✓' if rest_ok  else '✗  VIOLATED'}")
    print(f"    ≤10% weekly ramp     : {'✓' if ramp_ok  else '✗  VIOLATED'}")
    print(f"    ≥1 run of {LONG_RUN_KM:.0f} km     : {'✓' if long_ok  else '✗  VIOLATED'}")
    print(f"    No run > {MAX_RUN_KM:.0f} km       : {'✓' if cap_ok   else '✗  VIOLATED'}")
    print(f"    Week 1 ≈ {FIRST_WEEK_KM:.0f} km         : {'✓' if week1_ok else '✗  VIOLATED'}")

print("\n" + "=" * 55)
print("  CONSTRAINT APPROACH COMPARISON")
print("=" * 55)
# constraint_report("Hard constraints (repair)",   loads_hard)
constraint_report("Penalty constraints",         loads_penalty)
print("=" * 55)

# print(f'Iterations hard: {res_hard.nit}')
# print(f'Function evaluations hard: {res_hard.nfev}')
print(f'Iterations penalty: {res_penalty.nit}')
print(f'Function evaluations penalty: {res_penalty.nfev}')

# Point downstream plotting/printing at whichever result you want to inspect.
# Change this flag to 'penalty' to switch.
# INSPECT = 'hard'   # 'hard' | 'penalty'

# optimal_loads = loads_hard    if INSPECT == 'hard' else loads_penalty
# final_perf    = perf_hard     if INSPECT == 'hard' else perf_penalty
# final_g       = g_hard        if INSPECT == 'hard' else g_penalty
# final_h       = h_hard        if INSPECT == 'hard' else h_penalty
# final_k2      = k2_hard       if INSPECT == 'hard' else k2_penalty

# ─────────────────────────────────────────────
# 5. VISUALIZE Results
# ─────────────────────────────────────────────
import os
script_dir = os.path.dirname(os.path.abspath(__file__))



def plot_performance_dynamics(loads, perf, g, h, k2, INSPECT):
    days = np.arange(len(loads))
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    ax1.bar(days, loads, color='royalblue', alpha=0.6, label='Daily Load (km)')
    ax1.set_ylabel('Distance (km)')
    ax1.set_title(f'Optimized 16-Week Training Plan (Dual Annealing — {INSPECT} constraints)')
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
    plt.savefig(os.path.join(script_dir, f"simulated_annealing_performance_{INSPECT}v3.png"), dpi=150, bbox_inches='tight')
    # plt.show()


def plot_weekly_volume(loads, INSPECT):
    n_weeks = len(loads) // 7
    weekly_totals = [loads[w * 7:(w + 1) * 7].sum() for w in range(n_weeks)]

    plt.figure(figsize=(10, 5))
    plt.bar(range(1, n_weeks + 1), weekly_totals, color='orange', edgecolor='black', alpha=0.8)

    plt.plot(range(1, n_weeks + 1), weekly_totals, color='darkred', marker='o', linestyle='--', linewidth=1)

    plt.title(f'Total Weekly Training Volume ({INSPECT} constraints)')
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
    plt.savefig(os.path.join(script_dir, f"simulated_annealing_weekly_volume_{INSPECT}v3.png"), dpi=150, bbox_inches='tight')
    # plt.show()

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

def plot_convergence(history: List[float], INSPECT: str):
    """Plot best objective value vs. iteration to visualise SA convergence."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history, color='steelblue', linewidth=1.5, label='Best objective (− race-day perf)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Best objective value')
    ax.set_title(f'Simulated Annealing — Convergence ({INSPECT} constraints)')
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, f"simulated_annealing_convergence_{INSPECT}v3.png"), dpi=150, bbox_inches='tight')

def print_and_show_plots():
    plt.show()

## plots hard constraints
# print_weekly_summary(loads_hard)
# print_detailed_summary(loads_hard)

# plot_performance_dynamics(loads_hard, perf_hard, g_hard, h_hard, k2_hard, 'hard')
# plot_weekly_volume(loads_hard, 'hard')

## plots penalty constraints
print_weekly_summary(loads_penalty)
print_detailed_summary(loads_penalty)
plot_performance_dynamics(loads_penalty, perf_penalty, g_penalty, h_penalty, k2_penalty, 'penalty')
plot_weekly_volume(loads_penalty, 'penalty')
plot_convergence(_convergence_history, 'penalty')

print_and_show_plots()