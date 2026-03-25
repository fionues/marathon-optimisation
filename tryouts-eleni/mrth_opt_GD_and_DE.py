import numpy as np
import numdifftools as nd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple
import math
from scipy.optimize import differential_evolution

# ─────────────────────────────────────────────
# 1.  MODEL PARAMETERS
# ─────────────────────────────────────────────
@dataclass
class BussoParams:
    p0:   float = 50.0    # baseline performance (AU) -> AU: arbitrary units, normalized to 100, so 50 means "halfway to peak performance"
    k1:   float = 0.1     # fitness gain factor (fixed, the magnitude of fitness gained by a unit of training) TODO: re-check number - depends on the athlete
    k3:   float = 0.0005  # fatigue sensitivity multiplier (drives dynamic k2: the magnitude of fatigue incurred by a unit of training) TODO: re-check number - depends on the athlete
    tau1: float = 45.0    # fitness decay constant (days)
    tau2: float = 15.0    # fatigue decay constant (days)
    tau3: float = 38.0    # fatigue sensitivity decay constant (days)

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
    d3 = math.exp(-1.0 / params.tau3) # every day, the athlete retains 97.4% of their compounding fatigue sensitivity from the day before

    k2_prev = 0.0 # Fatigue sensitivity starts at 0
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
# 3.  Gradient Descent Optimization of Busso Training Simulation
# ─────────────────────────────────────────────

# gradient descent constraints:
# (projection: taking a point that is outside an allowed area and moving it to the nearest "legal" point)

def project_non_negative(loads: np.ndarray) -> np.ndarray:
    """Ensures all training loads are 0 or greater."""
    return np.maximum(0, loads)

def project_first_week_volume(loads: np.ndarray, target_volume: float = 40.0) -> np.ndarray:
    """Forces the sum of the first 7 days to equal the target_volume.
        A 40 km first week is realistic for an intermediate athlete."""
    loads = loads.copy()
    first_week_sum = loads[:7].sum()

    if first_week_sum > 0:                          # Avoid division by zero
        scale = target_volume / first_week_sum
        loads[:7] *= scale
    else:
        loads[:7] = target_volume / 7.0             # Fallback: distribute evenly if sum is 0

    return loads

# Limit weekly load increase (10% rule for injury prevention)
def project_ramp_rate_weekly(loads, max_weekly_increase=0.10):
    loads = loads.copy()
    for w in range(1, len(loads) // 7):
        prev_sum = loads[(w-1)*7 : w*7].sum()
        curr_sum = loads[w*7 : (w+1)*7].sum()
        if curr_sum > prev_sum * (1 + max_weekly_increase):
            scale = (prev_sum * (1 + max_weekly_increase)) / curr_sum
            loads[w*7 : (w+1)*7] *= scale
    return loads

def project_constraints(loads: np.ndarray) -> np.ndarray:
    """ Applies all constraints to the load array. """

    loads_valid = project_non_negative(loads)
    loads_anchored = project_first_week_volume(loads_valid)
    loads_projected = project_ramp_rate_weekly(loads_anchored)

    return loads_projected

# Initial load schedule and hyperparameters
loads_init = np.ones(n_days) * 5     # starting point: constant daily load of 5 km
rate = 0.05                          # learning rate alpha
precision = 1e-6                     # convergence threshold
delta_loss = 1000.0
max_iters = 5000
iters = 0

# dg = nd.Gradient(busso_objective)
# loads = loads_init.copy()
# obj_history = [busso_objective(loads)]
#
# while delta_loss > precision and iters < max_iters:
#     loads -= rate * dg(loads)            # gradient descent step
#     loads = project_constraints(loads)
#     obj_history.append(busso_objective(loads))
#     delta_loss = abs(obj_history[-2] - obj_history[-1])
#     iters += 1
#
# print(f"\tIterations: {iters}  Race-day performance: {-obj_history[-1]:.4f}")
#
# # ─────────────────────────────────────────────
# # 4. VISUALIZATION AND PLOTTING
# # ─────────────────────────────────────────────
#
# final_perf, final_g, final_h, final_k2 = simulate_busso(loads, params_busso)
#
# n_weeks = n_days // 7
# weekly_load = [loads[w*7 : (w+1)*7].sum() for w in range(n_weeks)]
#
# fig, axs = plt.subplots(2, 2, figsize=(14, 10))
#
# axs[0, 0].plot(obj_history, color='purple', linewidth=2)
# axs[0, 0].set_title('Algorithm Convergence', fontsize=12, fontweight='bold')
# axs[0, 0].set_xlabel('Iteration')
# axs[0, 0].set_ylabel('Objective Value (Negative Performance)')
# axs[0, 0].grid(True, linestyle='--', alpha=0.6)
#
# axs[0, 1].bar(range(n_days), loads, color='skyblue', edgecolor='black', alpha=0.8)
# axs[0, 1].set_title('Optimized Daily Training Loads', fontsize=12, fontweight='bold')
# axs[0, 1].set_xlabel('Day of Training Block')
# axs[0, 1].set_ylabel('Daily Load (km)')
# axs[0, 1].grid(axis='y', linestyle='--', alpha=0.6)
#
# axs[1, 0].bar(range(1, n_weeks + 1), weekly_load, color='coral', edgecolor='black', alpha=0.8)
# axs[1, 0].set_title('Weekly Training Volume', fontsize=12, fontweight='bold')
# axs[1, 0].set_xlabel('Week')
# axs[1, 0].set_ylabel('Total Weekly Load (km)')
# axs[1, 0].set_xticks(range(1, n_weeks + 1))
# axs[1, 0].grid(axis='y', linestyle='--', alpha=0.6)
#
# axs[1, 1].plot(final_g, label='Fitness (g)', color='green', linewidth=2)
# axs[1, 1].plot(final_h, label='Fatigue (h)', color='red', linewidth=2)
# axs[1, 1].plot(final_perf, label='Performance (p)', color='blue', linewidth=2, linestyle='--')
# axs[1, 1].set_title('Busso Model: Physiological States', fontsize=12, fontweight='bold')
# axs[1, 1].set_xlabel('Day of Training Block')
# axs[1, 1].set_ylabel('Arbitrary Units (AU)')
# axs[1, 1].legend(loc='upper left')
# axs[1, 1].grid(True, linestyle='--', alpha=0.6)
#
# plt.tight_layout()
# plt.show()

# GD is not converging after 5000 iterations (and takes long to run...)
# Plus, we now need to introduce more constraints:
# A. at least one day per week should be 0 km (reality-check for athlete's every-day life)
#   this constraint is not handled well by algorithms that require differentiation (GD, SLSQP, Simulated Annealing)
#   -> try an evolutionary algorithm (does not use derivatives): Differential Evolution
# B. at least one 32 km run must be performed (reality-check when the target event is a marathon run)

# ─────────────────────────────────────────────
# 5. Differential Evolution
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

    # First week volume anchored to 40 km
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
            scale = (prev_sum * 1.10) / curr_sum if curr_sum > 0 else 0 # TODO: recheck trapping to 0
            loads[w*7 : (w+1)*7] *= scale

    # At least one 32 km run
    # We find the day the optimizer already assigned the highest load, and boost it to 32 km
    # if it hasn't organically reached that distance.
    max_day_idx = np.argmax(loads)
    if loads[max_day_idx] < 32.0:
        loads[max_day_idx] = 32.0

    # No tiny runs (under 5 km)
    small_run_mask = (loads > 0) & (loads < 5.0)
    snap_to_zero = small_run_mask & (loads < 2.5)
    loads[snap_to_zero] = 0.0
    snap_to_five = small_run_mask & (loads >= 2.5)
    loads[snap_to_five] = 5.0

    # No couch-potato taper:
    # Ensure the final weeks (15 and 16) have some runs to maintain 'feel'
    # TODO: implement this

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

print(f"Final Race-Day Performance: {final_perf[-1]:.2f} AU")

# ─────────────────────────────────────────────
# 5. VISUALIZE DE Results
# ─────────────────────────────────────────────

def plot_performance_dynamics(loads, perf, g, h, k2):
    days = np.arange(len(loads))
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    ax1.bar(days, loads, color='royalblue', alpha=0.6, label='Daily Load (km)')
    ax1.set_ylabel('Distance (km)')
    ax1.set_title('Optimized 16-Week Training Plan (DE)')
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