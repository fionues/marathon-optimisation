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
    p0:   float = 50.0   # baseline performance (AU) -> au: arbitrary units. normalized to 100, so 50 means "halfway to peak performance"
    k1b:  float = 0.008   # baseline gain scaling factor (k̄1)
    k2:   float = 1.8    # fatigue gain factor (fixed)
    tau1: float = 45.0   # fitness time constant (days)
    tau2: float = 15.0   # fatigue time constant (days)
    tau3: float = 38.0   # gain adaptation time constant (days)

# ─────────────────────────────────────────────
# 2.  BUSSO VDR MODEL
# ─────────────────────────────────────────────
def simulate_busso(loads: np.ndarray,
                   params: BussoParams
                   ) -> np.ndarray:
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
    k1      : variable gain factor k1(t)
    """
    n = len(loads)
    g    = np.zeros(n)   # fitness
    h    = np.zeros(n)   # fatigue
    k1   = np.zeros(n)   # variable gain
    perf = np.zeros(n)

    d1 = math.exp(-1.0 / params.tau1)
    d2 = math.exp(-1.0 / params.tau2)
    d3 = math.exp(-1.0 / params.tau3)

    # Initialise k1 at its baseline value
    k1_prev = params.k1b
    g_prev  = 0.0
    h_prev  = 0.0

    for t in range(n):
        w = loads[t]

        # Performance uses k1 from the *previous* day (pre-load)
        perf[t] = params.p0 + k1_prev * g_prev - params.k2 * h_prev

        # Update state variables
        g[t]  = g_prev  * d1 + w
        h[t]  = h_prev  * d2 + w
        k1[t] = k1_prev * d3 + params.k1b * w   # variable gain driven by load

        g_prev  = g[t]
        h_prev  = h[t]
        k1_prev = k1[t]

    return perf


params_busso = BussoParams()
n_days = 112  # training horizon leading up to race day

def busso_objective(loads: np.ndarray) -> float:
    """Negative race-day performance (minimizing this maximizes performance)."""
    perf = simulate_busso(loads, params_busso)
    return -perf[-1]


###  Gradient Descent Optimization of Busso Training Simulation ####
print('Gradient Descent Optimization of Busso Training Simulation ...')

# gradient descent constraints:
# Cap total weekly load
def project_weekly_load(loads, weekly_cap=15.0):
    loads = loads.copy()
    n_weeks = len(loads) // 7
    for w in range(n_weeks):
        s, e = w * 7, (w + 1) * 7
        week_sum = loads[s:e].sum()
        if week_sum > weekly_cap:
            loads[s:e] *= weekly_cap / week_sum  # scale down proportionally
    return loads

# Limit day-to-day load increases (injury prevention)
def project_ramp_rate_weekly(loads, max_weekly_increase=0.10):
    loads = loads.copy()
    for w in range(1, len(loads) // 7):
        prev_sum = loads[(w-1)*7 : w*7].sum()
        curr_sum = loads[w*7 : (w+1)*7].sum()
        if curr_sum > prev_sum * (1 + max_weekly_increase):
            scale = (prev_sum * (1 + max_weekly_increase)) / curr_sum
            loads[w*7 : (w+1)*7] *= scale
    return loads
    
def project_constraints(loads, weekly_cap=15.0, max_daily_increase=0.10):
    loads = np.clip(loads, 0, None)              # non-negative
    loads = project_weekly_load(loads, weekly_cap)
    loads = project_ramp_rate_weekly(loads, max_daily_increase)
    return loads

# Initial load schedule and hyperparameters
loads_init = np.ones(n_days) * 0.5       # starting point: constant daily load
rate = 0.01                              # learning rate alpha
precision = 1e-6                         # convergence threshold
delta_loss = 1000.0
max_iters = 500                          # maximum number of iterations
iters = 0                                # iteration counter

dg = nd.Gradient(busso_objective)
loads = loads_init.copy()
obj_history = [busso_objective(loads)]

while delta_loss > precision and iters < max_iters:
    loads -= rate * dg(loads)            # gradient descent step
    loads = project_constraints(loads)
    obj_history.append(busso_objective(loads))
    delta_loss = abs(obj_history[-2] - obj_history[-1])
    iters += 1

print(f"\tIterations: {iters}  Race-day performance: {-obj_history[-1]:.4f}")


# ################################  E4: Scipy solvers ################################
from scipy.optimize import minimize

# L-BFGS-B cannot handle constraints
# gradient_busso = nd.Jacobian(busso_objective)

# res_bfgs = minimize(
#     busso_objective,
#     loads_init,
#     method='L-BFGS-B',
#     jac=lambda x: gradient_busso(x).ravel(),
#     bounds=[(0, None)] * n_days,          # non-negative
#     constraints=constraints
# )
# print(f'Maximal race-day performance (L-BFGS-B): {-res_bfgs.fun:.4f}')

# perf_scipy = simulate_busso(res_bfgs.x, params_busso)

# SLSQP constraints:
# Cap total weekly load
constraints = [
    {'type': 'ineq', 'fun': lambda x, i=i: 80 - x[i*7:(i+1)*7].sum()}
    for i in range(n_days // 7)
]
# Limit day-to-day load increases (injury prevention)
constraints += [
    {
        'type': 'ineq',
        'fun': lambda x, w=w: 1.1 * x[(w-1)*7 : w*7].sum() - x[w*7 : (w+1)*7].sum()
    }
    for w in range(1, n_days // 7)   # <10% weekly ramp
]

res_slsqp = minimize(
    busso_objective,
    loads_init,
    method='SLSQP',
    bounds=[(0, 3)] * n_days,        # list of (min, max) per variable
    constraints=constraints,
    options={'ftol': 1e-9, 'maxiter': 1000}
)

perf_scipy_slsqp = simulate_busso(res_slsqp.x, params_busso)
print(f'Maximal race-day performance (SLSQP): {-res_slsqp.fun:.4f}')


# ##########################   Scipy Simulated Annealing  ###########################
from scipy.optimize import dual_annealing

def busso_objective_penalized(loads, weekly_cap=15.0, ramp_rate=0.10, penalty_weight=1e4):
    perf = -simulate_busso(loads, params_busso)[-1]
    
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
    bounds=[(0, 3)] * n_days,
    maxiter=1000,
    seed=42
)

print(f'Dual Annealing performance: {-res_da.fun:.4f}')


# ###### plots #########
import os
script_dir = os.path.dirname(os.path.abspath(__file__))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot([-o for o in obj_history])
axes[0].set_xlabel('Iterations')
axes[0].set_ylabel('Race-day Performance')
axes[0].set_title('Gradient Descent Convergence (Busso)')

perf_gd = simulate_busso(loads, params_busso)
axes[1].bar(range(n_days), loads, alpha=0.6, label='Optimized Loads (GD)')
axes[1].plot(perf_gd, color='r', label='Performance (GD)')
axes[1].set_xlabel('Day')
axes[1].set_title('Optimized Load Schedule & Performance Trajectory')
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "marathon_optimisation_gradient.png"), dpi=150, bbox_inches='tight')
# plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

x = np.arange(n_days)
w = 0.6  # bar width
axes[0].bar(x - w, loads,        width=w, alpha=0.7, label='Optimal Loads (GD)')
axes[0].bar(x,     res_slsqp.x,  width=w, alpha=0.7, label='Optimal Loads (SLSQP)')
axes[0].bar(x + w, res_da.x,     width=w, alpha=0.7, label='Optimal Loads (Dual Annealing)')
axes[0].set_xlabel('Day')
axes[0].set_ylabel('Training Load')
axes[0].set_title('Optimal Load Schedules')
axes[0].legend()

# axes[1].plot(perf_scipy, label=f"L-BFGS-B  (perf={-res_bfgs.fun:.2f})")
axes[1].plot(perf_gd, linestyle='--', label=f"GD  (perf={-obj_history[-1]:.2f})")
axes[1].plot(perf_scipy_slsqp, label=f"SLSQP  (perf={-res_slsqp.fun:.2f})")
axes[1].plot(simulate_busso(res_da.x, params_busso), label=f"Dual Annealing  (perf={-res_da.fun:.2f})")
axes[1].set_xlabel('Day')
axes[1].set_ylabel('Performance')
axes[1].set_title('Performance Trajectories')
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "marathon_optimisation_scipy.png"), dpi=150, bbox_inches='tight')
plt.show()