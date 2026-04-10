import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from scipy.optimize import differential_evolution

from busso_model import (
    simulate_busso, params_busso, n_days,
    RAMP_RATE, FIRST_WEEK_KM, LONG_RUN_KM, MAX_RUN_KM, MARATHON_KM,
)
from plots import (
    save_all_plots,
    print_weekly_summary, print_detailed_summary,
)

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, 'output')
os.makedirs(output_dir, exist_ok=True)

from dataclasses import dataclass, field
from typing import List, Tuple
import time

@dataclass
class ConvergenceLogger:
    """Records (nfev, best_value, elapsed_time) at each callback."""
    label: str
    trace: List[Tuple[int, float, float]] = field(default_factory=list)
    _start: float = field(default_factory=time.time, init=False, repr=False)

    def reset(self):
        self.trace = []
        self._start = time.time()

    def best_values(self):
        return [v for _, v, _ in self.trace]

    def nfevs(self):
        return [n for n, _, _ in self.trace]


# ─────────────────────────────────────────────
# HARD CONSTRAINTS (repair / projection) with taper caps
# ─────────────────────────────────────────────
def apply_all_constraints(loads: np.ndarray) -> np.ndarray:
    """Forces the optimizer's guess to comply with human training rules."""
    loads = np.maximum(0, loads)  # Enforce non-negative loads

    # At least one rest day (0 km) per week
    for w in range(len(loads) // 7):
        week_slice = slice(w*7, (w+1)*7)
        min_day_idx = np.argmin(loads[week_slice])
        loads[w*7 + min_day_idx] = 0.0

    # First week volume anchored to FIRST_WEEK_KM
    first_week_sum = loads[:7].sum()
    if first_week_sum > 0:
        loads[:7] *= (FIRST_WEEK_KM / first_week_sum)
    else:
        loads[:7] = FIRST_WEEK_KM / 6.0
        loads[np.argmin(loads[:7])] = 0.0

    # Max RAMP_RATE week-on-week increase to prevent injury
    for w in range(1, len(loads) // 7):
        prev_sum = loads[(w-1)*7 : w*7].sum()
        curr_sum = loads[w*7 : (w+1)*7].sum()
        if curr_sum > prev_sum * (1 + RAMP_RATE):
            scale = (prev_sum * (1 + RAMP_RATE)) / curr_sum if curr_sum > 0 else 0
            loads[w*7 : (w+1)*7] *= scale

    # Specific taper caps (Weeks 14, 15, 16)
    wk13_sum = loads[12 * 7: 13 * 7].sum()

    # Week 14: ≤ 80 % of Week 13
    wk14_slice = slice(13 * 7, 14 * 7)
    if loads[wk14_slice].sum() > wk13_sum * 0.80:
        loads[wk14_slice] *= (wk13_sum * 0.80) / loads[wk14_slice].sum()

    # Week 15: ≤ 60 % of Week 13
    wk15_slice = slice(14 * 7, 15 * 7)
    if loads[wk15_slice].sum() > wk13_sum * 0.60:
        loads[wk15_slice] *= (wk13_sum * 0.60) / loads[wk15_slice].sum()

    # Week 16 (excluding race day): ≤ 35 % of Week 13
    wk16_slice = slice(15 * 7, 16 * 7 - 1)
    if loads[wk16_slice].sum() > wk13_sum * 0.35:
        loads[wk16_slice] *= (wk13_sum * 0.35) / loads[wk16_slice].sum()

    # At least one run of LONG_RUN_KM (boost the day with highest load if needed)
    max_day_idx = np.argmax(loads)
    if loads[max_day_idx] < LONG_RUN_KM:
        loads[max_day_idx] = LONG_RUN_KM

    # No tiny runs (under 5 km) outside race week
    is_not_taper   = np.arange(len(loads)) < (n_days - 7)
    small_run_mask = (loads > 0) & (loads < 5.0) & is_not_taper
    loads[small_run_mask & (loads < 2.5)]  = 0.0
    loads[small_run_mask & (loads >= 2.5)] = 5.0

    # Hard cap per single run
    loads = np.minimum(loads, MAX_RUN_KM)

    # Race day
    loads[-1] = MARATHON_KM

    return loads


def de_objective(raw_loads: np.ndarray) -> float:
    """Repair to feasible region, simulate, return negative race-day performance."""
    repaired_loads = apply_all_constraints(raw_loads.copy())
    perf, _, _, _ = simulate_busso(repaired_loads, params_busso)
    return -perf[-1]

class TrackingObjective:
    """Wraps an objective function, logging every call."""
    def __init__(self, fn, logger: ConvergenceLogger):
        self.fn = fn
        self.logger = logger
        self.nfev = 0
        self.best = np.inf

    def __call__(self, x):
        val = self.fn(x)
        self.nfev += 1
        if val < self.best:
            self.best = val
            elapsed = time.time() - self.logger._start
            self.logger.trace.append((self.nfev, val, elapsed))
        return val

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_convergence(loggers: List[ConvergenceLogger], title="Convergence Curve"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"SA": "#E07B39", "DE": "#3A7EBF"}

    for logger in loggers:
        nfevs  = logger.nfevs()
        values = logger.best_values()
        color  = colors.get(logger.label, "grey")

        # Plot vs NFEv
        axes[0].plot(nfevs, values, label=logger.label, color=color, linewidth=1.8)
        # Plot vs time
        times = [t for _, _, t in logger.trace]
        axes[1].plot(times, values, label=logger.label, color=color, linewidth=1.8)

    for ax, xlabel in zip(axes, ["Function Evaluations (NFEv)", "Wall Time (s)"]):
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Best Objective Value (neg. performance)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))

    axes[0].set_title("Convergence vs. Budget")
    axes[1].set_title("Convergence vs. Time")
    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ─────────────────────────────────────────────
# RUN OPTIMISER
# ─────────────────────────────────────────────
bounds = [(0, MAX_RUN_KM) for _ in range(n_days)]

if __name__ == "__main__":
    _convergence_history: List[float] = []
    _best_so_far = [np.inf]

    def _de_callback(xk, convergence):
        current = de_objective(xk)
        if current < _best_so_far[0]:
            _best_so_far[0] = current
        _convergence_history.append(_best_so_far[0])

    de_logger  = ConvergenceLogger(label="DE")
    tracked_de = TrackingObjective(de_objective, de_logger)

    result = differential_evolution(
        tracked_de,
        bounds,
        strategy='best1bin',
        maxiter=1000,
        popsize=15,   # population = 15 × n_days individuals
        tol=0.01,
        disp=True,
        callback=_de_callback,
    )

    plot_convergence([de_logger], title="Differential Evolution Convergence")

    # ─────────────────────────────────────────────
    # RESULTS
    # ─────────────────────────────────────────────
    # Repair one final time to guarantee a fully feasible schedule
    optimal_loads = apply_all_constraints(result.x)
    final_perf, final_g, final_h, final_k2 = simulate_busso(optimal_loads, params_busso)

    print(f"Final Race-Day Performance: {final_perf[-1]:.2f} AU")

    rounded_loads      = np.rint(optimal_loads)
    rounded_loads[-1]  = MARATHON_KM

    print_weekly_summary(rounded_loads)
    print_detailed_summary(rounded_loads)

    save_all_plots(
        optimal_loads, final_perf, final_g, final_h, final_k2,
        _convergence_history, params_busso.k1,
        label='Differential Evolution', suffix='_de', save_dir=output_dir,
    )
    plt.show()
