import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import dual_annealing

from busso_model import (
    simulate_busso, params_busso, n_days, busso_objective,
    RAMP_RATE, FIRST_WEEK_KM, LONG_RUN_KM, MAX_RUN_KM, MARATHON_KM,
)
from plots import (
    save_all_plots,
    print_weekly_summary, print_detailed_summary, constraint_report,
)

script_dir     = os.path.dirname(os.path.abspath(__file__))
output_dir     = os.path.join(script_dir, 'output')
os.makedirs(output_dir, exist_ok=True)
PENALTY_WEIGHT = 1e4   # multiplier that converts constraint violations into cost

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

# ─────────────────────────────────────────────────────────────────────────────
# PENALTY CONSTRAINTS
# Infeasible solutions are allowed but penalised.
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
    penalty += max(0, LONG_RUN_KM - loads.max()) ** 2

    # Penalty 5b: no run above MAX_RUN_KM
    penalty += np.sum(np.maximum(0, loads - MAX_RUN_KM) ** 2)

    # Penalty 6: tapering — last 3 weeks must reduce volume relative to peak
    if n_weeks >= 4:
        peak_vol = max(loads[w*7:(w+1)*7].sum() for w in range(n_weeks - 3))
        for taper_idx, taper_cap in enumerate([0.80, 0.60, 0.35]):
            w = n_weeks - 3 + taper_idx
            week_sum = loads[w*7 : (w+1)*7].sum()
            penalty += max(0, week_sum - taper_cap * peak_vol) ** 2

    # Penalty 7: no load in the open interval (0, 5)
    in_gap = (loads > 0) & (loads < 5)
    penalty += np.sum(np.minimum(loads, 5 - loads)[in_gap] ** 2)

    return objective + PENALTY_WEIGHT * penalty


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


class TrackingObjective:
    def __init__(self, fn, logger: ConvergenceLogger, perf_fn=None):
        """
        fn      : the objective the optimizer sees (e.g. penalized)
        perf_fn : optional separate function for what gets logged (e.g. clean performance)
                  If None, logs fn's return value directly.
        """
        self.fn = fn
        self.perf_fn = perf_fn if perf_fn is not None else fn
        self.logger = logger
        self.nfev = 0
        self.best_perf = np.inf   # tracks best logged value (perf_fn)
        self.best_opt  = np.inf   # tracks best optimizer value (fn), for correctness

    def __call__(self, x):
        opt_val  = self.fn(x)           # penalized — returned to optimizer
        perf_val = self.perf_fn(x)      # clean performance — logged only
        self.nfev += 1

        if opt_val < self.best_opt:     # improvement by optimizer's metric
            self.best_opt = opt_val
            elapsed = time.time() - self.logger._start
            self.logger.trace.append((self.nfev, perf_val, elapsed))

        return opt_val                  # optimizer always gets the penalized value

    def reset(self):
        self.nfev = 0
        self.best_opt  = np.inf
        self.best_perf = np.inf
        self.logger.reset()

# ─────────────────────────────────────────────────────────────────────────────
# RUN OPTIMISER
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running Dual Annealing — penalty constraints …")

    PATIENCE   = 50      # stop after this many callbacks with no improvement
    TOLERANCE  = 0.0001  # minimum improvement to count as progress
    _best_so_far = [np.inf]
    _no_improve_count = [0]

    def _sa_callback(x, f, context):
        if _best_so_far[0] - f > TOLERANCE:
            _best_so_far[0] = f
            _no_improve_count[0] = 0
        else:
            _no_improve_count[0] += 1

        if _no_improve_count[0] >= PATIENCE:
            print(f"Early stopping: no improvement > {TOLERANCE} for {PATIENCE} callbacks.")
            return True  # signals dual_annealing to stop
        return False
    
    sa_logger  = ConvergenceLogger(label="SA")
    tracked_sa = TrackingObjective(
        fn=busso_objective_penalty,
        logger=sa_logger,
        perf_fn=busso_objective,
    )

    res_penalty = dual_annealing(
        tracked_sa,
        bounds=[(0, MAX_RUN_KM)] * n_days,
        maxiter=500,
        seed=42
    )

    sa_logger.trace.append((res_penalty.nfev, res_penalty.fun, time.time() - sa_logger._start))

    # ─────────────────────────────────────────────────────────────────────────────
    # RESULTS
    # ─────────────────────────────────────────────────────────────────────────────
    loads_penalty       = res_penalty.x
    loads_penalty[-1]   = MARATHON_KM
    perf_penalty, g_penalty, h_penalty, k2_penalty = simulate_busso(loads_penalty, params_busso)

    print("\n" + "=" * 55)
    print("  CONSTRAINT APPROACH COMPARISON")
    print("=" * 55)
    constraint_report("Penalty constraints", loads_penalty)
    print("=" * 55)

    print(f'Iterations:            {res_penalty.nit}')
    print(f'Function evaluations:  {res_penalty.nfev}')
    print(f"Final Race-Day Performance: {perf_penalty[-1]:.2f} AU")

    print_weekly_summary(loads_penalty)
    print_detailed_summary(loads_penalty)

    plot_convergence([sa_logger], title="Simulated Annealing Convergence")

    # save_all_plots(
    #     loads_penalty, perf_penalty, g_penalty, h_penalty, k2_penalty,
    #     [], params_busso.k1,
    #     label='Simulated Annealing', suffix='_sa', save_dir=output_dir,
    # )
    # plt.show()
