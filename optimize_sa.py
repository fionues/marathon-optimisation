import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from scipy.optimize import dual_annealing

from busso_model import (
    simulate_busso, params_busso, n_days,
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


# ─────────────────────────────────────────────────────────────────────────────
# RUN OPTIMISER
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running Dual Annealing — penalty constraints …")

    _convergence_history: List[tuple[float, float]] = []
    _best_perf_so_far = [-np.inf]
    _nfev = [0]

    def _counted_objective(loads: np.ndarray) -> float:
        _nfev[0] += 1
        perf, _, _, _ = simulate_busso(loads, params_busso)
        race_day_perf = perf[-1]
        if race_day_perf > _best_perf_so_far[0]:
            _best_perf_so_far[0] = race_day_perf
        _convergence_history.append((_nfev[0], _best_perf_so_far[0]))
        return busso_objective_penalty(loads)

    PATIENCE   = 10    # stop after this many callbacks with no improvement
    TOLERANCE  = 0.001  # minimum improvement to count as progress
    _no_improve_count = [0]
    _last_best = [-np.inf]

    def _sa_callback(x, f, context):
        perf, _, _, _ = simulate_busso(x, params_busso)
        race_day_perf = perf[-1]
        if race_day_perf > _best_perf_so_far[0]:
            _best_perf_so_far[0] = race_day_perf
        _convergence_history.append((_nfev[0], _best_perf_so_far[0]))

        if _best_perf_so_far[0] - _last_best[0] > TOLERANCE:
            _last_best[0] = _best_perf_so_far[0]
            _no_improve_count[0] = 0
        else:
            _no_improve_count[0] += 1

        if _no_improve_count[0] >= PATIENCE:
            print(f"Early stopping: no improvement > {TOLERANCE} for {PATIENCE} callbacks.")
            return True  # signals dual_annealing to stop
        return False

    res_penalty = dual_annealing(
        _counted_objective,
        bounds=[(0, MAX_RUN_KM)] * n_days,
        maxiter=1000,
        seed=42,
        callback=_sa_callback,
    )

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

    save_all_plots(
        loads_penalty, perf_penalty, g_penalty, h_penalty, k2_penalty,
        _convergence_history, params_busso.k1,
        label='Simulated Annealing', suffix='_sa', save_dir=output_dir,
    )
    # save_all_plots(
    #     _convergence_history, label='Simulated Annealing', suffix='_sa', save_dir=output_dir,
    # )
    plt.show()
