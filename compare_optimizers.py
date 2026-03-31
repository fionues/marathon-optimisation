"""
compare_optimizers.py
=====================
Compare robustness and wall-clock time between Differential Evolution (DE)
and Simulated Annealing (SA) on the Busso marathon-training optimisation.

For each algorithm, N_RUNS independent runs are performed with no fixed
seeds, so each run draws fresh randomness from the OS.  The following
metrics are collected per run:

  - race-day performance (AU)
  - total plan distance (km)
  - wall-clock time (s)
  - number of function evaluations

Summary statistics (mean, std, min, max) are then printed and plotted.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, dual_annealing

from busso_model import (
    simulate_busso, params_busso, n_days,
    RAMP_RATE, FIRST_WEEK_KM, LONG_RUN_KM, MAX_RUN_KM, MARATHON_KM,
)

script_dir = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# EXPERIMENT SETTINGS
# ─────────────────────────────────────────────
N_RUNS        = 10     # independent runs per algorithm
SA_MAXITER    = 100    # keep identical to optimize_sa.py
DE_MAXITER    = 500    # reduced from 1000 for tractable comparison time
DE_POPSIZE    = 15
PENALTY_WEIGHT = 1e4
bounds        = [(0, MAX_RUN_KM)] * n_days


# ─────────────────────────────────────────────
# CONSTRAINT HELPERS (self-contained copies so this file runs standalone)
# ─────────────────────────────────────────────

def _repair_sa(loads: np.ndarray) -> np.ndarray:
    """SA repair: no explicit taper caps (those are enforced via penalty)."""
    loads = np.maximum(0, loads)

    for w in range(len(loads) // 7):
        min_idx = np.argmin(loads[w*7:(w+1)*7])
        loads[w*7 + min_idx] = 0.0

    first_sum = loads[:7].sum()
    if first_sum > 0:
        loads[:7] *= FIRST_WEEK_KM / first_sum
    else:
        loads[:7] = FIRST_WEEK_KM / 6.0
        loads[np.argmin(loads[:7])] = 0.0

    for w in range(1, len(loads) // 7):
        prev = loads[(w-1)*7:w*7].sum()
        curr = loads[w*7:(w+1)*7].sum()
        if curr > prev * (1 + RAMP_RATE):
            loads[w*7:(w+1)*7] *= (prev * (1 + RAMP_RATE)) / curr if curr > 0 else 0

    max_idx = np.argmax(loads)
    if loads[max_idx] < LONG_RUN_KM:
        loads[max_idx] = LONG_RUN_KM

    loads = np.minimum(loads, MAX_RUN_KM)
    loads = np.where((loads > 0) & (loads < 5), np.where(loads < 2.5, 0.0, 5.0), loads)
    loads[-1] = MARATHON_KM
    return loads


def _repair_de(loads: np.ndarray) -> np.ndarray:
    """DE repair: includes explicit taper caps for weeks 14–16."""
    loads = np.maximum(0, loads)

    for w in range(len(loads) // 7):
        min_idx = np.argmin(loads[w*7:(w+1)*7])
        loads[w*7 + min_idx] = 0.0

    first_sum = loads[:7].sum()
    if first_sum > 0:
        loads[:7] *= FIRST_WEEK_KM / first_sum
    else:
        loads[:7] = FIRST_WEEK_KM / 6.0
        loads[np.argmin(loads[:7])] = 0.0

    for w in range(1, len(loads) // 7):
        prev = loads[(w-1)*7:w*7].sum()
        curr = loads[w*7:(w+1)*7].sum()
        if curr > prev * (1 + RAMP_RATE):
            loads[w*7:(w+1)*7] *= (prev * (1 + RAMP_RATE)) / curr if curr > 0 else 0

    wk13 = loads[12*7:13*7].sum()
    for sl, cap in [(slice(13*7, 14*7), 0.80),
                    (slice(14*7, 15*7), 0.60),
                    (slice(15*7, 16*7 - 1), 0.35)]:
        if loads[sl].sum() > wk13 * cap:
            loads[sl] *= (wk13 * cap) / loads[sl].sum()

    max_idx = np.argmax(loads)
    if loads[max_idx] < LONG_RUN_KM:
        loads[max_idx] = LONG_RUN_KM

    is_not_taper   = np.arange(len(loads)) < (n_days - 7)
    small_mask     = (loads > 0) & (loads < 5.0) & is_not_taper
    loads[small_mask & (loads < 2.5)]  = 0.0
    loads[small_mask & (loads >= 2.5)] = 5.0

    loads = np.minimum(loads, MAX_RUN_KM)
    loads[-1] = MARATHON_KM
    return loads


# ─────────────────────────────────────────────
# OBJECTIVE FUNCTIONS
# ─────────────────────────────────────────────

def _sa_objective(loads: np.ndarray) -> float:
    perf, _, _, _ = simulate_busso(loads, params_busso)
    objective = -perf[-1]
    penalty   = 0.0
    n_weeks   = len(loads) // 7

    penalty += np.sum(np.minimum(loads, 0) ** 2)
    for w in range(n_weeks):
        penalty += min(loads[w*7:(w+1)*7]) ** 2
    penalty += (loads[:7].sum() - FIRST_WEEK_KM) ** 2
    for w in range(1, n_weeks):
        prev = loads[(w-1)*7:w*7].sum()
        curr = loads[w*7:(w+1)*7].sum()
        penalty += max(0, curr - prev * (1 + RAMP_RATE)) ** 2
    penalty += max(0, LONG_RUN_KM - loads.max()) ** 2
    penalty += np.sum(np.maximum(0, loads - MAX_RUN_KM) ** 2)
    if n_weeks >= 4:
        peak = max(loads[w*7:(w+1)*7].sum() for w in range(n_weeks - 3))
        for ti, cap in enumerate([0.80, 0.60, 0.35]):
            wv = loads[(n_weeks - 3 + ti)*7:(n_weeks - 2 + ti)*7].sum()
            penalty += max(0, wv - cap * peak) ** 2
    in_gap = (loads > 0) & (loads < 5)
    penalty += np.sum(np.minimum(loads, 5 - loads)[in_gap] ** 2)

    return objective + PENALTY_WEIGHT * penalty


def _de_objective(loads: np.ndarray) -> float:
    repaired = _repair_de(loads.copy())
    perf, _, _, _ = simulate_busso(repaired, params_busso)
    return -perf[-1]


# ─────────────────────────────────────────────
# BENCHMARK LOOP
# ─────────────────────────────────────────────

def _run_sa() -> dict:
    t0  = time.perf_counter()
    res = dual_annealing(_sa_objective, bounds=bounds, maxiter=SA_MAXITER)
    elapsed = time.perf_counter() - t0

    loads        = res.x.copy()
    loads[-1]    = MARATHON_KM
    perf, _, _, _ = simulate_busso(loads, params_busso)
    return {
        "perf":  perf[-1],
        "dist":  loads.sum(),
        "time":  elapsed,
        "nfev":  res.nfev,
        "nit":   res.nit,
    }


def _run_de() -> dict:
    t0  = time.perf_counter()
    res = differential_evolution(
        _de_objective, bounds=bounds,
        strategy='best1bin', maxiter=DE_MAXITER,
        popsize=DE_POPSIZE, tol=0.01,
    )
    elapsed = time.perf_counter() - t0

    loads        = _repair_de(res.x.copy())
    perf, _, _, _ = simulate_busso(loads, params_busso)
    return {
        "perf":  perf[-1],
        "dist":  loads.sum(),
        "time":  elapsed,
        "nfev":  res.nfev,
        "nit":   res.nit,
    }


def _collect(run_fn, label: str) -> list[dict]:
    results = []
    for i in range(N_RUNS):
        print(f"  [{label}] run {i+1}/{N_RUNS} …", flush=True)
        results.append(run_fn())
    return results


# ─────────────────────────────────────────────
# STATISTICS HELPERS
# ─────────────────────────────────────────────

def _stats(values: list, key: str) -> dict:
    arr = np.array([v[key] for v in values])
    return {"mean": arr.mean(), "std": arr.std(), "min": arr.min(), "max": arr.max(), "all": arr}


def _print_stats(label: str, results: list) -> None:
    print(f"\n{'─'*55}")
    print(f"  {label}  ({len(results)} runs)")
    print(f"{'─'*55}")
    for key, unit in [("perf", "AU"), ("dist", "km"), ("time", "s"), ("nfev", ""), ("nit", "")]:
        s = _stats(results, key)
        print(f"  {key:<6}  mean={s['mean']:>10.3f}{unit}  std={s['std']:>9.3f}  "
              f"min={s['min']:>9.3f}  max={s['max']:>9.3f}")


# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────

def _plot_comparison(sa_results: list, de_results: list) -> None:
    metrics   = [("perf", "Race-day performance (AU)"),
                 ("time", "Wall-clock time (s)"),
                 ("nfev", "Function evaluations")]
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    colors    = {"SA": "steelblue", "DE": "darkorange"}

    for ax, (key, ylabel) in zip(axes, metrics):
        sa_vals = np.array([r[key] for r in sa_results])
        de_vals = np.array([r[key] for r in de_results])

        # Box plots
        bp = ax.boxplot(
            [sa_vals, de_vals],
            labels=["SA", "DE"],
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
        )
        for patch, color in zip(bp["boxes"], [colors["SA"], colors["DE"]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # Overlay individual points (jittered)
        rng = np.random.default_rng(0)
        for pos, vals in enumerate([sa_vals, de_vals], start=1):
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(pos + jitter, vals, color="black", alpha=0.6, s=25, zorder=3)

        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle(
        f"SA (maxiter={SA_MAXITER}) vs DE (maxiter={DE_MAXITER}, popsize={DE_POPSIZE})\n"
        f"{N_RUNS} independent runs each",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, "comparison_boxplots.png"), dpi=150, bbox_inches="tight")


def _plot_variability(sa_results: list, de_results: list) -> None:
    """Scatter of race-day performance vs time for all runs."""
    fig, ax = plt.subplots(figsize=(8, 5))

    sa_perf = np.array([r["perf"] for r in sa_results])
    de_perf = np.array([r["perf"] for r in de_results])
    sa_time = np.array([r["time"] for r in sa_results])
    de_time = np.array([r["time"] for r in de_results])

    ax.scatter(sa_time, sa_perf, color="steelblue",  s=80, label="SA", zorder=3)
    ax.scatter(de_time, de_perf, color="darkorange", s=80, label="DE", zorder=3)

    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("Race-day performance (AU)")
    ax.set_title("Performance vs. Runtime across all runs")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, "comparison_scatter.png"), dpi=150, bbox_inches="tight")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nRunning {N_RUNS} SA runs  (maxiter={SA_MAXITER}) …")
    sa_results = _collect(_run_sa, "SA")

    print(f"\nRunning {N_RUNS} DE runs  (maxiter={DE_MAXITER}, popsize={DE_POPSIZE}) …")
    de_results = _collect(_run_de, "DE")

    print("\n" + "=" * 55)
    print("  ROBUSTNESS & TIMING COMPARISON")
    print("=" * 55)
    _print_stats("Simulated Annealing (SA)", sa_results)
    _print_stats("Differential Evolution (DE)", de_results)

    sa_perf = np.array([r["perf"] for r in sa_results])
    de_perf = np.array([r["perf"] for r in de_results])
    print(f"\n  Best SA performance  : {sa_perf.max():.3f} AU  (seed {np.argmax(sa_perf)})")
    print(f"  Best DE performance  : {de_perf.max():.3f} AU  (seed {np.argmax(de_perf)})")
    print(f"  SA coefficient of variation (perf): {sa_perf.std()/sa_perf.mean()*100:.2f} %")
    print(f"  DE coefficient of variation (perf): {de_perf.std()/de_perf.mean()*100:.2f} %")

    _plot_comparison(sa_results, de_results)
    _plot_variability(sa_results, de_results)
    plt.show()
