"""
compare_optimizers.py
=====================
Compare robustness and wall-clock time between Differential Evolution (DE)
and Simulated Annealing (SA) on the Busso marathon-training optimisation.

Objective functions and repair heuristics are imported directly from
optimize_de.py and optimize_sa.py so there is no duplication.

For each algorithm, N_RUNS independent runs are performed with no fixed
seeds (fresh OS randomness every run).  The following metrics are collected:

  - race-day performance (AU)
  - total plan volume (km)
  - peak weekly volume (km)
  - wall-clock time (s)
  - number of function evaluations (nfev)

Summary statistics (mean, std, min, max) are printed and four plots are saved.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, dual_annealing
import time

from busso_model import simulate_busso, params_busso, n_days, MAX_RUN_KM, MARATHON_KM

# Import objective functions and repair helpers directly from the individual
# optimiser scripts (no code duplication).
from optimize_de import (
    de_objective,
    apply_all_constraints as de_repair,
    bounds as de_bounds,
)
from optimize_sa import busso_objective_penalty

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "output")
os.makedirs(output_dir, exist_ok=True)

# ─────────────────────────────────────────────
# EXPERIMENT SETTINGS
# ─────────────────────────────────────────────
N_RUNS      = 10
SA_MAXITER  = 100    # enough for convergence to be 0 without excessive runtime
DE_MAXITER  = 500    # DE stops after tolerance is met, so this is a max cap rather than the max iterations
DE_POPSIZE  = 15
DE_TOLERANCE = 0.01  # DE stops when population's best and worst are within this % of each other

sa_bounds = [(0, MAX_RUN_KM)] * n_days


# ─────────────────────────────────────────────
# CONVERGENCE TRACKING WRAPPER
# ─────────────────────────────────────────────
def _tracked_objective(fn, perf_fn=None):
    """
    Returns (tracked_fn, get_trace) where:
      - tracked_fn  : pass to the optimizer
      - get_trace   : call after optimization to get (nfevs, perfs, times)
    """
    _perf_fn   = perf_fn if perf_fn is not None else fn
    state      = {"nfev": 0, "best_opt": np.inf, "trace": [], "start": time.time()}

    def tracked_fn(x):
        opt_val  = fn(x)
        perf_val = _perf_fn(x)
        state["nfev"] += 1
        if opt_val < state["best_opt"]:
            state["best_opt"] = opt_val
            state["trace"].append((state["nfev"], perf_val, time.time() - state["start"]))
        return opt_val

    def get_trace():
        nfevs = [n for n, _, _ in state["trace"]]
        perfs = [p for _, p, _ in state["trace"]]
        times = [t for _, _, t in state["trace"]]
        return nfevs, perfs, times

    return tracked_fn, get_trace

# ─────────────────────────────────────────────
# SINGLE-RUN HELPERS
# ─────────────────────────────────────────────

def _sa_perf_fn(x):
    """race-day performance for an SA candidate."""
    perf, _, _, _ = simulate_busso(x.copy(), params_busso)
    return -perf[-1]


def _stopping_callback(tol=0.001, patience=50):
    """
    Stops dual_annealing when best value hasn't improved by `tol`
    over the last `patience` callback calls.
    """
    state = {"best": np.inf, "stagnant_count": 0}

    def callback(x, f, context):
        if f < state["best"] - tol:
            state["best"] = f
            state["stagnant_count"] = 0
        else:
            state["stagnant_count"] += 1

        return state["stagnant_count"] >= patience  # True = stop

    return callback


def _run_sa() -> dict:
    tracked_sa, get_sa_trace = _tracked_objective(
        fn=busso_objective_penalty,
        perf_fn=_sa_perf_fn,
    )
    res = dual_annealing(tracked_sa, bounds=sa_bounds, maxiter=SA_MAXITER, callback=_stopping_callback())
    sa_nfevs, sa_perfs, _ = get_sa_trace()
    t0      = time.perf_counter()
    elapsed = time.perf_counter() - t0

    loads = res.x.copy()
    loads[-1] = MARATHON_KM
    perf, _, _, _ = simulate_busso(loads, params_busso)
    n_w = len(loads) // 7
    return {
        "perf":      perf[-1],
        "dist":      loads.sum(),
        "peak_week": max(loads[w * 7:(w + 1) * 7].sum() for w in range(n_w)),
        "time":      elapsed,
        "nfev":      res.nfev,
        "nit":       res.nit,
        "conv":      list(zip(sa_nfevs, sa_perfs))
    }


def _run_de() -> dict:
    tracked_de, get_de_trace = _tracked_objective(
        fn=de_objective,
    )
    t0      = time.perf_counter()
    res     = differential_evolution(
        tracked_de,
        de_bounds,
        strategy="best1bin",
        maxiter=DE_MAXITER,
        popsize=DE_POPSIZE,
        tol=DE_TOLERANCE,
    )
    de_nfevs, de_perfs, _ = get_de_trace()
    elapsed = time.perf_counter() - t0

    loads = de_repair(res.x.copy())
    perf, _, _, _ = simulate_busso(loads, params_busso)
    n_w = len(loads) // 7
    return {
        "perf":      perf[-1],
        "dist":      loads.sum(),
        "peak_week": max(loads[w * 7:(w + 1) * 7].sum() for w in range(n_w)),
        "time":      elapsed,
        "nfev":      res.nfev,
        "nit":       res.nit,
        "conv":      list(zip(de_nfevs, de_perfs))
    }


def _collect(run_fn, label: str) -> list:
    results = []
    for i in range(N_RUNS):
        print(f"  [{label}] run {i + 1}/{N_RUNS} …", flush=True)
        results.append(run_fn())
    return results


# ─────────────────────────────────────────────
# STATISTICS HELPERS
# ─────────────────────────────────────────────

def _stats(results: list, key: str) -> dict:
    arr = np.array([r[key] for r in results])
    return {"mean": arr.mean(), "std": arr.std(), "min": arr.min(), "max": arr.max(), "all": arr}


def _print_stats(label: str, results: list) -> None:
    print(f"\n{'─' * 55}")
    print(f"  {label}  ({len(results)} runs)")
    print(f"{'─' * 55}")
    for key, unit in [("perf", " AU"), ("dist", " km"), ("peak_week", " km"),
                      ("time", " s"), ("nfev", ""), ("nit", "")]:
        s = _stats(results, key)
        print(f"  {key:<9}  mean={s['mean']:>10.3f}{unit}  std={s['std']:>9.3f}  "
              f"min={s['min']:>9.3f}  max={s['max']:>9.3f}")


# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────

def _plot_boxplots(sa_results: list, de_results: list) -> None:
    """Side-by-side box plots for the four key metrics."""
    metrics = [
        ("perf",      "Race-day performance (AU)"),
        ("dist",      "Total training volume (km)"),
        ("time",      "Wall-clock time (s)"),
        ("nfev",      "Function evaluations"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))
    colors = {"SA": "steelblue", "DE": "darkorange"}
    rng = np.random.default_rng(0)

    for ax, (key, ylabel) in zip(axes, metrics):
        sa_vals = np.array([r[key] for r in sa_results])
        de_vals = np.array([r[key] for r in de_results])

        bp = ax.boxplot(
            [sa_vals, de_vals],
            labels=["SA", "DE"],
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
        )
        for patch, color in zip(bp["boxes"], [colors["SA"], colors["DE"]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        for pos, vals in enumerate([sa_vals, de_vals], start=1):
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(pos + jitter, vals, color="black", alpha=0.6, s=25, zorder=3)

        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle(
        f"Simulated Annealing vs Differential Evolution — {N_RUNS} independent runs each",
        fontsize=11,
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "robustness_boxplots.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")


def _plot_scatter(sa_results: list, de_results: list) -> None:
    """Performance vs. wall-clock time scatter for all runs."""
    fig, ax = plt.subplots(figsize=(7, 5))

    sa_perf = np.array([r["perf"] for r in sa_results])
    de_perf = np.array([r["perf"] for r in de_results])
    sa_time = np.array([r["time"] for r in sa_results])
    de_time = np.array([r["time"] for r in de_results])

    ax.scatter(sa_time, sa_perf, color="steelblue",  s=80, label="Simulated Annealing", zorder=3)
    ax.scatter(de_time, de_perf, color="darkorange", s=80, label="Differential Evolution", zorder=3)

    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("Race-day performance (AU)")
    ax.set_title("Performance vs. Runtime across all runs")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "robustness_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")


def _plot_run_profiles(sa_results: list, de_results: list) -> None:
    """Performance across run index to visualise consistency."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    runs = np.arange(1, N_RUNS + 1)

    for ax, results, label, color in [
        (axes[0], sa_results, "Simulated Annealing", "steelblue"),
        (axes[1], de_results, "Differential Evolution", "darkorange"),
    ]:
        perfs = np.array([r["perf"] for r in results])
        mean  = perfs.mean()
        ax.bar(runs, perfs, color=color, alpha=0.7, label=label)
        ax.axhline(mean, color="black", linewidth=1.5, linestyle="--", label=f"Mean = {mean:.3f}")
        ax.set_xlabel("Run index")
        ax.set_ylabel("Race-day performance (AU)")
        ax.set_title(label)
        ax.set_xticks(runs)
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Race-day performance across independent runs", fontsize=11)
    plt.tight_layout()
    path = os.path.join(output_dir, "robustness_per_run.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")


def _plot_volume_comparison(sa_results: list, de_results: list) -> None:
    """Total and peak weekly volume distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    rng = np.random.default_rng(1)

    for ax, key, ylabel in [
        (axes[0], "dist",      "Total training volume (km)"),
        (axes[1], "peak_week", "Peak weekly volume (km)"),
    ]:
        sa_vals = np.array([r[key] for r in sa_results])
        de_vals = np.array([r[key] for r in de_results])

        bp = ax.boxplot(
            [sa_vals, de_vals],
            labels=["SA", "DE"],
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
        )
        for patch, color in zip(bp["boxes"], ["steelblue", "darkorange"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        for pos, vals in enumerate([sa_vals, de_vals], start=1):
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(pos + jitter, vals, color="black", alpha=0.6, s=25, zorder=3)

        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Training volume distributions — SA vs DE", fontsize=11)
    plt.tight_layout()
    path = os.path.join(output_dir, "robustness_volume.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")


def _plot_convergence(sa_results: list, de_results: list) -> None:
    """Solution quality (race-day performance) vs cumulative function evaluations.

    Each run is plotted as a thin line; the mean across runs is shown as a
    thick line.  Both algorithms are overlaid on the same axes for direct
    comparison.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for results, label, color in [
        (sa_results, "Simulated Annealing", "steelblue"),
        (de_results, "Differential Evolution", "darkorange"),
    ]:
        # Build a common nfev grid from 1 to the max nfev across all runs
        max_nfev = max(r["conv"][-1][0] for r in results)
        grid = np.linspace(1, max_nfev, 500)

        interp_curves = []
        for r in results:
            nfevs, perfs = zip(*r["conv"])
            nfevs = np.array(nfevs, dtype=float)
            perfs = np.array(perfs, dtype=float)
            # Forward-fill: for each grid point take the best perf seen so far
            interp = np.interp(grid, nfevs, perfs)
            interp_curves.append(interp)
            ax.plot(grid, interp, color=color, alpha=0.15, linewidth=0.8)

        mean_curve = np.mean(interp_curves, axis=0)
        ax.plot(grid, mean_curve, color=color, linewidth=2.2, label=f"{label} (mean)")

    ax.set_xlabel("Cumulative function evaluations")
    ax.set_ylabel("Best race-day performance (AU)")
    ax.set_title(
        f"Convergence curves — solution quality vs function evaluations\n"
        f"({N_RUNS} independent runs per algorithm; thin lines = individual runs)"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "robustness_convergence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nRunning {N_RUNS} SA runs  (maxiter={SA_MAXITER}) …")
    sa_results = _collect(_run_sa, "SA")

    print(f"\nRunning {N_RUNS} DE runs  (maxiter={DE_MAXITER}, popsize={DE_POPSIZE}) …")
    de_results = _collect(_run_de, "DE")

    # ── Print summary ──────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  ROBUSTNESS & TIMING COMPARISON")
    print("=" * 55)
    _print_stats("Simulated Annealing (SA)", sa_results)
    _print_stats("Differential Evolution (DE)", de_results)

    sa_perf = np.array([r["perf"] for r in sa_results])
    de_perf = np.array([r["perf"] for r in de_results])
    sa_dist = np.array([r["dist"] for r in sa_results])
    de_dist = np.array([r["dist"] for r in de_results])
    sa_peak = np.array([r["peak_week"] for r in sa_results])
    de_peak = np.array([r["peak_week"] for r in de_results])

    print(f"\n  Best SA performance  : {sa_perf.max():.3f} AU  (run {np.argmax(sa_perf) + 1})")
    print(f"  Best DE performance  : {de_perf.max():.3f} AU  (run {np.argmax(de_perf) + 1})")
    print(f"  SA coeff. of variation (perf): {sa_perf.std() / sa_perf.mean() * 100:.2f} %")
    print(f"  DE coeff. of variation (perf): {de_perf.std() / de_perf.mean() * 100:.2f} %")
    print(f"\n  SA total volume : {sa_dist.mean():.1f} ± {sa_dist.std():.1f} km  "
          f"(min {sa_dist.min():.1f}, max {sa_dist.max():.1f})")
    print(f"  DE total volume : {de_dist.mean():.1f} ± {de_dist.std():.1f} km  "
          f"(min {de_dist.min():.1f}, max {de_dist.max():.1f})")
    print(f"  SA peak weekly  : {sa_peak.mean():.1f} ± {sa_peak.std():.1f} km  "
          f"(min {sa_peak.min():.1f}, max {sa_peak.max():.1f})")
    print(f"  DE peak weekly  : {de_peak.mean():.1f} ± {de_peak.std():.1f} km  "
          f"(min {de_peak.min():.1f}, max {de_peak.max():.1f})")

    # ── Generate plots ─────────────────────────────────────────────────
    print("\nGenerating plots …")
    _plot_boxplots(sa_results, de_results)
    _plot_scatter(sa_results, de_results)
    _plot_run_profiles(sa_results, de_results)
    _plot_volume_comparison(sa_results, de_results)
    _plot_convergence(sa_results, de_results)
    plt.show()
