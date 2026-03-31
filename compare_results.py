"""
compare_results.py
==================
Run SA and DE once each, then produce a single figure that puts their
outputs side-by-side for direct comparison.  Results are also printed
as a concise summary table.

What is compared
----------------
Quantitative metrics
  - Race-day performance (AU)          — main objective
  - Total training volume (km)         — overall load
  - Peak weekly volume (km)            — highest training week
  - Number of rest days                — recovery compliance
  - Longest single run (km)            — long-run compliance
  - Taper compliance (wk 14/15/16)     — injury prevention
  - Wall-clock runtime (s)             — computational cost
  - Function evaluations               — efficiency

Visual comparison (one combined figure, saved to output/)
  - Daily loads        : SA vs DE side by side
  - Weekly volume      : both overlaid on the same axes
  - Performance (p)    : both overlaid
  - Fitness (g) & Fatigue (h) : both overlaid
  - k2 dynamics        : both overlaid
  - Convergence        : SA and DE side by side (own x-axes)
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import dual_annealing, differential_evolution

from busso_model import (
    simulate_busso, params_busso, n_days,
    RAMP_RATE, FIRST_WEEK_KM, LONG_RUN_KM, MAX_RUN_KM, MARATHON_KM,
)

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, 'output')
os.makedirs(output_dir, exist_ok=True)

PENALTY_WEIGHT = 1e4

# ─────────────────────────────────────────────
# CONSTRAINT HELPERS
# ─────────────────────────────────────────────

def _repair_sa(loads: np.ndarray) -> np.ndarray:
    """SA-style repair: no explicit taper caps (handled via penalty)."""
    loads = np.maximum(0, loads)
    for w in range(len(loads) // 7):
        idx = np.argmin(loads[w*7:(w+1)*7])
        loads[w*7 + idx] = 0.0
    s = loads[:7].sum()
    if s > 0:
        loads[:7] *= FIRST_WEEK_KM / s
    else:
        loads[:7] = FIRST_WEEK_KM / 6.0
        loads[np.argmin(loads[:7])] = 0.0
    for w in range(1, len(loads) // 7):
        prev = loads[(w-1)*7:w*7].sum()
        curr = loads[w*7:(w+1)*7].sum()
        if curr > prev * (1 + RAMP_RATE):
            loads[w*7:(w+1)*7] *= (prev * (1 + RAMP_RATE)) / curr
    max_idx = np.argmax(loads)
    if loads[max_idx] < LONG_RUN_KM:
        loads[max_idx] = LONG_RUN_KM
    loads = np.minimum(loads, MAX_RUN_KM)
    loads = np.where((loads > 0) & (loads < 5), np.where(loads < 2.5, 0.0, 5.0), loads)
    loads[-1] = MARATHON_KM
    return loads


def _repair_de(loads: np.ndarray) -> np.ndarray:
    """DE-style repair: includes explicit taper caps for weeks 14–16."""
    loads = np.maximum(0, loads)
    for w in range(len(loads) // 7):
        idx = np.argmin(loads[w*7:(w+1)*7])
        loads[w*7 + idx] = 0.0
    s = loads[:7].sum()
    if s > 0:
        loads[:7] *= FIRST_WEEK_KM / s
    else:
        loads[:7] = FIRST_WEEK_KM / 6.0
        loads[np.argmin(loads[:7])] = 0.0
    for w in range(1, len(loads) // 7):
        prev = loads[(w-1)*7:w*7].sum()
        curr = loads[w*7:(w+1)*7].sum()
        if curr > prev * (1 + RAMP_RATE):
            loads[w*7:(w+1)*7] *= (prev * (1 + RAMP_RATE)) / curr
    wk13 = loads[12*7:13*7].sum()
    for sl, cap in [(slice(13*7, 14*7), 0.80),
                    (slice(14*7, 15*7), 0.60),
                    (slice(15*7, 16*7 - 1), 0.35)]:
        if loads[sl].sum() > wk13 * cap:
            loads[sl] *= (wk13 * cap) / loads[sl].sum()
    max_idx = np.argmax(loads)
    if loads[max_idx] < LONG_RUN_KM:
        loads[max_idx] = LONG_RUN_KM
    is_not_taper = np.arange(len(loads)) < (n_days - 7)
    sm = (loads > 0) & (loads < 5.0) & is_not_taper
    loads[sm & (loads < 2.5)]  = 0.0
    loads[sm & (loads >= 2.5)] = 5.0
    loads = np.minimum(loads, MAX_RUN_KM)
    loads[-1] = MARATHON_KM
    return loads


# ─────────────────────────────────────────────
# OBJECTIVE FUNCTIONS
# ─────────────────────────────────────────────

def _sa_objective(loads: np.ndarray) -> float:
    perf, _, _, _ = simulate_busso(loads, params_busso)
    obj     = -perf[-1]
    penalty = 0.0
    n_weeks = len(loads) // 7
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
    return obj + PENALTY_WEIGHT * penalty


def _de_objective(loads: np.ndarray) -> float:
    repaired = _repair_de(loads.copy())
    perf, _, _, _ = simulate_busso(repaired, params_busso)
    return -perf[-1]


# ─────────────────────────────────────────────
# RUN BOTH OPTIMISERS
# ─────────────────────────────────────────────
bounds = [(0, MAX_RUN_KM)] * n_days

print("Running Simulated Annealing …")
sa_conv: list[float] = []
_sa_best = [np.inf]

def _sa_cb(x, f, context):
    if f < _sa_best[0]:
        _sa_best[0] = f
    sa_conv.append(_sa_best[0])

t0 = time.perf_counter()
sa_res = dual_annealing(_sa_objective, bounds=bounds, maxiter=100, callback=_sa_cb)
sa_time = time.perf_counter() - t0

print("Running Differential Evolution …")
de_conv: list[float] = []
_de_best = [np.inf]

def _de_cb(xk, convergence):
    val = _de_objective(xk)
    if val < _de_best[0]:
        _de_best[0] = val
    de_conv.append(_de_best[0])

t0 = time.perf_counter()
de_res = differential_evolution(
    _de_objective, bounds=bounds,
    strategy='best1bin', maxiter=1000, popsize=15, tol=0.01,
    disp=True, callback=_de_cb,
)
de_time = time.perf_counter() - t0

# ─────────────────────────────────────────────
# POST-PROCESS
# ─────────────────────────────────────────────
sa_loads = sa_res.x.copy()
sa_loads[-1] = MARATHON_KM
sa_perf, sa_g, sa_h, sa_k2 = simulate_busso(sa_loads, params_busso)

de_loads = _repair_de(de_res.x.copy())
de_perf, de_g, de_h, de_k2 = simulate_busso(de_loads, params_busso)

# ─────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────

def _weekly(loads):
    return [loads[w*7:(w+1)*7].sum() for w in range(len(loads) // 7)]

def _taper_ok(loads):
    wv = _weekly(loads)
    peak = max(wv[:-3])
    return (wv[-3] <= peak * 0.80 + 1e-3,
            wv[-2] <= peak * 0.60 + 1e-3,
            wv[-1] <= peak * 0.35 + 1e-3)

def _rest_days(loads):
    return int((loads == 0).sum())

def _check(ok): return '✓' if ok else '✗'

print("\n" + "=" * 62)
print(f"  {'METRIC':<35} {'SA':>10} {'DE':>10}")
print("=" * 62)

sa_loads_without_race = sa_loads[:-1]

rows = [
    ("Race-day performance (AU)",      f"{sa_perf[-1]:.3f}",         f"{de_perf[-1]:.3f}"),
    ("Total training volume (km)",     f"{sa_loads_without_race.sum():.1f}",       f"{sa_loads_without_race.sum():.1f}"),
    ("Peak weekly volume (km)",        f"{max(_weekly(sa_loads_without_race)):.1f}", f"{max(_weekly(sa_loads_without_race)):.1f}"),
    ("Longest single run (km)",        f"{sa_loads_without_race.max():.1f}",       f"{sa_loads_without_race.max():.1f}"),
    ("Rest days",                      f"{_rest_days(sa_loads_without_race)}",     f"{_rest_days(sa_loads_without_race)}"),
    ("Taper wk14 ≤80% of peak ✓/✗",  _check(_taper_ok(sa_loads_without_race)[0]), _check(_taper_ok(sa_loads_without_race)[0])),
    ("Taper wk15 ≤60% of peak ✓/✗",  _check(_taper_ok(sa_loads_without_race)[1]), _check(_taper_ok(sa_loads_without_race)[1])),
    ("Taper wk16 ≤35% of peak ✓/✗",  _check(_taper_ok(sa_loads_without_race)[2]), _check(_taper_ok(sa_loads_without_race)[2])),
    ("Wall-clock time (s)",            f"{sa_time:.1f}",              f"{de_time:.1f}"),
    ("Function evaluations",           f"{sa_res.nfev}",              f"{de_res.nfev}"),
    ("Iterations / generations",       f"{sa_res.nit}",               f"{de_res.nit}"),
]

for metric, sa_val, de_val in rows:
    print(f"  {metric:<35} {sa_val:>10} {de_val:>10}")

print("=" * 62)

# ─────────────────────────────────────────────
# COMBINED COMPARISON FIGURE
# ─────────────────────────────────────────────
days    = np.arange(n_days)
n_weeks = n_days // 7
sa_wv   = _weekly(sa_loads)
de_wv   = _weekly(de_loads)
weeks   = range(1, n_weeks + 1)

fig = plt.figure(figsize=(18, 20))
fig.suptitle("SA vs DE — Head-to-Head Comparison", fontsize=15, fontweight='bold', y=0.99)
gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.50, wspace=0.35)

SA_COLOR = 'steelblue'
DE_COLOR = 'darkorange'

# ── Row 0: Daily loads side by side ─────────────────────────────────────────
ax_sa_loads = fig.add_subplot(gs[0, 0])
ax_sa_loads.bar(days, sa_loads, color=SA_COLOR, alpha=0.7)
ax_sa_loads.set_title('Daily Loads — SA')
ax_sa_loads.set_ylabel('Distance (km)')
ax_sa_loads.set_xlabel('Day')
ax_sa_loads.grid(axis='y', linestyle='--', alpha=0.3)

ax_de_loads = fig.add_subplot(gs[0, 1], sharey=ax_sa_loads)
ax_de_loads.bar(days, de_loads, color=DE_COLOR, alpha=0.7)
ax_de_loads.set_title('Daily Loads — DE')
ax_de_loads.set_xlabel('Day')
ax_de_loads.grid(axis='y', linestyle='--', alpha=0.3)

# ── Row 1: Weekly volume overlaid | Performance p overlaid ──────────────────
ax_wv = fig.add_subplot(gs[1, 0])
ax_wv.plot(weeks, sa_wv, color=SA_COLOR, marker='o', linewidth=2, label='SA')
ax_wv.plot(weeks, de_wv, color=DE_COLOR, marker='s', linewidth=2, label='DE')
ax_wv.set_title('Weekly Volume')
ax_wv.set_xlabel('Week')
ax_wv.set_ylabel('km')
ax_wv.legend()
ax_wv.grid(alpha=0.3)

ax_perf = fig.add_subplot(gs[1, 1])
ax_perf.plot(days, sa_perf, color=SA_COLOR, linewidth=2, label='SA performance')
ax_perf.plot(days, de_perf, color=DE_COLOR, linewidth=2, label='DE performance')
ax_perf.set_title('Race-Day Performance Curve')
ax_perf.set_xlabel('Day')
ax_perf.set_ylabel('AU')
ax_perf.legend()
ax_perf.grid(alpha=0.3)

# ── Row 2: Fitness & Fatigue overlaid | k2 overlaid ─────────────────────────
ax_fit = fig.add_subplot(gs[2, 0])
ax_fit.plot(days, sa_g, color=SA_COLOR, linewidth=2,   linestyle='-',  label='SA fitness (g)')
ax_fit.plot(days, de_g, color=DE_COLOR, linewidth=2,   linestyle='-',  label='DE fitness (g)')
ax_fit.plot(days, sa_h, color=SA_COLOR, linewidth=1.5, linestyle='--', label='SA fatigue (h)')
ax_fit.plot(days, de_h, color=DE_COLOR, linewidth=1.5, linestyle='--', label='DE fatigue (h)')
ax_fit.set_title('Fitness (solid) & Fatigue (dashed)')
ax_fit.set_xlabel('Day')
ax_fit.set_ylabel('AU')
ax_fit.legend(fontsize=7, ncol=2)
ax_fit.grid(alpha=0.3)

ax_k2 = fig.add_subplot(gs[2, 1])
ax_k2.plot(days, sa_k2, color=SA_COLOR, linewidth=2, label='SA k2')
ax_k2.plot(days, de_k2, color=DE_COLOR, linewidth=2, label='DE k2')
ax_k2.axhline(y=params_busso.k1, color='grey', linestyle='--', linewidth=1.5,
              label=f'k1 = {params_busso.k1}')
ax_k2.set_title('Fatigue Sensitivity $k_2$ vs. Fitness Factor $k_1$')
ax_k2.set_xlabel('Day')
ax_k2.set_ylabel('Multiplier')
ax_k2.legend(fontsize=8)
ax_k2.grid(alpha=0.3)

# ── Row 3: Convergence side by side ─────────────────────────────────────────
ax_sa_conv = fig.add_subplot(gs[3, 0])
ax_sa_conv.plot(sa_conv, color=SA_COLOR, linewidth=1.5)
ax_sa_conv.set_title('Convergence — SA')
ax_sa_conv.set_xlabel('Iteration')
ax_sa_conv.set_ylabel('Best objective (−perf)')
ax_sa_conv.grid(alpha=0.3)

ax_de_conv = fig.add_subplot(gs[3, 1])
ax_de_conv.plot(de_conv, color=DE_COLOR, linewidth=1.5)
ax_de_conv.set_title('Convergence — DE')
ax_de_conv.set_xlabel('Generation')
ax_de_conv.set_ylabel('Best objective (−perf)')
ax_de_conv.grid(alpha=0.3)

plt.savefig(os.path.join(output_dir, 'comparison.png'), dpi=150, bbox_inches='tight')
print(f"\nComparison figure saved to {os.path.join(output_dir, 'comparison.png')}")
plt.show()
