"""
compare_results.py
==================
Run SA and DE 10 times each, then produce a single figure that puts their
outputs side-by-side for direct comparison.  Results are also printed
as a concise summary table showing mean ± std across all runs.

What is compared
----------------
Quantitative metrics (mean ± std over 10 runs)
  - Race-day performance (AU)          — main objective
  - Total training volume (km)         — overall load
  - Peak weekly volume (km)            — highest training week
  - Wall-clock runtime (s)             — computational cost
  - Function evaluations               — efficiency

Visual comparison (one combined figure, saved to output/)
  - Daily loads        : boxplots across 10 runs (SA vs DE side by side)
  - Weekly volume      : mean line + min/max band for both overlaid
  - Performance (p)    : mean line + min/max band for both overlaid
  - Convergence        : mean line + min/max band, SA and DE side by side
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

N_RUNS       = 10
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
# HELPER: pad convergence curves to equal length
# ─────────────────────────────────────────────

def _pad_curves(curves: list[list[float]]) -> np.ndarray:
    """Pad each convergence curve with its final value so all have the same length."""
    max_len = max(len(c) for c in curves)
    padded = np.array([c + [c[-1]] * (max_len - len(c)) for c in curves])
    return padded


def _weekly(loads: np.ndarray) -> list[float]:
    return [loads[w*7:(w+1)*7].sum() for w in range(len(loads) // 7)]


# ─────────────────────────────────────────────
# RUN BOTH OPTIMISERS N_RUNS TIMES
# ─────────────────────────────────────────────
bounds = [(0, MAX_RUN_KM)] * n_days

sa_all_loads  : list[np.ndarray]   = []
sa_all_perf   : list[np.ndarray]   = []
sa_all_wv     : list[list[float]]  = []
sa_all_conv   : list[list[float]]  = []
sa_all_times  : list[float]        = []
sa_all_nfev   : list[int]          = []
sa_all_nit    : list[int]          = []

for run in range(N_RUNS):
    print(f"SA run {run + 1}/{N_RUNS} …")
    _conv: list[float] = []
    _best = [np.inf]

    def _sa_cb(x, f, context):
        if f < _best[0]:
            _best[0] = f
        _conv.append(_best[0])

    t0 = time.perf_counter()
    res = dual_annealing(_sa_objective, bounds=bounds, maxiter=200, callback=_sa_cb)
    elapsed = time.perf_counter() - t0

    loads = res.x.copy()
    loads[-1] = MARATHON_KM
    perf, _, _, _ = simulate_busso(loads, params_busso)

    sa_all_loads.append(loads)
    sa_all_perf.append(perf)
    sa_all_wv.append(_weekly(loads))
    sa_all_conv.append(_conv)
    sa_all_times.append(elapsed)
    sa_all_nfev.append(res.nfev)
    sa_all_nit.append(res.nit)

de_all_loads  : list[np.ndarray]   = []
de_all_perf   : list[np.ndarray]   = []
de_all_wv     : list[list[float]]  = []
de_all_conv   : list[list[float]]  = []
de_all_times  : list[float]        = []
de_all_nfev   : list[int]          = []
de_all_nit    : list[int]          = []

for run in range(N_RUNS):
    print(f"DE run {run + 1}/{N_RUNS} …")
    _conv_de: list[float] = []
    _best_de = [np.inf]

    def _de_cb(xk, convergence):
        val = _de_objective(xk)
        if val < _best_de[0]:
            _best_de[0] = val
        _conv_de.append(_best_de[0])

    t0 = time.perf_counter()
    res = differential_evolution(
        _de_objective, bounds=bounds,
        strategy='best1bin', maxiter=1000, popsize=15, tol=0.01,
        disp=False, callback=_de_cb,
    )
    elapsed = time.perf_counter() - t0

    loads = _repair_de(res.x.copy())
    perf, _, _, _ = simulate_busso(loads, params_busso)

    de_all_loads.append(loads)
    de_all_perf.append(perf)
    de_all_wv.append(_weekly(loads))
    de_all_conv.append(_conv_de)
    de_all_times.append(elapsed)
    de_all_nfev.append(res.nfev)
    de_all_nit.append(res.nit)

# ─────────────────────────────────────────────
# AGGREGATE ACROSS RUNS
# ─────────────────────────────────────────────
sa_loads_mat  = np.array(sa_all_loads)   # (N_RUNS, n_days)
sa_perf_mat   = np.array(sa_all_perf)    # (N_RUNS, n_days)
sa_wv_mat     = np.array(sa_all_wv)      # (N_RUNS, n_weeks)
sa_conv_mat   = _pad_curves(sa_all_conv) # (N_RUNS, max_iter)

de_loads_mat  = np.array(de_all_loads)
de_perf_mat   = np.array(de_all_perf)
de_wv_mat     = np.array(de_all_wv)
de_conv_mat   = _pad_curves(de_all_conv)

sa_final_perf = sa_perf_mat[:, -1]
de_final_perf = de_perf_mat[:, -1]
sa_train_vol  = sa_loads_mat[:, :-1].sum(axis=1)
de_train_vol  = de_loads_mat[:, :-1].sum(axis=1)
sa_peak_wv    = sa_wv_mat[:, :-1].max(axis=1)
de_peak_wv    = de_wv_mat[:, :-1].max(axis=1)

# ─────────────────────────────────────────────
# SUMMARY TABLE  (mean ± std)
# ─────────────────────────────────────────────

def _ms(arr):
    return f"{np.mean(arr):.2f} ± {np.std(arr):.2f}"

print("\n" + "=" * 72)
print(f"  {'METRIC':<38} {'SA (mean ± std)':>15} {'DE (mean ± std)':>15}")
print("=" * 72)

rows = [
    ("Race-day performance (AU)",        _ms(sa_final_perf),  _ms(de_final_perf)),
    ("Total training volume (km)",       _ms(sa_train_vol),   _ms(de_train_vol)),
    ("Peak weekly volume (km)",          _ms(sa_peak_wv),     _ms(de_peak_wv)),
    ("Wall-clock time (s)",              _ms(sa_all_times),   _ms(de_all_times)),
    ("Function evaluations",             _ms(sa_all_nfev),    _ms(de_all_nfev)),
    ("Iterations / generations",         _ms(sa_all_nit),     _ms(de_all_nit)),
]

for metric, sa_val, de_val in rows:
    print(f"  {metric:<38} {sa_val:>15} {de_val:>15}")

print("=" * 72)

# ─────────────────────────────────────────────
# COMBINED COMPARISON FIGURE
# ─────────────────────────────────────────────
days    = np.arange(n_days)
n_weeks = n_days // 7
weeks   = np.arange(1, n_weeks + 1)

SA_COLOR = 'steelblue'
DE_COLOR = 'darkorange'

fig = plt.figure(figsize=(18, 20))
fig.suptitle(f"SA vs DE — {N_RUNS}-Run Comparison", fontsize=15, fontweight='bold', y=0.99)
gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.50, wspace=0.35)


# ── Row 0: Daily loads — boxplots across N_RUNS ─────────────────────────────
ax_sa_loads = fig.add_subplot(gs[0, 0])
bp_sa = ax_sa_loads.boxplot(
    sa_loads_mat.T.tolist(),
    positions=days,
    widths=0.6,
    patch_artist=True,
    showfliers=False,
    boxprops=dict(facecolor=SA_COLOR, alpha=0.5),
    medianprops=dict(color='black', linewidth=1.5),
    whiskerprops=dict(color=SA_COLOR),
    capprops=dict(color=SA_COLOR),
)
ax_sa_loads.set_title(f'Daily Loads — SA ({N_RUNS} runs, boxplot)')
ax_sa_loads.set_ylabel('Distance (km)')
ax_sa_loads.set_xlabel('Day')
ax_sa_loads.grid(axis='y', linestyle='--', alpha=0.3)

ax_de_loads = fig.add_subplot(gs[0, 1], sharey=ax_sa_loads)
bp_de = ax_de_loads.boxplot(
    de_loads_mat.T.tolist(),
    positions=days,
    widths=0.6,
    patch_artist=True,
    showfliers=False,
    boxprops=dict(facecolor=DE_COLOR, alpha=0.5),
    medianprops=dict(color='black', linewidth=1.5),
    whiskerprops=dict(color=DE_COLOR),
    capprops=dict(color=DE_COLOR),
)
ax_de_loads.set_title(f'Daily Loads — DE ({N_RUNS} runs, boxplot)')
ax_de_loads.set_xlabel('Day')
ax_de_loads.grid(axis='y', linestyle='--', alpha=0.3)


# ── Row 1: Weekly volume overlaid with range band ───────────────────────────
ax_wv = fig.add_subplot(gs[1, 0])
sa_wv_mean = sa_wv_mat.mean(axis=0)
sa_wv_min  = sa_wv_mat.min(axis=0)
sa_wv_max  = sa_wv_mat.max(axis=0)
de_wv_mean = de_wv_mat.mean(axis=0)
de_wv_min  = de_wv_mat.min(axis=0)
de_wv_max  = de_wv_mat.max(axis=0)

ax_wv.fill_between(weeks, sa_wv_min, sa_wv_max, color=SA_COLOR, alpha=0.20)
ax_wv.fill_between(weeks, de_wv_min, de_wv_max, color=DE_COLOR, alpha=0.20)
ax_wv.plot(weeks, sa_wv_mean, color=SA_COLOR, marker='o', linewidth=2, label='SA mean')
ax_wv.plot(weeks, de_wv_mean, color=DE_COLOR, marker='s', linewidth=2, label='DE mean')
ax_wv.set_title('Weekly Volume (mean + range)')
ax_wv.set_xlabel('Week')
ax_wv.set_ylabel('km')
ax_wv.legend()
ax_wv.grid(alpha=0.3)


# ── Row 1: Performance curve overlaid with range band ───────────────────────
ax_perf = fig.add_subplot(gs[1, 1])
sa_perf_mean = sa_perf_mat.mean(axis=0)
sa_perf_min  = sa_perf_mat.min(axis=0)
sa_perf_max  = sa_perf_mat.max(axis=0)
de_perf_mean = de_perf_mat.mean(axis=0)
de_perf_min  = de_perf_mat.min(axis=0)
de_perf_max  = de_perf_mat.max(axis=0)

ax_perf.fill_between(days, sa_perf_min, sa_perf_max, color=SA_COLOR, alpha=0.20)
ax_perf.fill_between(days, de_perf_min, de_perf_max, color=DE_COLOR, alpha=0.20)
ax_perf.plot(days, sa_perf_mean, color=SA_COLOR, linewidth=2, label='SA mean')
ax_perf.plot(days, de_perf_mean, color=DE_COLOR, linewidth=2, label='DE mean')
ax_perf.set_title('Race-Day Performance Curve (mean + range)')
ax_perf.set_xlabel('Day')
ax_perf.set_ylabel('AU')
ax_perf.legend()
ax_perf.grid(alpha=0.3)


# ── Row 2: Race-day performance distribution — violin / strip ────────────────
ax_viol = fig.add_subplot(gs[2, 0])
vp = ax_viol.violinplot(
    [sa_final_perf, de_final_perf],
    positions=[1, 2],
    showmedians=True,
    showextrema=True,
)
for i, (body, color) in enumerate(zip(vp['bodies'], [SA_COLOR, DE_COLOR])):
    body.set_facecolor(color)
    body.set_alpha(0.6)
vp['cmedians'].set_color('black')
vp['cmedians'].set_linewidth(2)
ax_viol.set_xticks([1, 2])
ax_viol.set_xticklabels(['SA', 'DE'])
ax_viol.set_title(f'Race-Day Performance Distribution ({N_RUNS} runs)')
ax_viol.set_ylabel('AU')
ax_viol.grid(axis='y', alpha=0.3)

# scatter individual points on top
rng = np.random.default_rng(42)
ax_viol.scatter(rng.uniform(0.85, 1.15, N_RUNS), sa_final_perf,
                color=SA_COLOR, zorder=3, s=40, alpha=0.8)
ax_viol.scatter(rng.uniform(1.85, 2.15, N_RUNS), de_final_perf,
                color=DE_COLOR, zorder=3, s=40, alpha=0.8)

# ── Row 2: Runtime distribution ─────────────────────────────────────────────
ax_rt = fig.add_subplot(gs[2, 1])
ax_rt.boxplot(
    [sa_all_times, de_all_times],
    labels=['SA', 'DE'],
    patch_artist=True,
    boxprops=dict(facecolor='lightgrey'),
    medianprops=dict(color='black', linewidth=2),
)
ax_rt.set_title(f'Wall-Clock Runtime ({N_RUNS} runs)')
ax_rt.set_ylabel('seconds')
ax_rt.grid(axis='y', alpha=0.3)


# ── Row 3: Convergence — mean + range, side by side ─────────────────────────
ax_sa_conv = fig.add_subplot(gs[3, 0])
sa_conv_mean = sa_conv_mat.mean(axis=0)
sa_conv_min  = sa_conv_mat.min(axis=0)
sa_conv_max  = sa_conv_mat.max(axis=0)
iters_sa     = np.arange(sa_conv_mat.shape[1])
ax_sa_conv.fill_between(iters_sa, sa_conv_min, sa_conv_max, color=SA_COLOR, alpha=0.25, label='min–max range')
ax_sa_conv.plot(iters_sa, sa_conv_mean, color=SA_COLOR, linewidth=1.5, label='mean')
ax_sa_conv.set_title(f'Convergence — SA ({N_RUNS} runs)')
ax_sa_conv.set_xlabel('Iteration')
ax_sa_conv.set_ylabel('Best objective (−perf)')
ax_sa_conv.legend(fontsize=8)
ax_sa_conv.grid(alpha=0.3)

ax_de_conv = fig.add_subplot(gs[3, 1])
de_conv_mean = de_conv_mat.mean(axis=0)
de_conv_min  = de_conv_mat.min(axis=0)
de_conv_max  = de_conv_mat.max(axis=0)
iters_de     = np.arange(de_conv_mat.shape[1])
ax_de_conv.fill_between(iters_de, de_conv_min, de_conv_max, color=DE_COLOR, alpha=0.25, label='min–max range')
ax_de_conv.plot(iters_de, de_conv_mean, color=DE_COLOR, linewidth=1.5, label='mean')
ax_de_conv.set_title(f'Convergence — DE ({N_RUNS} runs)')
ax_de_conv.set_xlabel('Generation')
ax_de_conv.set_ylabel('Best objective (−perf)')
ax_de_conv.legend(fontsize=8)
ax_de_conv.grid(alpha=0.3)


plt.savefig(os.path.join(output_dir, 'comparison_10runs.png'), dpi=150, bbox_inches='tight')
print(f"\nComparison figure saved to {os.path.join(output_dir, 'comparison_10runs.png')}")
plt.show()
