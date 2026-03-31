import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List

from busso_model import RAMP_RATE, LONG_RUN_KM, MAX_RUN_KM, FIRST_WEEK_KM

# ─────────────────────────────────────────────
# TEXT SUMMARIES
# ─────────────────────────────────────────────

def print_weekly_summary(loads: np.ndarray) -> None:
    print("\n" + "=" * 45)
    print(f"{'WEEK':<6} | {'TOTAL KM':<10} | {'LONGEST RUN':<12} | {'REST'}")
    print("-" * 45)
    for w in range(len(loads) // 7):
        week_data = loads[w * 7: (w + 1) * 7]
        total_km  = week_data.sum()
        max_run   = week_data.max()
        has_rest  = 0.0 in week_data
        print(f"{w + 1:<6} | {total_km:<10.1f} | {max_run:<12.1f} | {'Yes' if has_rest else 'No'}")
    print("=" * 45)


def print_detailed_summary(loads: np.ndarray) -> None:
    print("\n" + "=" * 65)
    print(f"{'WEEK':<5} | {'TOTAL':<8} | {'DAILY BREAKDOWN (km)':<40}")
    print("-" * 65)
    for w in range(len(loads) // 7):
        week_data = loads[w * 7: (w + 1) * 7]
        total_km  = week_data.sum()
        day_str   = "  ".join([f"{d:4.1f}" for d in week_data])
        print(f"W{w + 1:>2}   | {total_km:>6.1f} | {day_str}")
    print("=" * 65)
    print(f"TOTAL PLAN DISTANCE: {loads.sum():.1f} km")


def constraint_report(label: str, loads: np.ndarray) -> None:
    """Print a feasibility check for a given load vector."""
    n_weeks  = len(loads) // 7
    rest_ok  = all((loads[w*7:(w+1)*7] == 0).any() for w in range(n_weeks))
    ramp_ok  = all(
        loads[w*7:(w+1)*7].sum() <= loads[(w-1)*7:w*7].sum() * (1 + RAMP_RATE) + 1e-6
        for w in range(1, n_weeks)
    )
    long_ok  = loads.max() >= LONG_RUN_KM - 1e-6
    cap_ok   = (loads[:-1] <= MAX_RUN_KM + 1e-6).all()   # exclude race day
    week1_ok = abs(loads[:7].sum() - FIRST_WEEK_KM) < 1.0

    print(f"\n  {label}")
    print(f"    Total plan distance  : {loads.sum():.1f} km")
    print(f"    Rest day every week  : {'✓' if rest_ok  else '✗  VIOLATED'}")
    print(f"    ≤10% weekly ramp     : {'✓' if ramp_ok  else '✗  VIOLATED'}")
    print(f"    ≥1 run of {LONG_RUN_KM:.0f} km     : {'✓' if long_ok  else '✗  VIOLATED'}")
    print(f"    No run > {MAX_RUN_KM:.0f} km       : {'✓' if cap_ok   else '✗  VIOLATED'}")
    print(f"    Week 1 ≈ {FIRST_WEEK_KM:.0f} km         : {'✓' if week1_ok else '✗  VIOLATED'}")


# ─────────────────────────────────────────────
# PLOTS — unified (save with suffix _sa / _de when save_dir is given)
# ─────────────────────────────────────────────

def plot_daily_loads(
    loads: np.ndarray,
    label: str,
    suffix: str = '',
    save_dir: str = None,
) -> None:
    days = np.arange(len(loads))
    plt.figure(figsize=(12, 5))
    plt.bar(days, loads, color='royalblue', alpha=0.6, label='Daily Load (km)')
    plt.ylabel('Distance (km)')
    plt.title(f'Optimized 16-Week Training Plan — Daily Loads ({label})')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"daily_loads{suffix}.png"), dpi=150, bbox_inches='tight')


def plot_weekly_volume(
    loads: np.ndarray,
    label: str,
    suffix: str = '',
    save_dir: str = None,
) -> None:
    n_weeks       = len(loads) // 7
    weekly_totals = [loads[w * 7:(w + 1) * 7].sum() for w in range(n_weeks)]

    plt.figure(figsize=(10, 5))
    plt.bar(range(1, n_weeks + 1), weekly_totals, color='orange', edgecolor='black', alpha=0.8)
    plt.plot(range(1, n_weeks + 1), weekly_totals, color='darkred', marker='o', linestyle='--', linewidth=1)
    plt.title(f'Total Weekly Training Volume ({label})')
    plt.xlabel('Week Number')
    plt.ylabel('Total Distance (km)')
    plt.xticks(range(1, n_weeks + 1))
    plt.grid(axis='y', linestyle=':', alpha=0.6)

    max_vol  = max(weekly_totals)
    max_week = weekly_totals.index(max_vol) + 1
    plt.annotate(
        f'Peak: {max_vol:.1f}km',
        xy=(max_week, max_vol),
        xytext=(max_week, max_vol + 5),
        arrowprops=dict(facecolor='black', shrink=0.05),
        ha='center',
    )

    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"weekly_volume{suffix}.png"), dpi=150, bbox_inches='tight')


def plot_performance_dynamics(
    loads: np.ndarray,
    perf: np.ndarray,
    g: np.ndarray,
    h: np.ndarray,
    label: str,
    suffix: str = '',
    save_dir: str = None,
) -> None:
    days = np.arange(len(loads))
    plt.figure(figsize=(12, 5))
    plt.plot(days, g,    color='green', linewidth=2,   label='Fitness (g) - Gain')
    plt.plot(days, h,    color='red',   linewidth=2,   label='Fatigue (h) - Drain')
    plt.plot(days, perf, color='black', linewidth=2.5, label='Performance (p)')
    plt.ylabel('AU')
    plt.xlabel('Days')
    plt.title(f'Busso Model: Performance as a function of Fitness vs. Fatigue ({label})')
    plt.legend(loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"performance_dynamics{suffix}.png"), dpi=150, bbox_inches='tight')


def plot_convergence(
    history: List[float],
    label: str,
    suffix: str = '',
    save_dir: str = None,
) -> None:
    """Plot best objective value vs. iteration to visualise convergence."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history, color='steelblue', linewidth=1.5, label='Best objective (− race-day perf)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Best objective value')
    ax.set_title(f'Convergence ({label})')
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"convergence{suffix}.png"), dpi=150, bbox_inches='tight')


def plot_k1_and_k2(
    loads: np.ndarray,
    k2: np.ndarray,
    k1_val: float,
    label: str = '',
    suffix: str = '',
    save_dir: str = None,
) -> None:
    days = np.arange(len(loads))
    plt.figure(figsize=(12, 5))
    plt.axhline(y=k1_val, color='blue', linestyle='--', linewidth=2,
                label=f'Fitness Factor (k1 = {k1_val})')
    plt.plot(days, k2, color='purple', linewidth=2, label='Fatigue Sensitivity (k2)')
    plt.xlabel('Days')
    plt.ylabel('Multiplier Value')
    title = 'Busso Model: Dynamic Fatigue Sensitivity ($k_2$) vs. Constant Fitness Factor ($k_1$)'
    if label:
        title += f' ({label})'
    plt.title(title)
    plt.legend(loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"k1_k2{suffix}.png"), dpi=150, bbox_inches='tight')


# ─────────────────────────────────────────────
# COMBINED — save all plots individually, then show together
# ─────────────────────────────────────────────

def save_all_plots(
    loads: np.ndarray,
    perf: np.ndarray,
    g: np.ndarray,
    h: np.ndarray,
    k2: np.ndarray,
    convergence_history: List[float],
    k1_val: float,
    label: str,
    suffix: str,
    save_dir: str,
) -> None:
    """
    Call all 5 plot functions, saving each to its own PNG file.
    Figures are kept open so the caller can follow with plt.show().
    """
    plot_daily_loads(loads,  label=label, suffix=suffix, save_dir=save_dir)
    plot_weekly_volume(loads, label=label, suffix=suffix, save_dir=save_dir)
    plot_performance_dynamics(loads, perf, g, h, label=label, suffix=suffix, save_dir=save_dir)
    plot_convergence(convergence_history, label=label, suffix=suffix, save_dir=save_dir)
    plot_k1_and_k2(loads, k2, k1_val, label=label, suffix=suffix, save_dir=save_dir)
