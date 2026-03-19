"""
Marathon Training Optimizer
===========================
Uses Busso's Variable Dose-Response (VDR) model to predict performance
and Simulated Annealing (SA) to find the optimal 16-week training plan.

Busso VDR Model (Busso, 2003)
------------------------------
Unlike the fixed Banister model, the VDR model makes the fitness gain
factor k1 itself a function of recent training history. This captures
the well-known phenomenon that the same training dose produces diminishing
returns as the athlete adapts — and greater gains after a rest period.

State equations (discrete, daily):
  p(t)  = p0 + k1(t-1)*g(t) - k2*h(t)

  g(t)  = g(t-1)*exp(-1/τ1) + w(t)          # fitness impulse
  h(t)  = h(t-1)*exp(-1/τ2) + w(t)          # fatigue impulse
  k1(t) = k1(t-1)*exp(-1/τ3) + k̄1*w(t)     # variable gain (adapts to load)

Where:
  w(t)  = training load on day t (TRIMP or arbitrary units)
  τ1    = fitness time constant        (~45 days)
  τ2    = fatigue time constant        (~15 days)
  τ3    = gain adaptation time constant (~38 days, Busso 2003)
  k̄1   = baseline gain scaling factor
  k2    = fatigue gain factor (fixed)
  p0    = initial performance baseline

Key insight: k1(t) decays when training is reduced (gain potential increases
after rest) and is consumed/saturated when training is heavy (diminishing
returns). This makes it fundamentally more realistic than Banister's fixed k1.

References:
  Busso T. (2003). Variable dose-response relationship between exercise
  training and performance. Med Sci Sports Exerc, 35(7):1188-1195.
"""

import numpy as np
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
    p0:   float = 50.0   # baseline performance (AU)
    k1b:  float = 0.08   # baseline gain scaling factor (k̄1)
    k2:   float = 1.8    # fatigue gain factor (fixed)
    tau1: float = 45.0   # fitness time constant (days)
    tau2: float = 15.0   # fatigue time constant (days)
    tau3: float = 38.0   # gain adaptation time constant (days)


# ─────────────────────────────────────────────
# 2.  BUSSO VDR MODEL SIMULATION
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

    return perf, g, h, k1


# ─────────────────────────────────────────────
# 3.  OBJECTIVE FUNCTION
# ─────────────────────────────────────────────
def objective(loads: np.ndarray,
              params: BussoParams,
              race_day: int,
              max_daily_load:    float = 20.0,
              max_weekly_load:   float = 80.0,
              start_weekly_load: float = 28.0,
              max_weekly_increase: float = 0.10) -> float:
    """
    Maximise predicted race-day performance subject to physiological
    and practical training constraints. Violations are penalised.
    """
    perf, g, h, k1 = simulate_busso(loads, params)
    score = perf[race_day]

    n_weeks = len(loads) // 7
    weekly  = np.array([loads[w*7:(w+1)*7].sum() for w in range(n_weeks)])

    # # ── P1: daily load cap
    # over_daily = np.maximum(0, loads - max_daily_load)
    # score -= 50.0 * np.sum(over_daily ** 2)

    # ── P2: weekly volume cap
    over_weekly = np.maximum(0, weekly - max_weekly_load)
    score -= 20.0 * np.sum(over_weekly ** 2)

    # # ── P3: anchor week-1 volume to a realistic starting point
    # score -= 40.0 * (weekly[0] - start_weekly_load) ** 2

    # # ── P4: progressive overload — enforce ≤10% increase AND
    # #        penalise premature volume drops (outside taper window)
    # for w in range(1, n_weeks):
    #     if weekly[w-1] > 1e-3:
    #         change = (weekly[w] - weekly[w-1]) / weekly[w-1]
    #         if change > max_weekly_increase:
    #             score -= 80.0 * (change - max_weekly_increase) ** 2
    #         elif change < -max_weekly_increase and w < n_weeks - 2:
    #             score -= 40.0 * (change + max_weekly_increase) ** 2

    # # ── P5: taper — meaningful reduction in final 2 weeks
    # taper_ratio = weekly[-1] / (weekly[-2] + 1e-9)
    # if taper_ratio > 0.6:
    #     score -= 60.0 * (taper_ratio - 0.6) ** 2

    # ── P6: non-negative loads
    score -= 1000.0 * np.sum(np.minimum(0, loads) ** 2)

    return score


# ─────────────────────────────────────────────
# 4.  INITIAL PLAN
# ─────────────────────────────────────────────
def _initial_plan(n_days: int, max_load: float,
                  start_fraction: float = 0.35) -> np.ndarray:
    """Periodised starting plan: linear build → peak (wk12) → taper."""
    loads = np.zeros(n_days)
    n_weeks = n_days // 7
    day_pattern = np.array([0.18, 0.15, 0.20, 0.15, 0.18, 0.14, 0.0])
    for w in range(n_weeks):
        if w < 12:
            frac = start_fraction + (0.85 - start_fraction) * (w / 11)
        else:
            frac = 0.85 * max(0.40, 1 - (w - 11) * 0.22)
        loads[w*7:(w+1)*7] = max_load * frac * day_pattern
    return loads


# ─────────────────────────────────────────────
# 5.  SIMULATED ANNEALING OPTIMISER
# ─────────────────────────────────────────────
def simulated_annealing(
    params:          BussoParams,
    n_days:          int   = 112,
    race_day:        int   = 111,
    T_init:          float = 5.0,
    T_final:         float = 0.005,
    n_iter:          int   = 60_000,
    max_daily_load:  float = 20.0,
    start_fraction:  float = 0.35,
    seed:            int   = 42,
) -> Tuple[np.ndarray, List[float]]:
    """
    Simulated Annealing optimisation of daily training loads.

    Three neighbourhood moves:
      (A) Perturb a single day   – fine-grained local search
      (B) Swap two days          – redistributes load without changing volume
      (C) Scale an entire week   – changes total weekly volume

    Cooling: geometric schedule  T(i) = T_init * alpha^i
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    loads = _initial_plan(n_days, max_daily_load, start_fraction)

    cur_score  = objective(loads, params, race_day, max_daily_load)
    best_loads = loads.copy()
    best_score = cur_score
    history    = [cur_score]

    alpha = (T_final / T_init) ** (1.0 / n_iter)
    T     = T_init

    for i in range(n_iter):
        T *= alpha
        new_loads = loads.copy()
        move = rng.random()

        if move < 0.50:
            # (A) Perturb single day
            day = rng.randint(0, n_days - 1)
            delta = rng.gauss(0, 1.5)
            new_loads[day] = float(np.clip(new_loads[day] + delta, 0, max_daily_load))

        elif move < 0.75:
            # (B) Swap two days
            d1 = rng.randint(0, n_days - 1)
            d2 = rng.randint(0, n_days - 1)
            new_loads[d1], new_loads[d2] = new_loads[d2], new_loads[d1]

        else:
            # (C) Scale a whole week
            wk = rng.randint(0, n_days // 7 - 1)
            scale = rng.gauss(1.0, 0.04)
            new_loads[wk*7:(wk+1)*7] = np.clip(
                new_loads[wk*7:(wk+1)*7] * scale, 0, max_daily_load)

        new_score = objective(new_loads, params, race_day, max_daily_load)
        delta_e   = new_score - cur_score

        if delta_e > 0 or rng.random() < math.exp(delta_e / (T + 1e-12)):
            loads     = new_loads
            cur_score = new_score
            if cur_score > best_score:
                best_score = cur_score
                best_loads = loads.copy()

        if i % 5000 == 0:
            history.append(best_score)

    return best_loads, history


# ─────────────────────────────────────────────
# 6.  VISUALISATION
# ─────────────────────────────────────────────
def plot_results(loads: np.ndarray,
                 params: BussoParams,
                 history: List[float]):
    perf, g, h, k1 = simulate_busso(loads, params)

    days   = np.arange(len(loads))
    n_w    = len(loads) // 7
    weeks  = np.arange(1, n_w + 1)
    weekly = np.array([loads[w*7:(w+1)*7].sum() for w in range(n_w)])

    fig = plt.figure(figsize=(17, 13))
    fig.suptitle(
        "16-Week Marathon Training Optimisation\n"
        "Busso Variable Dose-Response Model + Simulated Annealing",
        fontsize=14, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.35)

    # (A) Daily load
    ax1 = fig.add_subplot(gs[0, :])
    bar_colors = ['#d62728' if d % 7 == 6 else '#1f77b4' for d in days]
    ax1.bar(days, loads, color=bar_colors, alpha=0.75, width=0.8)
    ax1.set_xlabel("Day")
    ax1.set_ylabel("Training Load (AU)")
    ax1.set_title("(A) Optimised Daily Training Load  (red = rest day)")
    ax1.set_xticks(np.arange(0, len(days), 7))
    ax1.set_xticklabels([f"W{w+1}" for w in range(n_w)], fontsize=8)
    ax1.grid(axis='y', alpha=0.3)

    # (B) Weekly volume
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.bar(weeks, weekly, color='#2ca02c', alpha=0.8)
    ax2.set_xlabel("Week")
    ax2.set_ylabel("Weekly Volume (AU)")
    ax2.set_title("(B) Weekly Training Volume")
    ax2.grid(axis='y', alpha=0.3)

    # (C) Fitness, Fatigue, Variable Gain k1
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(days, g,  label='Fitness g(t)',       color='#2ca02c', lw=2)
    ax3.plot(days, h,  label='Fatigue h(t)',        color='#d62728', lw=2)
    ax3b = ax3.twinx()
    ax3b.plot(days, k1, label='Variable gain k1(t)', color='#ff7f0e',
              lw=1.5, ls='--')
    ax3b.set_ylabel("k1(t)", color='#ff7f0e')
    ax3b.tick_params(axis='y', labelcolor='#ff7f0e')
    ax3.axvline(111, color='k', ls=':', lw=1.5)
    ax3.set_xlabel("Day")
    ax3.set_ylabel("Model Units")
    ax3.set_title("(C) Fitness, Fatigue & Variable Gain k1(t)")
    lines1, labs1 = ax3.get_legend_handles_labels()
    lines2, labs2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc='upper left')
    ax3.grid(alpha=0.3)

    # (D) Predicted performance
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(days, perf, color='#9467bd', lw=2.5)
    ax4.axvline(111, color='k', ls='--', lw=1.5, label='Race day')
    ax4.scatter([111], [perf[111]], color='gold', zorder=5, s=120,
                label=f"Race perf = {perf[111]:.2f} AU")
    ax4.set_xlabel("Day")
    ax4.set_ylabel("Performance (AU)")
    ax4.set_title("(D) Predicted Performance (VDR Model)")
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)

    # (E) SA convergence
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(history, color='#ff7f0e', lw=2)
    ax5.set_xlabel("SA Checkpoint (×5 000 iters)")
    ax5.set_ylabel("Best Objective Score")
    ax5.set_title("(E) Simulated Annealing Convergence")
    ax5.grid(alpha=0.3)

    plt.savefig("busso_marathon_optimisation.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Figure saved → busso_marathon_optimisation.png")


# ─────────────────────────────────────────────
# 7.  REPORTING
# ─────────────────────────────────────────────
def print_weekly_plan(loads: np.ndarray, params: BussoParams):
    perf, g, h, k1 = simulate_busso(loads, params)
    n_weeks = len(loads) // 7
    print("\n" + "="*72)
    print(f"{'Wk':>3} {'Vol':>6} {'Mon':>5} {'Tue':>5} {'Wed':>5} {'Thu':>5} "
          f"{'Fri':>5} {'Sat':>5} {'Sun':>5}  {'Perf':>7}  {'k1':>7}")
    print("-"*72)
    for w in range(n_weeks):
        d   = loads[w*7:(w+1)*7]
        end = (w+1)*7 - 1
        print(f"{w+1:>3} {d.sum():>6.1f} {d[0]:>5.1f} {d[1]:>5.1f} {d[2]:>5.1f} "
              f"{d[3]:>5.1f} {d[4]:>5.1f} {d[5]:>5.1f} {d[6]:>5.1f}  "
              f"{perf[end]:>7.3f}  {k1[end]:>7.4f}")
    print("="*72)
    print(f"\n▶ Race day (day 111) summary:")
    print(f"  Predicted performance : {perf[111]:.3f} AU")
    print(f"  Fitness  g(t)         : {g[111]:.3f}")
    print(f"  Fatigue  h(t)         : {h[111]:.3f}")
    print(f"  Variable gain k1(t)   : {k1[111]:.4f}")
    print(f"  Form  (g - h)         : {g[111] - h[111]:.3f}\n")


# ─────────────────────────────────────────────
# 8.  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    params = BussoParams(
        p0   = 50.0,   # baseline performance
        k1b  = 0.08,   # baseline gain scaling (k̄1) — key VDR parameter
        k2   = 1.8,    # fatigue gain factor
        tau1 = 45.0,   # fitness time constant (days)
        tau2 = 15.0,   # fatigue time constant (days)
        tau3 = 38.0,   # gain adaptation time constant (days) — Busso 2003
    )

    print("Running Simulated Annealing with Busso VDR model …")
    print("(60 000 iterations – ~10–20 s depending on hardware)\n")

    best_loads, sa_history = simulated_annealing(
        params         = params,
        n_days         = 112,
        race_day       = 111,
        T_init         = 5.0,
        T_final        = 0.005,
        n_iter         = 60_000,
        max_daily_load = 20.0,
        start_fraction = 0.35,
        seed           = 42,
    )

    print_weekly_plan(best_loads, params)
    plot_results(best_loads, params, sa_history)