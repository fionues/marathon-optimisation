"""
Marathon Training Optimization via Simulated Annealing
using the Busso Variable-Dose Model.

Key design decisions vs. previous version:
  - Decision variables: 16 weekly volumes  (not 112 daily loads)
  - Within-week distribution: fixed realistic template
  - Objective: Busso performance  −  quadratic injury penalty
    → optimizer must balance load vs. injury risk (no trivial max-everything)
  - Taper is part of the optimization, not hardcoded
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─── Busso model parameters ───────────────────────────────────────────────────
P0   = 50.0    # baseline performance
K1   = 1.0     # fitness gain factor
K2   = 2.0     # fatigue gain factor
TAU1 = 45.0    # fitness decay time-constant (days)
TAU2 = 15.0    # fatigue decay time-constant (days)

# ─── Injury penalty ───────────────────────────────────────────────────────────
# Quadratic penalty on weekly volume above a "safe" threshold.
# Forces the optimizer to spread load rather than always max out.
K3             = 0.003   # injury penalty weight  (tune this!)
INJURY_THRESH  = 70.0    # km/week — below this, no penalty

# ─── Training plan dimensions ─────────────────────────────────────────────────
N_WEEKS = 16

# ─── Weekly volume bounds ─────────────────────────────────────────────────────
VOL_MIN = 20.0    # km/week minimum (don't undertrain)
VOL_MAX = 90.0    # km/week hard cap
VOL_START = 40.0  # realistic starting volume

# ─── 10% rule ─────────────────────────────────────────────────────────────────
TEN_PERCENT_RULE = 1.10

# ─── Within-week distribution template ───────────────────────────────────────
# Day indices: Mon=0, Tue=1, Wed=2, Thu=3, Fri=4, Sat=5, Sun=6
#
#   Mon  — rest         (0%)
#   Tue  — easy run     (15%)
#   Wed  — medium run   (20%)
#   Thu  — easy run     (15%)
#   Fri  — rest         (0%)
#   Sat  — LONG run     (32%)
#   Sun  — medium run   (18%)
#
# Fractions sum to 1.0
WEEKLY_TEMPLATE = np.array([0.00, 0.15, 0.20, 0.15, 0.00, 0.32, 0.18])
DAY_LABELS      = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

# Absolute per-day caps (km) — safety ceiling even after template scaling
DAY_CAPS = np.array([0.0, 12.0, 15.0, 12.0, 0.0, 35.0, 20.0])

# ─── SA hyper-parameters ──────────────────────────────────────────────────────
T_INIT   = 10.0
T_MIN    = 1e-4
ALPHA    = 0.9995
N_ITER   = 100_000
PERTURB_STD = 3.0    # km std-dev on weekly volume


# ══════════════════════════════════════════════════════════════════════════════
# Expand weekly volumes → daily loads
# ══════════════════════════════════════════════════════════════════════════════

def expand_to_daily(vols: np.ndarray) -> np.ndarray:
    """
    vols : shape (16,) — weekly volumes in km
    returns W : shape (16, 7) — daily loads in km
    """
    W = np.zeros((N_WEEKS, 7))
    for n, v in enumerate(vols):
        raw = WEEKLY_TEMPLATE * v
        # Clip each day to its absolute cap; redistribute excess to other days
        for d in range(7):
            if raw[d] > DAY_CAPS[d]:
                excess      = raw[d] - DAY_CAPS[d]
                raw[d]      = DAY_CAPS[d]
                # Spread excess proportionally to uncapped days
                uncapped    = [j for j in range(7)
                               if j != d and DAY_CAPS[j] > 0 and raw[j] < DAY_CAPS[j]]
                if uncapped:
                    share = excess / len(uncapped)
                    for j in uncapped:
                        raw[j] = min(raw[j] + share, DAY_CAPS[j])
        W[n] = raw
    return W


# ══════════════════════════════════════════════════════════════════════════════
# Busso model
# ══════════════════════════════════════════════════════════════════════════════

def busso_performance(vols: np.ndarray) -> float:
    """
    Compute Busso performance on race day (day 112).
    """
    W      = expand_to_daily(vols)
    loads  = W.flatten()        # length 112, chronological
    T_race = len(loads)

    fitness = fatigue = 0.0
    for i, w in enumerate(loads):
        dt      = T_race - i
        fitness += w * np.exp(-dt / TAU1)
        fatigue += w * np.exp(-dt / TAU2)

    return P0 + K1 * fitness - K2 * fatigue


# ══════════════════════════════════════════════════════════════════════════════
# Injury penalty  (baked into objective — not a hard constraint)
# ══════════════════════════════════════════════════════════════════════════════

def injury_penalty(vols: np.ndarray) -> float:
    """
    Quadratic penalty on weekly volumes above INJURY_THRESH.
    Penalises spikes much harder than moderate overload:
      excess=10 → penalty ∝ 100
      excess=20 → penalty ∝ 400
    """
    excess = np.maximum(0.0, vols - INJURY_THRESH)
    return K3 * np.sum(excess ** 2)


# ══════════════════════════════════════════════════════════════════════════════
# Constraint penalty  (hard structural rules)
# ══════════════════════════════════════════════════════════════════════════════

def constraint_penalty(vols: np.ndarray) -> float:
    p = 0.0

    for n, v in enumerate(vols):
        # Volume bounds
        if v < VOL_MIN:
            p += 300.0 * (VOL_MIN - v)
        if v > VOL_MAX:
            p += 300.0 * (v - VOL_MAX)

        # 10% rule
        if n > 0:
            cap = TEN_PERCENT_RULE * vols[n - 1]
            if v > cap:
                p += 200.0 * (v - cap)

    return p


# ══════════════════════════════════════════════════════════════════════════════
# Combined objective  (SA minimises this)
# ══════════════════════════════════════════════════════════════════════════════

def objective(vols: np.ndarray) -> float:
    perf     = busso_performance(vols)
    inj_pen  = injury_penalty(vols)
    con_pen  = constraint_penalty(vols)
    # Maximise performance, minimise injury risk and constraint violations
    return -(perf - inj_pen) + con_pen


# ══════════════════════════════════════════════════════════════════════════════
# Initial solution
# ══════════════════════════════════════════════════════════════════════════════

def initial_volumes() -> np.ndarray:
    """
    Ramp up for 13 weeks, taper for 3.
    Respects 10% rule and VOL bounds by construction.
    """
    vols = np.zeros(N_WEEKS)
    for n in range(N_WEEKS):
        if n < 13:
            vols[n] = min(VOL_START * (1.08 ** n), VOL_MAX)
        else:
            vols[n] = vols[12] * (0.75 ** (n - 12))
        vols[n] = max(vols[n], VOL_MIN)
    return vols


# ══════════════════════════════════════════════════════════════════════════════
# Neighbourhood move — single weekly volume perturbation
# ══════════════════════════════════════════════════════════════════════════════

def neighbour(vols: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    v_new = vols.copy()
    n     = rng.integers(0, N_WEEKS)
    delta = rng.normal(0, PERTURB_STD)
    v_new[n] = np.clip(v_new[n] + delta, VOL_MIN, VOL_MAX)
    return v_new


# ══════════════════════════════════════════════════════════════════════════════
# Simulated Annealing
# ══════════════════════════════════════════════════════════════════════════════

def simulated_annealing(seed: int = 42):
    rng    = np.random.default_rng(seed)
    vols   = initial_volumes()
    E      = objective(vols)

    vols_best = vols.copy()
    E_best    = E

    T       = T_INIT
    history = []

    for it in range(N_ITER):
        v_cand = neighbour(vols, rng)
        E_cand = objective(v_cand)
        dE     = E_cand - E

        if dE < 0 or rng.random() < np.exp(-dE / T):
            vols = v_cand
            E    = E_cand
            if E < E_best:
                vols_best = vols.copy()
                E_best    = E

        T = max(T * ALPHA, T_MIN)

        if it % 2000 == 0:
            history.append(-E_best)

    return vols_best, history


# ══════════════════════════════════════════════════════════════════════════════
# Constraint verification
# ══════════════════════════════════════════════════════════════════════════════

def verify(vols: np.ndarray):
    print("\n── Constraint Verification ──────────────────────────────────────")
    ok = True
    for n, v in enumerate(vols):
        if v < VOL_MIN - 1e-6:
            print(f"  ❌ Week {n+1}: volume {v:.1f} < min {VOL_MIN}")
            ok = False
        if v > VOL_MAX + 1e-6:
            print(f"  ❌ Week {n+1}: volume {v:.1f} > max {VOL_MAX}")
            ok = False
        if n > 0 and v > TEN_PERCENT_RULE * vols[n-1] + 1e-6:
            print(f"  ❌ Week {n+1}: 10% rule violated "
                  f"({v:.1f} > 1.1 × {vols[n-1]:.1f})")
            ok = False
    if ok:
        print("  ✅ All constraints satisfied.")
    print("─────────────────────────────────────────────────────────────────")


# ══════════════════════════════════════════════════════════════════════════════
# Visualisation
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(vols: np.ndarray, history: list):
    W = expand_to_daily(vols)

    fig = plt.figure(figsize=(16, 11), facecolor='#0f0f0f')
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. Heatmap ────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor('#1a1a1a')

    for n in range(N_WEEKS):
        for d in range(7):
            load = W[n, d]
            cap  = DAY_CAPS[d] if DAY_CAPS[d] > 0 else 1.0
            if load < 0.5:
                c = '#2a2a2a'
            elif d == 5:      # long run — orange ramp
                intensity = min(load / 35.0, 1.0)
                r = int(180 + 75 * intensity)
                c = f'#{r:02x}6b35'
            elif d in [2, 6]: # medium runs — teal-ish
                intensity = min(load / 20.0, 1.0)
                g = int(100 + 100 * intensity)
                c = f'#1d{g:02x}75'
            else:             # easy runs — blue ramp
                intensity = min(load / 12.0, 1.0)
                b = int(100 + 155 * intensity)
                c = f'#4a9e{b:02x}'

            rect = plt.Rectangle([n - 0.45, d - 0.45], 0.9, 0.9,
                                  facecolor=c, edgecolor='#0f0f0f', lw=0.5)
            ax1.add_patch(rect)
            if load > 0.5:
                ax1.text(n, d, f'{load:.0f}', ha='center', va='center',
                         fontsize=6.5, color='white', fontweight='bold')

    ax1.set_xlim(-0.5, N_WEEKS - 0.5)
    ax1.set_ylim(-0.5, 6.5)
    ax1.set_xticks(range(N_WEEKS))
    ax1.set_xticklabels([f'W{n+1}' for n in range(N_WEEKS)],
                        color='#aaaaaa', fontsize=8)
    ax1.set_yticks(range(7))
    ax1.set_yticklabels(DAY_LABELS, color='#aaaaaa', fontsize=9)
    ax1.set_title('Optimised Training Plan  —  Daily Load (km)',
                  color='white', fontsize=13, pad=10)
    for spine in ax1.spines.values():
        spine.set_edgecolor('#333333')

    # ── 2. Weekly volume + long run ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor('#1a1a1a')

    weeks    = range(N_WEEKS)
    bar_cols = ['#4a9eff' if n < 13 else '#ff6b35' for n in weeks]
    ax2.bar(weeks, vols, color=bar_cols, edgecolor='#0f0f0f', lw=0.5,
            label='Weekly volume')
    ax2.plot(weeks, W[:, 5], 'o--', color='#ff9f40', lw=1.5, ms=4,
             label='Long run (Sat)')
    ax2.axvline(12.5, color='#ff6b35', lw=1.5, linestyle='--', alpha=0.6,
                label='Taper start')
    ax2.set_xticks(range(N_WEEKS))
    ax2.set_xticklabels([f'W{n+1}' for n in range(N_WEEKS)],
                        color='#aaaaaa', fontsize=7, rotation=45)
    ax2.set_ylabel('km', color='#aaaaaa')
    ax2.set_title('Weekly Volume & Long Run', color='white', fontsize=11)
    ax2.legend(fontsize=8, facecolor='#2a2a2a',
               labelcolor='white', edgecolor='#444444')
    ax2.tick_params(colors='#555555')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#333333')

    # ── 3. Busso fitness / fatigue curves ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor('#1a1a1a')

    loads    = W.flatten()
    n_days   = len(loads)
    T_race   = n_days + 1
    fit_arr  = np.zeros(T_race)
    fat_arr  = np.zeros(T_race)

    for t in range(1, T_race + 1):
        f = fa = 0.0
        for i, w in enumerate(loads[:t]):
            dt = t - i
            f  += w * np.exp(-dt / TAU1)
            fa += w * np.exp(-dt / TAU2)
        if t <= T_race:
            fit_arr[t - 1] = K1 * f
            fat_arr[t - 1] = K2 * fa

    days = np.arange(1, T_race + 1)
    ax3.plot(days, fit_arr,  color='#4a9eff', lw=1.5, label='Fitness')
    ax3.plot(days, fat_arr,  color='#ff6b35', lw=1.5, label='Fatigue')
    ax3.plot(days, fit_arr - fat_arr,
             color='#aaffaa', lw=2.0, label='Form (fitness − fatigue)')
    ax3.axvline(n_days, color='white', lw=1.2, linestyle='--', alpha=0.5,
                label='Race day')
    ax3.set_xlabel('Day', color='#aaaaaa')
    ax3.set_title('Busso Fitness / Fatigue / Form', color='white', fontsize=11)
    ax3.legend(fontsize=8, facecolor='#2a2a2a',
               labelcolor='white', edgecolor='#444444')
    ax3.tick_params(colors='#555555')
    for spine in ax3.spines.values():
        spine.set_edgecolor('#333333')

    # ── 4. SA convergence ─────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, :])
    ax4.set_facecolor('#1a1a1a')

    iters = np.arange(len(history)) * 2000
    ax4.plot(iters, history, color='#4a9eff', lw=1.5)
    ax4.fill_between(iters, min(history), history,
                     alpha=0.15, color='#4a9eff')
    ax4.set_xlabel('Iteration', color='#aaaaaa')
    ax4.set_ylabel('Best Performance (Busso)', color='#aaaaaa')
    ax4.set_title('SA Convergence', color='white', fontsize=11)
    ax4.tick_params(colors='#555555')
    for spine in ax4.spines.values():
        spine.set_edgecolor('#333333')

    plt.suptitle(
        'Marathon Training Optimisation  ·  Weekly-Volume SA + Busso + Injury Penalty',
        color='white', fontsize=13, y=1.005, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Running Simulated Annealing (weekly-volume space) …\n")
    vols_opt, history = simulated_annealing(seed=42)
    W_opt             = expand_to_daily(vols_opt)

    perf    = busso_performance(vols_opt)
    inj     = injury_penalty(vols_opt)
    net     = perf - inj

    print(f"Busso performance on race day : {perf:.3f}")
    print(f"Injury penalty                : {inj:.3f}")
    print(f"Net objective (perf − injury) : {net:.3f}")
    print(f"Total training volume         : {W_opt.sum():.1f} km")
    print(f"Peak weekly volume            : {vols_opt.max():.1f} km  "
          f"(week {vols_opt.argmax()+1})")

    verify(vols_opt)

    print("\nWeekly breakdown (km):")
    print(f"{'Wk':<4} {'Vol':>6}  "
          f"{'Mon':>5} {'Tue':>5} {'Wed':>5} {'Thu':>5} "
          f"{'Fri':>5} {'Sat(L)':>7} {'Sun':>5}")
    print("─" * 58)
    for n in range(N_WEEKS):
        row = W_opt[n]
        tag = '← taper' if n >= 13 else ''
        print(f"W{n+1:<3} {vols_opt[n]:6.1f}  "
              + "  ".join(f"{v:5.1f}" for v in row)
              + f"  {tag}")

    plot_results(vols_opt, history)