"""
Marathon Training Optimization via Simulated Annealing
using the Busso Variable-Dose Model.

State:  W[n, d]  — training load (km) on week n, day d  (n=0..15, d=0..6)
        d=0: Monday, ..., d=4: Friday, d=5: Saturday, d=6: Sunday

Busso model:
    fitness(t)  = k1 * Σ_{i<t} W_i * exp(-(t - i) / tau1)
    fatigue(t)  = k2 * Σ_{i<t} W_i * exp(-(t - i) / tau2)
    performance(t) = p0 + fitness(t) - fatigue(t)

Goal: maximise performance on race day (day after last training week).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─── Busso model parameters (typical literature values) ──────────────────────
P0    = 50.0   # baseline performance
K1    = 1.0    # fitness gain factor
K2    = 2.0    # fatigue gain factor  (>K1 so overtraining hurts)
TAU1  = 45.0   # fitness decay time constant (days)
TAU2  = 15.0   # fatigue decay time constant (days)

# ─── Training plan dimensions ────────────────────────────────────────────────
N_WEEKS = 16
N_DAYS  = 7      # Mon=0 … Sun=6
WEEKDAYS   = [0, 1, 2, 3, 4]   # Mon–Fri
WEEKEND    = [5, 6]             # Sat, Sun

# ─── Constraint parameters ───────────────────────────────────────────────────
MAX_WEEKDAY_LOAD  = 12.0   # km × intensity  (constraint 1)
MAX_WEEKEND_LOAD  = 35.0   # long run hard cap per weekend day
MIN_REST_PER_WEEK = 1      # at least 1 rest day  (constraint 2)
TEN_PERCENT_RULE  = 1.10   # week-over-week volume cap  (constraint 3)
INITIAL_WEEKLY_KM = 40.0   # realistic starting volume for non-novice

# ─── SA hyper-parameters ─────────────────────────────────────────────────────
T_INIT      = 5.0
T_MIN       = 1e-4
ALPHA       = 0.995        # geometric cooling
N_ITER      = 200_000      # total moves
PERTUB_STD  = 2.0          # km std-dev of Gaussian perturbation


# ══════════════════════════════════════════════════════════════════════════════
# Helper: flatten / unflatten plan
# ══════════════════════════════════════════════════════════════════════════════

def plan_to_daily(W: np.ndarray) -> np.ndarray:
    """Return 1-D array of daily loads in chronological order (shape: 112,)."""
    return W.flatten()                  # row-major: week0_mon, …, week15_sun


# ══════════════════════════════════════════════════════════════════════════════
# Busso model
# ══════════════════════════════════════════════════════════════════════════════

def busso_performance(W: np.ndarray) -> float:
    """
    Compute performance on race day (= day 112, i.e. day after last training).
    W : shape (N_WEEKS, N_DAYS)
    """
    loads = plan_to_daily(W)   # length 112
    T_race = len(loads)        # race is on day 112 (index 112)

    fitness  = 0.0
    fatigue  = 0.0
    for i, w in enumerate(loads):
        dt = T_race - i        # days before race
        fitness  += w * np.exp(-dt / TAU1)
        fatigue  += w * np.exp(-dt / TAU2)

    return P0 + K1 * fitness - K2 * fatigue


# ══════════════════════════════════════════════════════════════════════════════
# Constraint checks & penalty
# ══════════════════════════════════════════════════════════════════════════════

def penalty(W: np.ndarray) -> float:
    """
    Returns a non-negative penalty for constraint violations.
    Larger violation → larger penalty.
    """
    p = 0.0

    for n in range(N_WEEKS):
        week = W[n]                        # shape (7,)
        rest_days = np.sum(week == 0.0)

        # ── Constraint 1a: weekday loads ≤ MAX_WEEKDAY_LOAD ──────────────────
        for d in WEEKDAYS:
            excess = week[d] - MAX_WEEKDAY_LOAD
            if excess > 0:
                p += 500.0 * excess

        # ── Constraint 1b: weekend loads ≤ MAX_WEEKEND_LOAD ──────────────────
        for d in WEEKEND:
            excess = week[d] - MAX_WEEKEND_LOAD
            if excess > 0:
                p += 500.0 * excess

        # ── Constraint 2: ≥ 1 rest day per week ──────────────────────────────
        deficit = MIN_REST_PER_WEEK - rest_days
        if deficit > 0:
            p += 500.0 * deficit

        # ── Constraint 3: 10 % rule ───────────────────────────────────────────
        if n > 0:
            vol_prev = W[n-1].sum()
            vol_curr = week.sum()
            excess = vol_curr - TEN_PERCENT_RULE * vol_prev
            if excess > 0:
                p += 200.0 * excess

    # ── Constraint 4: non-negativity (W ≥ 0) ─────────────────────────────────
    neg_mask = W < 0
    p += 1000.0 * (-W[neg_mask]).sum()

    return p


def objective(W: np.ndarray) -> float:
    """We maximise performance − penalty  →  SA minimises the negative."""
    return -(busso_performance(W) - penalty(W))


# ══════════════════════════════════════════════════════════════════════════════
# Initial feasible solution
# ══════════════════════════════════════════════════════════════════════════════

def initial_plan() -> np.ndarray:
    """
    Build a simple hand-crafted feasible starting plan:
    - 1 long run on Saturday (~30 % of weekly volume)
    - 1 medium run on Sunday  (~20 %)
    - 3–4 weekday runs, rest on Mon & another day
    - Weekly volume grows by 10 % each week (respects 10 % rule)
    - Taper: last 3 weeks volume decreases
    """
    W = np.zeros((N_WEEKS, N_DAYS))

    for n in range(N_WEEKS):
        # Taper in final 3 weeks
        if n < 13:
            vol = INITIAL_WEEKLY_KM * (1.05 ** n)   # gentle build
        else:
            vol = INITIAL_WEEKLY_KM * (1.05 ** 12) * (0.75 ** (n - 12))

        # Distribute: Sat=30%, Sun=20%, Tue/Wed/Thu = ~16% each, rest Mon/Fri
        W[n, 5] = 0.30 * vol   # Saturday long run
        W[n, 6] = 0.20 * vol   # Sunday medium
        W[n, 1] = 0.17 * vol   # Tuesday
        W[n, 2] = 0.17 * vol   # Wednesday
        W[n, 3] = 0.16 * vol   # Thursday
        # Monday (0) and Friday (4) = rest → 0

        # Clip weekday loads to constraint 1
        for d in WEEKDAYS:
            W[n, d] = min(W[n, d], MAX_WEEKDAY_LOAD)

    return W


# ══════════════════════════════════════════════════════════════════════════════
# Neighbourhood move
# ══════════════════════════════════════════════════════════════════════════════

def neighbour(W: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Generate a neighbouring plan by one of three move types:
      A) Perturb a single day's load (most common)
      B) Shift load between two days in the same week
      C) Force a rest day swap (helps satisfy constraint 2)
    """
    W_new = W.copy()
    move = rng.choice(['perturb', 'shift', 'rest_swap'], p=[0.6, 0.25, 0.15])

    n = rng.integers(0, N_WEEKS)

    if move == 'perturb':
        d = rng.integers(0, N_DAYS)
        delta = rng.normal(0, PERTUB_STD)
        W_new[n, d] = max(0.0, W_new[n, d] + delta)
        # Re-clip weekday constraint
        if d in WEEKDAYS:
            W_new[n, d] = min(W_new[n, d], MAX_WEEKDAY_LOAD)

    elif move == 'shift':
        # Move load from one day to another within same week
        d1, d2 = rng.choice(N_DAYS, size=2, replace=False)
        amount = rng.uniform(0, W_new[n, d1])
        W_new[n, d1] -= amount
        W_new[n, d2] += amount
        for d in [d1, d2]:
            if d in WEEKDAYS:
                W_new[n, d] = min(W_new[n, d], MAX_WEEKDAY_LOAD)
        W_new[n, d1] = max(0.0, W_new[n, d1])
        W_new[n, d2] = max(0.0, W_new[n, d2])

    else:  # rest_swap: zero out a busy day, add its load to a weekend day
        busy = [d for d in range(N_DAYS) if W_new[n, d] > 0]
        if busy:
            d_zero = rng.choice(busy)
            load   = W_new[n, d_zero]
            W_new[n, d_zero] = 0.0
            d_recv = rng.choice(WEEKEND)
            W_new[n, d_recv] = min(W_new[n, d_recv] + load, MAX_WEEKEND_LOAD)

    return W_new


# ══════════════════════════════════════════════════════════════════════════════
# Simulated Annealing
# ══════════════════════════════════════════════════════════════════════════════

def simulated_annealing(seed: int = 42):
    rng   = np.random.default_rng(seed)
    W     = initial_plan()
    E     = objective(W)

    W_best = W.copy()
    E_best = E

    T = T_INIT
    history = []   # track best performance over iterations

    for it in range(N_ITER):
        W_cand = neighbour(W, rng)
        E_cand = objective(W_cand)

        dE = E_cand - E

        # Accept if better, or probabilistically if worse
        if dE < 0 or rng.random() < np.exp(-dE / T):
            W = W_cand
            E = E_cand

            if E < E_best:
                W_best = W.copy()
                E_best = E

        # Cool down
        T = max(T * ALPHA, T_MIN)

        if it % 5000 == 0:
            history.append(-E_best)   # store performance (not negated obj)

    return W_best, -E_best, history


# ══════════════════════════════════════════════════════════════════════════════
# Visualisation
# ══════════════════════════════════════════════════════════════════════════════

DAY_LABELS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

def plot_results(W: np.ndarray, history: list):
    fig = plt.figure(figsize=(16, 10), facecolor='#0f0f0f')
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    colors = {
        'weekday': '#4a9eff',
        'weekend': '#ff6b35',
        'rest':    '#2a2a2a',
    }

    # ── 1. Heatmap of daily loads ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor('#1a1a1a')

    for n in range(N_WEEKS):
        for d in range(N_DAYS):
            load = W[n, d]
            if load < 0.5:
                c = colors['rest']
            elif d in WEEKEND:
                intensity = min(load / MAX_WEEKEND_LOAD, 1.0)
                r = int(255 * intensity)
                c = f'#{r:02x}6b35'
            else:
                intensity = min(load / MAX_WEEKDAY_LOAD, 1.0)
                r = int(74 + 100 * intensity)
                c = f'#{r:02x}9eff'

            rect = plt.Rectangle([n - 0.45, d - 0.45], 0.9, 0.9,
                                  facecolor=c, edgecolor='#0f0f0f', lw=0.5)
            ax1.add_patch(rect)
            if load > 1:
                ax1.text(n, d, f'{load:.0f}', ha='center', va='center',
                         fontsize=6.5, color='white', fontweight='bold')

    ax1.set_xlim(-0.5, N_WEEKS - 0.5)
    ax1.set_ylim(-0.5, N_DAYS - 0.5)
    ax1.set_xticks(range(N_WEEKS))
    ax1.set_xticklabels([f'W{n+1}' for n in range(N_WEEKS)],
                        color='#aaaaaa', fontsize=8)
    ax1.set_yticks(range(N_DAYS))
    ax1.set_yticklabels(DAY_LABELS, color='#aaaaaa', fontsize=9)
    ax1.set_title('Optimised Training Plan  —  Daily Load (km)',
                  color='white', fontsize=13, pad=10)
    ax1.tick_params(colors='#555555')
    for spine in ax1.spines.values():
        spine.set_edgecolor('#333333')

    # ── 2. Weekly volume bar chart ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor('#1a1a1a')

    weekly_vols = W.sum(axis=1)
    bar_colors  = ['#4a9eff' if n < 13 else '#ff6b35' for n in range(N_WEEKS)]
    ax2.bar(range(N_WEEKS), weekly_vols, color=bar_colors, edgecolor='#0f0f0f', lw=0.5)
    ax2.axvline(12.5, color='#ff6b35', lw=1.5, linestyle='--', alpha=0.7,
                label='Taper start')
    ax2.set_xticks(range(N_WEEKS))
    ax2.set_xticklabels([f'W{n+1}' for n in range(N_WEEKS)],
                        color='#aaaaaa', fontsize=7, rotation=45)
    ax2.set_ylabel('km / week', color='#aaaaaa')
    ax2.set_title('Weekly Volume', color='white', fontsize=11)
    ax2.tick_params(colors='#555555')
    ax2.legend(fontsize=8, facecolor='#2a2a2a', labelcolor='white',
               edgecolor='#444444')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#333333')
    ax2.yaxis.label.set_color('#aaaaaa')

    # ── 3. SA convergence ────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor('#1a1a1a')

    iters = np.arange(len(history)) * 5000
    ax3.plot(iters, history, color='#4a9eff', lw=1.5)
    ax3.fill_between(iters, min(history), history, alpha=0.15, color='#4a9eff')
    ax3.set_xlabel('Iteration', color='#aaaaaa')
    ax3.set_ylabel('Best Performance (Busso)', color='#aaaaaa')
    ax3.set_title('SA Convergence', color='white', fontsize=11)
    ax3.tick_params(colors='#555555')
    for spine in ax3.spines.values():
        spine.set_edgecolor('#333333')

    plt.suptitle('Marathon Training Optimisation  ·  Simulated Annealing + Busso Model',
                 color='white', fontsize=14, y=1.01, fontweight='bold')
    plt.show()

    # plt.savefig('./outputs/marathon_training_plan.png',
    #             dpi=150, bbox_inches='tight', facecolor='#0f0f0f')
    # plt.close()
    print("Plot saved.")


# ══════════════════════════════════════════════════════════════════════════════
# Constraint verification (for reporting)
# ══════════════════════════════════════════════════════════════════════════════

def verify_constraints(W: np.ndarray):
    print("\n── Constraint Verification ──────────────────────────────────────")
    all_ok = True

    for n in range(N_WEEKS):
        week = W[n]
        rest = int(np.sum(week < 0.5))

        # C1: weekday loads
        for d in WEEKDAYS:
            if week[d] > MAX_WEEKDAY_LOAD + 1e-6:
                print(f"  ❌ C1 violated  week {n+1} {DAY_LABELS[d]}: "
                      f"{week[d]:.1f} km > {MAX_WEEKDAY_LOAD} km")
                all_ok = False

        # C2: rest days
        if rest < MIN_REST_PER_WEEK:
            print(f"  ❌ C2 violated  week {n+1}: only {rest} rest day(s)")
            all_ok = False

        # C3: 10 % rule
        if n > 0:
            vol_prev = W[n-1].sum()
            vol_curr = week.sum()
            if vol_curr > TEN_PERCENT_RULE * vol_prev + 1e-6:
                print(f"  ❌ C3 violated  week {n+1}: "
                      f"{vol_curr:.1f} > 1.1 × {vol_prev:.1f} = "
                      f"{TEN_PERCENT_RULE * vol_prev:.1f}")
                all_ok = False

    # C4: non-negativity
    if np.any(W < -1e-6):
        print("  ❌ C4 violated: negative loads found")
        all_ok = False

    if all_ok:
        print("  ✅ All constraints satisfied.")
    print("─────────────────────────────────────────────────────────────────")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Running Simulated Annealing …")
    W_opt, perf_opt, history = simulated_annealing(seed=42)

    print(f"\nOptimised Busso performance on race day: {perf_opt:.3f}")
    print(f"Total training volume: {W_opt.sum():.1f} km")
    print(f"Peak weekly volume:    {W_opt.sum(axis=1).max():.1f} km")

    verify_constraints(W_opt)

    print("\nWeekly breakdown:")
    print(f"{'Week':<6} {'Mon':>5} {'Tue':>5} {'Wed':>5} {'Thu':>5} "
          f"{'Fri':>5} {'Sat':>5} {'Sun':>5} {'Total':>7} {'Rest':>5}")
    print("─" * 58)
    for n in range(N_WEEKS):
        row   = W_opt[n]
        total = row.sum()
        rest  = int(np.sum(row < 0.5))
        vals  = "  ".join(f"{v:5.1f}" for v in row)
        print(f"W{n+1:<4}  {vals}  {total:6.1f}  {rest:4d}")

    plot_results(W_opt, history)
    print("\nDone. Outputs saved to /mnt/user-data/outputs/")
