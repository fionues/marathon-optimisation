"""
Marathon Training Optimization via Simulated Annealing
using the CTL/ATL (Chronic/Acute Training Load) model.

Model overview
──────────────
Every day you run, you accumulate Training Stress Score (TSS).
TSS is a single number capturing both volume and intensity:

    TSS = (duration_h × intensity²) × 100
        ≈ km × intensity_factor          (we use km as a proxy here)

From daily TSS two exponential moving averages are tracked:

    CTL(t) = CTL(t−1) + (TSS(t) − CTL(t−1)) / τ_ctl    # "fitness"
    ATL(t) = ATL(t−1) + (TSS(t) − ATL(t−1)) / τ_atl    # "fatigue"

    TSB(t) = CTL(t) − ATL(t)                             # "form"

Where:
    τ_ctl = 42 days  (fitness builds and decays slowly)
    τ_atl =  7 days  (fatigue builds and decays quickly)

Optimisation goal
─────────────────
Maximise CTL on race day (arrive as fit as possible)
subject to:
    1. TSB on race day ≥ TSB_TARGET  (arrive fresh, not exhausted)
    2. ATL never exceeds ATL_MAX     (avoid injury / overtraining)
    3. Weekly volume grows ≤ 10%     (10% rule)
    4. Weekly volume ≥ VOL_MIN       (don't undertrain)
    5. Daily loads follow the weekly template (1 long / 2 medium / 2 easy / 2 rest)

Decision variables
──────────────────
16 weekly TSS values  v[0] … v[15]
Daily loads are derived from v[n] via a fixed within-week template.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─── CTL/ATL model parameters ────────────────────────────────────────────────
TAU_CTL = 42.0     # fitness time-constant (days)
TAU_ATL =  7.0     # fatigue time-constant (days)

CTL_0   =  20.0    # starting CTL (trained amateur, not a beginner)
ATL_0   =  20.0    # starting ATL (same — assumed in a steady state)

# ─── Race-day targets ─────────────────────────────────────────────────────────
TSB_TARGET =  5.0   # minimum TSB on race day (positive = fresh)
ATL_MAX    = 80.0   # hard ceiling on daily ATL (overtraining guard)

# ─── Training plan ───────────────────────────────────────────────────────────
N_WEEKS   = 16
VOL_MIN   = 25.0    # minimum weekly TSS-equivalent (km proxy)
VOL_MAX   = 100.0   # absolute hard cap (safety)
VOL_START = 40.0    # week 1 volume
TEN_PCT   = 1.10    # 10% rule multiplier

# ─── Within-week template (fractions must sum to 1.0) ────────────────────────
# Mon=rest, Tue=easy, Wed=medium, Thu=easy, Fri=rest, Sat=LONG, Sun=medium
TEMPLATE = np.array([0.00, 0.15, 0.20, 0.15, 0.00, 0.32, 0.18])
DAY_CAPS = np.array([0.0,  12.0, 15.0, 12.0,  0.0, 35.0, 20.0])
DAY_LABELS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

# ─── SA hyper-parameters ──────────────────────────────────────────────────────
T_INIT      = 15.0
T_MIN       = 1e-4
ALPHA       = 0.9997
N_ITER      = 150_000
PERTURB_STD = 3.0

# ─── Penalty weights ──────────────────────────────────────────────────────────
W_TSB  = 500.0   # penalty per unit TSB below target on race day
W_ATL  = 300.0   # penalty per unit ATL above ATL_MAX
W_10PC = 200.0   # penalty per unit volume violating 10% rule
W_MIN  = 300.0   # penalty per unit below VOL_MIN
W_MAX  = 300.0   # penalty per unit above VOL_MAX


# ══════════════════════════════════════════════════════════════════════════════
# Expand weekly volumes → daily loads
# ══════════════════════════════════════════════════════════════════════════════

def expand_to_daily(vols: np.ndarray) -> np.ndarray:
    """vols (16,) → W (16, 7) daily loads via fixed template + day caps."""
    W = np.zeros((N_WEEKS, 7))
    for n, v in enumerate(vols):
        raw = TEMPLATE * v
        for d in range(7):
            if raw[d] > DAY_CAPS[d] and DAY_CAPS[d] > 0:
                excess  = raw[d] - DAY_CAPS[d]
                raw[d]  = DAY_CAPS[d]
                others  = [j for j in range(7)
                           if j != d and DAY_CAPS[j] > 0 and raw[j] < DAY_CAPS[j]]
                if others:
                    share = excess / len(others)
                    for j in others:
                        raw[j] = min(raw[j] + share, DAY_CAPS[j])
        W[n] = raw
    return W


# ══════════════════════════════════════════════════════════════════════════════
# CTL / ATL simulation
# ══════════════════════════════════════════════════════════════════════════════

def simulate(vols: np.ndarray):
    """
    Simulate CTL, ATL, TSB for all 112 training days + race day.

    Returns
    -------
    ctl : (113,)  CTL at end of each day (index 112 = race day)
    atl : (113,)
    tsb : (113,)
    """
    loads = expand_to_daily(vols).flatten()   # 112 values
    n     = len(loads) + 1                    # +1 for race day (TSS=0)

    ctl = np.zeros(n)
    atl = np.zeros(n)

    ctl[0] = CTL_0
    atl[0] = ATL_0

    for t in range(1, n):
        tss     = loads[t - 1] if t <= len(loads) else 0.0
        ctl[t]  = ctl[t-1] + (tss - ctl[t-1]) / TAU_CTL
        atl[t]  = atl[t-1] + (tss - atl[t-1]) / TAU_ATL

    tsb = ctl - atl
    return ctl, atl, tsb


# ══════════════════════════════════════════════════════════════════════════════
# Objective  (SA minimises this)
# ══════════════════════════════════════════════════════════════════════════════

def objective(vols: np.ndarray) -> float:
    ctl, atl, tsb = simulate(vols)

    # ── Reward: maximise CTL on race day ─────────────────────────────────────
    score = ctl[-1]

    # ── Penalty 1: TSB on race day must be ≥ TSB_TARGET ──────────────────────
    pen = W_TSB * max(0.0, TSB_TARGET - tsb[-1])

    # ── Penalty 2: ATL must never exceed ATL_MAX ─────────────────────────────
    pen += W_ATL * np.sum(np.maximum(0.0, atl - ATL_MAX))

    # ── Penalty 3: 10% rule ───────────────────────────────────────────────────
    for n in range(1, N_WEEKS):
        excess = vols[n] - TEN_PCT * vols[n - 1]
        if excess > 0:
            pen += W_10PC * excess

    # ── Penalty 4: volume bounds ──────────────────────────────────────────────
    pen += W_MIN * np.sum(np.maximum(0.0, VOL_MIN - vols))
    pen += W_MAX * np.sum(np.maximum(0.0, vols - VOL_MAX))

    return -(score - pen)


# ══════════════════════════════════════════════════════════════════════════════
# Initial solution — gentle ramp then taper
# ══════════════════════════════════════════════════════════════════════════════

def initial_volumes() -> np.ndarray:
    vols = np.zeros(N_WEEKS)
    for n in range(N_WEEKS):
        if n < 13:
            vols[n] = min(VOL_START * (1.08 ** n), VOL_MAX)
        else:
            vols[n] = vols[12] * (0.75 ** (n - 12))
        vols[n] = max(vols[n], VOL_MIN)
    return vols


# ══════════════════════════════════════════════════════════════════════════════
# Neighbourhood move
# ══════════════════════════════════════════════════════════════════════════════

def neighbour(vols: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    v = vols.copy()
    n = rng.integers(0, N_WEEKS)
    v[n] = np.clip(v[n] + rng.normal(0, PERTURB_STD), VOL_MIN, VOL_MAX)
    return v


# ══════════════════════════════════════════════════════════════════════════════
# Simulated Annealing
# ══════════════════════════════════════════════════════════════════════════════

def simulated_annealing(seed: int = 42):
    rng  = np.random.default_rng(seed)
    vols = initial_volumes()
    E    = objective(vols)

    best_vols = vols.copy()
    best_E    = E
    T         = T_INIT
    history   = []

    for it in range(N_ITER):
        v_cand = neighbour(vols, rng)
        E_cand = objective(v_cand)
        dE     = E_cand - E

        if dE < 0 or rng.random() < np.exp(-dE / T):
            vols = v_cand
            E    = E_cand
            if E < best_E:
                best_vols = vols.copy()
                best_E    = E

        T = max(T * ALPHA, T_MIN)
        if it % 3000 == 0:
            history.append(-best_E)

    return best_vols, history


# ══════════════════════════════════════════════════════════════════════════════
# Reporting
# ══════════════════════════════════════════════════════════════════════════════

def report(vols: np.ndarray):
    ctl, atl, tsb = simulate(vols)
    W = expand_to_daily(vols)

    print("\n── Results ──────────────────────────────────────────────────────")
    print(f"  CTL on race day : {ctl[-1]:.1f}  (fitness)")
    print(f"  ATL on race day : {atl[-1]:.1f}  (fatigue)")
    print(f"  TSB on race day : {tsb[-1]:.1f}  (form — target ≥ {TSB_TARGET})")
    print(f"  Peak ATL        : {atl.max():.1f}  (limit {ATL_MAX})")
    print(f"  Total volume    : {W.sum():.0f} km")
    print(f"  Peak weekly vol : {vols.max():.1f} km  (week {vols.argmax()+1})")

    print("\n── Constraint check ─────────────────────────────────────────────")
    ok = True
    if tsb[-1] < TSB_TARGET - 0.1:
        print(f"  ❌ TSB on race day {tsb[-1]:.1f} < target {TSB_TARGET}"); ok=False
    if atl.max() > ATL_MAX + 0.1:
        print(f"  ❌ Peak ATL {atl.max():.1f} > limit {ATL_MAX}"); ok=False
    for n in range(1, N_WEEKS):
        if vols[n] > TEN_PCT * vols[n-1] + 0.1:
            print(f"  ❌ Week {n+1}: 10% rule violated"); ok=False
    if ok:
        print("  ✅ All constraints satisfied.")

    print("\n── Weekly breakdown ─────────────────────────────────────────────")
    print(f"{'Wk':<4} {'Vol':>6}  {'Mon':>5} {'Tue':>5} {'Wed':>5} "
          f"{'Thu':>5} {'Fri':>5} {'Sat':>5} {'Sun':>5}  {'CTL':>6} {'ATL':>6} {'TSB':>6}")
    print("─" * 80)
    for n in range(N_WEEKS):
        day_idx = n * 7 + 7          # end-of-week CTL/ATL index
        row = W[n]
        tag = ' ← taper' if n >= 13 else ''
        print(f"W{n+1:<3} {vols[n]:6.1f}  "
              + "  ".join(f"{v:5.1f}" for v in row)
              + f"  {ctl[day_idx]:6.1f} {atl[day_idx]:6.1f} {tsb[day_idx]:6.1f}"
              + tag)


# ══════════════════════════════════════════════════════════════════════════════
# Visualisation
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(vols: np.ndarray, history: list):
    ctl, atl, tsb = simulate(vols)
    W = expand_to_daily(vols)

    fig = plt.figure(figsize=(16, 13), facecolor='#0f0f0f')
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    days = np.arange(len(ctl))

    # ── 1. Daily load heatmap ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor('#1a1a1a')
    for n in range(N_WEEKS):
        for d in range(7):
            load = W[n, d]
            if load < 0.5:
                c = '#2a2a2a'
            elif d == 5:
                intensity = min(load / 35.0, 1.0)
                r = int(180 + 75 * intensity)
                c = f'#{r:02x}6b35'
            elif d in [2, 6]:
                intensity = min(load / 20.0, 1.0)
                g = int(100 + 100 * intensity)
                c = f'#1d{g:02x}75'
            else:
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
    for sp in ax1.spines.values(): sp.set_edgecolor('#333333')

    # ── 2. CTL / ATL / TSB over time ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    ax2.set_facecolor('#1a1a1a')
    ax2.plot(days, ctl, color='#4a9eff', lw=2.0, label='CTL — fitness (42d avg)')
    ax2.plot(days, atl, color='#ff6b35', lw=2.0, label='ATL — fatigue (7d avg)')
    ax2.plot(days, tsb, color='#aaffaa', lw=1.5, linestyle='--',
             label='TSB — form (CTL − ATL)')
    ax2.axhline(0,          color='#555555', lw=0.8, linestyle=':')
    ax2.axhline(TSB_TARGET, color='#aaffaa', lw=0.8, linestyle=':',
                alpha=0.5, label=f'TSB target ({TSB_TARGET})')
    ax2.axhline(ATL_MAX,    color='#ff6b35', lw=0.8, linestyle=':',
                alpha=0.5, label=f'ATL limit ({ATL_MAX})')
    ax2.axvline(112,        color='white',   lw=1.2, linestyle='--',
                alpha=0.6, label='Race day')
    # shade taper
    ax2.axvspan(91, 112, alpha=0.08, color='#ff6b35', label='Taper zone')
    ax2.set_xlabel('Day', color='#aaaaaa')
    ax2.set_ylabel('Load units', color='#aaaaaa')
    ax2.set_title('CTL / ATL / TSB  over training cycle', color='white', fontsize=11)
    ax2.legend(fontsize=8, facecolor='#2a2a2a', labelcolor='white',
               edgecolor='#444444', ncol=4)
    ax2.tick_params(colors='#555555')
    for sp in ax2.spines.values(): sp.set_edgecolor('#333333')

    # ── 3. Weekly volume ──────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.set_facecolor('#1a1a1a')
    bar_cols = ['#4a9eff' if n < 13 else '#ff6b35' for n in range(N_WEEKS)]
    ax3.bar(range(N_WEEKS), vols, color=bar_cols, edgecolor='#0f0f0f', lw=0.5)
    ax3.axvline(12.5, color='#ff6b35', lw=1.5, linestyle='--', alpha=0.6)
    ax3.set_xticks(range(N_WEEKS))
    ax3.set_xticklabels([f'W{n+1}' for n in range(N_WEEKS)],
                        color='#aaaaaa', fontsize=7, rotation=45)
    ax3.set_ylabel('km / week', color='#aaaaaa')
    ax3.set_title('Weekly Volume  (blue = build, orange = taper)',
                  color='white', fontsize=11)
    ax3.tick_params(colors='#555555')
    for sp in ax3.spines.values(): sp.set_edgecolor('#333333')

    # ── 4. SA convergence ────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.set_facecolor('#1a1a1a')
    iters = np.arange(len(history)) * 3000
    ax4.plot(iters, history, color='#4a9eff', lw=1.5)
    ax4.fill_between(iters, min(history), history, alpha=0.15, color='#4a9eff')
    ax4.set_xlabel('Iteration', color='#aaaaaa')
    ax4.set_ylabel('CTL on race day', color='#aaaaaa')
    ax4.set_title('SA Convergence', color='white', fontsize=11)
    ax4.tick_params(colors='#555555')
    for sp in ax4.spines.values(): sp.set_edgecolor('#333333')

    plt.suptitle(
        'Marathon Training Optimisation  ·  CTL/ATL Model + Simulated Annealing',
        color='white', fontsize=13, y=1.005, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Running Simulated Annealing (CTL/ATL model) …\n")
    vols_opt, history = simulated_annealing(seed=42)

    report(vols_opt)
    plot_results(vols_opt, history)