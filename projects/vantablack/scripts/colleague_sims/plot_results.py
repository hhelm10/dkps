"""Figures for the three simulation experiments. Reads results_E1.csv and results_E2.csv
(falls back to results_pilot.csv for E1 so the paper compiles before the full run)."""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, '..', 'figures')
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})

def load(name, fallback=None):
    p = os.path.join(HERE, name)
    if not os.path.exists(p) and fallback: p = os.path.join(HERE, fallback)
    if not os.path.exists(p): return None
    df = pd.read_csv(p)
    for c in ['drift', 'S2_size', 'correct', 'nom_auc']:
        if c not in df: df[c] = np.nan
    if 'grid' in df and 'exp' not in df: df['exp'] = 'E1'
    return df

def style(ax, xlabel, ylabel, title, ylim=(-0.02, 1.02)):
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, loc='left', fontsize=9.5, fontweight='bold')
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(length=0)
    if ylim: ax.set_ylim(*ylim)

def rate_curve(ax, sub, x, y, label, color, band=True):
    g = sub.groupby(x)[y].agg(['mean', 'size']).reset_index().sort_values(x)
    se = np.sqrt(g['mean'] * (1 - g['mean']) / g['size'])
    ax.plot(g[x], g['mean'], marker='o', ms=4, lw=1.6, color=color, label=label)
    if band: ax.fill_between(g[x], g['mean'] - se, g['mean'] + se, color=color, alpha=0.15, lw=0)

def mean_curve(ax, sub, x, y, label, color, ls='-'):
    g = sub.groupby(x)[y].agg(['mean', 'sem']).reset_index().sort_values(x)
    ax.plot(g[x], g['mean'], marker='o', ms=4, lw=1.6, color=color, label=label, ls=ls)
    ax.fill_between(g[x], g['mean'] - g['sem'], g['mean'] + g['sem'], color=color, alpha=0.15, lw=0)

# =============================== Experiment 1: detection (main: one row; appendix: remaining factors)
E1 = load('results_E1.csv', 'results_pilot.csv')
def factor_panel(ax, grid, col, title, lab, vals):
    sub = E1[(E1.grid == grid) | (E1.grid == 'A_delta')]
    for k, v in enumerate(vals):
        s = sub[np.isclose(sub[col].astype(float), v) & sub.delta.isin([0.0, 0.5, 1.0, 1.5])]
        if len(s): rate_curve(ax, s, 'delta', 'reject', f'{lab}={v:g}', C[k % len(C)], band=False)
    ax.axhline(0.05, color=MUTED, lw=0.9, ls=(0, (4, 3))); style(ax, r'$\delta$', 'rejection rate', title); ax.legend(fontsize=6, loc='lower right')
if E1 is not None:
    base = E1[E1.grid == 'A_delta']; null = E1[E1.delta == 0.0]
    fig, axes = plt.subplots(1, 4, figsize=(6.8, 2.1))
    ax = axes[0]
    rate_curve(ax, base, 'delta', 'reject', 'rejection', C[0])
    if base['correct'].notna().any(): rate_curve(ax, base, 'delta', 'correct', r'correct (Jaccard $\geq .5$)', C[1], band=False)
    ax.axhline(0.05, color=MUTED, lw=0.9, ls=(0, (4, 3))); style(ax, r'$\delta$', 'rate', '(a) base configuration'); ax.legend(fontsize=6.5, loc='lower right')
    factor_panel(axes[1], 'C_m', 'm', '(b) responses $m$', 'm', [3.0, 5.0, 10.0, 20.0, 40.0])
    factor_panel(axes[2], 'E_sigma_q', 'sigma_q', r'(c) query noise $\sigma_q$ (0 = black box)', r'$\sigma_q$', [0.0, 0.5, 1.0, 2.0, 4.0])
    ax = axes[3]; t1 = null.groupby('grid')['reject'].agg(['mean', 'size']).reset_index()
    ax.bar(range(len(t1)), t1['mean'], color=C[0], width=0.6)
    ax.errorbar(range(len(t1)), t1['mean'], yerr=np.sqrt(t1['mean'] * (1 - t1['mean']) / t1['size']), fmt='none', ecolor=INK2, lw=0.8)
    ax.axhline(0.05, color=MUTED, lw=0.9, ls=(0, (4, 3))); ax.set_xticks(range(len(t1))); ax.set_xticklabels([g.split('_')[0] for g in t1['grid']], fontsize=7)
    style(ax, 'null cells (grid)', 'Type I at 0.05', '(d) Type I error', ylim=(0, max(0.15, float(t1['mean'].max()) * 1.3)))
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'sim_detection.pdf')); plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(6.8, 2.1))
    factor_panel(axes[0], 'B_N', 'N', '(a) agents $N$', 'N', [20, 50, 100, 200])
    factor_panel(axes[1], 'D_Ssize', 'S_size', '(b) size $|S|$', '|S|', [2, 3, 5, 10, 15])
    factor_panel(axes[2], 'F_alpha', 'alpha', r'(c) unpairedness $\alpha$', r'$\alpha$', [0.1, 0.2, 1.0, 5.0])
    factor_panel(axes[3], 'G_drift', 'drift', '(d) common drift', 'sd', [0.0, 0.5, 1.0])
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'sim_detection_appendix.pdf')); plt.close(fig)
    print("E1 Type I overall: %.3f (n=%d)" % (null.reject.mean(), len(null)))

# =============================== Experiment 2: membership
E2 = load('results_E2.csv')
if E2 is not None:
    fig, axes = plt.subplots(1, 4, figsize=(6.8, 2.1))
    ax = axes[0]; sub = E2[E2.grid == 'H_Ssize']
    for k, v in enumerate([3, 5, 10, 15]):
        s = sub[sub.S_size == v]
        if len(s): mean_curve(ax, s, 'delta', 'jaccard', f'|S|={v}', C[k])
    style(ax, r'$\delta$', 'Jaccard', '(a) membership vs $|S|$'); ax.legend(fontsize=6.5, loc='lower right')
    ax = axes[1]; s = sub[sub.S_size == 5]
    for k, (y, lab) in enumerate([('precision', 'precision'), ('recall', 'recall'), ('nom_auc', 'nomination AUC')]):
        if len(s): mean_curve(ax, s, 'delta', y, lab, C[k])
    style(ax, r'$\delta$', '', '(b) precision, recall, ranking'); ax.legend(fontsize=6.5, loc='lower right')
    ax = axes[2]; sub = E2[E2.grid == 'I_two']
    for k, v in enumerate([3, 5]):
        s = sub[sub.S2_size == v]
        if len(s):
            mean_curve(ax, s, 'delta', 'jaccard', f'$S$, $|S_2|$={v}', C[k])
            mean_curve(ax, s, 'delta', 'S2_jaccard', f'$S_2$, $|S_2|$={v}', C[k], ls='--')
    style(ax, r'$\delta$', 'Jaccard', '(c) two subnetworks'); ax.legend(fontsize=6, loc='upper left')
    ax = axes[3]; sub = E2[E2.grid == 'J_scale']
    for k, (N, S) in enumerate([(50, 5), (100, 10), (200, 20)]):
        s = sub[(sub.N == N) & (sub.S_size == S)]
        if len(s): mean_curve(ax, s, 'delta', 'jaccard', f'N={N}, |S|={S}', C[k])
    style(ax, r'$\delta$', 'Jaccard', '(d) scale, $|S|=N/10$'); ax.legend(fontsize=6.5, loc='lower right')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'sim_membership.pdf')); plt.close(fig)

print("done")
