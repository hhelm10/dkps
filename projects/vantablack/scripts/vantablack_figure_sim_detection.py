"""Figure: detection simulations, 2x3 panels.
  a  calibration under H0: the empirical Pr[P <= p] of the p-values of all rho = 0 replicates
     pooled, against the diagonal (an exact test lies on it), with a pointwise 95% binomial band
  b-e  rejection rate vs swarm size, amount of content, subnetwork size, time; one colour per
     effect size rho, solid = the scan, thin dashed = the oracle that knows S (the known-S test on
     the same data and shuffle null); rho = 0 (gray) is the Type I error at alpha = 0.05.
  Legends: colour (dots) = pairwise agent correlation rho; line style (black) = estimation regime
  f  number of subnetworks K (60 colluders split into K groups): share of the K groups recovered by
     peeling (a reported subnetwork with Jaccard >= 0.5), solid; oracle: share of the groups whose
     known-S test rejects, dashed

    python vantablack_figure_sim_detection.py [--stage pilot]

Reads data/results/sim_detection_<stage>.csv (sim_detection.py); writes
artifacts/vantablack_figure_sim_detection[_<stage>].{png,pdf}.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pipeline.paths import RESULTS, ART, REPO
from pipeline.figstyle import apply_print_sizes, below_axes  # noqa: E402
sys.path.insert(0, os.path.join(REPO, 'scripts'))
import hv_style as hv  # noqa: E402

from sim_detection import BASE, SWEEPS, K_TOTAL  # noqa: E402

PANELS = [('N', 'b) Number of agents (N)', 'N'),
          ('m', 'c) Content per agent (m)', 'm'),
          ('S_size', 'd) Subnetwork size (|S|)', '|S|'),
          ('T', 'e) Timesteps (T)', 'T'),
          ('n_groups', 'f) Number of subnetworks (K)', 'K')]


def cell_rates(df, sweep, col):
    """Rows for one panel: the sweep's own cells plus the shared base cell (not for K)."""
    d = df[(df.sweep == sweep) | ((df.sweep == 'base') & (sweep != 'n_groups'))]
    if sweep == 'S_size':                                              # no null cells off the base: no Type I line
        d = d[d.rho > 0]
    d = d.assign(x=d[sweep]).dropna(subset=[col])
    g = d.groupby(['rho', 'x'])[col].agg(['mean', 'count']).reset_index()
    g['se'] = np.sqrt(g['mean'].clip(0, 1) * (1 - g['mean'].clip(0, 1)) / g['count'])
    return g


def centered_legends(fig, groups, y=0.0, gap=0.04, **kw):
    """Place titled legends side by side on one row, the row centred on the figure."""
    legs = [fig.legend(handles=h, title=t, loc='upper left', ncol=len(h), bbox_to_anchor=(0, y), **kw)
            for t, h in groups]
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    widths = [inv.transform(l.get_window_extent())[:, 0].ptp() for l in legs]
    x = 0.5 - (sum(widths) + gap * (len(legs) - 1)) / 2
    for l, w in zip(legs, widths):
        l.set_bbox_to_anchor((x, y), transform=fig.transFigure)
        x += w + gap
    return legs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='pilot')
    a = ap.parse_args()
    df = pd.read_csv(os.path.join(RESULTS, f'sim_detection_{a.stage}.csv'))
    hv.apply()
    apply_print_sizes(hv, 17)
    rhos = sorted(r for r in df.rho.unique() if r > 0)
    cols = {r: hv.CMAP_BLUE(v) for r, v in zip(rhos, np.linspace(0.35, 1.0, len(rhos)))}
    gray = hv.ROLES['baseline_gray']['color']
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.4))
    axes = axes.ravel()

    # a) calibration under H0
    ax = axes[0]
    p0 = np.sort(df.loc[df.rho == 0, 'p'].values)
    n0 = len(p0)
    grid = np.linspace(0, 1, 401)
    band = 1.96 * np.sqrt(grid * (1 - grid) / max(n0, 1))
    ax.fill_between(grid, np.clip(grid - band, 0, 1), np.clip(grid + band, 0, 1), color=hv.EDGE, alpha=0.5, lw=0)
    ax.plot([0, 1], [0, 1], color=hv.REFLINE, ls=':', lw=1.2)
    ax.step(np.r_[0, p0, 1], np.r_[0, np.arange(1, n0 + 1) / n0, 1], where='post', color=gray, lw=3.2)
    ax.set_xlim(0, 1); ax.set_ylim(-0.02, 1.02)
    ax.set_title('a) CDF of p-value under the null', loc='left', fontsize=hv.SIZES['subtitle'])
    ax.set_xlabel('p', fontsize=hv.SIZES['annot'])
    ax.set_ylabel('Pr[P ≤ p]', fontsize=hv.SIZES['annot'])

    # b-f
    for ax, (sweep, title, xlabel) in zip(axes[1:], PANELS):
        multi = sweep == 'n_groups'
        col, col_o = ('recovered', 'recovered_known') if multi else ('reject', 'reject_known')
        g, go = cell_rates(df, sweep, col), cell_rates(df, sweep, col_o)
        if not multi:
            h = g[g.rho == 0].sort_values('x')
            if len(h):
                ax.plot(h.x, h['mean'], marker='o', ms=4, color=gray, lw=3.0)
                ax.fill_between(h.x, h['mean'] - h.se, h['mean'] + h.se, color=gray, alpha=0.12, lw=0)
        for rho in rhos:
            h = g[g.rho == rho].sort_values('x')
            ho = go[go.rho == rho].sort_values('x')
            ax.plot(ho.x, ho['mean'], color=cols[rho], ls='--', lw=2.4, alpha=0.9)
            ax.plot(h.x, h['mean'], marker='o', ms=5, color=cols[rho], lw=3.4)
            ax.fill_between(h.x, h['mean'] - h.se, h['mean'] + h.se, color=cols[rho], alpha=0.15, lw=0)
        if not multi:
            ax.axhline(0.05, color=hv.REFLINE, ls=':', lw=1.2)
        ax.set_xscale('log')
        ax.set_xticks(SWEEPS[sweep]); ax.set_xticklabels([str(v) for v in SWEEPS[sweep]])
        ax.minorticks_off()
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(title, loc='left', fontsize=hv.SIZES['subtitle'])
        ax.set_xlabel(xlabel, fontsize=hv.SIZES['annot'])
        if sweep in ('N', 'S_size'):
            ax.set_ylabel('rejection rate at α = 0.05', fontsize=hv.SIZES['annot'])
        if multi:
            ax.set_ylabel('share of subnetworks recovered', fontsize=hv.SIZES['annot'])

    dot = lambda c, lab: Line2D([], [], color=c, ls='None', marker='o', ms=14, label=lab)
    h_rho = [dot(gray, 'ρ = 0 (Type I error)')] + [dot(cols[r], f'ρ = {r:g}') for r in rhos]
    h_reg = [Line2D([], [], color='black', lw=3.4, label='scan (S unknown)'),
             Line2D([], [], color='black', lw=2.4, ls='--', label='oracle (S known)')]
    kw = dict(fontsize=hv.SIZES['annot'], title_fontsize=hv.SIZES['annot'], frameon=False)
    fig.tight_layout()
    centered_legends(fig, [('pairwise agent correlation', h_rho), ('estimation regime', h_reg)], y=below_axes(fig), **kw)
    name = 'vantablack_figure_sim_detection' + ('' if a.stage == 'full' else f'_{a.stage}')
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(ART, f'{name}.{ext}'), dpi=200, bbox_inches='tight')
    print(f'wrote artifacts/{name}.png/.pdf')


if __name__ == '__main__':
    main()
