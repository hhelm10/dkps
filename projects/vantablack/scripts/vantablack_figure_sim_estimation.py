"""Figure: membership estimation simulations, 2x4 panels, columns as in the detection figure
(swarm size, amount of content, subnetwork size, time). Top row: precision of the estimated
subnetwork S_hat (share of its members that are colluders); bottom row: recall (share of the
colluders in S_hat). Colour (dots in the legend) = pairwise agent correlation rho > 0; line
style (black in the legend) = estimation regime: solid = the scan, thin dashed = the oracle that
knows S (the same membership rule scored against the true group;
sim_oracle_membership.py). All replicates count, rejected or not (precision is undefined, and
left out, when S_hat is empty).

    python vantablack_figure_sim_estimation.py [--stage pilot]

Reads data/results/sim_detection_<stage>.csv (sim_detection.py) and, if present,
data/results/sim_oracle_membership_<stage>.csv; writes
artifacts/vantablack_figure_sim_estimation[_<stage>].{png,pdf}.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pipeline.paths import RESULTS, ART, REPO
sys.path.insert(0, os.path.join(REPO, 'scripts'))
import hv_style as hv  # noqa: E402

from sim_detection import BASE, SWEEPS  # noqa: E402
PANELS = [('N', 'a) Number of agents (N)', 'N'),
          ('m', 'b) Amount of content per agent (m)', 'm'),
          ('S_size', 'c) Size of subnetwork (|S|)', '|S|'),
          ('T', 'd) Timesteps (T)', 'T')]

ROWS = [('precision', 'precision of $\\hat S$'), ('recall', 'recall of $\\hat S$')]


def cell_means(df, sweep, METRIC):
    d = df[(df.sweep == sweep) | (df.sweep == 'base')]
    d = d.assign(x=d[sweep]).dropna(subset=[METRIC])
    g = d.groupby(['rho', 'x'])[METRIC].agg(['mean', 'std', 'count']).reset_index()
    g['se'] = g['std'].fillna(0) / np.sqrt(g['count'])
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
    df = pd.read_csv(os.path.join(RESULTS, f'sim_detection_{a.stage}.csv'), dtype={'value': str})
    df = df[df.rho > 0]
    orc = os.path.join(RESULTS, f'sim_oracle_membership_{a.stage}.csv')
    if os.path.exists(orc):
        o = pd.read_csv(orc, dtype={'value': str})
        key = ['sweep', 'value', 'rho', 'rep']
        df = df.assign(value=df.value.fillna('')).merge(o.assign(value=o.value.fillna(''))[key + ['oracle_precision', 'oracle_recall']],
                                                         on=key, how='left')
    hv.apply()
    rhos = sorted(df.rho.unique())
    cols = {r: hv.CMAP_BLUE(v) for r, v in zip(rhos, np.linspace(0.35, 1.0, len(rhos)))}
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.4), sharey=True)
    for r, (metric, ylab) in enumerate(ROWS):
        for ax, (sweep, title, xlabel) in zip(axes[r], PANELS):
            g = cell_means(df, sweep, metric)
            go = cell_means(df, sweep, 'oracle_' + metric) if 'oracle_' + metric in df else None
            for rho in rhos:
                h = g[g.rho == rho].sort_values('x')
                if go is not None:
                    ho = go[go.rho == rho].sort_values('x')
                    ax.plot(ho.x, ho['mean'], color=cols[rho], ls='--', lw=2.4, alpha=0.9)
                ax.plot(h.x, h['mean'], marker='o', ms=5, color=cols[rho], lw=3.4, label=f'ρ = {rho:g}')
                ax.fill_between(h.x, h['mean'] - h.se, h['mean'] + h.se, color=cols[rho], alpha=0.15, lw=0)
            ax.set_xscale('log')
            ax.set_xticks(SWEEPS[sweep]); ax.set_xticklabels([str(v) for v in SWEEPS[sweep]])
            ax.minorticks_off()
            ax.set_ylim(-0.02, 1.02)
            if r == 0:
                ax.set_title(title, loc='left', fontsize=hv.SIZES['subtitle'])
            else:
                ax.set_xlabel(xlabel, fontsize=hv.SIZES['annot'])
        axes[r, 0].set_ylabel(ylab, fontsize=hv.SIZES['annot'])
    axes = axes[0]
    from matplotlib.lines import Line2D
    dot = lambda c, lab: Line2D([], [], color=c, ls='None', marker='o', ms=9, label=lab)
    h_rho = [dot(cols[r], f'ρ = {r:g}') for r in rhos]
    h_reg = [Line2D([], [], color='black', lw=3.4, label='scan (S unknown)'),
             Line2D([], [], color='black', lw=2.4, ls='--', label='oracle (S known)')]
    kw = dict(fontsize=hv.SIZES['annot'], title_fontsize=hv.SIZES['annot'], frameon=False)
    fig.tight_layout()
    centered_legends(fig, [('pairwise agent correlation', h_rho), ('estimation regime', h_reg)], y=0.0, **kw)
    name = 'vantablack_figure_sim_estimation' + ('' if a.stage == 'full' else f'_{a.stage}')
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(ART, f'{name}.{ext}'), dpi=200, bbox_inches='tight')
    print(f'wrote artifacts/{name}.png/.pdf')


if __name__ == '__main__':
    main()
