"""Figure: membership estimation simulations, 1x4 panels as in the detection figure
(swarm size, amount of content, subnetwork size, time). y is the chance-corrected
normalised mean reciprocal rank of the true members in the agent ranking (0 = random
ranking, 1 = members at ranks 1..|S|; sim_detection.rank_metrics), one line per effect
size rho > 0. All replicates count, rejected or not.

    python vantablack_figure_sim_estimation.py [--stage pilot]

Reads data/results/sim_detection_<stage>.csv (sim_detection.py); writes
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
from vantablack_figure_sim_detection import PANELS  # noqa: E402

METRIC = 'nmrr_cc'


def cell_means(df, sweep):
    d = df[(df.sweep == sweep) | (df.sweep == 'base')]
    d = d.assign(x=d[sweep]).dropna(subset=[METRIC])
    g = d.groupby(['rho', 'x'])[METRIC].agg(['mean', 'std', 'count']).reset_index()
    g['se'] = g['std'].fillna(0) / np.sqrt(g['count'])
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='pilot')
    a = ap.parse_args()
    df = pd.read_csv(os.path.join(RESULTS, f'sim_detection_{a.stage}.csv'))
    df = df[df.rho > 0]
    hv.apply()
    rhos = sorted(df.rho.unique())
    cols = {r: hv.CMAP_BLUE(v) for r, v in zip(rhos, np.linspace(0.35, 1.0, len(rhos)))}
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6), sharey=True)
    for ax, (sweep, title, xlabel) in zip(axes, PANELS):
        g = cell_means(df, sweep)
        for rho in rhos:
            h = g[g.rho == rho].sort_values('x')
            ax.plot(h.x, h['mean'], marker='o', ms=5, color=cols[rho], lw=2.4, label=f'ρ = {rho:g}')
            ax.fill_between(h.x, h['mean'] - h.se, h['mean'] + h.se, color=cols[rho], alpha=0.15, lw=0)
        ax.axhline(0, color=hv.REFLINE, ls=':', lw=1.2)
        ax.axvline(BASE[sweep], color=hv.EDGE, lw=6, zorder=0)
        ax.set_xscale('log')
        ax.set_xticks(SWEEPS[sweep]); ax.set_xticklabels([str(v) for v in SWEEPS[sweep]])
        ax.minorticks_off()
        ax.set_ylim(-0.05, 1.02)
        ax.set_title(title, loc='left', fontsize=hv.SIZES['subtitle'])
        ax.set_xlabel(xlabel, fontsize=hv.SIZES['annot'])
    axes[0].set_ylabel('normalised MRR (chance-corrected)', fontsize=hv.SIZES['annot'])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=len(l), fontsize=hv.SIZES['annot'], bbox_to_anchor=(0.5, -0.07))
    reps = int(df.groupby(['sweep', 'value', 'rho']).size().median())
    fig.text(0.995, 0.005, f'{a.stage}: ~{reps} replicates per point, ±1 SE; dotted = chance; shaded bar = base value',
             ha='right', va='bottom', fontsize=10, color=hv.INK_MUTE)
    fig.tight_layout()
    name = 'vantablack_figure_sim_estimation' + ('' if a.stage == 'full' else f'_{a.stage}')
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(ART, f'{name}.{ext}'), dpi=200, bbox_inches='tight')
    print(f'wrote artifacts/{name}.png/.pdf')


if __name__ == '__main__':
    main()
