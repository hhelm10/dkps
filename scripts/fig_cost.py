"""Artifact 6a: cost vs error, two panels (SWE-bench Verified,
Terminal-Bench 2.0), random and adaptive on the same axes.

X: dollars to evaluate one new system at a nominal $2/agent-run (HAL
aggregate $1.84/rollout; published range ~$0.13-4), log scale, with the
full-benchmark cost as a reference line. Y: MAE (+/- 1 SEM bands).
Curves: Sample Score (random), adaptive IRT (score-only frontier),
blend random (focus), blend adaptive (anchor). Protocol:
leave-one-family-out.

Reads figures/{q100,tb2}_protocols.json + {q100,tb2}_adaptive_family.json.
Writes figures/fig_cost.png.
"""
import json
import sys

import matplotlib
matplotlib.use('Agg')
import numpy as np

sys.path.insert(0, 'scripts')
import hv_style
hv_style.apply()
import matplotlib.pyplot as plt

COST_RUN = 2.0
MS = (1, 3, 5, 10, 20)
PANELS = [
    ('SWE-bench Verified', 'figures/q100_protocols.json',
     'figures/q100_adaptive_family.json', 500),
    ('Terminal-Bench 2.0', 'figures/tb2_protocols.json',
     'figures/tb2_adaptive_family.json', 89 * 5),
]


def series(by_m, key):
    mae = np.array([by_m[str(m)][key]['mae'] for m in MS])
    sem = np.array([by_m[str(m)][key].get(
        'sem', (by_m[str(m)][key]['ci'][1]
                - by_m[str(m)][key]['ci'][0]) / 3.92) for m in MS])
    return mae, sem


def main():
    SZ = hv_style.SIZES
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, (title, rand_f, adap_f, full_runs) in zip(axes, PANELS):
        rand = json.load(open(rand_f))['protocols']['family']['by_m']
        adap = json.load(open(adap_f))['by_m']
        cost = np.array(MS) * COST_RUN
        full_cost = full_runs * COST_RUN

        # in this figure line style encodes the probe regime:
        # dashed = random probes, solid = adaptive probes
        curves = [
            (rand, 'sample', 'Sample Score (random)', 'baseline_gray', '--'),
            (adap, 'sample', 'Sample Score (adaptive)', 'baseline_gray', '-'),
            (rand, 'irt', 'IRT (random)', 'comparator', '--'),
            (adap, 'irt', 'IRT (adaptive)', 'comparator', '-'),
            (rand, 'blend', 'qubric + IRT blend (random)', 'anchor', '--'),
            (adap, 'blend', 'qubric + IRT blend (adaptive)', 'anchor', '-'),
        ]
        for src, key, label, role, ls in curves:
            st = hv_style.ROLES[role]
            lw = 3.4 if role == 'anchor' and ls == '-' else 2.4
            mae, sem = series(src, key)
            ax.plot(cost, mae, color=st['color'], ls=ls, lw=lw,
                    marker='o', ms=4, label=label,
                    zorder=4 if role == 'anchor' else 3)
            ax.fill_between(cost, mae - sem, mae + sem, color=st['color'],
                            alpha=.13, lw=0, zorder=2)
        ax.axvline(full_cost, color=hv_style.REFLINE, ls='--', lw=1.2,
                   zorder=1)
        ax.text(full_cost * .88, .235,
                f'full benchmark\n\\${full_cost:,.0f}',
                ha='right', va='top', fontsize=SZ['annot'],
                color=hv_style.INK_MUTE)
        ax.set_xscale('log')
        ax.set_xticks([2, 10, 40, 200, 1000])
        ax.set_xticklabels(['\\$2', '\\$10', '\\$40', '\\$200', '\\$1,000'])
        ax.set_xlim(1.5, max(full_cost * 1.6, 1500))
        ax.set_ylim(0, .25)
        ax.set_yticks([0, .1, .2])
        ax.set_title(title, fontsize=SZ['label'], color=hv_style.INK_TITLE)
        ax.set_xlabel('evaluation cost per new system '
                      '(\\$2/agent-run)', fontsize=SZ['subtitle'])
        ax.tick_params(labelsize=SZ['tick'])
    axes[0].set_ylabel('MAE$(\\hat{y}, y)$', fontsize=SZ['label'])
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, fontsize=SZ['legend'] - 1,
               handlelength=3.2, loc='lower center',
               bbox_to_anchor=(0.5, -0.075), ncol=3, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig('figures/fig_cost.png', dpi=200, bbox_inches='tight',
                pad_inches=0.03)
    print('wrote figures/fig_cost.png')


if __name__ == '__main__':
    main()
