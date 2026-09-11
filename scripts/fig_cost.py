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
    fig, axes = plt.subplots(1, 4, figsize=(22, 4.9))
    for ax, (title, rand_f, adap_f, full_runs) in zip(axes[:2], PANELS):
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
    axes[1].set_yticklabels([])
    axes[1].set_ylim(0, .25)
    axes[0].set_ylim(0, .25)

    # panels 3-4: pairwise decision accuracy (leave-two-out shared pools)
    pw = json.load(open('figures/pairwise_cost.json'))
    PW_SERIES = [('sample', 'Sample Score', 'baseline_gray'),
                 ('irt', 'IRT (2PL)', 'comparator'),
                 ('geom', 'qubric geometry', 'focus'),
                 ('blend', 'qubric + IRT blend', 'anchor')]
    for ax, (bkey, (title, _, _, full_runs)) in zip(
            axes[2:], zip(('swe', 'tb2'), PANELS)):
        cost = np.array(MS) * COST_RUN
        for key, label, role in PW_SERIES:
            st = hv_style.ROLES[role]
            acc = [pw[bkey][str(m)]['gap05'][key] for m in MS]
            ax.plot(cost, acc, color=st['color'], ls='--',
                    lw=st.get('lw', 2.6), marker='o', ms=4, label=label,
                    zorder=4 if role == 'anchor' else 3)
        ax.axhline(0.5, color=hv_style.REFLINE, ls=':', lw=1.1, zorder=1)
        ax.text(cost[-1], .505, 'chance', ha='right', va='bottom',
                fontsize=SZ['annot'] - 1, color=hv_style.INK_MUTE)
        ax.set_xscale('log')
        ax.set_xticks([2, 10, 40])
        ax.set_xticklabels(['\\$2', '\\$10', '\\$40'])
        ax.set_ylim(.45, 1.0)
        ax.set_title(f'{title}\n(ranking, random probes)',
                     fontsize=SZ['subtitle'], color=hv_style.INK_TITLE)
        ax.set_xlabel('cost per system (\\$2/run)',
                      fontsize=SZ['subtitle'] - 1)
        ax.tick_params(labelsize=SZ['tick'])
    axes[2].set_ylabel('pairwise accuracy\n(true gap $\\geq 0.05$)',
                       fontsize=SZ['subtitle'])
    axes[3].set_yticklabels([])

    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = axes[2].get_legend_handles_labels()
    seen = {}
    for h, l in zip(h1 + h2, l1 + l2):
        if l not in seen:
            seen[l] = h
    fig.legend(seen.values(), seen.keys(), fontsize=SZ['legend'] - 2,
               handlelength=3.0, loc='lower center',
               bbox_to_anchor=(0.5, -0.09), ncol=5, columnspacing=1.1)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig('figures/fig_cost.png', dpi=200, bbox_inches='tight',
                pad_inches=0.03)
    print('wrote figures/fig_cost.png')


if __name__ == '__main__':
    main()
