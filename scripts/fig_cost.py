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

        # line style = house method rule (dashed score-only, solid
        # embeddings); marker fill = probe regime (filled adaptive,
        # open random) -- consistent across all four panels
        curves = [
            (rand, 'sample', 'baseline_gray', False),
            (adap, 'sample', 'baseline_gray', True),
            (rand, 'irt', 'baseline_pale', False),
            (adap, 'irt', 'baseline_pale', True),
            (rand, 'blend', 'anchor', False),
            (adap, 'blend', 'anchor', True),
        ]
        for src, key, role, filled in curves:
            st = hv_style.ROLES[role]
            lw = 3.4 if role == 'anchor' and filled else 2.4
            mae, sem = series(src, key)
            ax.plot(mae, cost, color=st['color'], ls=st['ls'], lw=lw,
                    marker='o', ms=6.5,
                    markerfacecolor=st['color'] if filled else 'white',
                    markeredgecolor=st['color'], markeredgewidth=1.4,
                    zorder=4 if role == 'anchor' else 3)
            ax.fill_betweenx(cost, mae - sem, mae + sem,
                             color=st['color'], alpha=.13, lw=0, zorder=2)
        # full benchmark = exact scores (MAE 0) at full cost: a star,
        # not a reference line
        ax.plot([0], [full_cost], marker='*', ms=22, ls='none',
                markerfacecolor='#f59e0b',
                markeredgecolor=hv_style.INK_TITLE, markeredgewidth=1.1,
                zorder=5, clip_on=False)
        ax.text(.012, full_cost,
                f'full benchmark \\${full_cost:,.0f}',
                ha='left', va='center', fontsize=SZ['annot'],
                color=hv_style.INK_MUTE)
        ax.set_yscale('log')
        ax.set_yticks([2, 10, 40, 200, 1000])
        ax.set_yticklabels(['\\$2', '\\$10', '\\$40', '\\$200', '\\$1,000'])
        ax.set_ylim(1.5, max(full_cost * 1.6, 1500))
        ax.set_xlim(0, .25)
        ax.set_xticks([0, .1, .2])
        ax.set_title(title, fontsize=SZ['label'], color=hv_style.INK_TITLE)
        ax.set_xlabel('MAE$(\\hat{y}, y)$', fontsize=SZ['subtitle'])
        ax.tick_params(labelsize=SZ['tick'])
    axes[0].set_ylabel('evaluation cost per new system\n(\\$2/agent-run)',
                       fontsize=SZ['subtitle'])
    axes[1].set_yticklabels([])

    # panels 3-4: pairwise decision accuracy (leave-two-out shared pools)
    pw = json.load(open('figures/pairwise_cost.json'))
    PW_SERIES = [('sample', 'Sample Score', 'baseline_gray'),
                 ('irt', 'IRT (2PL)', 'baseline_pale'),
                 ('geom', 'qubric geometry', 'focus'),
                 ('blend', 'qubric + IRT blend', 'anchor')]
    for ax, (bkey, (title, _, _, full_runs)) in zip(
            axes[2:], zip(('swe', 'tb2'), PANELS)):
        cost = np.array(MS) * COST_RUN
        for key, label, role in PW_SERIES:
            st = hv_style.ROLES[role]
            acc = [pw[bkey][str(m)]['gap05'][key] for m in MS]
            ax.plot(acc, cost, color=st['color'], ls=st['ls'],
                    lw=st.get('lw', 2.6), marker='o', ms=6.5,
                    markerfacecolor='white',
                    markeredgecolor=st['color'], markeredgewidth=1.4,
                    label=label, zorder=4 if role == 'anchor' else 3)
        ax.axvline(0.5, color=hv_style.REFLINE, ls=':', lw=1.1, zorder=1)
        ax.text(.507, cost[-1], 'chance', ha='left', va='top', rotation=90,
                fontsize=SZ['annot'] - 1, color=hv_style.INK_MUTE)
        ax.set_yscale('log')
        ax.set_yticks([2, 10, 40])
        ax.set_yticklabels(['\\$2', '\\$10', '\\$40'])
        ax.set_ylim(1.6, 50)
        ax.set_xlim(.45, 1.0)
        ax.set_title(f'{title}\n(ranking, random probes)',
                     fontsize=SZ['subtitle'], color=hv_style.INK_TITLE)
        ax.set_xlabel('pairwise accuracy (true gap $\\geq 0.05$)',
                      fontsize=SZ['subtitle'] - 1)
        ax.tick_params(labelsize=SZ['tick'])
    axes[2].set_ylabel('cost per system (\\$2/run)',
                       fontsize=SZ['subtitle'])
    axes[3].set_yticklabels([])

    # split legends (orthogonal factors): method (color) + probe regime
    # (line style) under panels 1-2; ranking methods under panels 3-4
    from matplotlib.lines import Line2D
    ink = hv_style.INK
    meth = [Line2D([], [], color=hv_style.ROLES[r]['color'], lw=lw_,
                   ls=hv_style.ROLES[r]['ls'], label=lab)
            for r, lw_, lab in
            (('baseline_gray', 2.4, 'Sample Score'),
             ('baseline_pale', 2.4, 'IRT (2PL)'),
             ('anchor', 3.4, 'qubric + IRT blend'))]
    reg = [Line2D([], [], color=ink, lw=0, marker='o', ms=6.5,
                  markerfacecolor=ink, label='adaptive probes'),
           Line2D([], [], color=ink, lw=0, marker='o', ms=6.5,
                  markerfacecolor='white', markeredgecolor=ink,
                  markeredgewidth=1.4, label='random probes')]
    h2, l2 = axes[2].get_legend_handles_labels()
    fig.legend(handles=meth, fontsize=SZ['legend'] - 2, handlelength=3.0,
               loc='upper center', bbox_to_anchor=(0.26, 0.035), ncol=3,
               columnspacing=1.1, frameon=False)
    fig.legend(handles=reg, fontsize=SZ['legend'] - 2, handlelength=3.0,
               loc='upper center', bbox_to_anchor=(0.26, -0.035), ncol=2,
               columnspacing=1.1, frameon=False)
    fig.legend(h2, l2, fontsize=SZ['legend'] - 2, handlelength=3.0,
               loc='upper center', bbox_to_anchor=(0.76, 0.035), ncol=2,
               columnspacing=1.1, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig('figures/fig_cost.png', dpi=200, bbox_inches='tight',
                pad_inches=0.03)
    print('wrote figures/fig_cost.png')


if __name__ == '__main__':
    main()
