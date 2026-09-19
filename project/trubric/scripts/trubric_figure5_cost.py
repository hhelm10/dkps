"""Figure 5: cost-to-achieve, 1 x 4.

Panels 1-2: evaluation cost per new system (log y, $2/agent-run nominal)
vs MAE for {sample, IRT, trubric+IRT blend}; marker fill = probe regime;
gold star = full benchmark (exact scores at full cost). Panels 3-4:
cost vs leave-two-out pairwise accuracy (true gap >= 0.05) for {sample,
IRT, trubric geometry, blend}, random probes.

Render-only: reads the protocol/adaptive/pairwise caches in
project/trubric/data/.
"""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from trubric_common import DATA, ART, save_artifact, save_text_artifact  # noqa

import json
import sys

import matplotlib
matplotlib.use('Agg')
import numpy as np

import hv_style
hv_style.apply()
import matplotlib.pyplot as plt

COST_RUN = 2.0
MS = (1, 3, 5, 10, 20)
PANELS = [
    ('SWE-bench Verified', 'project/trubric/data/q100_protocols.json',
     'project/trubric/data/q100_adaptive_family.json', 500),
    ('Terminal-Bench 2.0', 'project/trubric/data/tb2_protocols.json',
     'project/trubric/data/tb2_adaptive_family.json', 89 * 5),
]


def series(by_m, key):
    mae = np.array([by_m[str(m)][key]['mae'] for m in MS])
    sem = np.array([by_m[str(m)][key].get(
        'sem', (by_m[str(m)][key]['ci'][1]
                - by_m[str(m)][key]['ci'][0]) / 3.92) for m in MS])
    return mae, sem


def main():
    SZ = hv_style.SIZES
    fig, axes = plt.subplots(1, 4, figsize=(14.2, 3.7))
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
        ax.text(.014, full_cost,
                f'full benchmark\n(~\\${full_cost:,.0f})',
                ha='left', va='center', fontsize=SZ['annot'],
                color=hv_style.INK_MUTE)
        ax.set_yscale('log')
        ax.set_yticks([2, 10, 40, 200, 1000])
        ax.set_yticklabels(['\\$2', '\\$10', '\\$40', '\\$200', '\\$1,000'])
        ax.set_ylim(1.5, max(full_cost * 1.6, 1500))
        ax.set_xlim(-0.014, .25)
        ax.set_xticks([0, .1, .2])
        ax.set_title(title, fontsize=SZ['label'], color=hv_style.INK_TITLE)
        ax.set_xlabel('MAE$(\\hat{y}, y)$', fontsize=SZ['subtitle'])
        ax.tick_params(labelsize=SZ['tick'])
    axes[0].set_ylabel('cost per new system\n(\\$2/agent-run)',
                       fontsize=SZ['subtitle'])
    axes[1].set_yticklabels([])

    # panels 3-4: pairwise decision accuracy (leave-two-out shared pools)
    pw = json.load(open('project/trubric/data/pairwise_cost.json'))
    PW_SERIES = [('sample', 'Sample Score', 'baseline_gray'),
                 ('irt', 'IRT (2PL)', 'baseline_pale'),
                 ('geom', 'trubric geometry', 'focus'),
                 ('blend', 'trubric + IRT blend', 'anchor')]
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
        ax.set_title(title, fontsize=SZ['label'],
                     color=hv_style.INK_TITLE)
        ax.set_xlabel('pairwise accuracy',
                      fontsize=SZ['subtitle'] - 1)
        ax.tick_params(labelsize=SZ['tick'])
    axes[2].set_ylabel('cost per system (\\$2/run)',
                       fontsize=SZ['subtitle'])
    axes[3].set_yticklabels([])

    # one legend for the whole figure: 4 methods + the 2 marker fills
    from matplotlib.lines import Line2D
    ink = hv_style.INK
    handles = [Line2D([], [], color=hv_style.ROLES[r]['color'], lw=lw_,
                      ls=hv_style.ROLES[r]['ls'], label=lab)
               for r, lw_, lab in
               (('baseline_gray', 2.4, 'Sample Score'),
                ('baseline_pale', 2.4, 'IRT (2PL)'),
                ('focus', 2.6, 'trubric geometry'),
                ('anchor', 3.4, 'trubric + IRT blend'))]
    handles += [Line2D([], [], color=ink, lw=0, marker='o', ms=6.5,
                       markerfacecolor=ink, label='adaptive probes'),
                Line2D([], [], color=ink, lw=0, marker='o', ms=6.5,
                       markerfacecolor='white', markeredgecolor=ink,
                       markeredgewidth=1.4, label='random probes')]
    fig.legend(handles=handles, fontsize=SZ['legend'] - 2,
               handlelength=2.6, loc='upper center',
               bbox_to_anchor=(0.5, 0.02), ncol=6,
               columnspacing=1.0, frameon=False)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save_artifact(fig, 'trubric_figure5_cost', dpi=200, pad=0.03)


if __name__ == '__main__':
    main()
