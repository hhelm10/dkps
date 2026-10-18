"""Recall curves for the cutoff-free ranking evaluation and a real-data
Figure 2 centred on one top-ranked real coordinated group.

    python plot_rank_eval.py
Reads data/derived/platform_jan31_feb17/rank_eval_curves.pkl (from rank_eval.py).
Writes artifacts/rank_recall.{png,pdf}, artifacts/figure2_real_coordinated_example.{png,pdf}
and writing/notes/real_coordinated_ranked.md (top disjoint real groups).
"""
import os
import pickle
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rank_eval import recall_curve
from pipeline.figure2 import draw

HERE = os.path.dirname(os.path.abspath(__file__))
from pipeline.paths import REPO, DERIVED, ART, NOTES  # noqa: E402
sys.path.insert(0, os.path.join(REPO, 'scripts'))
import hv_style as hv  # noqa: E402

D = os.path.join(DERIVED, 'platform_jan31_feb17')
REAL = 'res_platform_jan31_feb17_real_d100_coordinated.pkl'
METHODS = [('random', 'random order', 'baseline_pale'),
           ('intensity', 'content co-movement (mean cosine)', 'comparator'),
           ('coact', 'timing co-activity', 'slate'),
           ('either_rank', 'better of the two ranks', 'anchor')]


def curve_k(sc, labels_key, meth, K=100, n_rand=200):
    sets = [set(m) for m in sc['members']]
    pairs = set()
    for lab, m in sc['labels'].items():
        if lab.startswith(labels_key):
            pairs |= {(min(i, j), max(i, j)) for i in m for j in m if i < j}
    if meth == 'random':
        out = []
        for s in range(n_rand):
            order = np.random.default_rng(s).permutation(len(sets))[:K]
            _, rec = recall_curve(order, sets, pairs, 10 ** 9)
            out.append(np.pad(rec, (0, K - len(rec)), mode='edge'))
        return np.mean(out, 0)
    order = np.argsort(-sc[meth], kind='stable')[:K]
    _, rec = recall_curve(order, sets, pairs, 10 ** 9)
    return np.pad(rec, (0, K - len(rec)), mode='edge')


def main():
    hv.apply()
    c = pickle.load(open(os.path.join(D, 'rank_eval_curves.pkl'), 'rb'))['scores']
    K = 100
    x = np.arange(1, K + 1)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True)
    panels = [('ring', 'Planted ring, content shift 0.3', [k for k in c if '_a0.3_' in k]),
              ('ring', 'Planted ring, content shift 0.15', [k for k in c if '_a0.15_' in k]),
              ('siblings', 'Real sibling-name families', [REAL])]
    for ax, (lab, title, runs) in zip(axes, panels):
        for meth, name, role in METHODS:
            y = np.mean([curve_k(c[r], lab, meth, K) for r in runs], 0)
            st = hv.ROLES[role]
            ax.plot(x, y, color=st['color'], ls=st.get('ls', '-'), lw=2.4, label=name)
        ax.set_xscale('log')
        ax.set_title(title + (f' ({len(runs)} runs)' if len(runs) > 1 else ''),
                     fontsize=hv.SIZES['annot'] + 1)
        ax.set_xlabel('top-ranked groups inspected')
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel('known pairs recovered together')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, fontsize=hv.SIZES['annot'] - 2,
               bbox_to_anchor=(0.5, -0.1))
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(ART, f'rank_recall.{ext}'), bbox_inches='tight', dpi=200)
    plt.close(fig)

    # ranked disjoint real groups
    sc = c[REAL]
    res = pickle.load(open(os.path.join(D, REAL), 'rb'))['res']
    system = res['system']
    order = np.argsort(-sc['either_rank'], kind='stable')
    fam_of = {i: lab.split(':')[1] for lab, m in sc['labels'].items() for i in m}
    used, picked = set(), []
    for j in order:
        m = set(sc['members'][j])
        if used.isdisjoint(m):
            used |= m
            picked.append(j)
        if len(picked) >= 20:
            break
    lines = ['# Real coordinated groups, ranked (nothing planted)', '',
             'Groups of at most 100 agents from the whole-platform run, ordered by the better of '
             'their content co-movement rank and timing co-activity rank; overlapping lower-ranked '
             'groups are skipped. No cutoff: this is a ranked list for inspection. '
             'Agent names are public bot account names.', '',
             '| # | overall rank | candidate source | size | mean cosine | co-activity z | '
             'sibling family | members (first 8) |', '|---|---|---|---|---|---|---|---|']
    for n_, j in enumerate(picked, 1):
        m = sorted(sc['members'][j])
        fams = sorted(set(fam_of[i] for i in m if i in fam_of)) or ['']
        names = ', '.join(sorted(system[i] for i in m)[:8]) + (' …' if len(m) > 8 else '')
        lines.append(f"| {n_} | {list(order).index(j) + 1} | {sc['sources'][j]} | {len(m)} | "
                     f"{sc['intensity'][j]:.2f} | {sc['coact'][j]:.1f} | {', '.join(fams)} | {names} |")
    open(os.path.join(NOTES, 'real_coordinated_ranked.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    # Figure 2 centred on the highest-ranked group that is coordinated on both
    # signals (content cosine >= 0.5 and co-activity z >= 3)
    focus = next(j for j in picked if sc['intensity'][j] >= 0.5 and sc['coact'][j] >= 3)
    draw(res, os.path.join(ART, 'figure2_real_coordinated_example.png'),
         focus_idx=sc['members'][focus], other_sets=None,
         focus_label=f'focal group ({len(sc["members"][focus])} agents)')
    print('focal group:', sorted(system[i] for i in sc['members'][focus]))


if __name__ == '__main__':
    main()
