"""Fig 2 (radars), finalized: sensitivity profiles across seven embedders
on the q100 era (DeepSeek-Flash extractions, full q100 panel).

Spokes oriented desirable-OUTSIDE: content spokes (Task, Behavior) plot
normalized fidelity; authorship spokes (Identity, Model Family, Harness)
plot normalized invariance (chance at the rim). Dashed ring = chance on
every spoke. Reads figures/pillars_q100_*.json; 2x4 grid, legend in the
last cell; no suptitle (caption text lives in the paper).

Writes figures/radar_all.png.
"""
import glob
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, 'scripts')
import hv_style
hv_style.apply()

RAW = hv_style.ROLES['comparator']
QUB = hv_style.ROLES['focus']
BEH_MAX = 0.05

ORDER = ['text-embedding-3-small', 'nomic-ai_nomic-embed-text-v1.5',
         'BAAI_bge-large-en-v1.5', 'thenlper_gte-large',
         'intfloat_e5-large-v2', 'sentence-transformers_all-mpnet-base-v2',
         'sentence-transformers_all-MiniLM-L6-v2']
NICE = {'text-embedding-3-small': 'text-embedding-3-small',
        'nomic-ai_nomic-embed-text-v1.5': 'nomic-embed-text-v1.5',
        'BAAI_bge-large-en-v1.5': 'bge-large-en-v1.5',
        'thenlper_gte-large': 'gte-large',
        'intfloat_e5-large-v2': 'e5-large-v2',
        'sentence-transformers_all-mpnet-base-v2': 'all-mpnet-base-v2',
        'sentence-transformers_all-MiniLM-L6-v2': 'all-MiniLM-L6-v2'}

SPOKES = [('behavior', 'Behavior', 'keep'), ('task', 'Task', 'keep'),
          ('harness', 'Harness', 'inv'),
          ('family', 'Model\nFamily', 'inv'),
          ('identity_agg', 'Identity', 'inv')]


def coords(rep, ch):
    out = []
    for key, _, kind in SPOKES:
        v = rep[key]
        c = {'task': ch['task'], 'identity_agg': ch['identity'],
             'family': ch['family'], 'harness': ch['harness'],
             'behavior': 0.0}[key]
        if key == 'behavior':
            r = np.clip(v / BEH_MAX, 0, 1)
        elif kind == 'keep':
            r = np.clip((v - c) / (1 - c), 0, 1)
        else:
            r = np.clip(1 - (v - c) / (1 - c), 0, 1)
        out.append(float(r))
    return out


def main():
    panels = []
    for tag in ORDER:
        f = f'figures/pillars_q100_{tag}.json'
        if glob.glob(f):
            panels.append((NICE[tag], json.load(open(f))))
    if not panels:
        raise SystemExit('no pillars_q100_*.json found')
    print(f'{len(panels)} panels')

    ang = np.linspace(0, 2 * np.pi, len(SPOKES), endpoint=False)
    fig = plt.figure(figsize=(14.8, 7.6))
    for idx, (name, d) in enumerate(panels):
        r_, c_ = divmod(idx, 4)
        ax = fig.add_axes([.035 + c_ * .245, .52 - r_ * .48, .19, .40],
                          projection='polar')
        ax.set_rorigin(-0.35)
        ax.set_ylim(0, 1.0)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        for series, st, lw in (('raw', RAW, 2.4), ('qubric', QUB, 3.2)):
            r = coords(d['reps'][series], d['chance'])
            aa = np.concatenate([ang, ang[:1]])
            rr = np.array(r + r[:1])
            ax.plot(aa, rr, color=st['color'], lw=lw)
            ax.fill(aa, rr, color=st['color'], alpha=0.15)
        cc = [0 if kind == 'keep' else 1 for _, _, kind in SPOKES]
        ax.plot(np.concatenate([ang, ang[:1]]), np.array(cc + cc[:1]), '--',
                color=hv_style.REFLINE, lw=1.2)
        ax.set_xticks(ang)
        ax.set_xticklabels([lab for _, lab, _ in SPOKES],
                           fontsize=hv_style.SIZES['annot'],
                           color=hv_style.INK)
        ax.set_yticks([])
        ax.grid(color=hv_style.GRID, lw=0.8)
        ax.spines['polar'].set_color(hv_style.SPINE)
        ax.set_title(name, fontsize=hv_style.SIZES['subtitle'],
                     color=hv_style.INK_TITLE, pad=16)

    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([], [], color=RAW['color'], lw=2.4,
               label='raw trace embedding'),
        Line2D([], [], color=QUB['color'], lw=3.2, label='qubric'),
        Line2D([], [], color=hv_style.REFLINE, lw=1.2, ls='--',
               label='chance-level')],
        loc='center', bbox_to_anchor=(0.86, 0.26), frameon=False,
        fontsize=hv_style.SIZES['legend'], handlelength=3.2)
    fig.savefig('figures/radar_all.png', dpi=200, facecolor='white',
                bbox_inches='tight', pad_inches=0.05)
    print('wrote figures/radar_all.png')


if __name__ == '__main__':
    main()
