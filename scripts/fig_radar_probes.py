"""Radars on the collaborator's linear-probe metric (adopted per HH
2026-09-11): spokes are held-out balanced accuracy of a class-balanced
linear ridge probe (experiments/linear_probes), normalized
desirable-OUTSIDE. Content spokes (Task, Correctness) plot
(acc - chance)/(1 - chance); authorship spokes (Identity, Model Family,
Harness) plot 1 - (acc - chance)/(1 - chance), so chance sits at the rim.
Dashed ring = chance on every spoke. 7 embedder panels, raw vs qubric.

Reads figures/radar_probes.json (their runner's output schema).
Writes figures/radar_all.png.
"""
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
GEN = hv_style.ROLES['slate']
QUB = hv_style.ROLES['focus']

EMB = [('openai', 'text-embedding-3-small'),
       ('nomic', 'nomic-embed-text-v1.5'),
       ('bge', 'bge-large-en-v1.5'),
       ('gte', 'gte-large'),
       ('e5', 'e5-large-v2'),
       ('mpnet', 'all-mpnet-base-v2'),
       ('minilm', 'all-MiniLM-L6-v2')]
SPOKES = [('outcome', 'Correctness', 'keep'), ('task', 'Task', 'keep'),
          ('harness', 'Harness', 'inv'),
          ('vendor', 'Model\nFamily', 'inv'),
          ('system', 'System\nIdentity', 'inv')]


def main():
    table = {}
    for f in ('figures/radar_probes.json',
              'figures/radar_probes_gen.json'):
        try:
            d = json.load(open(f))
        except FileNotFoundError:
            continue
        for r in d['results']:
            table[(r['representation'], r['target'])] = (
                r['balanced_accuracy'], r['chance_balanced_accuracy'])

    def coords(rep):
        out = []
        for key, _, kind in SPOKES:
            acc, ch = table[(rep, key)]
            v = np.clip((acc - ch) / (1 - ch), 0, 1)
            out.append(float(v if kind == 'keep' else 1 - v))
        return out

    ang = np.linspace(0, 2 * np.pi, len(SPOKES), endpoint=False)
    fig = plt.figure(figsize=(14.8, 7.6))
    for idx, (short, name) in enumerate(EMB):
        r_, c_ = divmod(idx, 4)
        x0 = .035 + c_ * .245 + (.1225 if r_ == 1 else 0)
        ax = fig.add_axes([x0, .55 - r_ * .47, .19, .38],
                          projection='polar')
        ax.set_rorigin(-0.35)
        ax.set_ylim(0, 1.0)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        series_list = [(f'{short}_raw', RAW, 2.2),
                       (f'{short}_qubric', QUB, 3.2)]
        if (f'{short}_generic', 'system') in table:
            series_list.insert(1, (f'{short}_generic', GEN, 2.6))
        for series, st, lw in series_list:
            r = coords(series)
            aa = np.concatenate([ang, ang[:1]])
            rr = np.array(r + r[:1])
            ax.plot(aa, rr, color=st['color'], lw=lw)
            ax.fill(aa, rr, color=st['color'], alpha=0.15)
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
        Line2D([], [], color=RAW['color'], lw=2.2,
               label='raw trace embedding'),
        Line2D([], [], color=GEN['color'], lw=2.6,
               label='generic rubric'),
        Line2D([], [], color=QUB['color'], lw=3.2, label='qubric')],
        loc='lower center', bbox_to_anchor=(0.5, -0.01), ncol=3,
        frameon=False, fontsize=hv_style.SIZES['legend'],
        handlelength=3.2, columnspacing=2.0)
    fig.savefig('figures/radar_all.png', dpi=200, facecolor='white',
                bbox_inches='tight', pad_inches=0.05)
    print('wrote figures/radar_all.png (linear-probe metric)')


if __name__ == '__main__':
    main()
