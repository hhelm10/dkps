"""Figure 1: linear-probe radars.

Spokes are held-out balanced accuracy of class-balanced linear ridge
probes (experiments/linear_probes), chance-normalized so desirable is
OUTSIDE: content spokes (Correctness, Task) plot (acc-chance)/(1-chance);
authorship spokes (Harness, Model Family, System Identity) plot the
complement. 7 embedder panels x {raw, generic, trubric}.

Reads project/trubric/data/radar_probes{,_gen}.json (probe-runner
output schema). Render-only.
"""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from trubric_common import DATA, ART, save_artifact, save_text_artifact  # noqa

import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import hv_style
hv_style.apply()

RAW = hv_style.ROLES['comparator']
GEN = hv_style.ROLES['generic']
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
    for f in ('project/trubric/data/radar_probes.json',
              'project/trubric/data/radar_probes_gen.json'):
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
    fig = plt.figure(figsize=(14.8, 6.8))
    for idx, (short, name) in enumerate(EMB):
        r_, c_ = divmod(idx, 4)
        x0 = .035 + c_ * .245 + (.1225 if r_ == 1 else 0)
        ax = fig.add_axes([x0, .56 - r_ * .52, .19, .37],
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
               label='raw trace'),
        Line2D([], [], color=GEN['color'], lw=2.6,
               label='generic'),
        Line2D([], [], color=QUB['color'], lw=3.2, label='trubric')],
        loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=3,
        frameon=False, fontsize=hv_style.SIZES['legend'],
        handlelength=3.2, columnspacing=2.0)
    save_artifact(fig, 'trubric_figure1_radar', dpi=200, pad=0.05,
                  facecolor='white')


if __name__ == '__main__':
    main()
