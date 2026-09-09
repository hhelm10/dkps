"""Fig 1: sensitivity radars across seven embedders (Helivan Blues).

Reads figures/radar_all_data_v2.json + figures/family_metric.json. Spokes are
oriented desirable-OUTSIDE: content spokes (Task, Behavior) plot normalized
fidelity; authorship spokes (Identity, Model Family, Harness) plot normalized
invariance, so chance sits at the rim for them (dashed ring shows chance on
every spoke). Identity uses the aggregated test to match the heatmap.

House style: raw trace embedding -> comparator, qubric -> focus; chance ring
in reference gray; white ground.

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

RAW = hv_style.ROLES['comparator']['color']
QUB = hv_style.ROLES['focus']['color']

d = json.load(open('figures/radar_all_data_v2.json'))
fam = json.load(open('figures/family_metric.json'))
CH = d['chance']
CH['family'] = fam['chance_family']

SPOKES = [('behavior', 'Behavior', 'keep'), ('task', 'Task', 'keep'),
          ('scaf', 'Harness', 'inv'), ('family', 'Model\nFamily', 'inv'),
          ('provagg', 'Identity', 'inv')]
BEH_MAX = 0.05          # behavior deltas are small; fixed scale across panels


def coords(vals):
    out = []
    for key, _, kind in SPOKES:
        v, ch = vals[key], CH[key]
        if key == 'behavior':
            r = np.clip(v / BEH_MAX, 0, 1)
        elif kind == 'keep':
            r = np.clip((v - ch) / (1 - ch), 0, 1)
        else:
            r = np.clip(1 - (v - ch) / (1 - ch), 0, 1)
        out.append(r)
    return out


def chance_coords():
    return [0 if kind == 'keep' else 1 for _, _, kind in SPOKES]


panels = list(d['panels'].items())
for name, p in panels:
    p['raw']['family'] = fam['panels'][name]['raw']
    p['qubric']['family'] = fam['panels'][name]['qubric']

ang = np.linspace(0, 2 * np.pi, len(SPOKES), endpoint=False)
fig = plt.figure(figsize=(16.0, 8.4))
pos = [(0.045 + i * 0.245, 0.50) for i in range(4)] + \
      [(0.17 + i * 0.245, 0.045) for i in range(3)]

for (name, p), (x0, y0) in zip(panels, pos):
    ax = fig.add_axes([x0, y0, 0.165, 0.36], projection='polar')
    ax.set_rorigin(-0.35)
    ax.set_ylim(0, 1.0)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    for series, color in (('raw', RAW), ('qubric', QUB)):
        r = coords(p[series])
        aa = np.concatenate([ang, ang[:1]])
        rr = np.array(r + r[:1])
        ax.plot(aa, rr, color=color, lw=2.6)
        ax.fill(aa, rr, color=color, alpha=0.16)
    cc = chance_coords()
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
                 color=hv_style.INK_TITLE, pad=14)

fig.text(0.5, 0.965,
         'Sensitivity profiles across embedding models --- rim = ideal: '
         'faithful to content (Task, Behavior), invariant to authorship '
         '(Identity, Model Family, Harness)',
         ha='center', fontsize=hv_style.SIZES['subtitle'],
         fontweight='bold', color=hv_style.INK_TITLE)
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([], [], color=RAW, lw=2.6,
                           label='raw trace embedding'),
                    Line2D([], [], color=QUB, lw=2.6, label='qubric'),
                    Line2D([], [], color=hv_style.REFLINE, lw=1.2, ls='--',
                           label='chance-level')],
           loc='center', bbox_to_anchor=(0.92, 0.22), frameon=False,
           fontsize=hv_style.SIZES['legend'], handlelength=3.2)
fig.savefig('figures/radar_all.png', dpi=200, facecolor='white',
            bbox_inches='tight', pad_inches=0.05)
print('wrote figures/radar_all.png')
