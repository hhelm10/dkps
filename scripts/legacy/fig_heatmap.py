"""Pillar heatmap: representations x properties.

Rebuilt from committed data (figures/radar_all_data_v2.json,
figures/family_metric.json) after the original inline script was lost.
Changes vs the first committed figure (per HH 2026-08-28):
  - Identity collapsed to the aggregated (10-trace split-half) test only --
    the harder, honest variant.
  - Stability column: pipeline test-retest self-retrieval P@1 (same trace,
    resampled rubric creation + extraction; chance 1/107). Deterministic
    rows are exactly 1.0; free-form retest pending.

Cell text = raw metric. Color = desirability: for keep-columns (Task,
Behavior, Stability) distance above chance toward the column max; for
invariance columns 1 - distance above chance. Blue = ideal.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

INK, SURFACE, GRID = '#0A1638', '#f4f7fc', '#DFE6F2'
DESIR = LinearSegmentedColormap.from_list('desir', [
    (0.00, '#8f1d1d'), (0.45, '#e08a75'), (0.70, '#f3cebe'),
    (0.85, '#f6f4ef'), (1.00, '#2E5CA6')])

d = json.load(open('figures/radar_all_data_v2.json'))
fam = json.load(open('figures/family_metric.json'))
chance = d['chance']
chance['family'] = fam['chance_family']

ROWS = ['raw', 'head-only', 'tail-only', 'centered', 'free-form judge', 'qubric']
oai = d['panels']['text-embedding-3-small']
vals = {'raw': dict(oai['raw']), 'qubric': dict(oai['qubric'])}
for r in ('head-only', 'tail-only', 'centered', 'free-form judge'):
    vals[r] = dict(d['extras'][r])
vals['raw']['family'] = fam['panels']['text-embedding-3-small']['raw']
vals['qubric']['family'] = fam['panels']['text-embedding-3-small']['qubric']
for r in ('head-only', 'tail-only', 'centered', 'free-form judge'):
    vals[r]['family'] = fam['extras'][r]

COLS = [('task', 'Task', 'keep'), ('behavior', 'Behavior', 'keep'),
        ('provagg', 'Identity', 'inv'), ('family', 'Model\nFamily', 'inv'),
        ('scaf', 'Harness', 'inv')]

stab_path = 'figures/stability_column.json'
if os.path.exists(stab_path):
    stab = json.load(open(stab_path))
    for r in ROWS:
        vals[r]['stability'] = stab['rows'].get(r)
    chance['stability'] = stab['chance']
    COLS.append(('stability', 'Stability', 'keep'))

nR, nC = len(ROWS), len(COLS)
fig, ax = plt.subplots(figsize=(1.9 * nC + 2.4, 0.98 * nR + 1.5))
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

for j, (key, lab, kind) in enumerate(COLS):
    col = [vals[r][key] for r in ROWS if vals[r][key] is not None]
    ch = chance[key]
    vmax = max(col)
    for i, r in enumerate(ROWS):
        v = vals[r][key]
        if v is None:
            ax.add_patch(Rectangle((j, i), 0.97, 0.97, facecolor='#eef2f9',
                                   edgecolor=GRID, hatch='///', lw=0.8))
            ax.text(j + 0.485, i + 0.485, 'pending', ha='center', va='center',
                    fontsize=9, color='#9aa7c4', style='italic')
            continue
        if kind == 'keep':
            desir = np.clip((v - ch) / max(vmax - ch, 1e-9), 0, 1)
        else:
            desir = np.clip(1 - (v - ch) / max(1 - ch, 1e-9), 0, 1)
        color = DESIR(desir)
        lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        ax.add_patch(Rectangle((j, i), 0.97, 0.97, facecolor=color,
                               edgecolor=SURFACE, lw=1.5))
        ax.text(j + 0.485, i + 0.485, f'{v:.2f}', ha='center', va='center',
                fontsize=12, color='white' if lum < 0.55 else INK)

ax.set_xlim(-0.02, nC); ax.set_ylim(nR, -0.02)
ax.set_xticks([j + 0.485 for j in range(nC)])
ax.set_xticklabels([lab for _, lab, _ in COLS], fontsize=11, color=INK)
ax.set_yticks([i + 0.485 for i in range(nR)])
ax.set_yticklabels(ROWS, fontsize=11, color=INK)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)

sm = plt.cm.ScalarMappable(cmap=DESIR)
cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
cb.set_ticks([])
cb.set_label('desirability (1 = ideal for this pillar)', fontsize=10, color=INK)
cb.outline.set_visible(False)

ax.set_title('Pillar desirability across representations '
             '(cell = raw metric; blue = ideal, red = undesirable)',
             fontsize=13, color=INK, pad=14)
fig.tight_layout()
fig.savefig('figures/independence_heatmap.png', dpi=200, facecolor=SURFACE)
print('wrote figures/independence_heatmap.png with', nC, 'columns')
