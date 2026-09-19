"""Fig 3 right panel: extraction-judge x rubric-construction matrix.

Reads figures/judge_matrix.json (from scripts/judge_matrix.py). Cells colored
by desirability of the kNN leave-one-LLM-out MAE (navy = best, red = worst),
same ramp as the independence heatmap. Uncached judge/construction pairs are
hatched. Rubric writer is fixed at gpt-5.4-mini (the only cached writer);
filling that axis is queued with q150.
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

INK, SURFACE, GRID = '#0A1638', '#f4f7fc', '#DFE6F2'
DESIR = LinearSegmentedColormap.from_list('desir', [
    (0.00, '#8f1d1d'), (0.45, '#e08a75'), (0.70, '#f3cebe'),
    (0.85, '#f6f4ef'), (1.00, '#2E5CA6')])

CONSTRUCTIONS = [('blob', 'Free-form\nsummary'), ('generic', 'Generic\nrubric'),
                 ('verdict', 'Verdict\nquestions'), ('qubric', 'Qubric')]
JUDGES = [('gpt-oss-20b', 'gpt-oss-20b (open)'),
          ('gpt-4o-mini', 'gpt-4o-mini'),
          ('gpt-oss-120b', 'gpt-oss-120b (open)'),
          ('deepseek-v3.1', 'deepseek-v3.1 (open)'),
          ('gpt-5.4-nano', 'gpt-5.4-nano'),
          ('gpt-5.4-mini', 'gpt-5.4-mini')]

data = json.load(open('figures/judge_matrix.json'))
cells = data['cells']
maes = [v['knn_mae'] for v in cells.values()]
lo, hi = min(maes), max(maes)

fig, ax = plt.subplots(figsize=(7.2, 7.6))
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

for r, (jkey, jlab) in enumerate(JUDGES):
    for c, (ckey, clab) in enumerate(CONSTRUCTIONS):
        cell = cells.get(f'{ckey}|{jkey}')
        if cell is None:
            ax.add_patch(Rectangle((c, r), 0.96, 0.96, facecolor='#eef2f9',
                                   edgecolor=GRID, hatch='///', lw=0.8))
            ax.text(c + 0.48, r + 0.48, 'pending', ha='center', va='center',
                    fontsize=8.5, color='#9aa7c4', style='italic')
            continue
        d = (hi - cell['knn_mae']) / (hi - lo)
        color = DESIR(d)
        lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        tc = 'white' if lum < 0.55 else INK
        ax.add_patch(Rectangle((c, r), 0.96, 0.96, facecolor=color,
                               edgecolor=SURFACE, lw=2))
        ax.text(c + 0.48, r + 0.40, f"{cell['knn_mae']:.3f}", ha='center',
                va='center', fontsize=13, fontweight='bold', color=tc)
        ax.text(c + 0.48, r + 0.70, f"ridge {cell['ridge_mae']:.3f}",
                ha='center', va='center', fontsize=8, color=tc, alpha=0.85)

ax.set_xlim(-0.02, 4); ax.set_ylim(len(JUDGES), -0.02)
ax.set_xticks([c + 0.48 for c in range(4)])
ax.set_xticklabels([clab for _, clab in CONSTRUCTIONS], fontsize=10, color=INK)
ax.set_yticks([r + 0.48 for r in range(len(JUDGES))])
ax.set_yticklabels([jlab for _, jlab in JUDGES], fontsize=10, color=INK)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_ylabel('extraction judge', fontsize=10.5, color=INK)

fig.text(0.5, 0.965, 'Judge sensitivity: prediction error by construction and extractor',
         ha='center', fontsize=12.5, fontweight='bold', color=INK)
fig.text(0.5, 0.932,
         'Cells: leave-one-LLM-out kNN MAE (resolve rate), 107 systems x 20 instances; '
         'navy = lower error.',
         ha='center', fontsize=8.8, color='#4a5878')
fig.text(0.5, 0.045, 'Rubric writer fixed at gpt-5.4-mini (only cached writer); '
         'all cells embedded with nomic-embed-text-v1.5.',
         ha='center', fontsize=8.2, color='#4a5878')
fig.text(0.5, 0.018, 'Naive trace baseline (no judge): kNN 0.102.',
         ha='center', fontsize=8.2, color='#4a5878')
fig.subplots_adjust(top=0.895, bottom=0.13, left=0.22, right=0.97)
fig.savefig('figures/judge_matrix.png', dpi=200, facecolor=SURFACE)
print('wrote figures/judge_matrix.png')
