"""Expected savings vs error: what evaluation accuracy does each dollar buy?

X: evaluation cost for a new system as % of the full 500-instance benchmark
   (bottom) and dollars under a nominal $2/agent-run (top; published per-run
   costs span ~$0.1-10, HAL accounting).
Y: expected MAE of the predicted resolve rate (95% CI band where available).

Curves: sample score, adaptive 2PL IRT, qubric + adaptive-IRT blend.
Anchors: metadata-only predictors at ~zero marginal cost; the full benchmark
at 100% cost / 0 error (definitional).

Reads figures/q100_adaptive_table.json when present (final numbers), else
figures/q100_adaptive_blend.json. Writes figures/fig_savings.png.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

INK, SURFACE, GRID = '#0A1638', '#f4f7fc', '#DFE6F2'
NAVY, AMBER, SLATE = '#2E5CA6', '#D97706', '#6D93D6'

MS = [1, 3, 5, 10, 20]
COST_RUN = 2.0                       # nominal $/agent-run for the top axis
FULL = 500

if os.path.exists('figures/q100_adaptive_table.json'):
    d = json.load(open('figures/q100_adaptive_table.json'))['qubric']
    irt = [d[str(m)]['irt']['mae'] for m in MS]
    irt_ci = [d[str(m)]['irt']['ci'] for m in MS]
    bl = [d[str(m)]['blend']['mae'] for m in MS]
    bl_ci = [d[str(m)]['blend']['ci'] for m in MS]
else:
    d = json.load(open('figures/q100_adaptive_blend.json'))
    irt = [d[str(m)]['adaptive_irt'] for m in MS]
    irt_ci = None
    bl = [d[str(m)]['blend'] for m in MS]
    bl_ci = None

# sample score on random draws (labels only, quick recompute)
import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
from outcome_baselines import load_panel  # noqa: E402
systems, q100, y, B, allowed = load_panel(panel='data/judge/q100.json')
rng = np.random.default_rng(0)
samp = []
for m in MS:
    es = [np.abs(B[:, rng.choice(B.shape[1], m, replace=False)].mean(1) - y).mean()
          for _ in range(20)]
    samp.append(float(np.mean(es)))

x = np.array(MS) / FULL * 100

fig, ax = plt.subplots(figsize=(8.6, 5.4))
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

ax.plot(x, samp, 'o--', color=INK, lw=1.6, ms=5, label='sample score')
ax.plot(x, irt, 'o-', color=SLATE, lw=2, ms=5, label='adaptive 2PL IRT')
ax.plot(x, bl, 'o-', color=AMBER, lw=2.4, ms=6,
        label='qubric + adaptive-IRT blend')
if bl_ci:
    ax.fill_between(x, [c[0] for c in bl_ci], [c[1] for c in bl_ci],
                    color=AMBER, alpha=0.15, lw=0)
    ax.fill_between(x, [c[0] for c in irt_ci], [c[1] for c in irt_ci],
                    color=SLATE, alpha=0.12, lw=0)

ax.axhline(0.119, color='#8a97b5', lw=1.1, ls=':')
ax.text(3.9, 0.121, 'best metadata-only predictor (zero marginal cost)',
        fontsize=8, color='#5a6b8c', ha='right')
ax.plot([100], [0.0], marker='*', ms=14, color=NAVY, clip_on=False)
ax.text(96, 0.008, 'full benchmark\n(500 runs)', fontsize=8, color=NAVY,
        ha='right')

ax.set_xscale('log')
ax.set_xticks([0.2, 0.6, 1, 2, 4, 100])
ax.set_xticklabels(['0.2%', '0.6%', '1%', '2%', '4%', '100%'])
ax.set_xlabel('evaluation cost (% of full-benchmark agent runs)', fontsize=10)
ax.set_ylabel('expected MAE of predicted resolve rate', fontsize=10)
ax.set_ylim(0, 0.24)

sec = ax.secondary_xaxis('top',
                         functions=(lambda p: p / 100 * FULL * COST_RUN,
                                    lambda c: c / (FULL * COST_RUN) * 100))
sec.set_xticks([2, 10, 40, 1000])
sec.set_xticklabels(['$2', '$10', '$40', '$1000'])
sec.set_xlabel(f'nominal dollars at \\${COST_RUN:.0f} per agent run '
               '(published range \\$0.10\u2013\\$10)', fontsize=8.5,
               color='#4a5878')

for s in ax.spines.values():
    s.set_color(GRID)
ax.grid(color=GRID, lw=0.8)
ax.tick_params(labelsize=9)
ax.legend(frameon=False, fontsize=9.5, loc='upper right')
ax.set_title('Savings vs error: 99%+ of evaluation cost is avoidable at '
             '~0.05 MAE', fontsize=11.5, color=INK, pad=12)
fig.tight_layout()
fig.savefig('figures/fig_savings.png', dpi=200, facecolor=SURFACE)
print('wrote figures/fig_savings.png')
