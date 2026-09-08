"""Render figures/q100_tables.md as the styled two-table exhibit
(figures/q100_tables.png). Parses the markdown so table edits flow through.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

INK, SURFACE, GRID = '#0A1638', '#f4f7fc', '#DFE6F2'
NAVY, AMBER = '#2E5CA6', '#D97706'
AMBER_BG, NAVY_BG = '#fdf3e4', '#e8eef8'

md = open('figures/q100_tables.md').read()


def parse(block):
    rows = []
    for line in block.strip().split('\n'):
        if line.startswith('|') and '---' not in line:
            rows.append([c.strip().strip('*') for c in line.strip('|').split('|')])
    return rows


t1, t2 = md.split('## Informative')
T1 = parse(t1.split('\n', 2)[2])
T2 = parse(t2.split('\n', 2)[2])


def draw(ax, rows, title):
    n_r, n_c = len(rows), len(rows[0])
    col_w = [3.0] + [1.55] * (n_c - 1)
    xs = [0]
    for w in col_w:
        xs.append(xs[-1] + w)
    ax.set_xlim(0, xs[-1]); ax.set_ylim(n_r, -0.8); ax.axis('off')
    ax.text(0, -0.45, title, fontsize=13, fontweight='bold', color=INK)
    for ri, row in enumerate(rows):
        is_head = ri == 0
        is_delta = 'paired' in row[0]
        is_best = (not is_delta) and 'blend' in row[0] and 'qubric' in row[0]
        bg = AMBER_BG if is_best else (NAVY_BG if is_delta else
                                       ('white' if ri % 2 else '#f0f4fa'))
        if not is_head:
            ax.add_patch(Rectangle((0, ri), xs[-1], 0.96, facecolor=bg,
                                   edgecolor='none'))
        for ci, cell in enumerate(row):
            x = xs[ci] + (0.06 if ci == 0 else col_w[ci] / 2)
            ha = 'left' if ci == 0 else 'center'
            if is_head:
                ax.text(x, ri + 0.5, cell, ha=ha, va='center', fontsize=10,
                        fontweight='bold', color=INK)
                continue
            if ci == 0:
                ax.text(x, ri + 0.48, cell, ha=ha, va='center', fontsize=9,
                        color=AMBER if is_best else (NAVY if is_delta else INK),
                        fontweight='bold' if is_best else 'normal')
            else:
                parts = cell.replace('−', '-').split(' [')
                main = parts[0]
                ciTxt = ('[' + parts[1]) if len(parts) > 1 else ''
                col = AMBER if is_best else (NAVY if is_delta else INK)
                ax.text(x, ri + (0.32 if ciTxt else 0.48), main, ha='center',
                        va='center', fontsize=10.5, color=col,
                        fontweight='bold' if is_best else 'normal')
                if ciTxt:
                    ax.text(x, ri + 0.72, ciTxt, ha='center', va='center',
                            fontsize=7, color='#5a6b8c')
        ax.plot([0, xs[-1]], [ri, ri], color=GRID, lw=0.7, zorder=3)
    ax.plot([0, xs[-1]], [n_r, n_r], color=GRID, lw=0.7)


h1, h2 = len(T1), len(T2)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(13.8, 1.35 * (h1 + h2) + 2.2),
                             gridspec_kw={'height_ratios': [h1 + 1, h2 + 1]})
fig.patch.set_facecolor(SURFACE)
for a in (a1, a2):
    a.set_facecolor(SURFACE)
draw(a1, T1, 'Table 1 — Random probes (B=20 draws) — MAE [95% CI], '
             'leave-one-LLM-out, 107 systems')
draw(a2, T2, 'Table 2 — Informative / adaptive probes (per-target panels)')
fig.text(0.05, 0.012,
         'Constant references: predict-the-mean 0.134, same-scaffold mean 0.124, '
         'same-model mean 0.119.   Adaptive rows use simulated CAT item paths.\n'
         'q100 panel; judge (Models 1+2) DeepSeek-V4-Flash-0731 on pruned full '
         'traces; embedder text-embedding-3-small.',
         fontsize=8.5, color='#4a5878')
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig('figures/q100_tables.png', dpi=200, facecolor=SURFACE)
print('wrote figures/q100_tables.png')
