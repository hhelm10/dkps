"""Hero table (artifact 1, per HH: without the additional baselines).

Rows: raw-trace geometry, IRT (2PL), qubric geometry, qubric+IRT blend.
Columns: {SWE-bench Verified, Terminal-Bench 2.0} x {random, adaptive}
x m in {1, 5, 20}. Protocol: leave-one-family-out. Random = B=50 shared
draws; adaptive = per-target simulated CAT. Bold = best per column;
dagger on blend where the paired blend-IRT 95% CI excludes zero.

Reads figures/{q100,tb2}_protocols.json + {q100,tb2}_adaptive_family.json.
Writes figures/hero_table.md, notes/tables_hero.tex, figures/hero_table.png.
"""
import json
import sys

sys.path.insert(0, 'scripts')

MS = (1, 5, 20)
ROWS = [('sample', 'Sample Score'),
        ('raw', 'raw-trace geometry'),
        ('irt', 'IRT (2PL)'),
        ('geom', 'qubric geometry'),
        ('blend', 'qubric + IRT blend')]
SRC = {
    ('swe', 'random'): ('figures/q100_protocols.json', 'family'),
    ('swe', 'adaptive'): ('figures/q100_adaptive_family.json', None),
    ('tb2', 'random'): ('figures/tb2_protocols.json', 'family'),
    ('tb2', 'adaptive'): ('figures/tb2_adaptive_family.json', None),
}
BENCH = [('swe', 'SWE-bench Verified'), ('tb2', 'Terminal-Bench 2.0')]


def load_cell(bench, regime):
    path, proto = SRC[(bench, regime)]
    d = json.load(open(path))
    by_m = (d['protocols'][proto]['by_m'] if proto else d['by_m'])
    return by_m


def collect():
    """-> table[bench][regime][m] = {method: mae, 'star': bool}"""
    t = {}
    for b, _ in BENCH:
        t[b] = {}
        for reg in ('random', 'adaptive'):
            by_m = load_cell(b, reg)
            t[b][reg] = {}
            for m in MS:
                cell = by_m[str(m)]
                vals = {k: cell[k]['mae'] for k, _ in ROWS if k in cell}
                star = cell.get('delta_blend_irt', {}).get('ci',
                                                           [0, 1])[1] < 0
                t[b][reg][m] = (vals, star)
    return t


def fmt(v, best):
    s = f'{v:.3f}'
    return f'**{s}**' if best else s


def build_md(t):
    hdr = ['Method']
    for b, bl in BENCH:
        for reg in ('random', 'adaptive'):
            for m in MS:
                hdr.append(f'{bl.split()[0]} {reg[:4]}. m={m}')
    lines = ['| ' + ' | '.join(hdr) + ' |',
             '|' + '---|' * len(hdr)]
    for key, label in ROWS:
        row = [label]
        for b, _ in BENCH:
            for reg in ('random', 'adaptive'):
                for m in MS:
                    vals, star = t[b][reg][m]
                    best = min(vals.values())
                    s = fmt(vals[key], abs(vals[key] - best) < 5e-4)
                    if key == 'blend' and star:
                        s += '†'
                    row.append(s)
        lines.append('| ' + ' | '.join(row) + ' |')
    return '\n'.join(lines)


def build_tex(t):
    out = [r'\newcommand{\tablehero}{',
           r'\resizebox{\textwidth}{!}{%',
           r'\begin{tabular}{l' + 'S[table-format=1.3]' * 12 + '}',
           r'\toprule',
           r' & \multicolumn{6}{c}{\textbf{SWE-bench Verified}} & '
           r'\multicolumn{6}{c}{\textbf{Terminal-Bench 2.0}} \\',
           r'\cmidrule(lr){2-7} \cmidrule(lr){8-13}',
           r' & \multicolumn{3}{c}{random} & \multicolumn{3}{c}{adaptive}'
           r' & \multicolumn{3}{c}{random} & \multicolumn{3}{c}{adaptive} \\',
           r'\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10} '
           r'\cmidrule(lr){11-13}',
           r'\textit{Num.\ tasks $m$} & ' + ' & '.join(
               ['{%d}' % m for m in MS] * 4) + r' \\',
           r'\midrule']
    for key, label in ROWS:
        cells = []
        for b, _ in BENCH:
            for reg in ('random', 'adaptive'):
                for m in MS:
                    vals, star = t[b][reg][m]
                    best = min(vals.values())
                    v = vals[key]
                    s = f'{v:.3f}'
                    if abs(v - best) < 5e-4:
                        s = r'\bfseries ' + s
                    if key == 'blend' and star:
                        s += r'\rlap{$^\dagger$}'
                    cells.append(s)
        out.append(label.replace('+', '$+$') + ' & '
                   + ' & '.join(cells) + r' \\')
    out += [r'\bottomrule', r'\end{tabular}}', '}']
    return '\n'.join(out)


def build_png(t):
    import matplotlib
    matplotlib.use('Agg')
    import hv_style
    hv_style.apply()
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    n_data_rows = len(ROWS)
    fig, ax = plt.subplots(figsize=(13.6, 0.62 * (n_data_rows + 3) + 0.5))
    ax.axis('off')
    LAB_W = .195
    col_w = (1 - LAB_W) / 12
    xs = [LAB_W + col_w * j for j in range(13)]     # 12 data col edges
    n_rows = n_data_rows + 3                        # 2 header rows + m row
    row_h = 1 / (n_rows + 0.55)
    ys = [1 - row_h * r for r in range(n_rows + 1)]  # row top edges

    def cell(cx, cy, s, weight='normal', color=hv_style.INK, size=12,
             ha='center'):
        ax.text(cx, cy, s, weight=weight, color=color, fontsize=size,
                ha=ha, va='center', transform=ax.transAxes)

    def rect(x0, y0, w, h, fc):
        ax.add_patch(mpatches.Rectangle((x0, y0), w, h, fc=fc, ec='none',
                                        transform=ax.transAxes, zorder=0))

    # header wash
    rect(0, ys[3], 1, ys[0] - ys[3], hv_style.WASH)
    # zebra stripes on data rows
    for r in range(n_data_rows):
        if r % 2 == 1:
            rect(0, ys[4 + r], 1, row_h, '#f7f9fc')

    # header text
    cell((xs[0] + xs[6]) / 2, (ys[0] + ys[1]) / 2, 'SWE-bench Verified',
         'bold', hv_style.INK_TITLE, 13)
    cell((xs[6] + xs[12]) / 2, (ys[0] + ys[1]) / 2, 'Terminal-Bench 2.0',
         'bold', hv_style.INK_TITLE, 13)
    for j0, lab in ((0, 'random'), (3, 'adaptive'), (6, 'random'),
                    (9, 'adaptive')):
        cell((xs[j0] + xs[j0 + 3]) / 2, (ys[1] + ys[2]) / 2, lab,
             color=hv_style.INK_MUTE, size=12)
    cell(.008, (ys[2] + ys[3]) / 2, 'method  /  $m$ =', 'bold',
         hv_style.INK_MUTE, 11.5, ha='left')
    for j in range(12):
        cell((xs[j] + xs[j + 1]) / 2, (ys[2] + ys[3]) / 2, str(MS[j % 3]),
             color=hv_style.INK_MUTE, size=12)

    # data cells
    for r, (key, label) in enumerate(ROWS):
        cy = ys[3 + r] - row_h / 2
        is_anchor = key == 'blend'
        cell(.008, cy, label, 'bold' if is_anchor else 'normal',
             hv_style.ROLES['anchor']['color'] if is_anchor else hv_style.INK,
             12, ha='left')
        j = 0
        for b, _ in BENCH:
            for reg in ('random', 'adaptive'):
                for m in MS:
                    vals, star = t[b][reg][m]
                    best = min(vals.values())
                    v = vals[key]
                    isbest = abs(v - best) < 5e-4
                    s = f'{v:.3f}'
                    if key == 'blend' and star:
                        s += '†'
                    cell((xs[j] + xs[j + 1]) / 2, cy, s,
                         'bold' if isbest else 'normal',
                         hv_style.ROLES['anchor']['color'] if isbest
                         else hv_style.INK, 12)
                    j += 1

    # rules: horizontal
    def hline(yy, lw, color):
        ax.plot([0, 1], [yy, yy], color=color, lw=lw,
                transform=ax.transAxes, clip_on=False, zorder=3)
    hline(ys[0], 1.6, hv_style.INK_MUTE)
    hline(ys[1], .8, hv_style.EDGE)
    hline(ys[2], .8, hv_style.EDGE)
    hline(ys[3], 1.2, hv_style.INK_MUTE)
    for r in range(1, n_data_rows):
        hline(ys[3 + r], .7, hv_style.EDGE)
    hline(ys[3 + n_data_rows], 1.6, hv_style.INK_MUTE)
    # rules: vertical -- light between m columns, heavier between blocks
    for j in range(13):
        top = ys[1] if j % 3 == 0 else ys[2]
        major = j in (0, 6, 12)
        block = j % 3 == 0
        ax.plot([xs[j], xs[j]], [ys[3 + n_data_rows], top],
                color=hv_style.INK_MUTE if major else
                (hv_style.SPINE if block else hv_style.EDGE),
                lw=1.2 if major else (1.0 if block else .6),
                transform=ax.transAxes, clip_on=False, zorder=3)

    cell(.008, ys[3 + n_data_rows] - row_h * .55,
         'leave-one-family-out; random = 50 shared draws; adaptive = '
         'simulated CAT (Sample Score on CAT items is biased by design); '
         '† paired blend−IRT 95% CI excludes 0',
         color=hv_style.INK_MUTE, size=10, ha='left')
    fig.savefig('figures/hero_table.png', dpi=250, bbox_inches='tight',
                pad_inches=0.08)
    print('wrote figures/hero_table.png')


def main():
    t = collect()
    md = build_md(t)
    open('figures/hero_table.md', 'w').write(md + '\n')
    open('notes/tables_hero.tex', 'w').write(build_tex(t) + '\n')
    print(md)
    print('wrote figures/hero_table.md, notes/tables_hero.tex')
    build_png(t)


if __name__ == '__main__':
    main()
