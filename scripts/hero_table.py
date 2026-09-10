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
ROWS = [('raw', 'raw-trace geometry'),
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
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13.2, 2.9))
    ax.axis('off')
    ncol = 13
    x = [0.0, .20] + [.20 + .0665 * (i + 1) for i in range(12)]
    row_h = 1 / 7.4

    def cell(cx, cy, s, weight='normal', color=hv_style.INK, size=11.5,
             ha='center'):
        ax.text(cx, cy, s, weight=weight, color=color, fontsize=size,
                ha=ha, va='center', transform=ax.transAxes)

    y0 = 1 - row_h * .6
    cell((x[1] + x[4]) / 2 + .035, y0, 'SWE-bench Verified', 'bold',
         hv_style.INK_TITLE)
    cell((x[7] + x[10]) / 2 + .035, y0, 'Terminal-Bench 2.0', 'bold',
         hv_style.INK_TITLE)
    y1 = y0 - row_h
    for j, lab in ((1, 'random'), (4, 'adaptive'), (7, 'random'),
                   (10, 'adaptive')):
        cell((x[j + 1] + x[j + 3]) / 2, y1, lab, color=hv_style.INK_MUTE)
    y2 = y1 - row_h * .85
    cell(0.005, y2, 'method  /  m =', 'bold', hv_style.INK_MUTE, ha='left')
    for j in range(12):
        cell(x[j + 2] - .035, y2, str(MS[j % 3]), color=hv_style.INK_MUTE)
    ax.plot([0, 1], [y2 - row_h * .45] * 2, color=hv_style.SPINE, lw=1,
            transform=ax.transAxes, clip_on=False)
    for r, (key, label) in enumerate(ROWS):
        yy = y2 - row_h * (r + 1.1)
        is_anchor = key == 'blend'
        cell(0.005, yy, label, 'bold' if is_anchor else 'normal',
             hv_style.ROLES['anchor']['color'] if is_anchor else hv_style.INK,
             ha='left')
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
                    cell(x[j + 2] - .035, yy, s,
                         'bold' if isbest else 'normal',
                         hv_style.ROLES['anchor']['color'] if isbest
                         else hv_style.INK)
                    j += 1
    cell(0.005, y2 - row_h * 5.6,
         'leave-one-family-out; random = 50 shared draws; adaptive = '
         'simulated CAT; † paired blend−IRT 95% CI excludes 0',
         color=hv_style.INK_MUTE, size=9.5, ha='left')
    fig.savefig('figures/hero_table.png', dpi=250, bbox_inches='tight',
                pad_inches=0.06)
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
