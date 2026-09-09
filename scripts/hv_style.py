"""Helivan Blues house style (per HH style guide, 2026-09-09).

One blue, many lightnesses; line style encodes the paper's declared binary
attribute. This paper's role assignment (declare in Fig. 1 caption:
dashed = score-only, solid = uses trace embeddings):

  sample score        -> baseline_gray
  IRT (2PL)           -> baseline_pale
  raw-trace geometry  -> comparator
  qubric geometry     -> focus
  qubric + IRT blend  -> anchor
  (violet reserve unassigned so far)
"""
import matplotlib as mpl

INK_TITLE = '#0a2245'
INK = '#213c66'
INK_MUTE = '#486884'
SPINE = '#afbec6'
EDGE = '#d7dde6'
GRID = '#e5e7eb'
WASH = '#eef2f7'
REFLINE = '#94a3b8'

ROLES = {
    'baseline_gray': dict(color='#9ca3af', ls='--', lw=2.6),
    'baseline_pale': dict(color='#afbec6', ls='--', lw=2.6),
    'comparator':    dict(color='#93aacc', ls='-', lw=2.6),
    'slate':         dict(color='#486884', ls='--', lw=2.6),
    'focus':         dict(color='#3596ff', ls='-', lw=2.6),
    'anchor':        dict(color='#114471', ls='-', lw=3.6),
    'reserve':       dict(color='#6d5bd0'),
}

SIZES = dict(suptitle=17.5, panel_letter=16.2, label=15.6, subtitle=13.8,
             legend=13.8, tick=13.1, annot=11.5)

# sequential single-hue ramp in the brand blue (light -> navy)
CMAP_BLUE = mpl.colors.LinearSegmentedColormap.from_list(
    'hv_blues', ['#e8f0fb', '#93aacc', '#3596ff', '#114471'])
# the reserve concept's ramp (tint -> violet); this paper reserves violet
# for the benchmark score y
CMAP_RESERVE = mpl.colors.LinearSegmentedColormap.from_list(
    'hv_reserve', ['#e6e0f8', '#a795e3', '#6d5bd0', '#43307e'])

# Helivan diverging map (per HH): blue = negative / below center,
# helivan gold = positive / above center; center on zero (TwoSlopeNorm)
CMAP_DIV = mpl.colors.LinearSegmentedColormap.from_list('helivan_div', [
    '#19395e', '#065199', '#0c70cf', '#3894fc', '#7cb6fd', '#b5d6fe',
    '#f3f1ef',
    '#fecf99', '#fca72a', '#d08713', '#a3690d', '#774b06', '#4f3005',
])
try:
    mpl.colormaps.register(CMAP_DIV)
    mpl.colormaps.register(CMAP_DIV.reversed())  # 'helivan_div_r'
except ValueError:
    pass  # already registered in this process


def apply():
    mpl.rcParams.update({
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'axes.edgecolor': SPINE, 'axes.linewidth': 0.9,
        'axes.grid': True, 'axes.axisbelow': True,
        'grid.color': GRID, 'grid.linewidth': 0.8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.color': INK_MUTE, 'ytick.color': INK_MUTE,
        'axes.labelcolor': INK, 'axes.titlecolor': INK_TITLE,
        'text.color': INK, 'font.size': 11,
        'legend.frameon': False,
    })
