"""Fig 1: three-panel methods schematic.

(a) anatomy of a real trace excerpt -- authorship signal vs content signal
(b) qubric -- re-observation through an instance-specific rubric
(c) pillar taxonomy -- the conditioning variables of P(trace | task, agent, seed)

Excerpt in (a) is a lightly trimmed sweagent gpt-4 trace
(20240402_sweagent_gpt4 / django__django-11099).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, SURFACE, GRID = '#0A1638', '#f4f7fc', '#DFE6F2'
NAVY, AMBER, SLATE = '#2E5CA6', '#D97706', '#6D93D6'
RED_BG, NAVY_BG, GRAY_BG = '#f6dfd8', '#dde7f5', '#eceff5'
RED_TX = '#8f1d1d'

# (text, class) -- class: 'h' harness/authorship, 'c' content, 'm' meta comment
LINES = [
    ('swe_main', 'h', 'harness header'),
    ('First, we should locate the files that contain the', 'c', ''),
    ('`ASCIIUsernameValidator` ... in `contrib.auth.validators`.', 'c', 'intent'),
    ('action: find_file "validators.py" contrib/auth', 'h', 'tool syntax'),
    ('Directory contrib/auth not found', 'c', 'finding'),
    ('First, we should locate the files that contain the ...', 'h', 'format echo'),
    ('```', 'h', ''),
    ('find_file "validators.py" contrib/auth', 'h', ''),
    ('```', 'h', ''),
    ('{"open_file": "n/a", "working_dir": "/django__django"}', 'h', 'state dump'),
    ('Found 3 matches for "validators.py" in /django__django:', 'c', 'finding'),
    ('/django__django/django/contrib/auth/validators.py', 'c', ''),
    ('action: edit 11:11', 'h', 'tool syntax'),
    ("regex = r'\\A[\\w.@+-]+\\Z'   # was r'^[\\w.@+-]+$'", 'c', 'edit'),
    ('OBSERVATION: Your proposed edit has been applied.', 'h', 'harness echo'),
]

fig = plt.figure(figsize=(13.2, 5.0))
fig.patch.set_facecolor(SURFACE)
gs = fig.add_gridspec(1, 3, width_ratios=[1.22, 1.18, 1.0],
                      left=0.015, right=0.985, top=0.86, bottom=0.05, wspace=0.10)

# ---------------------------------------------------------------- (a) anatomy
ax = fig.add_subplot(gs[0])
ax.set_facecolor(SURFACE); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
ax.text(0.0, 1.075, '(a)  What an agentic trace is made of', fontsize=12,
        fontweight='bold', color=INK, transform=ax.transAxes)

y0, dy = 0.955, 0.058
for i, (txt, cls, note) in enumerate(LINES):
    y = y0 - i * dy
    bg = {'h': RED_BG, 'c': NAVY_BG}.get(cls, GRAY_BG)
    ax.add_patch(FancyBboxPatch((0.015, y - 0.023), 0.80, 0.046,
                                boxstyle='round,pad=0.004', mutation_aspect=0.5,
                                facecolor=bg, edgecolor='none'))
    ax.text(0.025, y, txt[:58], fontsize=6.6, family='monospace', color=INK,
            va='center')
    if note:
        ax.text(0.835, y, note, fontsize=6.4, color=RED_TX if cls == 'h' else NAVY,
                va='center', style='italic')
ax.add_patch(FancyBboxPatch((0.015, 0.02), 0.345, 0.052, boxstyle='round,pad=0.004',
                            facecolor=RED_BG, edgecolor='none'))
ax.text(0.03, 0.046, 'authorship: who wrote it', fontsize=7.5, color=RED_TX,
        va='center', fontweight='bold')
ax.add_patch(FancyBboxPatch((0.40, 0.02), 0.33, 0.052, boxstyle='round,pad=0.004',
                            facecolor=NAVY_BG, edgecolor='none'))
ax.text(0.415, 0.046, 'content: what happened', fontsize=7.5, color=NAVY,
        va='center', fontweight='bold')

# ---------------------------------------------------------------- (b) qubric
ax = fig.add_subplot(gs[1])
ax.set_facecolor(SURFACE); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
ax.text(0.0, 1.075, '(b)  qubric: re-observe via a rubric',
        fontsize=12, fontweight='bold', color=INK, transform=ax.transAxes)


def box(ax, x, y, w, h, text, fc, ec, tc=INK, fs=7.6, bold=False, ls='-'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012',
                                facecolor=fc, edgecolor=ec, lw=1.1, linestyle=ls))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs,
            color=tc, fontweight='bold' if bold else 'normal')


def arrow(ax, x0, y0, x1, y1, color=INK, lw=1.4):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>',
                                 mutation_scale=11, color=color, lw=lw,
                                 shrinkA=2, shrinkB=2))


box(ax, 0.03, 0.86, 0.40, 0.10, 'problem statement\n(public, no labels)', 'white', GRID)
box(ax, 0.56, 0.86, 0.40, 0.10, 'trace\n(any harness format)', 'white', GRID)
box(ax, 0.03, 0.64, 0.40, 0.10, 'rubric writer LLM', AMBER, AMBER, 'white', bold=True)
arrow(ax, 0.23, 0.86, 0.23, 0.745)
box(ax, 0.03, 0.42, 0.40, 0.115,
    'instance rubric\nwhat understanding, localization,\nrepro, editing, verification,\ncompletion mean HERE', 'white', AMBER, fs=6.4)
arrow(ax, 0.23, 0.64, 0.23, 0.55)
box(ax, 0.42, 0.245, 0.42, 0.10, 'judge LLM: extract per role', AMBER, AMBER,
    'white', bold=True)
arrow(ax, 0.43, 0.47, 0.55, 0.35)          # rubric -> judge
arrow(ax, 0.76, 0.86, 0.68, 0.35)          # trace -> judge
box(ax, 0.05, 0.135, 0.55, 0.09, '6 short factual descriptions', 'white', GRID, fs=7.2)
arrow(ax, 0.55, 0.245, 0.42, 0.225)
box(ax, 0.665, 0.135, 0.315, 0.09, 'embed per section\n(any off-the-shelf embedder)',
    'white', GRID, fs=6.4)
arrow(ax, 0.60, 0.18, 0.665, 0.18)
box(ax, 0.03, 0.0, 0.93, 0.09,
    'consensus-center per instance (− cross-system median)  →  concat',
    NAVY, NAVY, 'white', fs=7.4, bold=True)
arrow(ax, 0.82, 0.135, 0.70, 0.09)

# ---------------------------------------------------------------- (c) pillars
ax = fig.add_subplot(gs[2])
ax.set_facecolor(SURFACE); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
ax.text(0.0, 1.075, '(c)  Pillars = conditioning variables', fontsize=12,
        fontweight='bold', color=INK, transform=ax.transAxes)

ax.text(0.5, 0.955, 'trace  ~  P( trace | task, agent, seed )', ha='center',
        fontsize=9.5, color=INK, family='monospace')

box(ax, 0.04, 0.70, 0.27, 0.14, 'TASK\nwhich problem', NAVY_BG, NAVY, fs=7.2)
box(ax, 0.37, 0.56, 0.32, 0.28, '', RED_BG, RED_TX)  # container
ax.text(0.53, 0.80, 'AGENT', ha='center', fontsize=7.6, color=RED_TX, fontweight='bold')
for i, lab in enumerate(('Harness', 'Model family', 'Identity')):
    box(ax, 0.395, 0.715 - i * 0.072, 0.27, 0.058, lab, 'white', RED_TX, RED_TX, fs=6.6)
box(ax, 0.75, 0.70, 0.21, 0.14, 'SEED\nrun-to-run', GRAY_BG, '#9aa7c4', fs=7.2)

box(ax, 0.30, 0.33, 0.40, 0.11, 'TRACE', 'white', INK, fs=8.6, bold=True)
arrow(ax, 0.175, 0.70, 0.40, 0.44)
arrow(ax, 0.53, 0.56, 0.51, 0.44)
arrow(ax, 0.855, 0.70, 0.62, 0.44)

ax.text(0.06, 0.230, 'faithful → keep', fontsize=8, color=NAVY, fontweight='bold')
ax.text(0.06, 0.155, 'Task, Behavior', fontsize=7.4, color=INK)
ax.text(0.42, 0.230, 'invariant → chance', fontsize=8, color=RED_TX,
        fontweight='bold')
ax.text(0.42, 0.155, 'Identity, Model family,\nHarness', fontsize=7.4, color=INK,
        va='top')
ax.text(0.80, 0.230, 'noise floor', fontsize=8, color='#5a6b8c', fontweight='bold')
ax.text(0.80, 0.155, 'Reliability', fontsize=7.4, color=INK)
ax.text(0.02, 0.02, 'Outcome is inferred from content, not measured in it.',
        fontsize=6.8, color='#4a5878', style='italic')

fig.savefig('figures/fig1_schematic.png', dpi=200, facecolor=SURFACE)
print('wrote figures/fig1_schematic.png')
