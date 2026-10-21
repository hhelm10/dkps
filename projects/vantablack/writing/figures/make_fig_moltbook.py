"""Moltbook figure: what the method finds on Moltbook (PKPS finds only; other channels elsewhere).

Run: real_tstep.py on Jan 30 - Feb 9, T = 8 activity-time steps, 2,574 agents active in at least
6 of 8 steps, B = 199; subnetworks = the peeling rounds with p <= 0.05 (8 rounds, 4-10 agents each).
Positions, increments and pair scores are recomputed from the run's saved conditioned responses
with the run's settings and cached (data/derived/real_tstep/<tag>_fig_cache.npz); the cache
reproduces the run's d, static and missing counts.
  a  activity of the cohort per hour, with the 8 activity-time steps
  b  positions over time, one small panel per subnetwork: members (coloured) and a sample of other
     agents (grey), each from its first observed position, projected on the subnetwork's net
     displacement and the leading direction of all agents' net displacements
  c  repeated co-movement: the pair majority statistic among the members of the 8 subnetworks and
     a random sample of other agents, ordered by subnetwork (boxes, numbered)
  d  scan and permutation null, every round: largest set score in the size band of the round's
     S_hat, observed vs the round's 199 per-step permutations (recomputed with the run's seeds and
     cached in <tag>_fig_nulls.npz; each round reproduces its saved p-value)
  e  membership: each subnetwork's members' scores against its S_hat (coloured) and the scores of
     the other agents tested in that round (grey)

Subnetworks are numbered by peeling round; no agent or operator names appear.

    python writing/figures/make_fig_moltbook.py
"""
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(PROJ, 'scripts'))
from pipeline.paths import REPO, DERIVED  # noqa: E402
from pipeline.figstyle import apply_print_sizes, below_axes  # noqa: E402
sys.path.insert(0, os.path.join(REPO, 'scripts'))
import hv_style as hv  # noqa: E402
from pipeline import tstep  # noqa: E402
from pipeline.ingest_moltnet import ORIGIN  # noqa: E402

warnings.filterwarnings('ignore')
TAG = '01-30_02-09_T8_ms6'
OUT = os.path.join(HERE, 'fig_moltbook')
D_DIR = os.path.join(DERIVED, 'real_tstep')
# categorical palette (dataviz reference order; adjacent pairs pass the CVD checks), one per round
PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
C0, NAVY = hv.ROLES['baseline_pale']['color'], hv.ROLES['anchor']['color']
rng = np.random.default_rng(0)

old = pickle.load(open(os.path.join(D_DIR, f'{TAG}.pkl'), 'rb'))
res, summ = old['res'], old['summary']
N, T = len(old['cohort']), summ['T']
cache = np.load(os.path.join(D_DIR, f'{TAG}_fig_cache.npz'))
Z, D, Q = cache['Z'], cache['D'], cache['Q']
rows = pd.read_parquet(os.path.join(D_DIR, f'{TAG}_rows.parquet'), columns=['agent_ix', 'step', 't'])
rounds = [g for g in res['groups'] if g['p'] <= 0.05 and g['subnetwork']]
K = len(rounds)
grp = np.zeros(N, int)
for k, g in enumerate(rounds, 1):
    grp[g['members']] = k
members = np.concatenate([np.sort(g['members']) for g in rounds])
others = rng.choice(np.flatnonzero(grp == 0), 45, replace=False)
order = np.r_[members, others]
obs_unit = np.bincount(rows.agent_ix.values * T + rows.step.values, minlength=N * T).reshape(N, T) > 0

hv.apply()
apply_print_sizes(hv, 20)
fig = plt.figure(figsize=(20, 9.2))
gs = gridspec.GridSpec(2, 6, figure=fig, hspace=0.36, wspace=0.75, left=0.04, right=0.99, top=0.95, bottom=0.15)
bot = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[1, :], width_ratios=[0.0001, 0.9, 1.0, 1.05, 0.0001], wspace=0.9)
TITLES = []                                                             # (axes, text); placed per row below
title = lambda ax, s: TITLES.append((ax, s))
lab = dict(fontsize=hv.SIZES['annot'])

# a) activity and time steps
ax = fig.add_subplot(gs[0, 0:3])
t = ORIGIN + pd.to_timedelta(rows.t.values, unit='h')
hrs = pd.Series(1, index=t).resample('1h').sum()
bounds = pd.to_datetime(summ['step_bounds'])
for s in range(T):
    if s % 2 == 0:
        ax.axvspan(bounds[s], bounds[s + 1], color=hv.WASH, lw=0, zorder=0)
ax.fill_between(hrs.index, hrs.values, step='post', color=NAVY, alpha=0.85, lw=0, zorder=2)
ax.set_xlim(bounds[0], bounds[-1]); ax.set_ylim(0, hrs.max() * 1.05)
mids = bounds[:-1] + (bounds[1:] - bounds[:-1]) / 2
ax.set_xticks(mids); ax.set_xticklabels([str(s + 1) for s in range(T)], fontsize=hv.SIZES['tick'])
ax.tick_params(axis='x', length=0); ax.set_xlabel('t', **lab); ax.grid(axis='x', visible=False)
top = ax.secondary_xaxis('top')
top.xaxis.set_major_locator(mdates.DayLocator(interval=2)); top.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
top.tick_params(labelsize=hv.SIZES['tick'])
ax.set_ylabel('actions per hour', **lab)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v / 1000:g}k' if v else '0'))
title(ax, 'a) Activity and time steps')

# b) positions over time, one small panel per subnetwork: its members (coloured) and a sample of
#    other agents (grey), centred per step and projected on the subnetwork's net displacement and
#    the leading direction of all agents' net displacements
Zc = Z - np.nanmean(np.where(obs_unit[:, :, None], Z, np.nan), 0, keepdims=True)
Zm = np.where(obs_unit[:, :, None], Zc, np.nan)
first = np.array([np.flatnonzero(obs_unit[i])[0] for i in range(N)])
last = np.array([np.flatnonzero(obs_unit[i])[-1] for i in range(N)])
netall = Zc[np.arange(N), last] - Zc[np.arange(N), first]
bg = rng.choice(np.flatnonzero(grp == 0), 80, replace=False)
sub_b = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[0, 3:6], wspace=0.08, hspace=0.32)
for k, g in enumerate(rounds):
    a_ = fig.add_subplot(sub_b[k // 4, k % 4])
    u = netall[g['members']].mean(0); e1 = u / np.linalg.norm(u)
    W = netall - (netall @ e1)[:, None] * e1
    e2 = np.linalg.svd(W - W.mean(0), full_matrices=False)[2][0]
    Y = Zm @ np.stack([e1, e2], 1)
    Y = Y - np.nanmean(Y[g['members'], first[g['members']]], 0)          # members start near the origin
    for i in bg:
        y = Y[i] - Y[i, first[i]]
        a_.plot(y[:, 0], y[:, 1], color=C0, lw=0.7, alpha=0.5, zorder=1)
    for i in g['members']:
        y = Y[i] - Y[i, first[i]]
        a_.plot(y[:, 0], y[:, 1], color=PAL[k], lw=1.5, alpha=0.9, zorder=2)
        a_.scatter(y[last[i], 0], y[last[i], 1], s=16, color=PAL[k], edgecolors='white', linewidths=0.4, zorder=3)
    a_.set_xticks([]); a_.set_yticks([]); a_.grid(False); a_.set_facecolor(hv.WASH)
    for sp in ('left', 'bottom'):
        a_.spines[sp].set_visible(False)
    a_.set_xlabel(str(k + 1), fontsize=hv.SIZES['tick'], labelpad=2)
    if k == 0:
        title(a_, 'b) Positions over time')

# c) repeated co-movement among the subnetworks' members and a sample of other agents
ax = fig.add_subplot(bot[0, 1])
sub = Q[:, order][:, :, order]
Zp = tstep.pair_scores(sub).astype(float)
Zp[~np.isfinite(Zp)] = np.nan
CM = matplotlib.colors.LinearSegmentedColormap.from_list('ink', ['#ffffff', hv.EDGE, hv.INK_MUTE, hv.INK_TITLE])
im = ax.imshow(Zp, cmap=CM, vmin=-3, vmax=3, interpolation='nearest')
s0 = 0
ticks = []
for k, g in enumerate(rounds):
    n0 = len(g['members'])
    ax.add_patch(plt.Rectangle((s0 - 0.5, s0 - 0.5), n0, n0, fill=False, ec=PAL[k], lw=2.6))
    ticks.append(s0 + (n0 - 1) / 2); s0 += n0
ax.set_xticks([])
ax.set_yticks([])
ax.tick_params(length=0); ax.grid(False)
ax.set_xlabel('agents', **lab); ax.set_ylabel('agents', **lab)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.outline.set_visible(False); cb.ax.tick_params(labelsize=hv.SIZES['tick'])
title(ax, 'c) Repeated co-movement')

# d) scan and permutation null for every round: the largest set score in the size band of the
#    round's S_hat, observed (coloured) against the round's 199 per-step permutations (grey)
ax = fig.add_subplot(bot[0, 2])
NL = np.load(os.path.join(D_DIR, f'{TAG}_fig_nulls.npz'))
for k, g in enumerate(rounds):
    b = g['band']
    null = NL[f'nulls{k}'][:, b]; null = null[np.isfinite(null)]
    vp = ax.violinplot(null, positions=[k], vert=False, widths=0.8, showextrema=False)
    for pc in vp['bodies']:
        pc.set_facecolor(C0); pc.set_edgecolor('none'); pc.set_alpha(0.8)
    ax.plot([NL[f'obs{k}'][b]] * 2, [k - 0.38, k + 0.38], color=PAL[k], lw=3.4, solid_capstyle='butt', zorder=3)
ax.set_yticks(range(K)); ax.set_yticklabels([str(k + 1) for k in range(K)], fontsize=hv.SIZES['tick'])
ax.set_ylim(K - 0.5, -0.5)
ax.set_ylabel('subnetwork', **lab); ax.set_xlabel('largest set score', **lab)
title(ax, 'd) Scan and permutation null')

# e) membership: per round, members' scores against S_hat vs the other agents tested in that round
ax = fig.add_subplot(bot[0, 3])
keep = np.arange(N)
for k, g in enumerate(rounds):
    Qr = Q if k == 0 else Q[:, keep][:, :, keep]
    loc = np.searchsorted(keep, g['members'])
    z = tstep.member_z(Qr, loc, combine='sum')[0]
    isin = np.zeros(len(keep), bool); isin[loc] = True
    zo = z[~isin & np.isfinite(z)]
    vp = ax.violinplot(zo, positions=[k], vert=False, widths=0.8, showextrema=False)
    for pc in vp['bodies']:
        pc.set_facecolor(C0); pc.set_edgecolor('none'); pc.set_alpha(0.8)
    ax.scatter(z[loc], k + rng.uniform(-0.18, 0.18, len(loc)), s=30, color=PAL[k], edgecolors='white', linewidths=0.6, zorder=3)
    keep = np.setdiff1d(keep, g['members'])
ax.set_yticks(range(K)); ax.set_yticklabels([str(k + 1) for k in range(K)], fontsize=hv.SIZES['tick'])
ax.set_ylim(K - 0.5, -0.5)
ax.set_ylabel('subnetwork', **lab); ax.set_xlabel('agent score against $\\hat S$', **lab)
title(ax, 'e) Membership')

# panel titles: one height per row, just above the tallest element of the row
fig.canvas.draw()
rend = fig.canvas.get_renderer()
inv = fig.transFigure.inverted()
for row in ('ab', 'cde'):
    items = [(ax, s) for ax, s in TITLES if s[0] in row]
    top = max(inv.transform(ax.get_tightbbox(rend).corners())[:, 1].max() for ax, _ in items)
    for ax, s in items:
        fig.text(ax.get_position().x0, top + 0.008, s, ha='left', va='bottom', fontsize=hv.SIZES['subtitle'], color=hv.INK_TITLE)

# legends
dot = lambda c, l: Line2D([], [], color=c, ls='None', marker='o', ms=14, label=l)
h_ag = [dot(PAL[k], str(k + 1)) for k in range(K)] + [dot(C0, 'other agents')]
h_st = [Line2D([], [], color=hv.INK_TITLE, lw=3.4, label='observed'), Patch(color=C0, label='permutation null')]
kw = dict(fontsize=hv.SIZES['annot'], title_fontsize=hv.SIZES['annot'], frameon=False, handlelength=1.4, handletextpad=0.4, columnspacing=0.9)
Y = below_axes(fig)
legs = [fig.legend(handles=hh, title=tt, loc='upper left', ncol=len(hh), bbox_to_anchor=(0, Y), **kw)
        for tt, hh in (('subnetwork', h_ag), ('statistic', h_st))]
fig.canvas.draw()
inv = fig.transFigure.inverted()
widths = [np.ptp(inv.transform(l.get_window_extent())[:, 0]) for l in legs]
x = 0.5 - (sum(widths) + 0.025) / 2
for l, wd in zip(legs, widths):
    l.set_bbox_to_anchor((x, Y), transform=fig.transFigure); x += wd + 0.025

for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}.{ext}', dpi=200, bbox_inches='tight')
print('saved', OUT)
