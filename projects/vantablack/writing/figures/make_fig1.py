"""Figure 1: the whole method on one small synthetic swarm, every panel computed by the pipeline.

Swarm: N = 50 agents, T = 6 steps, m = 20 responses per agent per step, two colluding subnetworks
of 12 agents each, each with its own group process (rho = 0.5, pipeline/simgen.py, seed 3). The
pipeline is pipeline/tstep.py as run in the simulations (B = 199 permutations). Subnetwork 1 is
the one found in the first peeling round.
  a  the agents and an illustrative communication network, which the observer does not see
     (small-world graph; the members of each subnetwork are linked to each other)
  b  positions over time (projected on the two subnetworks' net displacements)
  c  increments at four steps (projected on the two subnetworks' mean increments at that step)
  d  per-step pair scores q_ij^(t) at the same four steps (agents ordered by subnetwork)
  e  repeated co-movement: the pair majority statistic
  f  scan and permutation null, first round: largest set score in S_hat's size band, observed vs
     the per-step shuffle
  g  membership in peeling rounds 1 and 2 (round 3 stops): agents' scores against that round's S_hat (among the
     agents not yet set aside) and the fitted two-groups model

    python writing/figures/make_fig1.py
"""
import os
import sys
import warnings

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(PROJ, 'scripts'))
from pipeline.paths import REPO  # noqa: E402
from pipeline.figstyle import apply_print_sizes, below_axes  # noqa: E402
sys.path.insert(0, os.path.join(REPO, 'scripts'))
import hv_style as hv  # noqa: E402
from pipeline.simgen import SimConfig, generate  # noqa: E402
from pipeline import tstep  # noqa: E402

warnings.filterwarnings('ignore')
OUT = os.path.join(HERE, 'fig1_illustration')
SEED, B, STEPS = 3, 199, (0, 1, 2, 3)
C1, C2, C0 = hv.ROLES['focus']['color'], '#d08713', hv.ROLES['baseline_pale']['color']
NAVY = hv.ROLES['anchor']['color']

# ---- pipeline, step by step (as tstep.run with peel)
data = generate(SimConfig(N=50, T=6, m=20, S_size=12, n_groups=2, rho=0.5, seed=SEED))
N, T = data['cfg']['N'], data['cfg']['T']
G1, G2 = (np.sort(g) for g in data['groups'])
R, _ = tstep.condition(data['U_obs'], data['X'], k=64)
pos = tstep.positions(R, data['agent'], data['step'], N, T, seed=SEED)
D = tstep.increments(pos)
D, kf = tstep.remove_factors(D)
d_eff = D.shape[1] * D.shape[2] - kf
Q = tstep.step_scores(D)
res = tstep.test(D, d_eff, B=B, seed=SEED, Q=Q)
S_hat, b_hat, score = tstep.estimate_rank(Q, res)
res.update(S_hat=S_hat, band_hat=b_hat, rank_score=score)
rounds = tstep.peel_groups(D, d_eff, first=res, max_groups=5, B=B, seed=SEED, stat='rank')
if len(set(rounds[0]['members'].tolist()) & set(G2.tolist())) > len(set(rounds[0]['members'].tolist()) & set(G1.tolist())):
    G1, G2 = G2, G1                                                     # subnetwork 1 = found first
print('rounds', [(g['p'], len(g['members']), len(set(g['members'].tolist()) & set(G1.tolist())),
                  len(set(g['members'].tolist()) & set(G2.tolist()))) for g in rounds])

grp = np.zeros(N, int); grp[G1] = 1; grp[G2] = 2
col = np.array([C0, C1, C2])[grp]
rest = np.flatnonzero(grp == 0)
order = np.r_[G1, G2, rest]
draw = np.r_[rest, G1, G2]                                              # members on top


def basis(u, v):
    """Orthonormal 2-D basis: u, then v with its u component removed."""
    e1 = u / np.linalg.norm(u)
    v = v - (v @ e1) * e1
    return np.stack([e1, v / np.linalg.norm(v)], 1)


def boxes(a, lw):
    for s0, n0, c in ((0, len(G1), C1), (len(G1), len(G2), C2)):
        a.add_patch(plt.Rectangle((s0 - 0.5, s0 - 0.5), n0, n0, fill=False, ec=c, lw=lw))


hv.apply()
apply_print_sizes(hv, 20)
fig = plt.figure(figsize=(20, 8.8))
gs = gridspec.GridSpec(2, 4, figure=fig, width_ratios=[1.3, 0.9, 0.92, 0.92], height_ratios=[1.22, 0.85], hspace=0.36, wspace=0.42, left=0.03, right=0.99, top=0.95, bottom=0.08)
bot = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[1, :], width_ratios=[0.0001, 0.8, 1.0, 1.05, 0.0001], wspace=0.98)
TITLES = []                                                             # (axes, text); placed per row below
title = lambda ax, s: TITLES.append((ax, s))
lab = dict(fontsize=hv.SIZES['annot'])
CM = matplotlib.colors.LinearSegmentedColormap.from_list('ink', ['#ffffff', hv.EDGE, hv.INK_MUTE, hv.INK_TITLE])
vq = 3.0

# a) agents and an illustrative communication network
ax = fig.add_subplot(gs[0, 0])
G = nx.watts_strogatz_graph(N, 4, 0.2, seed=3)
G = nx.relabel_nodes(G, dict(enumerate(np.random.default_rng(5).permutation(N))))
for S in (G1, G2):
    G.add_edges_from((int(i), int(j)) for k, i in enumerate(S) for j in S[k + 1:]
                     if np.random.default_rng(int(i * N + j)).random() < 0.3)
    G.add_edges_from((int(S[k]), int(S[k + 1])) for k in range(len(S) - 1))
lay = nx.spring_layout(G, seed=11, k=0.18, iterations=200)
within = lambda u, v: grp[u] > 0 and grp[u] == grp[v]
nx.draw_networkx_edges(G, lay, edgelist=[e for e in G.edges if not within(*e)], ax=ax, edge_color=hv.EDGE, width=1.0)
for g, c in ((1, C1), (2, C2)):
    nx.draw_networkx_edges(G, lay, edgelist=[(u, v) for u, v in G.edges if within(u, v) and grp[u] == g], ax=ax,
                           edge_color=c, width=1.8, alpha=0.9)
nx.draw_networkx_nodes(G, lay, nodelist=list(rest), ax=ax, node_color=C0, node_size=34, edgecolors='white', linewidths=0.7)
for S, c in ((G1, C1), (G2, C2)):
    nx.draw_networkx_nodes(G, lay, nodelist=list(S), ax=ax, node_color=c, node_size=56, edgecolors='white', linewidths=0.7)
ax.set_axis_off(); ax.margins(0.04)
title(ax, 'a) Latent communication topology')

# b) positions over time, centred per step, projected on the subnetworks' net displacements
ax = fig.add_subplot(gs[0, 1])
Z = pos['Z'] - pos['Z'].mean(0, keepdims=True)
Y = Z @ basis(Z[G1, -1].mean(0) - Z[G1, 0].mean(0), Z[G2, -1].mean(0) - Z[G2, 0].mean(0))
for i in draw:
    m = grp[i] > 0
    ax.plot(Y[i, :, 0], Y[i, :, 1], color=col[i], lw=1.6 if m else 0.9, alpha=0.8 if m else 0.45, zorder=2 if m else 1)
    ax.scatter(Y[i, -1, 0], Y[i, -1, 1], s=26 if m else 16, color=col[i], edgecolors='white', linewidths=0.6,
               zorder=4 if m else 3)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel('projection 1', **lab); ax.set_ylabel('projection 2', **lab)
title(ax, 'b) Positions over time')

# c) increments at four steps
sub = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0, 2], wspace=0.06, hspace=0.32)
lim = 0
axs = []
for k, t in enumerate(STEPS):
    a = fig.add_subplot(sub[k // 2, k % 2]); axs.append(a)
    V = D[:, t] @ basis(D[G1, t].mean(0), D[G2, t].mean(0))
    for i in draw:
        m = grp[i] > 0
        a.annotate('', xy=V[i], xytext=(0, 0), arrowprops=dict(arrowstyle='-|>', color=col[i], lw=1.2 if m else 0.8,
                                                              alpha=0.95 if m else 0.6, mutation_scale=7))
    lim = max(lim, np.abs(V).max())
    a.set_xticks([]); a.set_yticks([]); a.grid(False)
    a.set_xlabel(f't = {t + 1}', **lab)
    for s in ('left', 'bottom'):
        a.spines[s].set_visible(False)
    a.set_facecolor(hv.WASH)
for a in axs:
    a.set_xlim(-1.08 * lim, 1.08 * lim); a.set_ylim(-1.08 * lim, 1.08 * lim); a.set_aspect('equal')
title(axs[0], 'c) Movements')

# d) per-step pair scores at the same steps
sub = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0, 3], wspace=0.06, hspace=0.32)
axs = []
for k, t in enumerate(STEPS):
    a = fig.add_subplot(sub[k // 2, k % 2]); axs.append(a)
    a.imshow(Q[t][np.ix_(order, order)].astype(float), cmap=CM, vmin=-vq, vmax=vq, interpolation='nearest')
    boxes(a, 3.2)
    a.set_xticks([]); a.set_yticks([]); a.grid(False)
    a.set_xlabel(f't = {t + 1}', **lab)
title(axs[0], 'd) Pair scores')

# e) repeated co-movement: pair majority statistic
ax = fig.add_subplot(bot[0, 1])
Zp = tstep.pair_scores(Q).astype(float)
Zp[~np.isfinite(Zp)] = np.nan
im = ax.imshow(Zp[np.ix_(order, order)], cmap=CM, vmin=-vq, vmax=vq, interpolation='nearest')
boxes(ax, 4.6)
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
ax.set_xlabel('agents', **lab); ax.set_ylabel('agents', **lab)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.outline.set_visible(False); cb.ax.tick_params(labelsize=hv.SIZES['tick'])
title(ax, 'e) Repeated co-movement')

# f) scan and permutation null (first round, S_hat's band)
ax = fig.add_subplot(bot[0, 2])
b = rounds[0]['band']
null = res['nulls'][:, b]
null = null[np.isfinite(null)]
obs = res['obs'][b]
bins = np.linspace(min(null.min(), obs) - 0.3, max(null.max(), obs) + 0.3, 36)
ax.hist(null, bins=bins, color=C0, edgecolor='white', lw=0.6)
ax.axvline(obs, color=NAVY, lw=3.4)
ax.set_xlabel('largest set score', **lab); ax.set_ylabel('permutations', **lab)
title(ax, 'f) Scan and permutation null')

# g) membership in the first two peeling rounds: each round scores the agents not yet set aside
#    against its S_hat and fits the two-groups model
sub = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=bot[0, 3], hspace=0.12)
keep = np.arange(N)
axs = []
for r in range(2):
    a = fig.add_subplot(sub[r, 0], sharex=axs[0] if axs else None, sharey=axs[0] if axs else None); axs.append(a)
    Qr = Q if r == 0 else tstep.step_scores(D[keep])
    loc = np.searchsorted(keep, rounds[r]['members'])
    z = tstep.member_z(Qr, loc, combine='sum')[0]
    gk = grp[keep]
    w = tstep.two_groups(z)
    pi = w.mean(); mu1 = (w * z).sum() / w.sum(); mu0 = ((1 - w) * z).sum() / (1 - w).sum()
    sd = np.sqrt((w * (z - mu1) ** 2 + (1 - w) * (z - mu0) ** 2).mean())
    bins = np.linspace(-3.6, 4.1, 32)
    a.hist([z[gk == 0], z[gk == 2], z[gk == 1]], bins=bins, stacked=True, color=[C0, C2, C1], edgecolor='white', lw=0.6)
    xx = np.linspace(bins[0], bins[-1], 400)
    h = (bins[1] - bins[0]) * len(z)
    a.plot(xx, h * (1 - pi) * norm.pdf(xx, mu0, sd), color='black', ls='--', lw=2.4)
    a.plot(xx, h * pi * norm.pdf(xx, mu1, sd), color='black', ls='--', lw=2.4)
    thr = xx[np.argmax(pi * norm.pdf(xx, mu1, sd) > (1 - pi) * norm.pdf(xx, mu0, sd))]
    a.axvline(thr, color=hv.REFLINE, ls=':', lw=1.2)
    a.set_ylabel(f'round {r + 1}', **lab)
    if r == 0:
        plt.setp(a.get_xticklabels(), visible=False)
    keep = np.setdiff1d(keep, rounds[r]['members'])
axs[1].set_xlabel('agent score against $\\hat S$', **lab)
title(axs[0], 'g) Membership')

# panel titles: one height per row, just above the tallest element of the row
fig.canvas.draw()
rend = fig.canvas.get_renderer()
inv = fig.transFigure.inverted()
for row in ('abcd', 'efg'):
    items = [(ax, s) for ax, s in TITLES if s[0] in row]
    top = max(inv.transform(ax.get_tightbbox(rend).corners())[:, 1].max() for ax, _ in items)
    for ax, s in items:
        fig.text(ax.get_position().x0, top + 0.008, s, ha='left', va='bottom', fontsize=hv.SIZES['subtitle'], color=hv.INK_TITLE)

# legends
dot = lambda c, l: Line2D([], [], color=c, ls='None', marker='o', ms=14, label=l)
h_ag = [dot(C1, 'subnetwork 1'), dot(C2, 'subnetwork 2'), dot(C0, 'other agents')]
h_st = [Line2D([], [], color=NAVY, lw=3.4, label='observed'),
        Patch(color=C0, label='permutation null'),
        Line2D([], [], color='black', lw=2.4, ls='--', label='two-groups model')]
kw = dict(fontsize=hv.SIZES['annot'], title_fontsize=hv.SIZES['annot'], frameon=False, handlelength=1.4, handletextpad=0.4, columnspacing=0.9)
Y = below_axes(fig)
legs = [fig.legend(handles=hh, title=tt, loc='upper left', ncol=len(hh), bbox_to_anchor=(0, Y), **kw)
        for tt, hh in (('agents', h_ag), ('statistic', h_st))]
fig.canvas.draw()
inv = fig.transFigure.inverted()
widths = [np.ptp(inv.transform(l.get_window_extent())[:, 0]) for l in legs]
x = 0.5 - (sum(widths) + 0.025) / 2
for l, wd in zip(legs, widths):
    l.set_bbox_to_anchor((x, Y), transform=fig.transFigure); x += wd + 0.025

for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}.{ext}', dpi=200, bbox_inches='tight')
print('saved', OUT)
