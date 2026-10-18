"""Figure 1 for the paper, in the vantablack setting.
Latent system as in the one-pager (seed 7). Observation model: each agent acts only at irregular
times (heterogeneous rates; S acts in coordinated bursts after onset); each action yields one noisy
embedding; the agent's position is estimated from its actions in a trailing window; the topology
is latent (only co-activity is observable).
(a) agents and latent topology (solid = inferable from co-activity, dashed = not)
(b) estimated perspective space at t0 (disc = estimate uncertainty; hollow = no recent action)
(c) estimated perspective space at t > t0, with estimate trajectories
(d) centered displacements of the estimates, S highlighted, scan's selected set outlined
"""
import os, numpy as np, networkx as nx
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig1_illustration')
rng = np.random.default_rng(7)
C_B, C_A = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})

# ---- latent system
N, T, T_STAR = 20, 100, 40
S = np.array([3, 7, 11, 14, 18]); inS = np.zeros(N, bool); inS[S] = True; ben = np.where(~inS)[0]
psi = np.zeros((T, N, 2)); psi[0] = rng.normal(0, 1.0, (N, 2))
psi[0, S] = np.array([[-1.6, 0.4], [0.2, -1.5], [1.3, 0.9], [-0.4, 1.4], [0.9, -0.3]])
drift = np.array([0.006, -0.003]); vdir = np.array([1.0, 0.55]) / np.hypot(1.0, 0.55)
for t in range(1, T):
    step = drift + rng.normal(0, 0.045, (N, 2))
    if t >= T_STAR:
        r = min(1.0, (t - T_STAR) / 30); cen = psi[t - 1, S].mean(0)
        step[S] += 0.10 * r * (cen - psi[t - 1, S]) + 0.075 * r * vdir
    psi[t] = psi[t - 1] + step
G = nx.watts_strogatz_graph(N, 4, 0.25, seed=3)
for i in S:
    for j in S:
        if i < j: G.add_edge(i, j)
pos = nx.spring_layout(G, seed=11, k=0.55)

# ---- vantablack observation model
orng = np.random.default_rng(11)
rate = orng.uniform(0.08, 0.38, N); ACT_SD = 0.38; WIN = 8
act = orng.random((T, N)) < rate[None, :]
burst = (np.arange(T) >= T_STAR) & (orng.random(T) < 0.30); act[np.ix_(burst, S)] = True
acts = [np.where(act[:, n])[0] for n in range(N)]
emb = [psi[tt, n] + orng.normal(0, ACT_SD, (len(tt), 2)) for n, tt in enumerate(acts)]
est = np.full((T, N, 2), np.nan); nobs = np.zeros((T, N), int)
for n in range(N):
    for t in range(T):
        m = (acts[n] > t - WIN) & (acts[n] <= t); k = m.sum(); nobs[t, n] = k
        if k: est[t, n] = emb[n][m].mean(0)
t0, t1 = 20, 90

fig = plt.figure(figsize=(6.8, 1.95))
gs = gridspec.GridSpec(1, 6, figure=fig, wspace=0.08, width_ratios=[0.9, 0.08, 1, 0.9, 0.12, 1.0],
                       left=0.035, right=0.995, top=0.86, bottom=0.2)
ax_a = fig.add_subplot(gs[0, 0]); ax_b = fig.add_subplot(gs[0, 2]); ax_c = fig.add_subplot(gs[0, 3], sharey=ax_b); ax_d = fig.add_subplot(gs[0, 5])
def title(ax, s): ax.set_title(s, loc='left', fontsize=8.5, fontweight='bold', pad=3)

# (a) latent topology; an edge is inferable if the pair acted on the same step >= 7 times
seen = [(u, v) for u, v in G.edges if len(np.intersect1d(acts[u], acts[v])) >= 7]
unseen = [e for e in G.edges if e not in seen]
sS = [(u, v) for u, v in seen if inS[u] and inS[v]]; sO = [(u, v) for u, v in seen if not (inS[u] and inS[v])]
nx.draw_networkx_edges(G, pos, edgelist=unseen, ax=ax_a, edge_color=GRID, width=0.7, style=(0, (2, 3)), alpha=0.8)
nx.draw_networkx_edges(G, pos, edgelist=sO, ax=ax_a, edge_color=AXIS, width=1.0)
nx.draw_networkx_edges(G, pos, edgelist=sS, ax=ax_a, edge_color=C_A, width=1.6, alpha=0.9)
nx.draw_networkx_nodes(G, pos, nodelist=list(ben), ax=ax_a, node_color=C_B, node_size=38, edgecolors=SURF, linewidths=0.8)
nx.draw_networkx_nodes(G, pos, nodelist=list(S), ax=ax_a, node_color=C_A, node_size=52, edgecolors=SURF, linewidths=0.8)
ax_a.set_axis_off(); ax_a.margins(0.05); title(ax_a, '(a) agents, latent topology')
ax_a.legend(handles=[Line2D([], [], marker='o', color=C_A, ls='', ms=5, label='$S$'),
                     Line2D([], [], marker='o', color=C_B, ls='', ms=5, label='others'),
                     Line2D([], [], color=AXIS, lw=1.0, label='inferable edge'),
                     Line2D([], [], color=GRID, lw=0.7, ls=(0, (2, 3)), label='unobserved')],
            loc='lower left', frameon=False, fontsize=6, handletextpad=0.3, borderaxespad=0.0, labelspacing=0.2)

# (b), (c) estimated positions with uncertainty discs
allp = np.concatenate([psi[t0], psi[t1]])
xl = (allp[:, 0].min() - 0.45, allp[:, 0].max() + 0.45); yl = (allp[:, 1].min() - 0.45, allp[:, 1].max() + 0.45)
def ps(ax, t, ttl, trails=False):
    if trails:
        for n in S: e = est[t0:t + 1, n]; ax.plot(e[:, 0], e[:, 1], color=C_A, lw=0.7, alpha=0.35, zorder=1, drawstyle='steps-post')
        for n in ben: e = est[t0:t + 1, n]; ax.plot(e[:, 0], e[:, 1], color=C_B, lw=0.5, alpha=0.10, zorder=1, drawstyle='steps-post')
    for n in range(N):
        col = C_A if inS[n] else C_B; k = nobs[t, n]; sz = 30 if inS[n] else 22
        if k == 0:
            prev = est[:t + 1, n]; ok = ~np.isnan(prev[:, 0])
            if ok.any(): p = prev[ok][-1]; ax.scatter(p[0], p[1], s=sz, facecolors='none', edgecolors=col, linewidths=0.9, zorder=3)
            continue
        p = est[t, n]; ax.add_patch(Circle(p, 1.3 * ACT_SD / np.sqrt(k), facecolor=col, alpha=0.13, edgecolor='none', zorder=2))
        ax.scatter(p[0], p[1], s=sz, color=col, edgecolors=SURF, linewidths=0.6, zorder=4 if inS[n] else 3)
    ax.set_xlim(*xl); ax.set_ylim(*yl); ax.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True); ax.tick_params(length=0, labelsize=7)
    ax.set_xlabel(r'$\hat\psi$ dim. 1', labelpad=1); title(ax, ttl)
ps(ax_b, t0, r'(b) estimates $\hat\psi$, $t_0$'); ax_b.set_ylabel(r'$\hat\psi$ dim. 2', labelpad=1)
ps(ax_c, t1, r'(c) estimates $\hat\psi$, $t>t_0$', trails=True); plt.setp(ax_c.get_yticklabels(), visible=False)

# (d) centered displacements of the estimates (agents with no estimate at either time are omitted)
have = ~np.isnan(est[t0, :, 0]) & ~np.isnan(est[t1, :, 0])
D = est[t1] - est[t0]; D = D - np.nanmedian(D[have], axis=0)
ax_d.axhline(0, color=AXIS, lw=0.7); ax_d.axvline(0, color=AXIS, lw=0.7)
for n in ben:
    if have[n]: ax_d.annotate('', xy=D[n], xytext=(0, 0), arrowprops=dict(arrowstyle='-', color=C_B, lw=0.6, alpha=0.35))
for n in S:
    if have[n]: ax_d.annotate('', xy=D[n], xytext=(0, 0), arrowprops=dict(arrowstyle='-', color=C_A, lw=0.8, alpha=0.6))
bh = [n for n in ben if have[n]]; Sh = [n for n in S if have[n]]
ax_d.scatter(D[bh, 0], D[bh, 1], s=22, color=C_B, edgecolors=SURF, linewidths=0.6, zorder=3)
ax_d.scatter(D[Sh, 0], D[Sh, 1], s=30, color=C_A, edgecolors=SURF, linewidths=0.6, zorder=4)
mu = D[Sh].mean(0); cov = np.cov(D[Sh].T) + 1e-6 * np.eye(2); w, V = np.linalg.eigh(cov); ang = np.degrees(np.arctan2(V[1, -1], V[0, -1]))
ax_d.add_patch(Ellipse(mu, 2 * 2.0 * np.sqrt(w[-1]), 2 * 2.0 * np.sqrt(w[0]), angle=ang, fill=False, ls=(0, (3, 2)), lw=1.0, ec=INK2, zorder=5))
ax_d.text(0.03, 0.97, r'dashed: $\hat S$', transform=ax_d.transAxes, ha='left', va='top', fontsize=7, color=INK2)
lim = np.nanmax(np.abs(D[have])) * 1.25; ax_d.set_xlim(-lim, lim); ax_d.set_ylim(-lim, lim)
ax_d.grid(True, color=GRID, lw=0.5); ax_d.set_axisbelow(True); ax_d.tick_params(length=0, labelsize=7)
ax_d.set_xlabel(r'$\tilde\Delta$ dim. 1', labelpad=1); ax_d.set_ylabel(r'$\tilde\Delta$ dim. 2', labelpad=1)
title(ax_d, r'(d) displacements $\tilde\Delta_i$, scan')

fig.savefig(OUT + '.pdf'); fig.savefig(OUT + '.png', dpi=200); print('saved', OUT)
