"""Figure 2 of the one-pager, from pipeline output (synthetic or real).

(a) observed interaction graph (post-burn-in replies/votes), flagged sets
    highlighted; (b) estimated positions before onset with uncertainty
    discs (radius ~ sigma / sqrt(k)), hollow = no action in window;
    (c) positions late in the window with trajectories since (b);
    (d) top: cumulative platform score per agent; bottom: displacement of
    the windowed position from the burn-in estimate (held when idle).

2-D display axes: the adversarial set's mean displacement direction and
the leading principal direction of all positions orthogonal to it. This
is a display choice, the tests run in all d dimensions.
"""
import os
import sys
import numpy as np
from . import paths
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.insert(0, os.path.join(paths.REPO, 'scripts'))
import hv_style as hv  # noqa: E402

COL = dict(adv='#114471', coh='#6d5bd0', other='#afbec6')


def _roles(res):
    role = np.array(['other'] * len(res['system']), dtype=object)
    for f in res['flagged'][::-1]:
        role[f['members']] = 'adv' if f['adversarial'] else 'coh'
    return role


def _ffill(X, ok):
    X = X.copy()
    X[~ok] = np.nan
    for i in range(X.shape[0]):
        last = np.nan
        for j in range(X.shape[1]):
            if np.isnan(X[i, j]):
                X[i, j] = last
            else:
                last = X[i, j]
    return X


def draw(res, path, t_b=None, t_c=None, title=None, max_graph_nodes=300, max_other=400,
         truth_idx=(), focus_idx=None, focus_label='focal group', onset=None,
         other_sets=None):
    """truth_idx: system indices of a known (planted) ring, outlined in black.
    focus_idx: draw one chosen group (e.g. a real coordinated group from a
    ranking) in the focus colour, with its displacement as the display axis;
    other_sets (lists of indices) recede in the secondary colour."""
    hv.apply()
    S = hv.SIZES
    grid, t0, k, Psi = res['grid'], res['t0'], res['k'], res['Psi']
    if focus_idx is not None:
        role = np.array(['other'] * len(res['system']), dtype=object)
        for o_ in other_sets or []:
            role[list(o_)] = 'coh'
        role[list(focus_idx)] = 'adv'
        adv = [dict(members=list(focus_idx), onset=onset)]
    else:
        role = _roles(res)
        adv = [f for f in res['flagged'] if f['adversarial']]
    onset = adv[0]['onset'] if adv else None
    t_b = t_b or t0 - 8
    t_c = t_c or grid[-1] - 4
    jb, jc = np.argmin(abs(grid - t_b)), np.argmin(abs(grid - t_c))

    # display basis
    D = res['D']
    focus = adv[0]['members'] if adv else (res['flagged'][0]['members'] if res['flagged'] else [])
    act = (k > 0)
    P = Psi[act]
    mu = P.mean(0)
    if len(focus):
        u1 = D[focus].mean(0) - D.mean(0)
    else:
        u1 = np.linalg.svd(P - mu, full_matrices=False)[2][0]
    u1 /= np.linalg.norm(u1)
    Pr = (P - mu) - np.outer((P - mu) @ u1, u1)
    u2 = np.linalg.svd(Pr, full_matrices=False)[2][0]
    B = np.vstack([u1, u2]).T
    XY = (Psi - mu) @ B                     # (n, G, 2)
    sigma = np.sqrt(np.mean(np.var(P @ B, axis=0)))

    fig = plt.figure(figsize=(22, 6.4))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.1, 1, 1, 1.25], height_ratios=[1, 1],
                          wspace=0.28, hspace=0.32)
    axA = fig.add_subplot(gs[:, 0]); axB = fig.add_subplot(gs[:, 1])
    axC = fig.add_subplot(gs[:, 2], sharex=axB, sharey=axB)
    axD1 = fig.add_subplot(gs[0, 3]); axD2 = fig.add_subplot(gs[1, 3], sharex=axD1)
    order = np.argsort([{'other': 0, 'coh': 1, 'adv': 2}[r] for r in role], kind='stable')
    truth = set(int(i) for i in truth_idx)
    big = len(role) > 500
    if big:  # platform scale: show every flagged agent and a sample of the rest
        rs = np.random.default_rng(0)
        oth = np.flatnonzero(role == 'other')
        shown = set(rs.choice(oth, min(len(oth), max_other), replace=False).tolist()) | \
            set(np.flatnonzero(role != 'other').tolist()) | truth
        order = np.array([i for i in order if i in shown])

    # (a) interaction graph
    G = res['G_int'].copy()
    core = set(np.flatnonzero(role == 'adv')) | truth
    if G.number_of_nodes() > max_graph_nodes and core:
        # platform scale: the flagged adversarial set (and planted ring) with
        # its strongest observed neighbours
        nb = {}
        for u in core:
            if u in G:
                for v, e in G[u].items():
                    if v not in core:
                        nb[v] = nb.get(v, 0) + e['weight']
        keep = core | set(sorted(nb, key=lambda v: -nb[v])[:150])
        G = G.subgraph([v for v in G.nodes if v in keep]).copy()
        G.add_nodes_from(v for v in core if v not in G)
    elif G.number_of_nodes() > max_graph_nodes:
        keep = set(np.flatnonzero(role != 'other')) | truth | set(
            sorted(G.nodes, key=lambda v: -G.degree(v, weight='weight'))[:max_graph_nodes])
        G = G.subgraph([v for v in G.nodes if v in keep]).copy()
    pos = nx.spring_layout(G, weight='weight', seed=1, k=1.6 / np.sqrt(max(G.number_of_nodes(), 1)))
    w = np.array([G[u][v]['weight'] for u, v in G.edges]) if G.number_of_edges() else np.array([])
    inside = [(u, v) for u, v in G.edges if role[u] == role[v] != 'other']
    rest = [(u, v) for u, v in G.edges if (u, v) not in set(inside)]
    nx.draw_networkx_edges(G, pos, edgelist=rest, ax=axA, edge_color=hv.EDGE, width=0.5, alpha=0.6)
    for r_ in ('coh', 'adv'):
        el = [(u, v) for u, v in inside if role[u] == r_]
        nx.draw_networkx_edges(G, pos, edgelist=el, ax=axA, edge_color=COL[r_], width=1.6, alpha=0.8)
    for r_, z, sz in (('other', 1, 22), ('coh', 2, 60), ('adv', 3, 70)):
        nl = [v for v in G.nodes if role[v] == r_]
        nx.draw_networkx_nodes(G, pos, nodelist=nl, ax=axA, node_size=sz, node_color=COL[r_],
                               edgecolors='white', linewidths=0.6).set_zorder(z)
    tn = [v for v in G.nodes if v in truth]
    if tn:
        nx.draw_networkx_nodes(G, pos, nodelist=tn, ax=axA, node_size=90, node_color='none',
                               edgecolors=hv.INK_TITLE, linewidths=1.4).set_zorder(5)
    axA.set_axis_off()
    axA.set_title('Interactions near $\\hat S$ after $t_0$' if big and core else 'Observed interactions after $t_0$',
                  fontsize=S['subtitle'])

    # (b), (c) perspective space
    def scatter(ax, j, trails_from=None):
        if trails_from is not None:
            for i in order:
                if role[i] == 'other':
                    continue
                a_ = np.flatnonzero(k[i, :trails_from + 1] > 0)
                b_ = np.flatnonzero(k[i, :j + 1] > 0)
                if len(a_) and len(b_):  # straight segment: last position at (b) -> now
                    p0, p1 = XY[i, a_[-1]], XY[i, b_[-1]]
                    ax.annotate('', xy=p1, xytext=p0, zorder=2,
                                arrowprops=dict(arrowstyle='-', color=COL[role[i]], lw=1.0, alpha=0.6))
        for i in order:
            c = COL[role[i]]
            x, y = XY[i, j]
            if k[i, j] > 0 and (not big or role[i] != 'other'):
                ax.add_patch(Circle((x, y), sigma / np.sqrt(k[i, j]), facecolor=c, alpha=0.10,
                                    edgecolor='none', zorder=1))
            if k[i, j] > 0:
                tr = i in truth
                ax.scatter(x, y, s=40 if (role[i] != 'other' or tr) else (6 if big else 18), color=c,
                           zorder=4 if tr else 3, edgecolors=hv.INK_TITLE if tr else 'white',
                           linewidths=1.4 if tr else (0.5 if not big else 0))
            elif k[i, j] == 0:  # hollow: no action in window, show last known position
                prev = np.flatnonzero(k[i, :j] > 0)
                if len(prev):
                    x, y = XY[i, prev[-1]]
                    ax.scatter(x, y, s=18, facecolors='none', edgecolors=c, linewidths=0.9, zorder=3)
        ax.set_xlabel('displacement axis of $\\hat S$', fontsize=S['annot'])
        ax.set_ylabel('orthogonal principal axis', fontsize=S['annot'])
        ax.tick_params(labelsize=S['annot'] - 2)
    scatter(axB, jb)
    axB.set_title(f'Before onset ($t={grid[jb]:.0f}$ h)', fontsize=S['subtitle'])
    scatter(axC, jc, trails_from=jb)
    axC.set_title(f'$t={grid[jc]:.0f}$ h, moves since $t={grid[jb]:.0f}$', fontsize=S['subtitle'])

    # (d) scores and displacement
    Y = res['Y']
    disp = np.linalg.norm(Psi - (Psi[:, :1] * 0 + _burn_mean(Psi, k, grid, t0)[:, None, :]), axis=2)
    disp = _ffill(disp, k > 0)
    Ys = Y - Y[:, [np.argmin(abs(grid - t0))]]
    if big:  # real platform scores: start at t0, robust limits
        Ys = np.where(grid[None, :] >= t0, Ys, np.nan)
    for ax, M in ((axD1, Ys), (axD2, disp)):
        for i in order:
            ax.plot(grid, M[i], color=COL[role[i]], lw=1.4 if role[i] != 'other' else 0.7,
                    alpha=0.9 if role[i] != 'other' else 0.5, zorder=3 if role[i] != 'other' else 1)
        ax.axvline(t0, color=hv.REFLINE, ls=':', lw=1.2)
        if onset is not None:
            ax.axvline(onset, color=COL['adv'], ls='--', lw=1.0)
        ax.tick_params(labelsize=S['annot'] - 2)
    axD1.set_xlim(res['cfg'].get('window_h') or grid[0], grid[-1])
    if big:
        vals = Ys[order][:, grid >= t0]
        lo_, hi_ = np.nanpercentile(vals, [1, 99.8])
        axD1.set_ylim(min(lo_, -1), hi_ * 1.08)
    axD1.set_ylabel('score since $t_0$', fontsize=S['annot'])
    axD2.set_ylabel('displacement', fontsize=S['annot'])
    axD2.set_xlabel('time (h)', fontsize=S['annot'])
    axD1.set_title('Platform score and displacement', fontsize=S['subtitle'])
    axD1.text(t0, axD1.get_ylim()[1], ' $t_0$', color=hv.INK_MUTE, va='top', fontsize=S['annot'] - 2)
    if onset is not None:
        axD1.text(onset, axD1.get_ylim()[1] * 0.85, ' $\\hat t^*$', color=COL['adv'],
                  va='top', fontsize=S['annot'] - 2)

    for ax, letter in zip((axA, axB, axC, axD1), 'abcd'):
        ax.text(-0.02, 1.07, f'({letter})', transform=ax.transAxes, fontsize=S['panel_letter'] - 4,
                fontweight='bold', color=hv.INK_TITLE, ha='right')
    from matplotlib.lines import Line2D
    coord = res['cfg'].get('gate') == 'coordinated'
    lab_a = focus_label if focus_idx is not None else (
        'coordinated, gaining score' if coord else 'flagged, adversarial')
    lab_c = 'other top-ranked groups' if focus_idx is not None else (
        'coordinated' if coord else 'flagged, cohesive only')
    handles = [Line2D([], [], marker='o', ls='', color=COL['adv'], label=lab_a),
               Line2D([], [], marker='o', ls='', color=COL['coh'], label=lab_c),
               Line2D([], [], marker='o', ls='', color=COL['other'], label='other agents'),
               Line2D([], [], marker='o', ls='', markerfacecolor='none', color=hv.INK_MUTE,
                      label='no action in window')]
    if focus_idx is not None and not other_sets:
        handles = [h_ for h_ in handles if h_.get_label() != lab_c]
    if truth:
        handles.append(Line2D([], [], marker='o', ls='', markerfacecolor='white',
                              markeredgecolor=hv.INK_TITLE, markeredgewidth=1.4,
                              label='planted ring (ground truth)'))
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), fontsize=S['annot'],
               bbox_to_anchor=(0.5, -0.04))
    if title:
        fig.suptitle(title, fontsize=S['subtitle'], color=hv.INK_TITLE, y=1.02)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches='tight', dpi=200)
    fig.savefig(os.path.splitext(path)[0] + '.pdf', bbox_inches='tight')
    plt.close(fig)


def _burn_mean(Psi, k, grid, t0):
    pre = (grid <= t0)[None, :] & (k > 0)
    w = pre[..., None]
    return (Psi * w).sum(1) / np.maximum(w.sum(1), 1)
