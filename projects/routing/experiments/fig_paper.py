"""Paper figures for the routing paper.

fig1_concept   two query-localized geometries (same models, two queries):
               the candidate nearest the flagship changes with the query
fig2_contract  certified volume vs tolerance at alpha = 10%, and the
               flagship-bill savings the certified volume buys
fig3_ablations pairing overlap, anchor cache density, candidate count,
               flagship rank

Reads the lever-sweep decision parquets (figs 2-3) and runs one geometry
pass for the concept figure. Outputs results/figures/*.{png,pdf}.

Run from repo root: pixi run python -m projects.routing.experiments.fig_paper
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .lever_sweep2 import _prep, batched_stats2, gate2
from .run_cost_routing import parse_params
from .run_helm import RESULTS

FIGDIR = os.path.join(RESULTS, 'figures')
# helivan.io palette: deep-blue monochrome + one bright accent
C = {'pairdev': '#114471', 'qa': '#3596ff', 'oracle': '#afbec6',
     'flag': '#090f20', 'pickA': '#3596ff', 'pickB': '#00477d',
     'grey': '#9ca3af', 'slate1': '#93aacc', 'slate2': '#114471',
     'panel': '#f6f7f9', 'edge': '#afbec6'}

# Figures are drawn at FINAL print size (<= 6.9 in full text width), so
# every font below survives 1:1 in the paper.
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 7,
    'axes.titlesize': 7.5,
    'axes.labelsize': 7,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': '#486884',
    'axes.labelcolor': '#213c66',
    'axes.titlecolor': '#0a2245',
    'text.color': '#213c66',
    'xtick.color': '#486884',
    'ytick.color': '#486884',
    'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5,
    'legend.frameon': False,
    'legend.fontsize': 6,
    'grid.color': '#e5e7eb',
    'grid.linewidth': 0.5,
    'grid.alpha': 0.9,
    'figure.facecolor': 'white',
    'lines.linewidth': 1.4,
    'lines.markersize': 3,
})


def _save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIGDIR, f'{name}.{ext}'), dpi=200,
                    bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {name}')


# ---------------------------------------------------------------- concept
def fig_concept():
    from .run_combined import load_combined
    X, Qu, rows, n, names = load_combined()
    sizes = [parse_params(nm.split(':', 1)[1]) for nm in names]
    _, suite_of, gmean, ranked = _prep(X, Qu, rows, n, names)
    flag = ranked['helm'][0]
    pool = {i for i in range(n) if suite_of[i] == 'helm'
            and sizes[i] is not None and sizes[i] <= 40.0} - {flag}

    rng = np.random.default_rng(0)
    helm = rows[rows.suite == 'helm']
    resp = helm.groupby('query')['model'].agg(set)
    full = [q for q, ms in resp.items() if pool | {flag} <= ms]
    tasks = helm.drop_duplicates('query').set_index('query')['task']
    cand_q = list(rng.choice(full, size=120, replace=False))

    is_hold = rows['query'].isin(cand_q).to_numpy()
    anchor_m = ~is_hold
    Xa = X[anchor_m]
    sa = rows['score'].to_numpy()[anchor_m]
    model_a = rows['model'].to_numpy()[anchor_m]
    ua = Qu[rows['code'].to_numpy()[anchor_m]]
    groups = [np.flatnonzero(model_a == m) for m in range(n)]
    i = rng.integers(0, len(ua), 20000)
    k = rng.integers(0, len(ua), 20000)
    kp = i != k
    med = float(np.median(np.linalg.norm(ua[i[kp]] - ua[k[kp]], axis=1)))
    hold_groups = list(rows[is_hold].groupby('query'))
    Ue = np.stack([Qu[g['code'].iloc[0]] for _, g in hold_groups])
    D2 = ((Ue[:, None, :] - ua[None, :, :]) ** 2).sum(-1)
    D2 = D2 - D2.min(axis=1, keepdims=True)
    W = np.exp(-D2 / (2.0 * (0.25 * med) ** 2)).astype(np.float32)
    phi, _, _, ok = batched_stats2(Xa, sa, groups, n, W)

    mods = sorted(pool) + [flag]
    picks, order = [], []
    for gi, (q, g) in enumerate(hold_groups):
        d = np.linalg.norm(phi[gi][sorted(pool)] - phi[gi][flag], axis=1)
        picks.append(sorted(pool)[int(np.argmin(d))])
        order.append((q, tasks[q]))
    # two queries from different datasets with different nearest candidates,
    # such that the pick stays nearest under the shared 2-D projection
    def dset(task):
        return task.split(':')[1]

    def project(a, b):
        Pa = phi[a][mods] - phi[a][mods].mean(axis=0, keepdims=True)
        Pb = phi[b][mods] - phi[b][mods].mean(axis=0, keepdims=True)
        P = np.concatenate([Pa, Pb])
        _, _, Vt = np.linalg.svd(P, full_matrices=False)
        P2 = P @ Vt[:2].T
        return P2[:len(mods)], P2[len(mods):]

    def faithful(pts, gi):
        d2 = np.linalg.norm(pts[:-1] - pts[-1], axis=1)
        return mods[int(np.argmin(d2))] == picks[gi]

    def legibility(A, B, iA, iB):
        # smallest separation among the highlighted relations, in span units
        span = max(np.ptp(np.concatenate([A, B]), axis=0))
        gaps = []
        for pts in (A, B):
            gaps.append(np.linalg.norm(pts[iA] - pts[iB]))
            gaps += [np.linalg.norm(pts[j] - pts[-1]) for j in (iA, iB)]
        return min(gaps) / span

    best = (-1.0, None)
    for a in range(len(order)):
        for b in range(a + 1, len(order)):
            if dset(order[a][1]) == dset(order[b][1]) \
                    or picks[a] == picks[b]:
                continue
            Pa, Pb = project(a, b)
            if not (faithful(Pa, a) and faithful(Pb, b)):
                continue
            ia, ib = mods.index(picks[a]), mods.index(picks[b])
            score = legibility(Pa, Pb, ia, ib)
            if score > best[0]:
                best = (score, (a, b, Pa, Pb))
    assert best[1] is not None, 'no projection-faithful query pair found'
    qa_i, qb_i, A, B = best[1]

    mA, mB = picks[qa_i], picks[qb_i]
    iA, iB = mods.index(mA), mods.index(mB)

    def short(m):
        s = names[m].split(':', 1)[1].split('_', 1)[-1]
        return s if len(s) <= 24 else s[:22] + '…'

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))
    for ax, pts, gi in ((axes[0], A, qa_i), (axes[1], B, qb_i)):
        near = mods.index(picks[gi])
        others = [j for j in range(len(mods) - 1) if j not in (iA, iB)]
        ax.scatter(pts[others, 0], pts[others, 1], s=16, color=C['grey'],
                   alpha=0.55, linewidths=0, zorder=2,
                   label='cached models')
        ax.set_facecolor(C['panel'])
        # dashed link flagship -> this query's nearest
        ax.plot([pts[-1, 0], pts[near, 0]], [pts[-1, 1], pts[near, 1]],
                ls=(0, (2, 3)), color='#486884', lw=1.1, zorder=3)
        for j, key, m in ((iA, 'pickA', mA), (iB, 'pickB', mB)):
            is_near = j == near
            off = (11, 12) if is_near else (9, -13)
            ax.scatter(*pts[j], s=70 if is_near else 52, color=C[key],
                       edgecolors='white', linewidths=0.9, zorder=5)
            if is_near:
                ax.scatter(*pts[j], s=190, facecolors='none',
                           edgecolors=C[key], linewidths=1.2, zorder=4)
            ax.annotate(short(m), pts[j], textcoords='offset points',
                        xytext=off, fontsize=5.8, color=C[key],
                        fontweight='bold' if is_near else 'normal')
        ax.scatter(*pts[-1], marker='*', s=230, color=C['flag'],
                   edgecolors='white', linewidths=0.7, zorder=6,
                   label='flagship')
        ax.annotate(short(flag), pts[-1], textcoords='offset points',
                    xytext=(-13, -17), fontsize=5.8, color=C['flag'],
                    ha='right')
        ax.set_title(f'query from {dset(order[gi][1])}')
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(C['edge'])
        ax.margins(0.10)
    import matplotlib.lines as mlines
    handles = [
        mlines.Line2D([], [], marker='*', ls='', ms=9, color=C['flag'],
                      label='flagship'),
        mlines.Line2D([], [], marker='o', ls='', ms=5, color=C['pickA'],
                      label=f'nearest for the {dset(order[qa_i][1])} query'),
        mlines.Line2D([], [], marker='o', ls='', ms=5, color=C['pickB'],
                      label=f'nearest for the {dset(order[qb_i][1])} query'),
        mlines.Line2D([], [], marker='o', ls='', ms=4, color=C['grey'],
                      alpha=0.55, label='other cached models'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.05, 1, 1.0])
    _save(fig, 'fig1_concept')


# ---------------------------------------------------------------- helpers
def _read(*stems):
    """Concat result parquets; silently skip missing ones (stabilization
    passes append *_b files as they land)."""
    dfs = []
    for s in stems:
        p = os.path.join(RESULTS, f'{s}.parquet')
        if os.path.exists(p):
            dfs.append(pd.read_parquet(p))
    return pd.concat(dfs, ignore_index=True)


def _bar(ax, labels, means, ses, colors, ylabel):
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=ses, width=0.62, color=colors, capsize=1.5,
           error_kw=dict(lw=0.7))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha='right', fontsize=5.8)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, axis='y')


# ---------------------------------------------------------------- contract
def _vol_curve(sub, eps_grid, alpha=0.10):
    """Median volume across seeds (the gate's outcome is bimodal:
    typical certification vs occasional conservative abstention)."""
    out = []
    for eps in eps_grid:
        per_seed = []
        for _, ss in sub.groupby('seed'):
            cc = ss[ss.split == 'cal']['conf'].to_numpy()
            ce = ss[ss.split == 'eval']['conf'].to_numpy()
            cc, ce = (np.where(np.isfinite(x), x, 1e6) for x in (cc, ce))
            off = gate2(cc, ss[ss.split == 'cal']['dev'].to_numpy(),
                        ce, eps, alpha)
            dv = ss[ss.split == 'eval']['dev'].to_numpy()
            per_seed.append((off.mean(),
                             (dv[off] > eps).mean() if off.any() else 0.0))
        per_seed = np.asarray(per_seed)
        cert = per_seed[:, 0] > 0.05
        out.append((np.median(per_seed[:, 0]),
                    per_seed[cert, 1].mean() if cert.any() else 0.0))
    return np.array(out)


def fig_contract():
    d1 = _read('gap_close_p800')
    eps_grid = np.arange(0.15, 0.75, 0.05)
    lev = pd.read_parquet(os.path.join(RESULTS, 'lever2_conf.parquet'))
    rd_dev = lev[(lev.pool == 'le13b') & (lev.method == 'random')
                 & (lev.split == 'eval')]['dev'].median()
    METHS = (('pd-cal', C['pairdev'], 1.8),
             ('qa', C['qa'], 1.4),
             ('oracle-conf', C['oracle'], 1.0))

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6))
    ax = axes[0]
    for pool, ls in (('le13b', '-'), ('all', '--')):
        sub = d1[d1.pool == pool]
        for meth, col, lw in METHS:
            v = _vol_curve(sub[sub.method == meth], eps_grid)
            lbl = 'oracle' if meth == 'oracle-conf' else meth
            pn = '\u226413B' if pool == 'le13b' else 'unrestr.'
            ax.plot(eps_grid, 100 * v[:, 0], ls, color=col, lw=lw,
                    label=f'{lbl} ({pn})')
    ax.axvline(rd_dev, color='k', lw=0.6, ls=':')
    ax.text(rd_dev + 0.01, 6, 'median deviation of\na random model swap',
            fontsize=5.5)
    ax.set_xlabel(r'tolerance $\varepsilon$ (embedding deviation)')
    ax.set_ylabel(r'certified traffic volume (%)  [$\alpha$ = 10%]')
    ax.set_title('(a) Contract frontier: volume certifiable\n'
                 r'at P(deviation > $\varepsilon$) $\leq$ 10%')
    ax.legend(fontsize=6, frameon=False, loc='upper left', ncol=2)
    ax.grid(alpha=0.3)

    ax = axes[1]
    sub = d1[d1.pool == 'le13b']
    for meth, key in (('pd-cal', 'pairdev'), ('qa', 'qa')):
        v = _vol_curve(sub[sub.method == meth], eps_grid)
        for R, lsr in ((25, '-'), (10, ':')):
            ax.plot(eps_grid, 100 * v[:, 0] * (1 - 1 / R), lsr,
                    color=C[key], lw=1.6 if R == 25 else 1.0,
                    label=f'{meth}, {R}x price ratio')
    ax.axvline(rd_dev, color='k', lw=0.6, ls=':')
    ax.set_xlabel(r'tolerance $\varepsilon$')
    ax.set_ylabel('flagship bill saved (%)')
    ax.set_title('(b) Savings from certified offloading\n'
                 r'($\leq$13B substitutes; score-free)')
    ax.legend(fontsize=6, frameon=False, loc='upper left')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, 'fig3_contract')


# --------------------------------------------------------------- ablations
def fig_ablations():
    df2 = pd.read_parquet(os.path.join(RESULTS, 'lever2_conf.parquet'))
    dfp = pd.read_parquet(os.path.join(RESULTS, 'lever2_pairing.parquet'))
    df1 = pd.read_parquet(os.path.join(RESULTS, 'lever_decisions.parquet'))
    d1 = df2[df2.variant == 'r1']
    EPS = np.array([0.5])

    fig, axes = plt.subplots(1, 4, figsize=(6.9, 1.95))

    ax = axes[0]                                   # pairing overlap
    xs = [100, 50, 0]
    for pool, ls in (('le13b', '-'), ('all', '--')):
        for meth, key in (('pairdev', 'pairdev'), ('qa', 'qa')):
            ys = []
            for al in ('aligned', 'independent', 'disjoint'):
                sub = dfp[(dfp.pool == pool) & (dfp.variant == al)
                          & (dfp.method == meth)]
                ys.append(100 * _vol_curve(sub, EPS)[0, 0]
                          if len(sub) else 0.0)
            ax.plot(xs, ys, ls, marker='o', color=C[key], lw=2,
                    label=f'{meth} ({pool})')
    ax.annotate('pairdev inadmissible\n(no shared anchors)', (0, 2),
                fontsize=7.5, ha='left', xytext=(8, 18),
                arrowprops=dict(arrowstyle='-', lw=0.7))
    ax.set_xlabel('flagship-candidate anchor overlap (%)')
    ax.set_ylabel('certified volume (%)')
    ax.set_title('(a) Pairing: guarantees need shared anchors\n'
                 '(HELM, fixed 50% cache per model)')
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(alpha=0.3)

    ax = axes[1]                                   # anchor cache density
    for pool, ls in (('le13b', '-'), ('all', '--')):
        sub = df1[(df1.pool == pool) & (df1.method == 'qa-0.25x')]
        xs, ys = [], []
        for af, ss in sub.groupby('anchor_frac'):
            xs.append(100 * af)
            ys.append(100 * _vol_curve(ss, EPS)[0, 0])
        ax.plot(xs, ys, ls, marker='o', color=C['qa'], lw=2,
                label=f'qa ({pool})')
    ax.set_xlabel('cached responses per model (% of full cache)')
    ax.set_ylabel('certified volume (%)')
    ax.set_title('(b) Cache density: volume keeps paying\nfor history')
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(alpha=0.3)

    ax = axes[2]                                   # candidate scaling
    ks, vol, odev = [], [], []
    for pool, K in [(f'k{k}', k) for k in (2, 4, 8, 16, 32, 64)] \
            + [('all', 137)]:
        sub = d1[(d1.pool == pool) & (d1.method == 'qa')]
        if not len(sub):
            continue
        ks.append(K)
        vol.append(100 * _vol_curve(sub, EPS)[0, 0])
        odev.append(d1[(d1.pool == pool) & (d1.method == 'oracle-pick')
                       & (d1.split == 'eval')]['dev'].mean())
    ax.plot(ks, vol, '-o', color=C['qa'], lw=2, label='qa certified volume')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('candidate pool size (unrestricted, random subsets)')
    ax.set_ylabel('certified volume (%)', color=C['qa'])
    ax2 = ax.twinx()
    ax2.plot(ks, odev, '-s', color=C['oracle'], lw=1.5,
             label='oracle-pick deviation')
    ax2.set_ylabel('hindsight-best deviation', color=C['oracle'])
    ax.set_title('(c) Candidate scaling: volume saturates,\n'
                 'the floor keeps falling')
    ax.grid(alpha=0.3)

    ax = axes[3]                                   # flagship rank
    width = 0.35
    ranks = ('r1', 'r5', 'rmed')
    lbl = ('best', '5th best', 'median')
    for j, (pool, col) in enumerate((('le13b', C['slate1']),
                                     ('all', C['slate2']))):
        ys = []
        for rk in ranks:
            sub = df2[(df2.pool == pool) & (df2.variant == rk)
                      & (df2.method == 'qa')]
            ys.append(100 * _vol_curve(sub, EPS)[0, 0])
        ax.bar(np.arange(3) + (j - 0.5) * width, ys, width, color=col,
               label=f'pool {pool}')
    ax.set_xticks(range(3))
    ax.set_xticklabels(lbl)
    ax.set_xlabel('deprecated target (rank by mean score)')
    ax.set_ylabel('certified volume (%)')
    ax.set_title('(d) Target rank: non-extreme targets certify\n'
                 'more traffic when the pool is rich')
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(alpha=0.3, axis='y')

    fig.suptitle(r'Ablations at contract ($\varepsilon$=0.5, '
                 r'$\alpha$=10%), 3-5 seeds', y=1.06)
    fig.tight_layout()
    _save(fig, 'fig3_ablations')


# --------------------------------------------------------------- selection
def fig2_selection():
    """PKPS selection minimizes mimicry; difficulty vs target rank,
    candidate count, and cache density. Pick-deviation units throughout."""
    fig, axes = plt.subplots(1, 4, figsize=(6.9, 1.95))

    ax = axes[0]                       # (a) combined-pool selection quality
    cm = _read('combined_mimicry')
    order = [('qa-0.25x', 'PKPS (label-free)', C['qa']),
             ('task*', 'task* (hidden labels)', '#114471'),
             ('static', 'static', C['slate1']),
             ('random', 'random', C['grey'])]
    per = cm[cm.method.isin([m for m, _, _ in order] + ['oracle'])] \
        .groupby(['method', 'seed'])['error'].mean().unstack()
    means = [per.loc[m].mean() for m, _, _ in order]
    ses = [per.loc[m].std() / np.sqrt(per.shape[1]) for m, _, _ in order]
    _bar(ax, [l for _, l, _ in order], means, ses,
         [c for _, _, c in order], 'mimicry error of the pick')
    orc = per.loc['oracle'].mean()  # panels b-d share this ylabel
    ax.axhline(orc, color=C['oracle'], lw=1.2, ls='--')
    ax.text(len(order) - 0.45, orc + 0.008, 'oracle floor', fontsize=5.5,
            ha='right', color='#486884')
    ax.set_title('(a) Label-free matches\nhidden task labels')

    d2 = _read('lever2_conf', 'lever2_conf_b')
    ev = d2[(d2.method == 'qa') & (d2.split == 'eval')]

    ax = axes[1]                       # (b) target rank vs neighbor floor
    lbl = {'r1': 'best', 'r5': '5th best', 'rmed': 'median'}
    orc_all = d2[(d2.method == 'oracle-pick') & (d2.split == 'eval')]
    for pool, ls in (('le13b', '-'), ('all', '--')):
        per = ev[ev.pool == pool].groupby(['variant', 'seed'])['dev'] \
            .mean().unstack().reindex(['r1', 'r5', 'rmed'])
        ax.errorbar(range(3), per.mean(1),
                    yerr=per.std(1) / np.sqrt(per.shape[1]), fmt=ls + 'o',
                    color=C['qa'], lw=1.4, ms=2.5, capsize=1.5,
                    label='\u226413B pool' if pool == 'le13b'
                    else 'unrestricted pool')
        orc = orc_all[orc_all.pool == pool].groupby('variant')['dev'] \
            .mean().reindex(['r1', 'r5', 'rmed'])
        ax.plot(range(3), orc, ls, color=C['oracle'], lw=1.0, ms=2,
                marker='s')
    ax.set_xticks(range(3))
    ax.set_xticklabels([lbl[v] for v in ('r1', 'r5', 'rmed')])
    ax.set_xlabel('target model (rank by mean score)')
    ax.set_title('(b) Error tracks the\nneighbor floor (grey)')
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)

    ax = axes[2]                       # (c) candidate count
    ev1 = d2[(d2.variant == 'r1') & (d2.split == 'eval')]
    ks, qa_m, qa_s, orc, rnd = [], [], [], [], []
    for pool, K in [(f'k{k}', k) for k in (2, 4, 8, 16, 32, 64)] \
            + [('all', 137)]:
        sq = ev1[(ev1.pool == pool) & (ev1.method == 'qa')]
        if not len(sq):
            continue
        per = sq.groupby('seed')['dev'].mean()
        ks.append(K)
        qa_m.append(per.mean())
        qa_s.append(per.std() / np.sqrt(len(per)))
        orc.append(ev1[(ev1.pool == pool)
                       & (ev1.method == 'oracle-pick')]['dev'].mean())
        rnd.append(ev1[(ev1.pool == pool)
                       & (ev1.method == 'random')]['dev'].mean())
    ax.plot(ks, rnd, ':', color=C['grey'], lw=1.0, ms=2, marker='o',
            label='random pick')
    ax.errorbar(ks, qa_m, yerr=qa_s, fmt='-o', color=C['qa'], lw=1.4,
                ms=2.5, capsize=1.5, label='PKPS pick')
    ax.plot(ks, orc, '-s', color=C['oracle'], lw=1.0, ms=2,
            label='hindsight best')
    ax.set_xscale('log', base=2)
    ax.set_xticks([2, 8, 32, 128])
    ax.set_xticklabels(['2', '8', '32', '128'])
    ax.set_xlabel('candidate models')
    ax.set_title('(c) Gains saturate; hindsight\nis selection-inflated')
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)

    ax = axes[3]                       # (d) cache density
    d1 = _read('lever_decisions', 'lever_decisions_b')
    sq = d1[(d1.method == 'qa-0.25x') & (d1.split == 'eval')]
    for pool, ls in (('le13b', '-'), ('all', '--')):
        per = sq[sq.pool == pool].groupby(['anchor_frac', 'seed'])['dev'] \
            .mean().unstack()
        ax.errorbar(100 * per.index, per.mean(1),
                    yerr=per.std(1) / np.sqrt(per.shape[1]), fmt=ls + 'o',
                    color=C['qa'], lw=1.4, ms=2.5, capsize=1.5,
                    label='\u226413B pool' if pool == 'le13b'
                    else 'unrestricted pool')
    ax.set_xlabel('cache size (% of full)')
    ax.set_title('(d) Cache density')
    ax.legend(fontsize=6, loc='lower left')
    ax.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, 'fig2_selection')


# ------------------------------------------------------------------ family
def _gate_stats(df, pool, methods, eps=0.5, alpha=0.10):
    from .gap_close import gate2
    rows = []
    for meth in methods:
        sub = df[(df.pool == pool) & (df.method == meth)]
        if not len(sub):
            continue
        per = []
        for _, ss in sub.groupby('seed'):
            cc = ss[ss.split == 'cal']['conf'].to_numpy()
            ce = ss[ss.split == 'eval']['conf'].to_numpy()
            cc, ce = (np.where(np.isfinite(x), x, 1e6) for x in (cc, ce))
            off = gate2(cc, ss[ss.split == 'cal']['dev'].to_numpy(), ce,
                        eps, alpha)
            dv = ss[ss.split == 'eval']['dev'].to_numpy()
            per.append((off.mean(),
                        (dv[off] > eps).mean() if off.any() else 0.0))
        per = np.asarray(per)
        cert = per[:, 0] > 0.05
        rows.append({'method': meth, 'vol': np.median(per[:, 0]),
                     'q25': np.quantile(per[:, 0], 0.25),
                     'q75': np.quantile(per[:, 0], 0.75),
                     'viol': per[cert, 1].mean() if cert.any() else 0.0,
                     'abstain': float((~cert).mean())})
    if not rows:
        return pd.DataFrame(columns=['vol', 'se', 'viol']) \
            .rename_axis('method')
    return pd.DataFrame(rows).set_index('method')


LADDER = [('qa', 'none\n(localized means)', C['qa']),
          ('var-norm', 'none', C['grey']),
          ('var-ub', 'none', C['grey']),
          ('pd30', '30 bought\nanchors/cand.', C['slate1']),
          ('pd100', '100 bought\nanchors/cand.', C['slate1']),
          ('pd-cal', 'calibration\nsample only', '#114471'),
          ('pairdev', 'full suite\npairing', '#486884'),
          ('oracle-conf', '(cheats)', C['oracle'])]


def fig4_family():
    """One estimator family, one coupling dial delta."""
    gc = _read('gap_close_p800')
    fig = plt.figure(figsize=(6.9, 2.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.3, 1.0])

    ax = fig.add_subplot(gs[0])        # (a) schematic: the coupling dial
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.84, r'$\hat D^2(m,f;q^*)=\frac{\sum_{j,l} k_\sigma(q_j,'
            r'q^*)\,k_\sigma(q_l,q^*)\,k_\delta(q_j,q_l)\,\|x_{mj}-x_{fl}'
            r'\|^2}{\mathrm{norm}}$', ha='center', fontsize=6.5)
    ax.annotate('', xy=(0.95, 0.60), xytext=(0.05, 0.60),
                arrowprops=dict(arrowstyle='->', color='#486884', lw=1.1))
    ax.text(0.5, 0.67, r'coupling bandwidth $\delta$', ha='center',
            fontsize=6.5, color='#213c66')
    for x, y, t, c in ((0.08, 0.49, r'$\delta\to\infty$' '\nqa, var-ub',
                        C['qa']),
                       (0.50, 0.27, r'finite $\delta$' '\nsoft-pairdev',
                        '#93aacc'),
                       (0.91, 0.49, r'$\delta\to 0$' '\npd-cal, pairdev',
                        '#114471')):
        ax.plot([x], [0.60], 'o', color=c, ms=4)
        ax.text(x, y, t, ha='center', va='top', fontsize=5.8, color=c)
    ax.text(0.5, 0.02, r'other dials: localization $\sigma$; anchor'
            ' measure', ha='center', fontsize=5.5, color='#486884')
    ax.set_title('(a) One family, one pairing dial')

    ax = fig.add_subplot(gs[1])        # (b) ladder bars, 15 seeds
    st = _gate_stats(gc, 'all', [m for m, _, _ in LADDER])
    labels, means, errs, cols = [], [], [[], []], []
    for m, note, c in LADDER:
        if m not in st.index:
            continue
        labels.append('oracle' if m == 'oracle-conf' else m)
        means.append(100 * st.loc[m, 'vol'])
        errs[0].append(100 * (st.loc[m, 'vol'] - st.loc[m, 'q25']))
        errs[1].append(100 * (st.loc[m, 'q75'] - st.loc[m, 'vol']))
        cols.append(c)
    _bar(ax, labels, means, errs, cols, 'certified volume (%)')
    for xi, m in enumerate(means):
        if m < 1.5:
            ax.text(xi, 1.5, '0', ha='center', fontsize=6,
                    color='#486884')
    ax.set_title('(b) Certified volume by member\n'
                 r'(median [IQR]; $\varepsilon$=0.5, $\alpha$=10%)')

    ax = fig.add_subplot(gs[2])        # (c) pairing overlap incl. soft
    pr = _read('lever2_pairing_softcorr2')
    xs = [100, 50, 0]
    for meth, col, lbl in (('pairdev', C['pairdev'], 'pairdev'),
                           ('soft-0.05x', '#93aacc', 'soft-pairdev'),
                           ('qa', C['qa'], 'qa')):
        ys = []
        for al in ('aligned', 'independent', 'disjoint'):
            st = _gate_stats(pr[pr.variant == al], 'all', [meth])
            ys.append(100 * st.loc[meth, 'vol'] if len(st) else np.nan)
        ax.plot(xs, ys, '-o', color=col, lw=1.4, ms=2.5, label=lbl)
    ax.annotate('pairdev inadmissible;\nsoft earns only its\n'
                r'$\delta\to0$ content', (0, 4), fontsize=5.5,
                xytext=(35, 8), arrowprops=dict(arrowstyle='-', lw=0.6))
    ax.set_xlabel('flagship-candidate overlap (%)')
    ax.set_ylabel('certified volume (%)')
    ax.set_title('(c) Exact pairs: the unique\nvariance-zero coupling')
    ax.legend(fontsize=6, loc='upper left')
    ax.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, 'fig4_family')


# ------------------------------------------------------------------- price
def fig5_price():
    """(a) certified volume vs paired-sample budget with N* prediction;
    (b) gate validity: achieved violation vs calibration size."""
    from .gap_close import gate2
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6))

    ax = axes[0]
    cur = _read('cal_curve_nested')
    ax.axvspan(50, 300, color='#e0edff', alpha=0.55, lw=0)
    ax.set_ylim(0, 104)
    ax.text(175, 102, 'gate under-calibrated\n(violation rides '
            r'$\approx\alpha$)', ha='center', va='top', fontsize=5.5,
            color='#486884')
    for pool, ls in (('all', '-'), ('le13b', '--')):
        sub = cur[(cur.pool == pool) & (cur.method == 'pd-cal')]
        g = sub.groupby('N')['vol']
        Ns = g.median().index.to_numpy()
        ax.fill_between(Ns, 100 * g.quantile(0.25), 100 * g.quantile(0.75),
                        color=C['pairdev'], alpha=0.15, lw=0)
        ax.plot(Ns, 100 * g.median(), ls + 'o', color=C['pairdev'],
                lw=1.6, ms=2.5,
                label='pd-cal (unrestricted)' if pool == 'all'
                else 'pd-cal (\u226413B)')
    ax.axvline(2 * 217, color='k', lw=0.6, ls=':')
    ax.text(2 * 217 + 18, 44, 'predicted knee\n'
            r'$2N^*=2s^2/(\kappa\tau^2)$', fontsize=5.5)
    ax.set_xlabel('total paired sample N (anchors + gate calibration)')
    ax.set_ylabel('certified volume (%), median [IQR]')
    ax.set_title('(a) The price of certification:\n'
                 'generations on your own traffic')
    ax.legend(fontsize=5.8, ncol=1, loc='lower right',
              bbox_to_anchor=(1.0, 0.0))
    ax.grid(alpha=0.3)

    ax = axes[1]
    gc = _read('gap_close_p800')
    sub = gc[(gc.pool == 'all') & (gc.method == 'pd-cal')]
    for rule, col, lbl in (('emp', C['pairdev'], 'empirical cutoff'),
                           ('ucb', C['qa'], 'Clopper-Pearson cutoff')):
        ns, vio, vols = [], [], []
        for nc in (50, 100, 200, 400):
            per = []
            for seed, ss in sub.groupby('seed'):
                cal = ss[ss.split == 'cal']
                cal = cal.sample(min(nc, len(cal)), random_state=int(seed))
                ev = ss[ss.split == 'eval']
                cc = np.where(np.isfinite(cal.conf), cal.conf, 1e6)
                ce = np.where(np.isfinite(ev.conf), ev.conf, 1e6)
                off = gate2(cc, cal['dev'].to_numpy(), ce, 0.5, 0.10,
                            rule=rule)
                dv = ev['dev'].to_numpy()
                per.append(((dv[off] > 0.5).mean() if off.any() else 0.0,
                            off.mean()))
            per = np.asarray(per)
            ns.append(nc)
            vio.append(per[:, 0].mean())
            vols.append(per[:, 1].mean())
        ax.plot(ns, 100 * np.asarray(vio), '-o', color=col, lw=1.4,
                ms=2.5, label=lbl)
        for x, y, v in zip(ns, 100 * np.asarray(vio), vols):
            ax.annotate(f'{v:.0%}', (x, y), textcoords='offset points',
                        xytext=(4, 4), fontsize=5.5, color=col)
    ax.axhline(10, color='k', lw=0.6, ls=':')
    ax.text(52, 10.3, r'budget $\alpha$', fontsize=5.5)
    ax.set_xlabel('gate calibration size')
    ax.set_ylabel('achieved violation (%)')
    ax.set_title('(b) Gate validity vs calibration size\n'
                 '(labels = certified volume)')
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, 'fig5_price')


# ------------------------------------------------------------- limitations
def fig6_novel():
    from .gap_close import gate2
    nv = _read('lever2_novel')
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.4))

    ax = axes[0]
    qa = nv[(nv.method == 'qa') & (nv.pool == 'all')]
    bins = np.linspace(0, np.quantile(qa.conf, 0.99), 40)
    for split, col, lbl in (('eval-seen', C['qa'], 'seen tasks'),
                            ('eval-novel', C['pairdev'],
                             'novel tasks (absent from cache)')):
        ax.hist(qa[qa.split == split]['conf'], bins=bins, density=True,
                histtype='stepfilled', alpha=0.45, color=col, label=lbl)
    from sklearn.metrics import roc_auc_score
    evq = qa[qa.split.str.startswith('eval')]
    auc = roc_auc_score((evq.split == 'eval-novel').astype(int), evq.conf)
    ax.text(0.97, 0.55, f'AUC(novel vs seen) = {auc:.2f}\n'
            'too weak to gate on', transform=ax.transAxes, ha='right',
            fontsize=5.8, color='#486884')
    ax.set_xlabel('PKPS confidence (predicted deviation)')
    ax.set_ylabel('density')
    ax.set_title('(a) Confidence carries little novelty signal')
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)

    ax = axes[1]
    width = 0.35
    for j, pool in enumerate(('le13b', 'all')):
        qa = nv[(nv.method == 'qa') & (nv.pool == pool)]
        stats = {}
        for lbl, evs in (('seen', 'eval-seen'), ('novel', 'eval-novel')):
            per = []
            for seed, ss in qa.groupby('seed'):
                cal = ss[ss.split == 'cal']
                ev = ss[ss.split == evs]
                off = gate2(cal['conf'].to_numpy(), cal['dev'].to_numpy(),
                            ev['conf'].to_numpy(), 0.5, 0.10)
                dv = ev['dev'].to_numpy()
                per.append((dv[off] > 0.5).mean() if off.any() else 0.0)
            stats[lbl] = np.mean(per)
        ax.bar(np.arange(2) + (j - 0.5) * width,
               [100 * stats['seen'], 100 * stats['novel']], width,
               color=[C['slate1'], C['slate2']][j],
               label='\u226413B pool' if pool == 'le13b'
               else 'unrestricted pool')
    ax.axhline(10, color='k', lw=0.6, ls=':')
    ax.text(-0.42, 10.3, r'budget $\alpha$', fontsize=5.5)
    ax.set_xticks(range(2))
    ax.set_xticklabels(['seen-task traffic', 'novel-task traffic'])
    ax.set_ylabel('achieved violation (%)')
    ax.set_title('(b) Novel-task traffic breaks\nthe contract silently')
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3, axis='y')

    fig.tight_layout()
    _save(fig, 'fig6_novel')


# ------------------------------------------------------------------ tables
T1_META = [
    ('qa', r'$\infty$ (means only)', 'full cache', 'none'),
    ('var-norm', r'$\infty$, volatility-scaled', 'full cache', 'none'),
    ('var-ub', r'$\infty$ (means+variances)', 'full cache', 'none'),
    ('pd10', '0 (exact)', '10 bought anchors/cand.', '10 generations'),
    ('pd30', '0 (exact)', '30 bought anchors/cand.', '30 generations'),
    ('pd100', '0 (exact)', '100 bought anchors/cand.', '100 generations'),
    ('pd-cal', '0 (exact)', 'calibration sample',
     'none beyond gate calibration'),
    ('pairdev', '0 (exact)', 'full paired suite', 'suite pairing'),
    ('oracle-conf', r'0, at $q^*$', '(peeks at eval)', 'cheating'),
]


def make_tables():
    from scipy.stats import spearmanr
    gc = _read('gap_close_p800')
    rows = []
    for meth, delta, anchors, extra in T1_META:
        rec = {'member': meth if meth != 'oracle-conf' else 'oracle',
               'coupling': delta, 'anchors': anchors, 'assumes': extra}
        for pool in ('le13b', 'all'):
            st = _gate_stats(gc, pool, [meth])
            if not len(st):
                continue
            rec[f'vol_{pool}'] = f"{100 * st.loc[meth, 'vol']:.0f}%"
            if pool == 'all':
                rec['viol'] = f"{st.loc[meth, 'viol']:.2f}"
                rec['abstain'] = f"{st.loc[meth, 'abstain']:.0%}"
                ev = gc[(gc.pool == pool) & (gc.method == meth)
                        & (gc.split == 'eval')]
                ce = np.where(np.isfinite(ev.conf), ev.conf, 1e6)
                rec['spearman'] = f'{spearmanr(ce, ev.dev).statistic:.2f}'
        rows.append(rec)
    t1 = pd.DataFrame(rows)

    lv = _read('lever2_conf', 'lever2_conf_b')
    ev = lv[(lv.variant == 'r1') & (lv.split == 'eval')]
    rows = []
    for meth in ('lead', 'random'):
        rec = {'method': meth}
        for eps in (0.1, 0.3, 0.5):
            for pool in ('le13b', 'all'):
                sub = ev[(ev.pool == pool) & (ev.method == meth)]
                rec[f'viol@{eps:g} ({pool})'] = \
                    f'{100 * (sub.dev > eps).mean():.0f}%'
        rows.append(rec)
    t2 = pd.DataFrame(rows)

    os.makedirs(os.path.join(RESULTS, 'tables'), exist_ok=True)
    for name, t in (('table1_family', t1), ('table2_baselines', t2)):
        t.to_csv(os.path.join(RESULTS, 'tables', f'{name}.csv'),
                 index=False)
        with open(os.path.join(RESULTS, 'tables', f'{name}.md'), 'w') as f:
            f.write(t.to_markdown(index=False))
        with open(os.path.join(RESULTS, 'tables', f'{name}.tex'), 'w') as f:
            f.write(t.to_latex(index=False, escape=False))
        print(f'wrote tables/{name}')
    return t1, t2


# =================================================================
# MAIN-BODY FIGURE SET (one hero: the PKPS router = localized pick +
# deviation-regression confidence + conformal gate). One question per
# figure; levers/full ladder/gate rules live in the appendix figures.
# =================================================================

HERO = '#114471'


def fig2_payoff():
    """Does it pay? Pick quality -> certified volume -> dollars."""
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.2))

    ax = axes[0]                                   # (a) the pick
    cm = _read('combined_mimicry')
    order = [('qa-0.25x', 'PKPS pick', C['qa']),
             ('task*', 'task labels', '#114471'),
             ('static', 'static', C['slate1']),
             ('random', 'random', C['grey'])]
    per = cm[cm.method.isin([m for m, _, _ in order] + ['oracle'])] \
        .groupby(['method', 'seed'])['error'].mean().unstack()
    means = [per.loc[m].mean() for m, _, _ in order]
    ses = [per.loc[m].std() / np.sqrt(per.shape[1]) for m, _, _ in order]
    _bar(ax, [l for _, l, _ in order], means, ses,
         [c for _, _, c in order], 'mimicry error of the pick')
    orc = per.loc['oracle'].mean()
    ax.axhline(orc, color=C['oracle'], lw=1.2, ls='--')
    ax.text(len(order) - 0.45, orc + 0.012, 'oracle', fontsize=5.5,
            ha='right', color='#486884')
    ax.set_title('(a) The pick: label-free,\nno task metadata')

    d1 = _read('gap_close_p800')
    eps_grid = np.arange(0.15, 0.75, 0.05)
    lev = _read('lever2_conf')
    rd_dev = lev[(lev.pool == 'le13b') & (lev.method == 'random')
                 & (lev.split == 'eval')]['dev'].median()

    ax = axes[1]                                   # (b) the contract
    for pool, ls, lbl in (('all', '-', 'unrestricted pool'),
                          ('le13b', '--', '\u226413B pool')):
        sub = d1[d1.pool == pool]
        v = _vol_curve(sub[sub.method == 'pd-cal'], eps_grid)
        ax.plot(eps_grid, 100 * v[:, 0], ls, color=HERO, lw=1.8,
                label=f'PKPS router ({lbl})')
        v = _vol_curve(sub[sub.method == 'oracle-conf'], eps_grid)
        ax.plot(eps_grid, 100 * v[:, 0], ls, color=C['oracle'], lw=1.0)
    ax.plot([], [], '-', color=C['oracle'], lw=1.0, label='oracle')
    ax.axvline(rd_dev, color='k', lw=0.6, ls=':')
    ax.text(rd_dev - 0.012, 20, 'typical model-\nswap deviation',
            fontsize=5.5, ha='right')
    ax.text(0.97, 0.06, 'leaderboard / static / random:\nno per-query '
            'signal, certify 0%', transform=ax.transAxes, ha='right',
            fontsize=5.5, color='#486884', style='italic')
    ax.set_xlabel(r'tolerance $\varepsilon$')
    ax.set_ylabel('certified volume (%)')
    ax.set_title('(b) The guarantee: median volume\n'
                 r'at P(dev > $\varepsilon$) $\leq$ 10%')
    ax.legend(fontsize=5.8, loc='upper left')
    ax.grid(alpha=0.3)

    ax = axes[2]                                   # (c) the money
    sub = d1[d1.pool == 'le13b']
    v = _vol_curve(sub[sub.method == 'pd-cal'], eps_grid)
    for R, lsr in ((25, '-'), (10, ':')):
        ax.plot(eps_grid, 100 * v[:, 0] * (1 - 1 / R), lsr, color=HERO,
                lw=1.8 if R == 25 else 1.2, label=f'{R}x price ratio')
    ax.axvline(rd_dev, color='k', lw=0.6, ls=':')
    ax.set_xlabel(r'tolerance $\varepsilon$')
    ax.set_ylabel('flagship bill saved (%)')
    ax.set_title('(c) The savings\n(\u226413B substitutes)')
    ax.legend(fontsize=5.8, loc='upper left')
    ax.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, 'fig2_payoff')


def fig3_why():
    """Why it works and what it minimally needs: the coupling ladder
    under ONE protocol, plus the controlled pairing ablation."""
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5))

    ax = axes[0]                                   # (a) condensed ladder
    gc = _read('gap_close_p800')
    members = [('qa', 'unpaired\n(means only)', C['qa']),
               ('pd-cal', 'calibration\nsample (ours)', HERO),
               ('pairdev', 'full suite\npairing', '#486884')]
    st = _gate_stats(gc, 'all', [m for m, _, _ in members]
                     + ['oracle-conf'])
    labels, meds, errs, cols = [], [], [[], []], []
    for m, lbl, c in members:
        labels.append(lbl)
        meds.append(100 * st.loc[m, 'vol'])
        errs[0].append(100 * (st.loc[m, 'vol'] - st.loc[m, 'q25']))
        errs[1].append(100 * (st.loc[m, 'q75'] - st.loc[m, 'vol']))
        cols.append(c)
    x = np.arange(len(labels))
    ax.bar(x, meds, yerr=errs, width=0.6, color=cols, capsize=1.5,
           error_kw=dict(lw=0.7))
    for xi, m in enumerate(meds):
        if m < 1.5:
            ax.text(xi, 2, '0', ha='center', fontsize=6, color='#486884')
    orc = 100 * st.loc['oracle-conf', 'vol']
    ax.axhline(orc, color=C['oracle'], lw=1.2, ls='--')
    ax.text(len(labels) - 0.5, orc + 2, 'oracle', fontsize=5.5,
            ha='right', color='#486884')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5.8)
    ax.set_ylabel('certified volume (%), median [IQR]')
    ax.set_title('(a) Confidence needs exact pairs;\n'
                 'the calibration sample suffices')
    ax.grid(alpha=0.3, axis='y')

    ax = axes[1]                                   # (b) pairing ablation
    pr = _read('lever2_pairing_softcorr2')
    xs = [100, 50, 0]
    for meth, col, lbl in (('pairdev', '#486884', 'full-pairing regr.'),
                           ('soft-0.05x', C['slate1'], 'soft-coupled'),
                           ('qa', C['qa'], 'unpaired (means)')):
        ys = []
        for al in ('aligned', 'independent', 'disjoint'):
            stp = _gate_stats(pr[pr.variant == al], 'all', [meth])
            ys.append(100 * stp.loc[meth, 'vol'] if len(stp) else np.nan)
        ax.plot(xs, ys, '-o', color=col, lw=1.4, ms=2.5, label=lbl)
    ax.annotate('inadmissible: no shared\nanchors to regress on',
                (0, 4), fontsize=5.5, xytext=(30, 8),
                arrowprops=dict(arrowstyle='-', lw=0.6))
    ax.set_xlabel('flagship\u2013candidate anchor overlap (%)')
    ax.set_ylabel('certified volume (%)')
    ax.set_title('(b) Controlled ablation: coupling\nmust be exact '
                 '(HELM, fixed cache)')
    ax.legend(fontsize=5.8, loc='upper left')
    ax.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, 'fig3_why')


def fig4_ops():
    """What it costs and when it fails: budget curve, gate validity,
    exchangeability breakage. All caveats live here by design."""
    from .gap_close import gate2 as _g2
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.1))

    ax = axes[0]                                   # (a) budget curve
    cur = _read('cal_curve_nested')
    ax.axvspan(50, 300, color='#e0edff', alpha=0.55, lw=0)
    ax.set_ylim(0, 104)
    ax.text(175, 102, 'gate unreliable', ha='center', va='top',
            fontsize=5.5, color='#486884')
    for pool, ls, lbl in (('all', '-', 'unrestricted'),
                          ('le13b', '--', '\u226413B')):
        sub = cur[(cur.pool == pool) & (cur.method == 'pd-cal')]
        g = sub.groupby('N')['vol']
        Ns = g.median().index.to_numpy()
        ax.fill_between(Ns, 100 * g.quantile(0.25),
                        100 * g.quantile(0.75), color=HERO, alpha=0.15,
                        lw=0)
        ax.plot(Ns, 100 * g.median(), ls + 'o', color=HERO, lw=1.6,
                ms=2.5, label=lbl)
    ax.axvline(2 * 217, color='k', lw=0.6, ls=':')
    ax.text(2 * 217 + 20, 40, 'predicted\nknee $2N^*$', fontsize=5.5)
    ax.set_xlabel('paired sample N (own-traffic queries)')
    ax.set_ylabel('certified volume (%)')
    ax.set_title('(a) Cost: generations on\nyour own traffic')
    ax.legend(fontsize=5.8, loc='lower right')
    ax.grid(alpha=0.3)

    ax = axes[1]                                   # (b) gate validity
    gc = _read('gap_close_p800')
    sub = gc[(gc.pool == 'all') & (gc.method == 'pd-cal')]
    for rule, col, lbl in (('emp', HERO, 'empirical cutoff'),
                           ('ucb', C['qa'], 'Clopper\u2013Pearson')):
        ns, vio, vols = [], [], []
        for nc in (50, 100, 200, 400):
            per = []
            for seed, ss in sub.groupby('seed'):
                cal = ss[ss.split == 'cal']
                cal = cal.sample(min(nc, len(cal)),
                                 random_state=int(seed))
                ev = ss[ss.split == 'eval']
                cc = np.where(np.isfinite(cal.conf), cal.conf, 1e6)
                ce = np.where(np.isfinite(ev.conf), ev.conf, 1e6)
                off = _g2(cc, cal['dev'].to_numpy(), ce, 0.5, 0.10,
                          rule=rule)
                dv = ev['dev'].to_numpy()
                per.append(((dv[off] > 0.5).mean() if off.any()
                            else 0.0, off.mean()))
            per = np.asarray(per)
            ns.append(nc)
            vio.append(per[:, 0].mean())
            vols.append(np.median(per[:, 1]))
        ax.plot(ns, 100 * np.asarray(vio), '-o', color=col, lw=1.4,
                ms=2.5, label=lbl)
        dy = -9 if rule == 'emp' else 4
        for xx, yy, v in zip(ns, 100 * np.asarray(vio), vols):
            ax.annotate(f'{v:.0%}', (xx, yy), textcoords='offset points',
                        xytext=(4, dy), fontsize=5.5, color=col)
    ax.axhline(10, color='k', lw=0.6, ls=':')
    ax.text(330, 10.25, r'budget $\alpha$', fontsize=5.5)
    ax.set_xlabel('gate calibration size')
    ax.set_ylabel('achieved violation (%)')
    ax.set_title('(b) Validity: violation vs\ncalibration (labels: '
                 'volume)')
    ax.legend(fontsize=5.8)
    ax.grid(alpha=0.3)

    ax = axes[2]                                   # (c) exchangeability
    nv = _read('lever2_novel')
    width = 0.35
    for j, (pool, lbl) in enumerate((('le13b', '\u226413B pool'),
                                     ('all', 'unrestricted pool'))):
        qa = nv[(nv.method == 'qa') & (nv.pool == pool)]
        stats = {}
        for k, evs in (('seen', 'eval-seen'), ('novel', 'eval-novel')):
            per = []
            for seed, ss in qa.groupby('seed'):
                cal = ss[ss.split == 'cal']
                ev = ss[ss.split == evs]
                off = _g2(cal['conf'].to_numpy(), cal['dev'].to_numpy(),
                          ev['conf'].to_numpy(), 0.5, 0.10)
                dv = ev['dev'].to_numpy()
                per.append((dv[off] > 0.5).mean() if off.any() else 0.0)
            stats[k] = np.mean(per)
        ax.bar(np.arange(2) + (j - 0.5) * width,
               [100 * stats['seen'], 100 * stats['novel']], width,
               color=[C['slate1'], '#114471'][j], label=lbl)
    ax.axhline(10, color='k', lw=0.6, ls=':')
    ax.set_xticks(range(2))
    ax.set_xticklabels(['seen tasks', 'novel tasks'])
    ax.set_ylabel('achieved violation (%)')
    ax.set_title('(c) Limit: novel-task traffic\nbreaks the contract')
    ax.legend(fontsize=5.8)
    ax.grid(alpha=0.3, axis='y')

    fig.tight_layout()
    _save(fig, 'fig4_ops')


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    # main-body set
    fig2_payoff()
    fig3_why()
    fig4_ops()
    make_tables()
    # appendix set
    fig2_selection()
    fig_contract()
    fig4_family()
    fig5_price()
    fig6_novel()
    fig_ablations()
    # concept last (data pass)
    fig_concept()


if __name__ == '__main__':
    main()
