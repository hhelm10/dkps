"""Ablations under the FIXED canonical protocol (leave-one-family-out,
SWE-bench Verified) — figure 4, rev 2 (per HH 2026-09-17):

  panel A  reference-library size: x = n in {10,20,40,70,107};
           IRT + qubric geometry only (no blend); m in {1,5,20} as a
           line-weight/alpha gradient per method
  panel B  probe selection: x = m; IRT + blend; filled = adaptive,
           open = random; solid n=107, dashed n=20
  panel C  embedding model: x = m; qubric geometry per encoder;
           solid n=107, dashed n=20

Estimator stack identical to fig_protocols: consensus centering, PKPS
kernel, per-draw pooled CV over sigma x ({kNN 3,5} u ridge-on-MDS
LOO-honest), pooled alpha blend, shared rng(0) draw stream. n-subsets
are nested (seeded rng(11) permutation).

Stages: python scripts/fig_ablations.py nsweep|regime20|embedders|render|all
Writes figures/ablations.json + figures/fig_ablations.{png,pdf}.
"""
import json
import os
import sys
from multiprocessing import get_context

import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
from outcome_baselines import ItemModel, load_panel  # noqa: E402
from dkps.traces.qubric import consensus_center  # noqa: E402

SIGS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
KS = (3, 5)
RIDGE = ((16, 0.1), (16, 1.0), (8, 0.1))
ALPHAS = np.linspace(0, 1, 101)
MS = (1, 3, 5, 10, 20)
MS_A = (1, 5, 20)
NS = (10, 20, 40, 70, 107)
B_DRAWS = 50
OUT_JSON = 'figures/ablations.json'
EMB = [('openai', 'text-emb-3-small',
        'data/judge/q100_emb_openai_small.npz', 'X'),
       ('nomic', 'nomic-v1.5',
        'data/judge/pillars_emb_q100-qspec-flash0731_'
        'nomic-ai_nomic-embed-text-v1.5.npz', 'Xq'),
       ('bge', 'bge-large',
        'data/judge/pillars_emb_q100-qspec-flash0731_'
        'BAAI_bge-large-en-v1.5.npz', 'Xq'),
       ('gte', 'gte-large',
        'data/judge/pillars_emb_q100-qspec-flash0731_'
        'thenlper_gte-large.npz', 'Xq'),
       ('e5', 'e5-large',
        'data/judge/pillars_emb_q100-qspec-flash0731_'
        'intfloat_e5-large-v2.npz', 'Xq'),
       ('mpnet', 'mpnet-base',
        'data/judge/pillars_emb_q100-qspec-flash0731_'
        'sentence-transformers_all-mpnet-base-v2.npz', 'Xq'),
       ('minilm', 'MiniLM-L6',
        'data/judge/pillars_emb_q100-qspec-flash0731_'
        'sentence-transformers_all-MiniLM-L6-v2.npz', 'Xq')]


def load_base():
    from pillars import vendor_tag
    systems, q100, y, B, _ = load_panel(panel='data/judge/q100.json')
    M, Q = B.shape
    labels = json.load(open('data/leaderboard/verified_labels.json'))
    fam = [vendor_tag(labels, s) for s in systems]
    allowed = np.array([[j != i and not (fam[i] and fam[j] == fam[i])
                         for j in range(M)] for i in range(M)])
    z = np.load('data/leaderboard/query_vecs_64.npz', allow_pickle=True)
    ids = [str(x) for x in z['ids']]
    V = np.asarray(z['vecs'], np.float32)[[ids.index(q) for q in q100]]
    D2q = ((V[:, None] - V[None]) ** 2).sum(-1)
    rng = np.random.default_rng(0)
    draws = {m: [rng.choice(Q, m, replace=False) for _ in range(B_DRAWS)]
             for m in MS}
    return systems, y, B, allowed, D2q, np.median(D2q), draws


def lib_mask(M, n):
    inlib = np.zeros(M, bool)
    inlib[np.random.default_rng(11).permutation(M)[:n]] = True
    return inlib


def load_X(path, key):
    M, Q = 107, 100
    return consensus_center(np.load(path)[key],
                            np.tile(np.arange(Q), M)) \
        .reshape(M, Q, -1).astype(np.float32)


def make_kern(X, D2q, med):
    kern = {}
    for s_ in SIGS:
        KQ = np.exp(-D2q / (2 * med / s_)).astype(np.float32)
        W = np.einsum('qp,jpd->jqd', KQ, X, optimize=True)
        kern[s_] = (KQ, W, np.einsum('jqd,jqd->j', X, W) / KQ.sum())
    return kern


def pkps_D(X, kern, cols, s_):
    M = len(X)
    KQ, W, Arr = kern[s_]
    Xi = X[:, cols].reshape(M, -1)
    A_tr = (Xi @ W[:, cols].reshape(M, -1).T) / KQ[cols].sum()
    KQc = KQ[np.ix_(cols, cols)]
    Wc = np.einsum('qp,jpd->jqd', KQc, X[:, cols], optimize=True)
    Att = np.einsum('jqd,jqd->j', X[:, cols], Wc) / KQc.sum()
    return np.sqrt(np.maximum(Att[:, None] + Arr[None] - 2 * A_tr, 0))


def _cv_select(D_of, cols_of, y, allowed):
    """Pooled-CV candidate stores; D_of[s_] maps panel tuple -> D."""
    M = len(y)
    cand = {}
    for s_ in SIGS:
        for k in KS:
            errs, store = [], {}
            for i in range(M):
                D = D_of[s_][tuple(cols_of[i])]
                refs = np.where(allowed[i])[0]
                Dref = D[:, refs].copy()
                for r, j in enumerate(refs):
                    Dref[j, r] = np.inf
                Dref[i] = D[i, refs]
                nn = np.argsort(Dref, 1)[:, :k]
                w = 1 / (np.take_along_axis(Dref, nn, 1) + 1e-12)
                store[i] = (w * y[refs][nn]).sum(1) / w.sum(1)
                errs.append(np.abs(store[i][refs] - y[refs]).mean())
            cand[(s_, k)] = (float(np.mean(errs)), store)
        for r_dim, alpha in RIDGE:
            errs, store = [], {}
            for i in range(M):
                D = D_of[s_][tuple(cols_of[i])]
                n_ = len(D)
                J = np.eye(n_) - 1 / n_
                Bm = -0.5 * J @ (D ** 2) @ J
                ew, ev = np.linalg.eigh(Bm)
                order = np.argsort(ew)[::-1][:r_dim]
                Z = ev[:, order] * np.sqrt(np.maximum(ew[order], 1e-12))
                refs = np.where(allowed[i])[0]
                Zr, yr = Z[refs], y[refs]
                G = Zr.T @ Zr + alpha * np.eye(r_dim)
                Gi = np.linalg.inv(G)
                beta = Gi @ (Zr.T @ (yr - yr.mean()))
                pred = np.clip(Z @ beta + yr.mean(), 0, 1)
                h = np.einsum('jr,rs,js->j', Zr, Gi, Zr)
                pr = Zr @ beta + yr.mean()
                loo = yr - (yr - pr) / np.maximum(1 - h, 1e-6)
                pred = pred.copy()
                pred[refs] = np.clip(loo, 0, 1)
                store[i] = pred
                errs.append(np.abs(pred[refs] - yr).mean())
            cand[(s_, 'r', r_dim, alpha)] = (float(np.mean(errs)), store)
    return min(cand.values(), key=lambda v: v[0])[1]


def geom_store(X, kern, y, allowed, cols):
    cols_of = {i: cols for i in range(len(y))}
    D_of = {s_: {tuple(cols): pkps_D(X, kern, cols, s_)} for s_ in SIGS}
    return _cv_select(D_of, cols_of, y, allowed)


def geom_store_adaptive(X, kern, y, allowed, cols_of):
    uniq = {tuple(cols_of[i]) for i in range(len(y))}
    D_of = {s_: {c: pkps_D(X, kern, np.asarray(c), s_) for c in uniq}
            for s_ in SIGS}
    return _cv_select(D_of, cols_of, y, allowed)


def pooled_alpha(models, B, y, allowed, cols_of, store):
    M = len(y)
    curves = np.zeros((M, len(ALPHAS)))
    for i in range(M):
        refs = np.where(allowed[i])[0]
        irt_ref = np.array([models[i].predict(cols_of[i],
                                              B[j, cols_of[i]])
                            for j in refs])
        curves[i] = np.abs(ALPHAS[None] * irt_ref[:, None]
                           + (1 - ALPHAS[None]) * store[i][refs, None]
                           - y[refs, None]).mean(0)
    return ALPHAS[int(curves.mean(0).argmin())]


def summarize(acc, M):
    return dict(mae=float(acc.mean()),
                sem=float(acc.std(ddof=1) / np.sqrt(M)),
                errs=[round(float(v), 6) for v in acc])


def stage_nsweep(out):
    """Panel A: irt + geom at n x m (no blend)."""
    systems, y, B, allowed, D2q, med, draws = load_base()
    M = len(y)
    X = load_X(EMB[0][2], EMB[0][3])
    kern = make_kern(X, D2q, med)
    res = out.setdefault('nsweepA', {})
    for n in NS:
        rn = res.setdefault(str(n), {})
        todo = [m for m in MS_A if str(m) not in rn
                or 'blend' not in rn[str(m)]]
        if not todo:
            print(f'A n={n}: cached', flush=True)
            continue
        allow_n = allowed & lib_mask(M, n)[None, :]
        models = [ItemModel(B[allow_n[i]], y[allow_n[i]], 10.0)
                  for i in range(M)]
        for m in todo:
            acc = {k: np.zeros(M) for k in ('irt', 'geom', 'blend')}
            for cols in draws[m]:
                cols_of = {i: cols for i in range(M)}
                irt_t = np.array([models[i].predict(cols, B[i, cols])
                                  for i in range(M)])
                store = geom_store(X, kern, y, allow_n, cols)
                a = pooled_alpha(models, B, y, allow_n, cols_of, store)
                geo_t = np.array([store[i][i] for i in range(M)])
                acc['irt'] += np.abs(irt_t - y) / B_DRAWS
                acc['geom'] += np.abs(geo_t - y) / B_DRAWS
                acc['blend'] += np.abs(a * irt_t + (1 - a) * geo_t
                                       - y) / B_DRAWS
            rn[str(m)] = {k: summarize(v, M) for k, v in acc.items()}
            json.dump(out, open(OUT_JSON, 'w'), indent=1)
            print(f'A n={n} m={m}:',
                  {k: round(v.mean(), 4) for k, v in acc.items()},
                  flush=True)
    return out


def stage_regime20(out):
    """Panel B extra: irt + blend at n=20, random and adaptive."""
    systems, y, B, allowed, D2q, med, draws = load_base()
    M = len(y)
    X = load_X(EMB[0][2], EMB[0][3])
    kern = make_kern(X, D2q, med)
    allow_n = allowed & lib_mask(M, 20)[None, :]
    models = [ItemModel(B[allow_n[i]], y[allow_n[i]], 10.0)
              for i in range(M)]

    res = out.setdefault('regime20', {})
    if 'random' in res and 'geom' not in res['random']['1']:
        del res['random']
    if 'adaptive' in res and 'geom' not in res['adaptive']['1']:
        del res['adaptive']
    if 'random' not in res:
        rr = {}
        for m in MS:
            acc = {k: np.zeros(M) for k in ('irt', 'geom', 'blend')}
            for cols in draws[m]:
                cols_of = {i: cols for i in range(M)}
                irt_t = np.array([models[i].predict(cols, B[i, cols])
                                  for i in range(M)])
                store = geom_store(X, kern, y, allow_n, cols)
                a = pooled_alpha(models, B, y, allow_n, cols_of, store)
                geo_t = np.array([store[i][i] for i in range(M)])
                acc['irt'] += np.abs(irt_t - y) / B_DRAWS
                acc['geom'] += np.abs(geo_t - y) / B_DRAWS
                acc['blend'] += np.abs(a * irt_t + (1 - a) * geo_t
                                       - y) / B_DRAWS
            rr[str(m)] = {k: summarize(v, M) for k, v in acc.items()}
            print(f'B rand20 m={m}:',
                  {k: round(v.mean(), 4) for k, v in acc.items()},
                  flush=True)
        res['random'] = rr
        json.dump(out, open(OUT_JSON, 'w'), indent=1)

    if 'adaptive' not in res:
        adaptive = [models[i].adaptive_path(B[i]) for i in range(M)]
        ra = {}
        for m in MS:
            cols_of = {i: np.array(adaptive[i][0][:m]) for i in range(M)}
            irt_t = np.array([adaptive[i][1][m - 1] for i in range(M)])
            store = geom_store_adaptive(X, kern, y, allow_n, cols_of)
            a = pooled_alpha(models, B, y, allow_n, cols_of, store)
            geo_t = np.array([store[i][i] for i in range(M)])
            e = {'irt': np.abs(irt_t - y),
                 'geom': np.abs(geo_t - y),
                 'blend': np.abs(a * irt_t + (1 - a) * geo_t - y)}
            ra[str(m)] = {k: summarize(v, M) for k, v in e.items()}
            print(f'B adap20 m={m}:',
                  {k: round(v.mean(), 4) for k, v in e.items()},
                  flush=True)
        res['adaptive'] = ra
        json.dump(out, open(OUT_JSON, 'w'), indent=1)
    return out


def _emb_work(args):
    short, path, key, n = args
    systems, y, B, allowed, D2q, med, draws = load_base()
    M = len(y)
    if n < M:
        allowed = allowed & lib_mask(M, n)[None, :]
    X = load_X(path, key)
    kern = make_kern(X, D2q, med)
    res = {}
    for m in MS:
        acc = np.zeros(M)
        for cols in draws[m]:
            store = geom_store(X, kern, y, allowed, cols)
            acc += np.abs(np.array([store[i][i] for i in range(M)])
                          - y) / B_DRAWS
        res[str(m)] = summarize(acc, M)
        print(f'C {short} n={n} m={m}: {acc.mean():.4f}', flush=True)
    return short, n, res


def stage_embedders(out):
    res107 = out.setdefault('embedders', {})
    res20 = out.setdefault('embedders20', {})
    todo = [(s, p, k, 107) for s, _, p, k in EMB if s not in res107]
    todo += [(s, p, k, 20) for s, _, p, k in EMB if s not in res20]
    if not todo:
        print('C: cached', flush=True)
        return out
    with get_context('fork').Pool(min(7, len(todo))) as pool:
        for short, n, r in pool.imap_unordered(_emb_work, todo):
            (res107 if n == 107 else res20)[short] = r
            json.dump(out, open(OUT_JSON, 'w'), indent=1)
    return out


def render(out):
    import matplotlib
    matplotlib.use('Agg')
    import hv_style
    hv_style.apply()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    SZ = hv_style.SIZES
    ink = hv_style.INK
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 3.9), sharey=True)
    M_STYLE = {1: dict(lw=1.5, alpha=.5), 5: dict(lw=2.4, alpha=.75),
               20: dict(lw=3.3, alpha=1.0)}

    # panel A: reference-library size; m as weight/alpha gradient
    ax = axes[0]
    ns = [int(n) for n in NS]
    for key, role in (('irt', 'baseline_pale'), ('geom', 'focus'),
                      ('blend', 'anchor')):
        st = hv_style.ROLES[role]
        for m in MS_A:
            mae = np.array([out['nsweepA'][str(n)][str(m)][key]['mae']
                            for n in ns])
            ax.plot(ns, mae, color=st['color'], ls=st['ls'],
                    marker='o', ms=4.5, **M_STYLE[m])
    ax.set_xscale('log')
    ax.set_xticks(ns)
    ax.set_xticklabels(ns)
    ax.set_xlabel('number of reference systems $n$',
                  fontsize=SZ['subtitle'])
    ax.set_title('reference-library size', fontsize=SZ['subtitle'],
                 color=hv_style.INK_TITLE)
    ax.set_ylabel('MAE$(\\hat{y}, y)$', fontsize=SZ['subtitle'])
    ax.legend(handles=[Line2D([], [], color=ink, label=f'$m={m}$',
                              **M_STYLE[m]) for m in MS_A],
              fontsize=SZ['annot'] - 1, frameon=False, handlelength=1.9,
              labelspacing=.3, loc='lower left',
              bbox_to_anchor=(0.02, 0.02))

    # panel B: probe selection; fill = regime, style = n
    ax = axes[1]
    rand107 = json.load(open('figures/q100_protocols.json'))[
        'protocols']['family']['by_m']
    adap107 = json.load(open('figures/q100_adaptive_family.json'))['by_m']
    srcs = {('random', 107): rand107, ('adaptive', 107): adap107,
            ('random', 20): out['regime20']['random'],
            ('adaptive', 20): out['regime20']['adaptive']}
    st = hv_style.ROLES['focus']
    for (regime, n), src in srcs.items():
        mae = np.array([src[str(m)]['geom']['mae'] for m in MS])
        filled = regime == 'adaptive'
        ax.plot(MS, mae, color=st['color'],
                ls='-' if n == 107 else '--',
                lw=2.6 if n == 107 else 2.0, marker='o', ms=6,
                markerfacecolor=st['color'] if filled else 'white',
                markeredgecolor=st['color'], markeredgewidth=1.3)
    ax.set_xscale('log')
    ax.set_xticks(MS)
    ax.set_xticklabels(MS)
    ax.set_xlabel('number of tasks $m$', fontsize=SZ['subtitle'])
    ax.set_title('probe selection (qubric geometry)',
                 fontsize=SZ['subtitle'], color=hv_style.INK_TITLE)

    # panel C: embedding model; style = n
    ax = axes[2]
    shades = hv_style.CMAP_BLUE(np.linspace(.35, 1.0, len(EMB)))
    for (short, nice, _, _), col in zip(EMB, shades):
        for src, ls in ((out['embedders'], '-'),
                        (out.get('embedders20', {}), '--')):
            r = src.get(short)
            if r is None:
                continue
            mae = np.array([r[str(m)]['mae'] for m in MS])
            ax.plot(MS, mae, color=col, lw=1.8, ls=ls, marker='o',
                    ms=3.5, label=nice if ls == '-' else None)
    ax.set_xscale('log')
    ax.set_xticks(MS)
    ax.set_xticklabels(MS)
    ax.set_xlabel('number of tasks $m$', fontsize=SZ['subtitle'])
    ax.set_title('embedding model (qubric geometry)',
                 fontsize=SZ['subtitle'], color=hv_style.INK_TITLE)
    ax.legend(fontsize=SZ['annot'] - 2, ncol=2, frameon=False,
              handlelength=1.4, labelspacing=.25, columnspacing=.8,
              loc='lower left', bbox_to_anchor=(0.02, 0.02))

    for ax in axes:
        ax.tick_params(labelsize=SZ['tick'])
    axes[0].set_ylim(0, .155)
    axes[0].set_yticks([0, .05, .1, .15])

    meth = [Line2D([], [], color=hv_style.ROLES[r]['color'],
                   ls=hv_style.ROLES[r]['ls'],
                   lw=hv_style.ROLES[r].get('lw', 2.6), label=lab)
            for r, lab in (('baseline_pale', 'IRT (2PL)'),
                           ('focus', 'qubric geometry'),
                           ('anchor', 'qubric + IRT blend'))]
    reg = [Line2D([], [], color=ink, lw=0, marker='o', ms=6,
                  markerfacecolor=ink, label='adaptive probes'),
           Line2D([], [], color=ink, lw=0, marker='o', ms=6,
                  markerfacecolor='white', markeredgecolor=ink,
                  markeredgewidth=1.3, label='random probes')]
    nn = [Line2D([], [], color=ink, lw=2.2, ls='-', label='$n=107$'),
          Line2D([], [], color=ink, lw=2.0, ls='--', label='$n=20$')]
    fig.legend(handles=meth + reg + nn, fontsize=SZ['legend'] - 3,
               handlelength=2.2, loc='upper center',
               bbox_to_anchor=(0.5, 0.02), ncol=7, columnspacing=.9,
               frameon=False)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    for ext in ('png', 'pdf'):
        fig.savefig(f'figures/fig_ablations.{ext}', dpi=200,
                    bbox_inches='tight', pad_inches=0.03)
    print('wrote figures/fig_ablations.png')


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    out = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    if stage in ('nsweep', 'all'):
        out = stage_nsweep(out)
    if stage in ('regime20', 'all'):
        out = stage_regime20(out)
    if stage in ('embedders', 'all'):
        out = stage_embedders(out)
    if stage in ('render', 'all'):
        render(out)
