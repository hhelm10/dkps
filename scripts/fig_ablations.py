"""Ablations under the FIXED canonical protocol (leave-one-family-out,
SWE-bench Verified) — the figure between the protocol grid and the cost
figure (per HH 2026-09-17):

  panel A  effect of reference-library size n (m=5, B=50 draws):
           IRT / qubric geometry / blend vs n in {10,20,40,70,107}
  panel B  effect of adaptive probes: IRT + blend vs m, filled=adaptive
           open=random (reads the existing protocol/adaptive jsons)
  panel C  effect of the embedding model: qubric geometry vs m for 7
           text encoders (openai + 6 local sentence-transformers)

Estimator stack identical to fig_protocols: consensus centering, PKPS
kernel, per-draw pooled CV over sigma x ({kNN 3,5} u ridge-on-MDS
LOO-honest), pooled alpha blend, shared rng(0) draw stream.

Stages: python scripts/fig_ablations.py nsweep|embedders|render|all
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
NS = (10, 20, 40, 70, 107)
M_FIX = 5
B_DRAWS = 50
OUT_JSON = 'figures/ablations.json'
EMB = [('openai', 'text-embedding-3-small',
        'data/judge/q100_emb_openai_small.npz', 'X'),
       ('nomic', 'nomic-embed-text-v1.5',
        'data/judge/pillars_emb_q100-qspec-flash0731_'
        'nomic-ai_nomic-embed-text-v1.5.npz', 'Xq'),
       ('bge', 'bge-large-en-v1.5',
        'data/judge/pillars_emb_q100-qspec-flash0731_'
        'BAAI_bge-large-en-v1.5.npz', 'Xq'),
       ('gte', 'gte-large',
        'data/judge/pillars_emb_q100-qspec-flash0731_'
        'thenlper_gte-large.npz', 'Xq'),
       ('e5', 'e5-large-v2',
        'data/judge/pillars_emb_q100-qspec-flash0731_'
        'intfloat_e5-large-v2.npz', 'Xq'),
       ('mpnet', 'all-mpnet-base-v2',
        'data/judge/pillars_emb_q100-qspec-flash0731_'
        'sentence-transformers_all-mpnet-base-v2.npz', 'Xq'),
       ('minilm', 'all-MiniLM-L6-v2',
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


def geom_store(X, kern, y, allowed, cols):
    """CV-selected per-target prediction stores (like fig_protocols)."""
    M = len(X)
    cand = {}
    for s_ in SIGS:
        D = pkps_D(X, kern, cols, s_)
        for k in KS:
            errs, store = [], {}
            for i in range(M):
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
        n_ = len(D)
        J = np.eye(n_) - 1 / n_
        Bm = -0.5 * J @ (D ** 2) @ J
        ew, ev = np.linalg.eigh(Bm)
        order = np.argsort(ew)[::-1]
        for r_dim, alpha in RIDGE:
            Z = ev[:, order[:r_dim]] * np.sqrt(
                np.maximum(ew[order[:r_dim]], 1e-12))
            errs, store = [], {}
            for i in range(M):
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


def summarize(acc, M):
    return dict(mae=float(acc.mean()),
                sem=float(acc.std(ddof=1) / np.sqrt(M)),
                errs=[round(float(v), 6) for v in acc])


def stage_nsweep(out):
    systems, y, B, allowed, D2q, med, draws = load_base()
    M = len(y)
    name, _, path, key = EMB[0]
    X = load_X(path, key)
    kern = make_kern(X, D2q, med)
    lib_order = np.random.default_rng(11).permutation(M)
    res = out.setdefault('nsweep', {})
    for n in NS:
        if str(n) in res:
            print(f'n={n}: cached', flush=True)
            continue
        inlib = np.zeros(M, bool)
        inlib[lib_order[:n]] = True
        allow_n = allowed & inlib[None, :]
        models = [ItemModel(B[allow_n[i]], y[allow_n[i]], 10.0)
                  for i in range(M)]
        acc = {k: np.zeros(M) for k in ('irt', 'geom', 'blend')}
        for cols in draws[M_FIX]:
            irt_t = np.array([models[i].predict(cols, B[i, cols])
                              for i in range(M)])
            store = geom_store(X, kern, y, allow_n, cols)
            curves = np.zeros((M, len(ALPHAS)))
            for i in range(M):
                refs = np.where(allow_n[i])[0]
                irt_ref = np.array([models[i].predict(cols, B[j, cols])
                                    for j in refs])
                curves[i] = np.abs(
                    ALPHAS[None] * irt_ref[:, None]
                    + (1 - ALPHAS[None]) * store[i][refs, None]
                    - y[refs, None]).mean(0)
            a = ALPHAS[int(curves.mean(0).argmin())]
            geo_t = np.array([store[i][i] for i in range(M)])
            acc['irt'] += np.abs(irt_t - y) / B_DRAWS
            acc['geom'] += np.abs(geo_t - y) / B_DRAWS
            acc['blend'] += np.abs(a * irt_t + (1 - a) * geo_t
                                   - y) / B_DRAWS
        res[str(n)] = {k: summarize(v, M) for k, v in acc.items()}
        json.dump(out, open(OUT_JSON, 'w'), indent=1)
        print(f'n={n}:', {k: round(v.mean(), 4) for k, v in acc.items()},
              flush=True)
    return out


def _emb_work(args):
    short, path, key = args
    systems, y, B, allowed, D2q, med, draws = load_base()
    M = len(y)
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
        print(f'{short} m={m}: {acc.mean():.4f}', flush=True)
    return short, res


def stage_embedders(out):
    res = out.setdefault('embedders', {})
    todo = [(s, p, k) for s, _, p, k in EMB if s not in res]
    if not todo:
        print('embedders: cached', flush=True)
        return out
    with get_context('fork').Pool(min(7, len(todo))) as pool:
        for short, r in pool.imap_unordered(_emb_work, todo):
            res[short] = r
            json.dump(out, open(OUT_JSON, 'w'), indent=1)
            print(f'{short}: done', flush=True)
    return out


def render(out):
    import matplotlib
    matplotlib.use('Agg')
    import hv_style
    hv_style.apply()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    SZ = hv_style.SIZES
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 3.9), sharey=True)

    # panel A: reference-library size
    ax = axes[0]
    ns = [int(n) for n in NS]
    for key, role in (('irt', 'baseline_pale'), ('geom', 'focus'),
                      ('blend', 'anchor')):
        st = hv_style.ROLES[role]
        mae = np.array([out['nsweep'][str(n)][key]['mae'] for n in ns])
        sem = np.array([out['nsweep'][str(n)][key]['sem'] for n in ns])
        ax.plot(ns, mae, color=st['color'], ls=st['ls'],
                lw=st.get('lw', 2.6), marker='o', ms=5)
        ax.fill_between(ns, mae - sem, mae + sem, color=st['color'],
                        alpha=.15, lw=0)
    ax.set_xscale('log')
    ax.set_xticks(ns)
    ax.set_xticklabels(ns)
    ax.set_xlabel('number of reference systems $n$',
                  fontsize=SZ['subtitle'])
    ax.set_title('reference-library size ($m=5$)',
                 fontsize=SZ['subtitle'], color=hv_style.INK_TITLE)
    ax.set_ylabel('MAE$(\\hat{y}, y)$', fontsize=SZ['subtitle'])

    # panel B: adaptive vs random probes
    ax = axes[1]
    rand = json.load(open('figures/q100_protocols.json'))[
        'protocols']['family']['by_m']
    adap = json.load(open('figures/q100_adaptive_family.json'))['by_m']
    for src, filled in ((rand, False), (adap, True)):
        for key, role in (('irt', 'baseline_pale'), ('blend', 'anchor')):
            st = hv_style.ROLES[role]
            mae = np.array([src[str(m)][key]['mae'] for m in MS])
            sem = np.array([src[str(m)][key].get('sem', 0) for m in MS])
            ax.plot(MS, mae, color=st['color'], ls=st['ls'],
                    lw=st.get('lw', 2.6), marker='o', ms=6.5,
                    markerfacecolor=st['color'] if filled else 'white',
                    markeredgecolor=st['color'], markeredgewidth=1.4)
            ax.fill_between(MS, mae - sem, mae + sem, color=st['color'],
                            alpha=.15, lw=0)
    ax.set_xscale('log')
    ax.set_xticks(MS)
    ax.set_xticklabels(MS)
    ax.set_xlabel('number of tasks $m$', fontsize=SZ['subtitle'])
    ax.set_title('probe selection', fontsize=SZ['subtitle'],
                 color=hv_style.INK_TITLE)

    # panel C: embedding model (qubric geometry only)
    ax = axes[2]
    shades = hv_style.CMAP_BLUE(np.linspace(.35, 1.0, len(EMB)))
    SHORT = {'openai': 'text-emb-3-small', 'nomic': 'nomic-v1.5',
             'bge': 'bge-large', 'gte': 'gte-large', 'e5': 'e5-large',
             'mpnet': 'mpnet-base', 'minilm': 'MiniLM-L6'}
    for (short, nice, _, _), col in zip(EMB, shades):
        r = out['embedders'][short]
        mae = np.array([r[str(m)]['mae'] for m in MS])
        lw = 3.0 if short == 'openai' else 1.8
        ax.plot(MS, mae, color=col, lw=lw, marker='o', ms=4,
                label=SHORT[short])
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
                   ls=hv_style.ROLES[r]['ls'], lw=hv_style.ROLES[r]
                   .get('lw', 2.6), label=lab)
            for r, lab in (('baseline_pale', 'IRT (2PL)'),
                           ('focus', 'qubric geometry'),
                           ('anchor', 'qubric + IRT blend'))]
    ink = hv_style.INK
    reg = [Line2D([], [], color=ink, lw=0, marker='o', ms=6.5,
                  markerfacecolor=ink, label='adaptive probes'),
           Line2D([], [], color=ink, lw=0, marker='o', ms=6.5,
                  markerfacecolor='white', markeredgecolor=ink,
                  markeredgewidth=1.4, label='random probes')]
    fig.legend(handles=meth + reg, fontsize=SZ['legend'] - 2,
               handlelength=2.6, loc='upper center',
               bbox_to_anchor=(0.5, 0.02), ncol=5, columnspacing=1.0,
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
    if stage in ('embedders', 'all'):
        out = stage_embedders(out)
    if stage in ('render', 'all'):
        render(out)
