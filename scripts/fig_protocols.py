"""Figure 2: protocol-robustness grid (per HH 2026-09-08).

Row 1 = SWE-bench Verified (q100 era); row 2 = Terminal-Bench (to be run;
rendered as placeholders). Columns = reference-exclusion protocols:
  col 1  leave-one-system-out   (only the target itself excluded;
                                 same-LLM and same-harness siblings allowed)
  col 2  leave-one-LLM-out      (canonical: references sharing the target's
                                 underlying LLM excluded)
  col 3  leave-one-harness-out  (references sharing the target's scaffold
                                 excluded; same-LLM allowed)

Each panel: MAE vs probe budget m under random probes (B=20 shared draws
across protocols) for population mean, sample score, 2PL IRT, qubric PKPS
geometry, and qubric + IRT blend. Kernel (sigma, k) and blend alpha are
pooled-selected per protocol per draw from honest reference errors, exactly
as in q100_loho.py / q100_final_table.py.

Stages: compute -> figures/q100_protocols.json; render -> figures/fig2_protocols.png.
Usage: python scripts/fig_protocols.py [compute|render|all]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
from outcome_baselines import ItemModel, load_panel  # noqa: E402
from pillars import harness_tag, vendor_tag  # noqa: E402
from dkps.traces.qubric import consensus_center  # noqa: E402

SIGS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
KS = (3, 5)
ALPHAS = np.linspace(0, 1, 101)
MS = (1, 3, 5, 10, 20)
B_DRAWS = 50
OUT_JSON = 'figures/q100_protocols.json'
OUT_PNG = 'figures/fig2_protocols.png'


def build_masks(systems, allowed_llm):
    M = len(systems)
    scaf = [harness_tag(s) for s in systems]
    labels = json.load(open('data/leaderboard/verified_labels.json'))
    fam = [vendor_tag(labels, s) for s in systems]
    loso = np.array([[j != i for j in range(M)] for i in range(M)])
    loho = np.array(
        [[j != i and not (scaf[i] and scaf[j] == scaf[i]) for j in range(M)]
         for i in range(M)])
    lofo = np.array(
        [[j != i and not (fam[i] and fam[j] == fam[i]) for j in range(M)]
         for i in range(M)])
    return {'system': loso, 'llm': allowed_llm, 'family': lofo,
            'harness': loho}


def main_compute():
    systems, q100, y, B, allowed_llm = load_panel(panel='data/judge/q100.json')
    M, Q = B.shape
    masks = build_masks(systems, allowed_llm)
    for name, a in masks.items():
        print(f'{name}: mean references per target {a.sum(1).mean():.1f}')

    Xc = consensus_center(np.load('data/judge/q100_emb_openai_small.npz')['X'],
                          np.tile(np.arange(Q), M)) \
        .reshape(M, Q, -1).astype(np.float32)
    z = np.load('data/leaderboard/query_vecs_64.npz', allow_pickle=True)
    ids = [str(x) for x in z['ids']]
    V = np.asarray(z['vecs'], np.float32)[[ids.index(q) for q in q100]]
    D2q = ((V[:, None] - V[None]) ** 2).sum(-1)
    med = np.median(D2q)
    kern = {}
    for s_ in SIGS:
        KQ = np.exp(-D2q / (2 * med / s_))
        W = np.einsum('qp,jpd->jqd', KQ, Xc)
        kern[s_] = (KQ, W, np.einsum('jqd,jqd->j', Xc, W) / KQ.sum())

    def pkps_D(cols, s_):
        KQ, W, Arr = kern[s_]
        A_tr = np.einsum('iqd,jqd->ij', Xc[:, cols], W[:, cols]) / KQ[cols].sum()
        KQc = KQ[np.ix_(cols, cols)]
        Wc = np.einsum('qp,jpd->jqd', KQc, Xc[:, cols])
        Att = np.einsum('iqd,iqd->i', Xc[:, cols], Wc) / KQc.sum()
        return np.sqrt(np.maximum(Att[:, None] + Arr[None] - 2 * A_tr, 0))

    rng_b = np.random.default_rng(1)

    def ci(e):
        v = np.array([e[rng_b.integers(0, M, M)].mean() for _ in range(2000)])
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    # shared probe draws across protocols so columns are comparable
    rng = np.random.default_rng(0)
    draws = {m: [rng.choice(Q, m, replace=False) for _ in range(B_DRAWS)]
             for m in MS}

    out = {'ms': list(MS), 'b_draws': B_DRAWS, 'protocols': {}}
    if os.path.exists(OUT_JSON):
        out = json.load(open(OUT_JSON))
    for name, allowed in masks.items():
        if name in out['protocols']:
            print(f'=== protocol: {name} cached, skipping ===')
            continue
        print(f'=== protocol: {name} ===')
        models = [ItemModel(B[allowed[i]], y[allowed[i]], 10.0)
                  for i in range(M)]
        pop_t = np.array([y[allowed[i]].mean() for i in range(M)])
        e_pop = np.abs(pop_t - y)

        def eval_cols(cols, irt_t):
            # pooled (sigma, k) from honest reference errors for this draw
            cand = {}
            for s_ in SIGS:
                D = pkps_D(cols, s_)
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
                # ridge on classical-MDS coords, LOO-honest ref preds
                # (same estimator family as tb2_eval; CV decides per draw)
                n_ = len(D)
                J = np.eye(n_) - 1 / n_
                Bmat = -0.5 * J @ (D ** 2) @ J
                ew, evec = np.linalg.eigh(Bmat)
                order_ = np.argsort(ew)[::-1]
                for r_dim, alpha in ((16, 0.1), (16, 1.0), (8, 0.1)):
                    Z = evec[:, order_[:r_dim]] * np.sqrt(
                        np.maximum(ew[order_[:r_dim]], 1e-12))
                    errs, store2 = [], {}
                    for i in range(M):
                        refs = np.where(allowed[i])[0]
                        Zr, yr = Z[refs], y[refs]
                        G = Zr.T @ Zr + alpha * np.eye(r_dim)
                        Ginv = np.linalg.inv(G)
                        beta = Ginv @ (Zr.T @ (yr - yr.mean()))
                        pred = np.clip(Z @ beta + yr.mean(), 0, 1)
                        h = np.einsum('jr,rs,js->j', Zr, Ginv, Zr)
                        pr = Zr @ beta + yr.mean()
                        loo = yr - (yr - pr) / np.maximum(1 - h, 1e-6)
                        pred = pred.copy()
                        pred[refs] = np.clip(loo, 0, 1)
                        store2[i] = pred
                        errs.append(np.abs(pred[refs] - yr).mean())
                    cand[(s_, 'r', r_dim, alpha)] = (float(np.mean(errs)),
                                                     store2)
            _, (_, store) = min(cand.items(), key=lambda kv: kv[1][0])
            curves = np.zeros((M, len(ALPHAS)))
            for i in range(M):
                refs = np.where(allowed[i])[0]
                irt_ref = np.array([models[i].predict(cols, B[j, cols])
                                    for j in refs])
                curves[i] = np.abs(ALPHAS[None] * irt_ref[:, None]
                                   + (1 - ALPHAS[None]) * store[i][refs, None]
                                   - y[refs, None]).mean(0)
            a = ALPHAS[int(curves.mean(0).argmin())]
            geo_t = np.array([store[i][i] for i in range(M)])
            return (np.abs(irt_t - y), np.abs(geo_t - y),
                    np.abs(a * irt_t + (1 - a) * geo_t - y))

        res = {'pop': dict(mae=float(e_pop.mean()), ci=ci(e_pop)), 'by_m': {}}
        for m in MS:
            acc = {n: np.zeros(M)
                   for n in ('sample', 'irt', 'geom', 'blend')}
            for cols in draws[m]:
                acc['sample'] += np.abs(B[:, cols].mean(1) - y) / B_DRAWS
                irt_t = np.array([models[i].predict(cols, B[i, cols])
                                  for i in range(M)])
                e_irt, e_geo, e_bl = eval_cols(cols, irt_t)
                acc['irt'] += e_irt / B_DRAWS
                acc['geom'] += e_geo / B_DRAWS
                acc['blend'] += e_bl / B_DRAWS
            res['by_m'][m] = {n: dict(mae=float(acc[n].mean()), ci=ci(acc[n]),
                                       sem=float(acc[n].std(ddof=1)
                                                 / np.sqrt(M)))
                              for n in acc}
            res['by_m'][m]['errs'] = {n: [round(float(v), 6) for v in acc[n]]
                                      for n in acc}
            d_bi = acc['blend'] - acc['irt']
            res['by_m'][m]['delta_blend_irt'] = dict(mean=float(d_bi.mean()),
                                                     ci=ci(d_bi))
            print(m, {n: round(acc[n].mean(), 4) for n in acc})
        out['protocols'][name] = res
        json.dump(out, open(OUT_JSON, 'w'), indent=2)  # checkpoint per protocol
    print(f'wrote {OUT_JSON}')


def main_raw():
    """Add a raw-trace-embedding geometry curve per protocol (the
    off-the-shelf null hypothesis): head+tail 8K-token slices of the
    unpruned render, text-embedding-3-small, median-center + L2 -- same
    kNN/kernel machinery and the same probe draws as main_compute."""
    systems, q100, y, B, allowed_llm = load_panel(panel='data/judge/q100.json')
    M, Q = B.shape
    masks = build_masks(systems, allowed_llm)
    HT = np.load('data/judge/q100_raw_emb_openai_small.npz')['HT'] \
        .reshape(M, Q, -1)
    Xc = HT - np.median(HT, axis=0, keepdims=True)
    Xc /= np.maximum(np.linalg.norm(Xc, axis=-1, keepdims=True), 1e-9)
    Xc = Xc.astype(np.float32)

    z = np.load('data/leaderboard/query_vecs_64.npz', allow_pickle=True)
    ids = [str(x) for x in z['ids']]
    V = np.asarray(z['vecs'], np.float32)[[ids.index(q) for q in q100]]
    D2q = ((V[:, None] - V[None]) ** 2).sum(-1)
    med = np.median(D2q)
    kern = {}
    for s_ in SIGS:
        KQ = np.exp(-D2q / (2 * med / s_))
        W = np.einsum('qp,jpd->jqd', KQ, Xc)
        kern[s_] = (KQ, W, np.einsum('jqd,jqd->j', Xc, W) / KQ.sum())

    def pkps_D(cols, s_):
        KQ, W, Arr = kern[s_]
        A_tr = np.einsum('iqd,jqd->ij', Xc[:, cols], W[:, cols]) / KQ[cols].sum()
        KQc = KQ[np.ix_(cols, cols)]
        Wc = np.einsum('qp,jpd->jqd', KQc, Xc[:, cols])
        Att = np.einsum('iqd,iqd->i', Xc[:, cols], Wc) / KQc.sum()
        return np.sqrt(np.maximum(Att[:, None] + Arr[None] - 2 * A_tr, 0))

    rng_b = np.random.default_rng(1)

    def ci(e):
        v = np.array([e[rng_b.integers(0, M, M)].mean() for _ in range(2000)])
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    rng = np.random.default_rng(0)
    draws = {m: [rng.choice(Q, m, replace=False) for _ in range(B_DRAWS)]
             for m in MS}

    out = json.load(open(OUT_JSON))
    for name, allowed in masks.items():
        if 'raw' in out['protocols'].get(name, {}).get('by_m', {}) \
                .get(str(MS[0]), {}):
            print(f'=== raw geometry: {name} cached, skipping ===')
            continue
        print(f'=== raw geometry: {name} ===')
        for m in MS:
            acc = np.zeros(M)
            for cols in draws[m]:
                cand = {}
                for s_ in SIGS:
                    D = pkps_D(cols, s_)
                    for k in KS:
                        errs, tgt = [], np.zeros(M)
                        for i in range(M):
                            refs = np.where(allowed[i])[0]
                            Dref = D[:, refs].copy()
                            for r, j in enumerate(refs):
                                Dref[j, r] = np.inf
                            Dref[i] = D[i, refs]
                            nn = np.argsort(Dref, 1)[:, :k]
                            w = 1 / (np.take_along_axis(Dref, nn, 1) + 1e-12)
                            pred = (w * y[refs][nn]).sum(1) / w.sum(1)
                            errs.append(np.abs(pred[refs] - y[refs]).mean())
                            tgt[i] = pred[i]
                        cand[(s_, k)] = (float(np.mean(errs)), tgt)
                    # LOO-honest ridge on MDS coords (parity with qubric)
                    n_ = len(D)
                    J = np.eye(n_) - 1 / n_
                    Bmat = -0.5 * J @ (D ** 2) @ J
                    ew, evec = np.linalg.eigh(Bmat)
                    order_ = np.argsort(ew)[::-1]
                    for r_dim, alpha in ((16, 0.1), (16, 1.0), (8, 0.1)):
                        Z = evec[:, order_[:r_dim]] * np.sqrt(
                            np.maximum(ew[order_[:r_dim]], 1e-12))
                        errs, tgt = [], np.zeros(M)
                        for i in range(M):
                            refs = np.where(allowed[i])[0]
                            Zr, yr = Z[refs], y[refs]
                            G = Zr.T @ Zr + alpha * np.eye(r_dim)
                            Ginv = np.linalg.inv(G)
                            beta = Ginv @ (Zr.T @ (yr - yr.mean()))
                            tgt[i] = float(np.clip(
                                Z[i] @ beta + yr.mean(), 0, 1))
                            h = np.einsum('jr,rs,js->j', Zr, Ginv, Zr)
                            pr = Zr @ beta + yr.mean()
                            loo = yr - (yr - pr) / np.maximum(1 - h, 1e-6)
                            errs.append(np.abs(np.clip(loo, 0, 1)
                                               - yr).mean())
                        cand[(s_, 'r', r_dim, alpha)] = \
                            (float(np.mean(errs)), tgt)
                tgt = min(cand.values(), key=lambda v: v[0])[1]
                acc += np.abs(tgt - y) / B_DRAWS
            out['protocols'][name]['by_m'][str(m)]['raw'] = dict(
                mae=float(acc.mean()), ci=ci(acc))
            out['protocols'][name]['by_m'][str(m)].setdefault(
                'errs', {})['raw'] = [round(float(v), 6) for v in acc]
            print(m, round(acc.mean(), 4))
        json.dump(out, open(OUT_JSON, 'w'), indent=2)
    print(f'wrote {OUT_JSON}')


def load_embeddings():
    """(Xc_qubric, Xr_raw, V64) for the q100 panel, both (M, Q, d)."""
    systems, q100, y, B, allowed_llm = load_panel(panel='data/judge/q100.json')
    M, Q = B.shape
    Xc = consensus_center(np.load('data/judge/q100_emb_openai_small.npz')['X'],
                          np.tile(np.arange(Q), M)) \
        .reshape(M, Q, -1).astype(np.float32)
    HT = np.load('data/judge/q100_raw_emb_openai_small.npz')['HT'] \
        .reshape(M, Q, -1)
    Xr = HT - np.median(HT, axis=0, keepdims=True)
    Xr /= np.maximum(np.linalg.norm(Xr, axis=-1, keepdims=True), 1e-9)
    z = np.load('data/leaderboard/query_vecs_64.npz', allow_pickle=True)
    ids = [str(x) for x in z['ids']]
    V = np.asarray(z['vecs'], np.float32)[[ids.index(q) for q in q100]]
    return systems, q100, y, B, allowed_llm, Xc, Xr.astype(np.float32), V


def main_adaptive():
    """Adaptive-probe (simulated CAT) version of the protocol grid.

    Per protocol: refit 2PL per target on the protocol's allowed refs,
    take each target's adaptive item path, then evaluate raw geometry,
    qubric geometry, and the qubric+IRT blend on the per-target panels
    with pooled (sigma, k, alpha) from honest reference errors -- the
    adaptive analogue of main_compute. Writes
    figures/q100_protocols_adaptive.json.
    """
    OUT = 'figures/q100_protocols_adaptive.json'
    systems, q100, y, B, allowed_llm, Xc, Xr, V = load_embeddings()
    M, Q = B.shape
    masks = build_masks(systems, allowed_llm)
    D2q = ((V[:, None] - V[None]) ** 2).sum(-1)
    med = np.median(D2q)

    kerns = {}
    for tag, X in (('qubric', Xc), ('raw', Xr)):
        for s_ in SIGS:
            KQ = np.exp(-D2q / (2 * med / s_))
            W = np.einsum('qp,jpd->jqd', KQ, X, optimize=True)
            kerns[(tag, s_)] = (KQ, W,
                                np.einsum('jqd,jqd->j', X, W) / KQ.sum())
    Xof = {'qubric': Xc, 'raw': Xr}

    def pkps_D(tag, cols, s_):
        X = Xof[tag]
        KQ, W, Arr = kerns[(tag, s_)]
        Xi = X[:, cols].reshape(M, -1)
        A_tr = (Xi @ W[:, cols].reshape(M, -1).T) / KQ[cols].sum()
        KQc = KQ[np.ix_(cols, cols)]
        Wc = np.einsum('qp,jpd->jqd', KQc, X[:, cols], optimize=True)
        Att = np.einsum('jqd,jqd->j', X[:, cols], Wc) / KQc.sum()
        return np.sqrt(np.maximum(Att[:, None] + Arr[None] - 2 * A_tr, 0))

    rng_b = np.random.default_rng(1)

    def ci(e):
        v = np.array([e[rng_b.integers(0, M, M)].mean() for _ in range(2000)])
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    out = {'ms': list(MS), 'protocols': {}}
    if os.path.exists(OUT):
        out = json.load(open(OUT))
    for name, allowed in masks.items():
        if name in out['protocols']:
            print(f'=== {name}: cached, skipping ===')
            continue
        print(f'=== adaptive protocol: {name} ===')
        models = [ItemModel(B[allowed[i]], y[allowed[i]], 10.0)
                  for i in range(M)]
        adaptive = [models[i].adaptive_path(B[i]) for i in range(M)]
        pop_t = np.array([y[allowed[i]].mean() for i in range(M)])

        def geom_errs(tag, cols_of):
            cand = {}
            for s_ in SIGS:
                Dc = {tuple(c): pkps_D(tag, np.asarray(c), s_)
                      for c in {tuple(cols_of[i]) for i in range(M)}}
                for k in KS:
                    errs, store = [], {}
                    for i in range(M):
                        D = Dc[tuple(cols_of[i])]
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
            return min(cand.values(), key=lambda v: v[0])[1]

        res = out['protocols'].setdefault(
            name, {'pop': dict(mae=float(np.abs(pop_t - y).mean()),
                               ci=ci(np.abs(pop_t - y))), 'by_m': {}})
        for m in MS:
            cols_of = {i: np.array(adaptive[i][0][:m]) for i in range(M)}
            irt_t = np.array([adaptive[i][1][m - 1] for i in range(M)])
            store = geom_errs('qubric', cols_of)
            store_raw = geom_errs('raw', cols_of)
            curves = np.zeros((M, len(ALPHAS)))
            for i in range(M):
                refs = np.where(allowed[i])[0]
                irt_ref = np.array([models[i].predict(cols_of[i],
                                                      B[j, cols_of[i]])
                                    for j in refs])
                curves[i] = np.abs(ALPHAS[None] * irt_ref[:, None]
                                   + (1 - ALPHAS[None]) * store[i][refs, None]
                                   - y[refs, None]).mean(0)
            a = ALPHAS[int(curves.mean(0).argmin())]
            geo_t = np.array([store[i][i] for i in range(M)])
            raw_t = np.array([store_raw[i][i] for i in range(M)])
            e = {'irt': np.abs(irt_t - y), 'geom': np.abs(geo_t - y),
                 'raw': np.abs(raw_t - y),
                 'blend': np.abs(a * irt_t + (1 - a) * geo_t - y)}
            res['by_m'][m] = {n: dict(mae=float(v.mean()), ci=ci(v))
                              for n, v in e.items()}
            print(m, {n: round(v.mean(), 4) for n, v in e.items()})
            json.dump(out, open(OUT, 'w'), indent=2)
    print(f'wrote {OUT}')


COLS = [('system', 'Leave-one-system-out'),
        ('family', 'Leave-one-family-out'),
        ('harness', 'Leave-one-harness-out')]
# Helivan Blues roles; dashed = score-only, solid = uses trace embeddings
import hv_style  # noqa: E402
SERIES = [('sample', 'Sample Score', 'baseline_gray'),
          ('irt', 'IRT (2PL)', 'baseline_pale'),
          ('raw', 'raw-trace geometry', 'comparator'),
          ('geom', 'qubric geometry', 'focus'),
          ('blend', 'qubric + IRT blend', 'anchor')]


def main_render(src=OUT_JSON, dst=OUT_PNG):
    import matplotlib
    matplotlib.use('Agg')
    hv_style.apply()
    import matplotlib.pyplot as plt

    SZ = hv_style.SIZES
    d = json.load(open(src))
    ms = d['ms']
    tb = None
    if os.path.exists('figures/tb2_protocols.json'):
        tb = json.load(open('figures/tb2_protocols.json'))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.6), sharex=True,
                             sharey='row')

    def draw_row(r, data):
        for c, (key, title) in enumerate(COLS):
            ax = axes[r, c]
            p = data['protocols'].get(key)
            if p is None:
                ax.text(.5, .5, 'pending', ha='center', va='center',
                        transform=ax.transAxes, color=hv_style.INK_MUTE)
                continue
            pop = p['pop']['mae']
            ax.axhline(pop, color=hv_style.REFLINE, lw=1.2, ls='--',
                       zorder=1)
            ax.text(ms[-1], pop, ' Pop. Mean', fontsize=SZ['annot'],
                    color=hv_style.INK_MUTE, va='bottom', ha='right')
            for name, label, role in SERIES:
                if name not in p['by_m'][str(ms[0])]:
                    continue
                st = hv_style.ROLES[role]
                mae = np.array([p['by_m'][str(m)][name]['mae'] for m in ms])
                # house style: +/- 1 SEM shading (fallback: derive SEM
                # from the stored 95% bootstrap CI width)
                sem = np.array([
                    p['by_m'][str(m)][name].get(
                        'sem',
                        (p['by_m'][str(m)][name]['ci'][1]
                         - p['by_m'][str(m)][name]['ci'][0]) / 3.92)
                    for m in ms])
                ax.plot(ms, mae, color=st['color'], ls=st['ls'],
                        lw=st.get('lw', 2.6), marker='o', ms=4, label=label,
                        zorder=4 if role == 'anchor' else 3)
                ax.fill_between(ms, mae - sem, mae + sem, color=st['color'],
                                alpha=.15, lw=0, zorder=2)
            if r == 0:
                ax.set_title(title, fontsize=SZ['label'],
                             color=hv_style.INK_TITLE)
            else:
                ax.set_xlabel('Number of tasks $m$', fontsize=SZ['label'])
            ax.set_xscale('log')
            ax.set_xticks(ms)
            ax.set_xticklabels(ms)
            ax.set_ylim(0, .25)
            ax.set_yticks([0, .1, .2])
            ax.tick_params(labelsize=SZ['tick'])

    draw_row(0, d)
    if tb is not None:
        draw_row(1, tb)
    else:
        for c in range(3):
            ax = axes[1, c]
            ax.set_facecolor(hv_style.WASH)
            ax.text(.5, .5, 'Terminal-Bench\n(to be run)', ha='center',
                    va='center', transform=ax.transAxes,
                    color=hv_style.INK_MUTE, fontsize=SZ['subtitle'])
            ax.set_xlabel('Number of tasks $m$', fontsize=SZ['label'])
    axes[0, 0].set_ylabel('SWE-bench Verified\nMAE$(\\hat{y}, y)$',
                          fontsize=SZ['label'])
    axes[1, 0].set_ylabel('Terminal-Bench 2.0\nMAE$(\\hat{y}, y)$',
                          fontsize=SZ['label'])
    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_, fontsize=SZ['legend'], handlelength=3.2,
               loc='lower center', bbox_to_anchor=(0.5, -0.015), ncol=5,
               columnspacing=1.4)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(dst, dpi=200, bbox_inches='tight', pad_inches=0.02)
    print(f'wrote {dst}')


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if stage in ('compute', 'all'):
        main_compute()
    if stage in ('raw', 'all'):
        main_raw()
    if stage in ('render', 'all'):
        main_render()
    if stage == 'adaptive':
        main_adaptive()
    if stage in ('adaptive', 'adaptive-render'):
        main_render('figures/q100_protocols_adaptive.json',
                    'figures/fig2b_protocols_adaptive.png')
