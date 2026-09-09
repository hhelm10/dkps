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
import sys

import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
from outcome_baselines import ItemModel, load_panel  # noqa: E402
from pillars import harness_tag  # noqa: E402
from dkps.traces.qubric import consensus_center  # noqa: E402

SIGS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
KS = (3, 5)
ALPHAS = np.linspace(0, 1, 101)
MS = (1, 3, 5, 10, 20)
B_DRAWS = 20
OUT_JSON = 'figures/q100_protocols.json'
OUT_PNG = 'figures/fig2_protocols.png'


def build_masks(systems, allowed_llm):
    M = len(systems)
    scaf = [harness_tag(s) for s in systems]
    loso = np.array([[j != i for j in range(M)] for i in range(M)])
    loho = np.array(
        [[j != i and not (scaf[i] and scaf[j] == scaf[i]) for j in range(M)]
         for i in range(M)])
    return {'system': loso, 'llm': allowed_llm, 'harness': loho}


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
    for name, allowed in masks.items():
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
            (s_b, k_b), (_, store) = min(cand.items(), key=lambda kv: kv[1][0])
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
            res['by_m'][m] = {n: dict(mae=float(acc[n].mean()), ci=ci(acc[n]))
                              for n in acc}
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
                tgt = min(cand.values(), key=lambda v: v[0])[1]
                acc += np.abs(tgt - y) / B_DRAWS
            out['protocols'][name]['by_m'][str(m)]['raw'] = dict(
                mae=float(acc.mean()), ci=ci(acc))
            print(m, round(acc.mean(), 4))
        json.dump(out, open(OUT_JSON, 'w'), indent=2)
    print(f'wrote {OUT_JSON}')


COLS = [('system', 'Leave-one-system-out'),
        ('llm', 'Leave-one-LLM-out'),
        ('harness', 'Leave-one-harness-out')]
SERIES = [('sample', 'Sample Score', '#8c8c8c', 'o'),
          ('raw', 'raw-trace geometry', '#555555', 'v'),
          ('irt', 'IRT (2PL)', '#e08214', 's'),
          ('geom', 'qubric geometry', '#2c7fb8', '^'),
          ('blend', 'qubric + IRT blend', '#c51b7d', 'D')]


def main_render():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    d = json.load(open(OUT_JSON))
    ms = d['ms']
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), sharex=True,
                             sharey='row')
    for c, (key, title) in enumerate(COLS):
        ax = axes[0, c]
        p = d['protocols'].get(key)
        if p is None:
            ax.text(.5, .5, 'pending', ha='center', va='center',
                    transform=ax.transAxes, color='.6')
            continue
        pop = p['pop']['mae']
        ax.axhline(pop, color='.75', lw=1.2, ls=':', zorder=1)
        ax.text(ms[-1], pop, ' Pop. Mean', fontsize=7, color='.45',
                va='bottom', ha='right')
        for name, label, color, marker in SERIES:
            if name not in p['by_m'][str(ms[0])]:
                continue
            mae = [p['by_m'][str(m)][name]['mae'] for m in ms]
            lo = [p['by_m'][str(m)][name]['ci'][0] for m in ms]
            hi = [p['by_m'][str(m)][name]['ci'][1] for m in ms]
            ax.plot(ms, mae, color=color, marker=marker, ms=4, lw=1.6,
                    label=label, zorder=3)
            ax.fill_between(ms, lo, hi, color=color, alpha=.12, lw=0, zorder=2)
        ax.set_title(title, fontsize=10)
        ax.set_xscale('log')
        ax.set_xticks(ms)
        ax.set_xticklabels(ms)
        ax.tick_params(labelsize=8)
        ax.grid(True, color='.92', lw=.6)
        ax.set_axisbelow(True)
    axes[0, 0].set_ylabel('SWE-bench Verified\nMAE', fontsize=9)
    axes[0, 0].set_ylim(0, .5)
    for c in range(3):
        ax = axes[1, c]
        ax.set_facecolor('.97')
        ax.text(.5, .5, 'Terminal-Bench\n(to be run)', ha='center',
                va='center', transform=ax.transAxes, color='.55', fontsize=10)
        ax.set_xscale('log')
        ax.set_xticks(ms)
        ax.set_xticklabels(ms)
        ax.set_yticks([])
        ax.tick_params(labelsize=8)
        ax.set_xlabel('Number of probe tasks $m$', fontsize=9)
    axes[1, 0].set_ylabel('Terminal-Bench\nMAE', fontsize=9)
    axes[0, 0].legend(fontsize=7.5, frameon=False, loc='upper right')
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200)
    print(f'wrote {OUT_PNG}')


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if stage in ('compute', 'all'):
        main_compute()
    if stage in ('raw', 'all'):
        main_raw()
    if stage in ('render', 'all'):
        main_render()
