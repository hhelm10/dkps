"""Additional q100 baselines (per HH 2026-09-04): the raw trace-embedding
null hypothesis under the identical blend protocol, plus CAPA.

  raw rep    head 32K + tail 32K chars of the UNPRUNED render
             (trace_texts_full_raw), embedded with text-embedding-3-small,
             per-instance median-centered + L2 -- the off-the-shelf default.
  regimes    informative (per-target Fisher panels) and random (B=20),
             pooled (sigma, k, alpha) on references, bootstrap CIs, paired
             deltas vs IRT and vs the qubric blend.
  capa       kappa-kNN over probe outcomes (F24), random B=20.

Stages: embed | eval.  Writes figures/q100_baselines.json.
"""
import argparse
import json
import os
import sys

import numpy as np
import requests
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
from outcome_baselines import ItemModel, load_panel  # noqa: E402

RAW = 'data/judge/trace_texts_full_raw'
EMB = 'data/judge/q100_raw_emb_openai_small.npz'
SIGS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
KS = (3, 5)
ALPHAS = np.linspace(0, 1, 101)
BOOT = 2000
MS = (1, 3, 5, 10, 20)
B_DRAWS = 20


def stage_embed(key):
    import time
    import tiktoken
    enc = tiktoken.get_encoding('cl100k_base')
    systems, q100, y, B, allowed = load_panel(panel='data/judge/q100.json')
    texts = []
    for s in systems:
        for q in q100:
            p = os.path.join(RAW, s, f'{q}.txt')
            t = open(p).read() if os.path.exists(p) else ' '
            toks = enc.encode(t, disallowed_special=())
            texts.append(enc.decode(toks[:8000]) or ' ')
            texts.append(enc.decode(toks[-8000:]) or ' ')
    rows = []
    for i in tqdm(range(0, len(texts), 16), desc='embed'):
        for attempt in range(8):
            try:
                r = requests.post('https://api.openai.com/v1/embeddings',
                                  json={'model': 'text-embedding-3-small',
                                        'input': [t or ' ' for t in texts[i:i+16]]},
                                  headers={'Authorization': f'Bearer {key}'},
                                  timeout=120)
            except requests.RequestException as e:
                print(f'net error at {i} (attempt {attempt}): {e}',
                      file=sys.stderr)
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code == 200:
                rows.extend(d['embedding'] for d in r.json()['data'])
                break
            if r.status_code == 400:
                raise RuntimeError(f'embed 400 at {i}: {r.text[:300]}')
            time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f'embed exhausted retries at {i}')
    E = np.asarray(rows, np.float32).reshape(len(texts) // 2, 2, -1)
    np.savez_compressed(EMB, HT=E)
    print('wrote', EMB, E.shape)


def stage_eval():
    systems, q100, y, B, allowed = load_panel(panel='data/judge/q100.json')
    M, Q = B.shape
    models = [ItemModel(B[allowed[i]], y[allowed[i]], 10.0) for i in range(M)]
    info_order = [mm.informative_order() for mm in models]
    HT = np.load(EMB)['HT'].reshape(M, Q, -1)
    inst = np.tile(np.arange(Q), M).reshape(M, Q)
    Xr = HT - np.median(HT, axis=0, keepdims=True)
    Xr /= np.maximum(np.linalg.norm(Xr, axis=-1, keepdims=True), 1e-9)
    Xc = Xr.astype(np.float32)

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

    def eval_cols(cols_of):
        irt_of = {i: np.array([models[j].predict(cols_of[i], B[j, cols_of[i]])
                               for j in range(M)]) for i in range(M)}
        cand = {}
        for s_ in SIGS:
            Dc = {}
            for i in range(M):
                key = tuple(cols_of[i])
                if key not in Dc:
                    Dc[key] = pkps_D(cols_of[i], s_)
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
        (s_b, k_b), (_, store) = min(cand.items(), key=lambda kv: kv[1][0])
        curves = np.zeros((M, len(ALPHAS)))
        for i in range(M):
            refs = np.where(allowed[i])[0]
            curves[i] = np.abs(ALPHAS[None] * irt_of[i][refs, None]
                               + (1 - ALPHAS[None]) * store[i][refs, None]
                               - y[refs, None]).mean(0)
        a = ALPHAS[int(curves.mean(0).argmin())]
        e = {n: np.zeros(M) for n in ('irt', 'geom', 'blend')}
        for i in range(M):
            e['irt'][i] = abs(irt_of[i][i] - y[i])
            e['geom'][i] = abs(store[i][i] - y[i])
            e['blend'][i] = abs(a * irt_of[i][i] + (1 - a) * store[i][i] - y[i])
        return e

    def capa_pred(i, cols):
        kk = B[i, cols]
        sims = np.full(M, -np.inf)
        for j in range(M):
            if not allowed[i][j]:
                continue
            obs = (kk == B[j, cols]).mean()
            exp = y[i] * y[j] + (1 - y[i]) * (1 - y[j])
            sims[j] = (obs - exp) / (1 - exp + 1e-9)
        nn = np.argsort(-sims)[:3]
        w = np.clip(sims[nn], 1e-6, None)
        return float(np.dot(w, y[nn]) / w.sum())

    rng = np.random.default_rng(0)
    out = {}

    def report(tag, e_sys):
        rng2 = np.random.default_rng(1)
        row = {}
        for n, errs in e_sys.items():
            vals = np.array([errs[rng2.integers(0, M, M)].mean()
                             for _ in range(BOOT)])
            row[n] = dict(mae=float(errs.mean()),
                          ci=[float(np.percentile(vals, 2.5)),
                              float(np.percentile(vals, 97.5))])
        print(tag, {n: round(r['mae'], 4) for n, r in row.items()})
        return row

    print('=== RAW rep, informative probes ===')
    out['raw_informative'] = {}
    for m in MS:
        e = eval_cols({i: np.array(info_order[i][:m]) for i in range(M)})
        out['raw_informative'][m] = report(f'm={m}', e)

    print('=== RAW rep, random probes (B=20) + CAPA ===')
    out['raw_random'] = {}
    out['capa_random'] = {}
    for m in MS:
        acc = {n: np.zeros(M) for n in ('irt', 'geom', 'blend')}
        e_capa = np.zeros(M)
        for b in range(B_DRAWS):
            cols = rng.choice(Q, m, replace=False)
            e = eval_cols({i: cols for i in range(M)})
            for n in acc:
                acc[n] += e[n] / B_DRAWS
            e_capa += np.array([abs(capa_pred(i, cols) - y[i])
                                for i in range(M)]) / B_DRAWS
        out['raw_random'][m] = report(f'm={m}', acc)
        out['capa_random'][m] = report(f'm={m} capa', {'capa': e_capa})

    json.dump(out, open('figures/q100_baselines.json', 'w'), indent=2)
    print('wrote figures/q100_baselines.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=('embed', 'eval'))
    args = ap.parse_args()
    load_dotenv('.env')
    if args.stage == 'embed':
        stage_embed(os.environ['OPENAI_API_KEY'])
    else:
        stage_eval()


if __name__ == '__main__':
    main()
