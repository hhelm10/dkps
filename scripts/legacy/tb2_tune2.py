"""TB2 tuning round 2 (per HH: 'did we re-optimize optimal pca for
queries and for traces?'): the two PCA choices inherited from SWE.

  qvec rank    16 | 64 | full (1536-d instruction embeddings, re-embedded
               here; PCA-64 was the SWE-CV'd convention)
  trace PCA    none | 64 | 256 (rank-r projection of the consensus-
               centered trace embeddings; r=64 was WORSE on SWE q100 --
               retired there, untested here)
crossed with kernel sigma in {2, 4, 8, 16} x kNN k in {5, 7, 15},
family-out, pooled reference selection, B=8 draws, m in {1, 5, 20}.
Writes figures/tb2_tune2.json.
"""
import json
import os
import sys
import time

import numpy as np
import requests
from dotenv import load_dotenv

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
from dkps.traces.qubric import consensus_center  # noqa: E402

SIGS = (2, 4, 8, 16)
KS = (5, 7, 15)
MS = (1, 5, 20)
B_DRAWS = 8
QRANKS = (16, 64, 'full')
TRANKS = (None, 64, 256)
DROP = {'Droid__GPT-5.3-Codex', 'Droid__Claude-Opus-4.6',
        'just-another-coding-agent__GLM-5'}
QE_CACHE = 'data/terminal_bench/tb2_instruction_emb_full.npz'


def instruction_embeddings(tasks, key):
    if os.path.exists(QE_CACHE):
        return np.load(QE_CACHE)['E']
    import tiktoken
    enc = tiktoken.get_encoding('cl100k_base')
    texts = []
    for t in tasks:
        toks = enc.encode(
            open(f'data/terminal_bench/tb2_tasks/{t}/instruction.md').read(),
            disallowed_special=())
        texts.append(enc.decode(toks[:8000]) or ' ')
    rows = []
    for i in range(0, len(texts), 8):
        for attempt in range(8):
            try:
                r = requests.post(
                    'https://api.openai.com/v1/embeddings',
                    json={'model': 'text-embedding-3-small',
                          'input': texts[i:i + 8]},
                    headers={'Authorization': f'Bearer {key}'}, timeout=120)
            except requests.RequestException:
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code == 200:
                rows.extend(x['embedding'] for x in r.json()['data'])
                break
            time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError('embed failed')
    E = np.asarray(rows, np.float32)
    np.savez_compressed(QE_CACHE, E=E)
    return E


def main():
    load_dotenv()
    d = json.load(open('data/terminal_bench/tb2_panel.json'))
    all_sys = [s for s in d['systems'] if s not in DROP]
    tasks = d['tasks']
    Q = len(tasks)
    cov = {s: sum(os.path.exists(f'data/terminal_bench/tb2_txt/{s}/{t}.txt')
                  for t in tasks) / Q for s in all_sys}
    systems = [s for s in all_sys if cov[s] >= 0.9]
    M = len(systems)
    sidx = [d['systems'].index(s) for s in systems]
    y = np.array(d['y'])[sidx]
    meta = d['meta']
    fams = [set(meta[s]['llm_tags']) for s in systems]
    allowed = np.array([[j != i and not (fams[i] and fams[i] & fams[j])
                         for j in range(M)] for i in range(M)])

    X0 = consensus_center(
        np.load('data/terminal_bench/tb2_emb_openai_small.npz')['X'],
        np.tile(np.arange(Q), len(all_sys))) \
        .reshape(len(all_sys), Q, -1)[
            [all_sys.index(s) for s in systems]].astype(np.float32)

    E = instruction_embeddings(tasks, os.environ['OPENAI_API_KEY'])
    Ec = E - E.mean(0, keepdims=True)
    U, S, _ = np.linalg.svd(Ec, full_matrices=False)

    def qvec(rank):
        if rank == 'full':
            return Ec
        return (U[:, :rank] * S[:rank]).astype(np.float32)

    def trace_X(rank):
        if rank is None:
            return X0
        F = X0.reshape(M * Q, -1)
        Uf, Sf, _ = np.linalg.svd(F - F.mean(0, keepdims=True),
                                  full_matrices=False)
        return (Uf[:, :rank] * Sf[:rank]).reshape(M, Q, -1) \
            .astype(np.float32)

    rng = np.random.default_rng(0)
    draws = {m: [rng.choice(Q, m, replace=False) for _ in range(B_DRAWS)]
             for m in MS}

    results = {}
    for qr in QRANKS:
        V = qvec(qr)
        D2q = ((V[:, None] - V[None]) ** 2).sum(-1)
        med = np.median(D2q)
        for tr in TRANKS:
            X = trace_X(tr)
            for s_ in SIGS:
                KQ = np.exp(-D2q / (2 * med / s_))
                W = np.einsum('qp,jpd->jqd', KQ, X, optimize=True)
                Arr = np.einsum('jqd,jqd->j', X, W) / KQ.sum()
                for m in MS:
                    for cols in draws[m]:
                        cols = np.asarray(cols)
                        Xi = X[:, cols].reshape(M, -1)
                        A_tr = (Xi @ W[:, cols].reshape(M, -1).T) \
                            / KQ[cols].sum()
                        KQc = KQ[np.ix_(cols, cols)]
                        Wc = np.einsum('qp,jpd->jqd', KQc, X[:, cols],
                                       optimize=True)
                        Att = np.einsum('jqd,jqd->j', X[:, cols], Wc) \
                            / KQc.sum()
                        D = np.sqrt(np.maximum(
                            Att[:, None] + Arr[None] - 2 * A_tr, 0))
                        for k in KS:
                            errs, tgts = [], []
                            for i in range(M):
                                refs = np.where(allowed[i])[0]
                                Dref = D[:, refs].copy()
                                for r_, j in enumerate(refs):
                                    Dref[j, r_] = np.inf
                                Dref[i] = D[i, refs]
                                nn = np.argsort(Dref, 1)[:, :k]
                                w = 1 / (np.take_along_axis(Dref, nn, 1)
                                         + 1e-12)
                                pred = (w * y[refs][nn]).sum(1) / w.sum(1)
                                errs.append(
                                    np.abs(pred[refs] - y[refs]).mean())
                                tgts.append(abs(pred[i] - y[i]))
                            key = (str(qr), str(tr), s_, k, m)
                            ref_e, tgt_e = np.mean(errs), np.mean(tgts)
                            a = results.setdefault(key, [0.0, 0.0])
                            a[0] += ref_e / B_DRAWS
                            a[1] += tgt_e / B_DRAWS
    out = {}
    for m in MS:
        sub = {k: v for k, v in results.items() if k[4] == m}
        best = min(sub, key=lambda k: sub[k][0])
        out[m] = dict(best=dict(qvec=best[0], trace_pca=best[1],
                                sigma=best[2], k=best[3],
                                ref=round(sub[best][0], 4),
                                target=round(sub[best][1], 4)),
                      top5=[dict(cfg=list(k[:4]), ref=round(v[0], 4),
                                 tgt=round(v[1], 4))
                            for k, v in sorted(sub.items(),
                                               key=lambda kv: kv[1][0])[:5]])
        print(f'm={m}: best qvec={best[0]} tracePCA={best[1]} '
              f'sigma={best[2]} k={best[3]} ref={sub[best][0]:.4f} '
              f'target={sub[best][1]:.4f}')
    json.dump(out, open('figures/tb2_tune2.json', 'w'), indent=2)
    print('wrote figures/tb2_tune2.json')


if __name__ == '__main__':
    main()
