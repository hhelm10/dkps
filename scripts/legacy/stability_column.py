"""Agent-seed stability per representation (heatmap Stability column).

Corpus: small cohort (14 models x 12 instances x 5 replicates). Metric:
replicate retrieval P@1 -- for each trace, among all other traces of the SAME
instance (13 other models x 5 reps + own 4 reps), is the nearest neighbor a
replicate of the same system? Chance = 4/69. High = the embedding measures
the system, not the run.

Representations mirror the heatmap rows, all embedded with nomic:
  raw        head(32K chars) + tail(32K) embeddings concatenated
  head-only / tail-only
  centered   raw minus per-instance median over all traces, L2
  free-form  judge description (deepseek), embedded
  qubric     6 sections (deepseek), embedded per section, per-(instance,
             section) median-centered, L2, concat

Writes figures/stability_column.json.
"""
import json
import os
import sys
from glob import glob

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dkps.traces import make_sentence_transformer_embed_fn  # noqa: E402

SECTIONS = ('understanding', 'localization', 'reproduction',
            'editing', 'verification', 'final_state')
TXT = 'data/judge/smallcohort_texts'       # compact render: judge inputs, key list
RICH = 'data/judge/smallcohort_texts_rich'  # with tool outputs: raw-embedding rows
JUDGE = 'deepseek-chat-v3.1'
CACHE = 'data/judge/smallcohort_emb_nomic_v2.npz'


def main():
    keys = []          # (model, rep, query)
    for p in sorted(glob(os.path.join(TXT, '*', '*', '*.txt'))):
        m, r, q = p.split(os.sep)[-3], p.split(os.sep)[-2], p.split(os.sep)[-1][:-4]
        keys.append((m, r, q))
    print(len(keys), 'traces')
    qdir = f'data/judge/smallcohort-qspec-{JUDGE}'
    fdir = f'data/judge/smallcohort-freeform-{JUDGE}'
    missing = [k for k in keys
               if not os.path.exists(os.path.join(qdir, k[0], k[1], f'{k[2]}.json'))
               or not os.path.exists(os.path.join(fdir, k[0], k[1], f'{k[2]}.txt'))]
    if missing:
        raise SystemExit(f'{len(missing)} traces lack judge outputs -- run '
                         'smallcohort_judge.py first')

    heads, tails, frees, secs = [], [], [], []
    for m, r, q in keys:
        t = open(os.path.join(RICH, m, r, f'{q}.txt')).read()
        heads.append(t[:32_000])
        tails.append(t[-32_000:])
        frees.append(open(os.path.join(fdir, m, r, f'{q}.txt')).read())
        try:
            d = json.loads(open(os.path.join(qdir, m, r, f'{q}.json')).read())
            if isinstance(d, list) and d:
                d = d[0]
        except json.JSONDecodeError:
            d = {}
        secs.append([str(d.get(s, '') or ' ') for s in SECTIONS])

    if os.path.exists(CACHE):
        z = np.load(CACHE)
        H, T, F, S = z['H'], z['T'], z['F'], z['S']
    else:
        embed = make_sentence_transformer_embed_fn()
        H = embed(heads).astype(np.float32)
        T = embed(tails).astype(np.float32)
        F = embed(frees).astype(np.float32)
        S = embed([s for row in secs for s in row]).astype(np.float32)
        S = S.reshape(len(keys), len(SECTIONS), -1)
        np.savez_compressed(CACHE, H=H, T=T, F=F, S=S)
    print('embeddings ready', H.shape, S.shape)

    models = sorted({k[0] for k in keys})
    queries = sorted({k[2] for k in keys})
    midx = {m: i for i, m in enumerate(models)}
    qidx = {q: i for i, q in enumerate(queries)}
    marr = np.array([midx[k[0]] for k in keys])
    qarr = np.array([qidx[k[2]] for k in keys])

    def center_per_instance(X):
        Y = X.copy()
        for qi in range(len(queries)):
            sel = qarr == qi
            Y[sel] -= np.median(Y[sel], axis=0, keepdims=True)
        n = np.linalg.norm(Y, axis=-1, keepdims=True)
        return Y / np.maximum(n, 1e-9)

    S_cent = np.zeros_like(S)
    for qi in range(len(queries)):
        sel = qarr == qi
        block = S[sel] - np.median(S[sel], axis=0, keepdims=True)
        n = np.linalg.norm(block, axis=-1, keepdims=True)
        S_cent[sel] = block / np.maximum(n, 1e-9)

    reps = {
        'raw': np.concatenate([H, T], axis=1),
        'head-only': H,
        'tail-only': T,
        'centered': center_per_instance(np.concatenate([H, T], axis=1)),
        'free-form judge': F,
        'qubric': S_cent.reshape(len(keys), -1),
    }

    def replicate_p1(X):
        hits, total = 0, 0
        for qi in range(len(queries)):
            sel = np.where(qarr == qi)[0]
            Xq = X[sel]
            Mq = marr[sel]
            D = np.linalg.norm(Xq[:, None, :] - Xq[None, :, :], axis=-1)
            np.fill_diagonal(D, np.inf)
            nn = D.argmin(axis=1)
            hits += int((Mq[nn] == Mq).sum())
            total += len(sel)
        return hits / total

    n_per = len(keys) / len(queries)
    chance = 4 / (n_per - 1)
    out = {'chance': chance, 'rows': {}}
    for name, X in reps.items():
        p1 = replicate_p1(X)
        out['rows'][name] = p1
        print(f'{name:18s} replicate P@1 {p1:.3f}  (chance {chance:.3f})')
    json.dump(out, open('figures/stability_column.json', 'w'), indent=2)
    print('wrote figures/stability_column.json')


if __name__ == '__main__':
    main()
