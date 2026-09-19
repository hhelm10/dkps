"""Fast viability test of the rubric approach: all 107 systems, 20 instances.

Embeds chunks for a seeded 20-instance subset (of the standard 150), builds
chunk-based representations (chunk-mean, rubric soft/hard, mass), and compares
against the head/tail baselines *on the same 20 instances* (those embeddings
are already cached). ~25 min at the 4.5M TPM budget.

Usage: python scripts/rubric_viability.py [--n-queries 20]
"""
import argparse
import hashlib
import json
import os
import sys
from glob import glob
from multiprocessing import Pool

import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dkps.traces import load_leaderboard_submission, make_openai_embed_fn
from dkps.traces.leaderboard import _extract_query_id
from dkps.traces.rubric import DEFAULT_RUBRIC, chunk_text, embed_rubric, rubric_pool
from dkps.traces.assemble import model_distance_matrix

CHUNK_CHARS = 4000
_labels_g = None
_wanted_g = None


def _load_one(sub_dir):
    traces = load_leaderboard_submission(sub_dir, labels=_labels_g)
    return (os.path.basename(sub_dir),
            {t.query_id: chunk_text(t.steps[0].assistant_text, CHUNK_CHARS)
             for t in traces if t.query_id in _wanted_g})


def main():
    global _labels_g, _wanted_g
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/leaderboard/verified')
    ap.add_argument('--labels', default='data/leaderboard/verified_labels.json')
    ap.add_argument('--cache-dir', default='.dkps_cache_lb')
    ap.add_argument('--embed-model', default='text-embedding-3-small')
    ap.add_argument('--n-queries', type=int, default=20)
    ap.add_argument('--tau', type=float, default=0.05)
    args = ap.parse_args()

    labels = json.load(open(args.labels))
    _labels_g = labels
    systems = [s for s in sorted(os.listdir(args.root))
               if len(glob(os.path.join(args.root, s, 'trajs', '*'))) >= 480
               and 'resolved' in labels.get(s, {})]
    y = {s: len(labels[s]['resolved']) / 500 for s in systems}

    per_sys = [{_extract_query_id(os.path.basename(os.path.normpath(p)))
                for p in glob(os.path.join(args.root, s, 'trajs', '*'))}
               for s in systems]
    q418 = sorted(set.intersection(*per_sys))
    rng = np.random.default_rng(0)
    q150 = sorted(rng.choice(q418, 150, replace=False))
    rng2 = np.random.default_rng(1)
    queries = sorted(rng2.choice(q150, args.n_queries, replace=False))
    _wanted_g = set(queries)
    M, Q = len(systems), len(queries)
    print(f'{M} systems x {Q} instances (subset of the standard 150)')

    embed_fn = make_openai_embed_fn(model=args.embed_model)
    ccfg = hashlib.sha1(f'openai/{args.embed_model}|chunks{CHUNK_CHARS}'.encode()
                        ).hexdigest()[:8]
    hcfg = hashlib.sha1(f'openai/{args.embed_model}|headtail8000'.encode()
                        ).hexdigest()[:8]

    def cpath(s, q):
        return os.path.join(args.cache_dir, s, f'{q}.{ccfg}.npz')

    todo_sys = [s for s in systems
                if any(not os.path.exists(cpath(s, q)) for q in queries)]
    print(f'{len(todo_sys)} systems need chunk embedding')
    if todo_sys:
        with Pool(16) as pool:
            it = pool.imap(_load_one,
                           [os.path.join(args.root, s) for s in todo_sys])
            for s, by_q in tqdm(it, total=len(todo_sys), desc='embed systems'):
                missing = [q for q in queries if not os.path.exists(cpath(s, q))]
                flat, spans = [], []
                for q in missing:
                    ch = by_q.get(q, [])
                    spans.append((len(flat), len(flat) + len(ch)))
                    flat.extend(ch)
                if not flat:
                    continue
                E = embed_fn(flat)
                for q, (a, b) in zip(missing, spans):
                    os.makedirs(os.path.dirname(cpath(s, q)), exist_ok=True)
                    np.savez_compressed(cpath(s, q), chunks=E[a:b])

    anchors = embed_rubric(DEFAULT_RUBRIC, embed_fn)
    S_r = len(DEFAULT_RUBRIC)

    dim = 1536
    reps = {'head+tail (baseline)': np.zeros((M, Q, 2 * dim)),
            'chunk_mean': np.zeros((M, Q, dim)),
            'rubric_soft': np.zeros((M, Q, S_r * dim)),
            'rubric_hard': np.zeros((M, Q, S_r * dim)),
            'mass': np.zeros((M, Q, S_r))}
    for i, s in enumerate(tqdm(systems, desc='build reps')):
        for j, q in enumerate(queries):
            with np.load(os.path.join(args.cache_dir, s, f'{q}.{hcfg}.npz')) as z:
                reps['head+tail (baseline)'][i, j] = np.concatenate([z['head'], z['tail']])
            with np.load(cpath(s, q)) as z:
                C = z['chunks']
            reps['chunk_mean'][i, j] = C.mean(0) if len(C) else 0
            sec, mass = rubric_pool(C, anchors, tau=args.tau)
            reps['rubric_soft'][i, j] = sec.ravel()
            sec_h, _ = rubric_pool(C, anchors, hard=True)
            reps['rubric_hard'][i, j] = sec_h.ravel()
            reps['mass'][i, j] = mass

    def knn_predict(dist_row, y_ref, k):
        nn = np.argsort(dist_row)[:k]
        w = 1.0 / (dist_row[nn] + 1e-12)
        return float(np.dot(w, y_ref[nn]) / w.sum())

    def evaluate(Xv, name):
        rep = {s: np.asarray(Xv[i], dtype=float) for i, s in enumerate(systems)}
        Xd = {s: v[:, None, :] for s, v in rep.items()}
        ss, D = model_distance_matrix(Xd)
        yv = np.array([y[s] for s in ss])
        row = f'{name:34s}'
        for k in (3, 5):
            preds = np.array([knn_predict(D[i][np.arange(M) != i],
                                          yv[np.arange(M) != i], k)
                              for i in range(M)])
            row += f'  k={k}: {np.abs(preds - yv).mean():.4f}/{spearmanr(preds, yv).statistic:.3f}'
        print(row)

    def centered(Xv):
        Xc = Xv - np.median(Xv, axis=0, keepdims=True)
        return Xc / np.maximum(np.linalg.norm(Xc, axis=-1, keepdims=True), 1e-9)

    print(f'\n=== raw geometries ({Q} instances) ===')
    for name, Xv in reps.items():
        evaluate(Xv, name)
    print(f'\n=== median-centered + L2 ===')
    for name, Xv in reps.items():
        evaluate(centered(Xv), name + ' [cent]')


if __name__ == '__main__':
    main()
