"""Embedding baselines on the full leaderboard corpus: first-8K-token and
last-8K-token slices of each rendered trace, embedded via the OpenAI API.

Representations: head8k, tail8k, head+tail (concat). Evaluation: leave-one-
system-out kNN prediction of the official Verified resolve rate from DKPS
distances (MAE + Spearman).

Usage:
    python scripts/leaderboard_baselines.py [--embed-model text-embedding-3-small]
"""
import argparse
import hashlib
import json
import os
import sys
from glob import glob
from multiprocessing import Pool

import numpy as np
import tiktoken
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dkps.traces import load_leaderboard_submission, make_openai_embed_fn
from dkps.traces.assemble import model_distance_matrix

ENC = tiktoken.get_encoding('cl100k_base')
MAX_TOK = 8000
PRETRIM = 40_000        # chars kept per side before tokenizing (>> 8K tokens)

_labels_g = None


def head_tail_texts(text):
    text = text or ' '
    head = ENC.decode(ENC.encode(text[:PRETRIM], disallowed_special=())[:MAX_TOK])
    tail = ENC.decode(ENC.encode(text[-PRETRIM:], disallowed_special=())[-MAX_TOK:])
    return head, tail


def _load_one(args):
    sub_dir, = args
    traces = load_leaderboard_submission(sub_dir, labels=_labels_g)
    out = {}
    for tr in traces:
        out[tr.query_id] = head_tail_texts(tr.steps[0].assistant_text)
    return os.path.basename(sub_dir), out


def knn_predict(dist_row, y_ref, k):
    nn = np.argsort(dist_row)[:k]
    w = 1.0 / (dist_row[nn] + 1e-12)
    return float(np.dot(w, y_ref[nn]) / w.sum())


def evaluate(rep_by_sys, y, k=3):
    X = {s: v[:, None, :] for s, v in rep_by_sys.items()}
    systems, D = model_distance_matrix(X)
    yv = np.array([y[s] for s in systems])
    preds = np.array([
        knn_predict(D[i][np.arange(len(systems)) != i],
                    yv[np.arange(len(systems)) != i], k)
        for i in range(len(systems))])
    return np.abs(preds - yv).mean(), spearmanr(preds, yv).statistic, systems, preds, yv


def main():
    global _labels_g
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/leaderboard/verified')
    ap.add_argument('--labels', default='data/leaderboard/verified_labels.json')
    ap.add_argument('--cache-dir', default='.dkps_cache_lb')
    ap.add_argument('--embed-model', default='text-embedding-3-small')
    ap.add_argument('--min-trajs', type=int, default=480)
    ap.add_argument('--load-workers', type=int, default=16)
    ap.add_argument('--api-workers', type=int, default=24)
    ap.add_argument('--max-queries', type=int, default=None,
                    help='seeded random subset of common instances (speed knob; '
                         'DKPS geometry is near-saturated in n_queries)')
    args = ap.parse_args()

    labels = json.load(open(args.labels))
    _labels_g = labels
    systems = [s for s in sorted(os.listdir(args.root))
               if len(glob(os.path.join(args.root, s, 'trajs', '*'))) >= args.min_trajs
               and 'resolved' in labels.get(s, {})]
    y = {s: len(labels[s]['resolved']) / 500 for s in systems}
    print(f'{len(systems)} systems (resolve rates '
          f'{min(y.values()):.2f}..{max(y.values()):.2f})')

    with Pool(args.load_workers) as pool:
        loaded = list(tqdm(
            pool.imap(_load_one, [(os.path.join(args.root, s),) for s in systems]),
            total=len(systems), desc='load+trim'))
    texts_by_sys = dict(loaded)

    common = set.intersection(*(set(v) for v in texts_by_sys.values()))
    queries = sorted(common)
    print(f'{len(queries)} instances common to all {len(systems)} systems')
    if len(queries) < 250:
        raise SystemExit('intersection too small; lower --min-trajs or inspect loaders')
    if args.max_queries and args.max_queries < len(queries):
        rng = np.random.default_rng(0)
        queries = sorted(rng.choice(queries, args.max_queries, replace=False))
        print(f'subsampled to {len(queries)} instances (--max-queries)')

    embed_fn = make_openai_embed_fn(model=args.embed_model,
                                    max_workers=args.api_workers)
    cfg = hashlib.sha1(f'{embed_fn.model_name}|headtail{MAX_TOK}'.encode()
                       ).hexdigest()[:8]

    def cache_path(s, q):
        return os.path.join(args.cache_dir, s, f'{q}.{cfg}.npz')

    todo = [(s, q) for s in systems for q in queries
            if not os.path.exists(cache_path(s, q))]
    print(f'{len(todo)} traces to embed ({2 * len(todo)} inputs)')
    CHUNK = 3000                      # traces per outer chunk (6000 API inputs)
    for i in tqdm(range(0, len(todo), CHUNK), desc='embed'):
        part = todo[i:i + CHUNK]
        flat = []
        for s, q in part:
            flat.extend(texts_by_sys[s][q])
        E = embed_fn(flat)
        for j, (s, q) in enumerate(part):
            path = cache_path(s, q)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez_compressed(path, head=E[2 * j], tail=E[2 * j + 1])

    reps = {'head8k': {}, 'tail8k': {}, 'head+tail': {}}
    for s in systems:
        H, T = [], []
        for q in queries:
            with np.load(cache_path(s, q)) as z:
                H.append(z['head'])
                T.append(z['tail'])
        H, T = np.array(H), np.array(T)
        reps['head8k'][s] = H
        reps['tail8k'][s] = T
        reps['head+tail'][s] = np.hstack([H, T])

    print(f'\n{"representation":12s} {"k":>2s} {"LOO MAE":>8s} {"spearman":>9s}')
    for name, rep in reps.items():
        for k in (3, 5):
            mae, rho, *_ = evaluate(rep, y, k=k)
            print(f'{name:12s} {k:2d} {mae:8.4f} {rho:9.3f}')

    _, _, sys_names, preds, yv = evaluate(reps['head+tail'], y, k=5)
    worst = np.argsort(-np.abs(preds - yv))[:8]
    print('\nlargest errors (head+tail, k=5):')
    for i in worst:
        print(f'  {sys_names[i]:55s} true={yv[i]:.2f} pred={preds[i]:.2f}')


if __name__ == '__main__':
    main()
