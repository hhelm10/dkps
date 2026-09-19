"""Representation bake-off on leaderboard traces with true resolved labels.

Representations per trace (all from the same off-the-shelf embedder):
  - whole        : head+tail-truncated rendered trace, one embedding (baseline)
  - chunk_mean   : mean of all chunk embeddings (no truncation, no structure)
  - rubric_soft  : rubric-anchored soft-pooled section vectors (dkps.traces.rubric)
  - rubric_hard  : same with hard argmax assignment
  - mass         : 6-dim rubric phase-profile alone (interpretable)

Evaluation: leave-one-system-out kNN (k=3) prediction of the official Verified
resolve rate from DKPS distances; MAE and Spearman(pred, true).

Usage:
    python scripts/leaderboard_reprs.py [--max-subs 12] [--chunk-chars 4000]
"""
import argparse
import hashlib
import json
import os
import sys
from glob import glob

import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dkps.traces import (load_leaderboard_submission, make_openai_embed_fn,
                         make_sentence_transformer_embed_fn)
from dkps.traces.canonicalize import head_tail
from dkps.traces.rubric import (DEFAULT_RUBRIC, chunk_text, embed_rubric,
                                rubric_pool)
from dkps.traces.assemble import model_distance_matrix


def knn_predict(dist_row, y_ref, k=3):
    nn = np.argsort(dist_row)[:k]
    w = 1.0 / (dist_row[nn] + 1e-12)
    return float(np.dot(w, y_ref[nn]) / w.sum())


def evaluate(rep_by_sys, y):
    """rep_by_sys: {sys: (n_queries, d)}. LOO kNN over systems."""
    X = {s: v[:, None, :] for s, v in rep_by_sys.items()}
    systems, D = model_distance_matrix(X)
    yv = np.array([y[s] for s in systems])
    preds = []
    for i in range(len(systems)):
        mask = np.ones(len(systems), bool)
        mask[i] = False
        preds.append(knn_predict(D[i][mask], yv[mask]))
    preds = np.array(preds)
    rho = spearmanr(preds, yv).statistic
    return np.abs(preds - yv).mean(), rho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/leaderboard/verified')
    ap.add_argument('--labels', default='data/leaderboard/verified_labels.json')
    ap.add_argument('--cache-dir', default='.dkps_cache_lb')
    ap.add_argument('--max-subs', type=int, default=12)
    ap.add_argument('--chunk-chars', type=int, default=4000)
    ap.add_argument('--whole-chars', type=int, default=32_000)
    ap.add_argument('--tau', type=float, default=0.05)
    ap.add_argument('--min-trajs', type=int, default=400)
    ap.add_argument('--embedder', choices=('local', 'openai'), default='local')
    ap.add_argument('--embed-model', default=None)
    args = ap.parse_args()

    labels = json.load(open(args.labels))
    synced = [s for s in sorted(os.listdir(args.root))
              if len(glob(os.path.join(args.root, s, 'trajs', '*'))) >= args.min_trajs
              and 'resolved' in labels.get(s, {})]
    y = {s: len(labels[s]['resolved']) / 500 for s in synced}
    order = sorted(synced, key=lambda s: y[s])
    if len(order) > args.max_subs:                 # even spread across scores
        idx = np.linspace(0, len(order) - 1, args.max_subs).round().astype(int)
        order = [order[i] for i in sorted(set(idx))]
    print(f'{len(synced)} synced systems; using {len(order)} '
          f'(resolve rates {y[order[0]]:.2f}..{y[order[-1]]:.2f})')

    traces_by_sys = {s: load_leaderboard_submission(os.path.join(args.root, s),
                                                    labels=labels)
                     for s in order}
    common = set.intersection(*({t.query_id for t in v}
                                for v in traces_by_sys.values()))
    queries = sorted(common)
    print(f'{len(queries)} instances common to all systems')
    if len(queries) < 100:
        for s, v in traces_by_sys.items():
            print(f'  {s}: {len(v)} traces, e.g. {sorted(t.query_id for t in v)[:2]}')
        raise SystemExit('query intersection too small -- check loaders above')

    if args.embedder == 'openai':
        embed_fn = make_openai_embed_fn(
            **({'model': args.embed_model} if args.embed_model else {}))
    else:
        embed_fn = make_sentence_transformer_embed_fn(
            **({'model_name': args.embed_model} if args.embed_model else {}))
    cfg = hashlib.sha1(
        f'{embed_fn.model_name}|{args.chunk_chars}|{args.whole_chars}'.encode()
    ).hexdigest()[:8]

    def cache_path(tr):
        return os.path.join(args.cache_dir, tr.model_id, f'{tr.query_id}.{cfg}.npz')

    # ---- batched embedding of chunks + whole for all cache misses ----------
    todo = []
    for s in order:
        for tr in traces_by_sys[s]:
            if tr.query_id in common and not os.path.exists(cache_path(tr)):
                todo.append(tr)
    if todo:
        texts, spans = [], []
        for tr in todo:
            chunks = chunk_text(tr.steps[0].assistant_text, args.chunk_chars)
            whole = head_tail(tr.steps[0].assistant_text or ' ', args.whole_chars)
            spans.append((len(texts), len(texts) + len(chunks)))
            texts.extend(chunks)
            texts.append(whole)
        print(f'embedding {len(texts)} texts for {len(todo)} traces')
        embs = []
        step = 2048
        for i in tqdm(range(0, len(texts), step), desc='embed'):
            embs.append(embed_fn(texts[i:i + step]))
        E = np.vstack(embs)
        for tr, (a, b) in zip(todo, spans):
            path = cache_path(tr)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez_compressed(path, chunks=E[a:b], whole=E[b])

    anchors = embed_rubric(DEFAULT_RUBRIC, embed_fn)

    # ---- build representations --------------------------------------------
    reps = {name: {} for name in
            ('whole', 'chunk_mean', 'rubric_soft', 'rubric_hard', 'mass')}
    for s in order:
        by_q = {t.query_id: t for t in traces_by_sys[s]}
        rows = {name: [] for name in reps}
        for q in queries:
            with np.load(cache_path(by_q[q])) as z:
                C, W = z['chunks'], z['whole']
            rows['whole'].append(W)
            rows['chunk_mean'].append(C.mean(0) if len(C) else np.zeros_like(W))
            sec, mass = rubric_pool(C, anchors, tau=args.tau)
            rows['rubric_soft'].append(sec.ravel())
            sec_h, mass_h = rubric_pool(C, anchors, hard=True)
            rows['rubric_hard'].append(sec_h.ravel())
            rows['mass'].append(mass)
        for name in reps:
            reps[name][s] = np.array(rows[name])

    print(f'\n{"representation":14s} {"dim":>7s} {"LOO MAE":>8s} {"spearman":>9s}')
    for name, rep in reps.items():
        mae, rho = evaluate(rep, y)
        print(f'{name:14s} {rep[order[0]].shape[1]:7d} {mae:8.4f} {rho:9.3f}')

    print('\ntrue scores:', {s: round(y[s], 3) for s in order})
    # phase profiles (mean mass per system) for eyeballing
    print('\nmean rubric mass per system (sections: '
          + ', '.join(DEFAULT_RUBRIC) + ')')
    for s in order:
        print(f'  {s:52s} ' + ' '.join(f'{v:.2f}' for v in reps['mass'][s].mean(0)))


if __name__ == '__main__':
    main()
