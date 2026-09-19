"""Chunk-level API embeddings for leaderboard traces (substrate for rubric,
dynamics, and alignment-based trace comparisons).

Chunks each rendered trace (~4K chars, line-aligned, full coverage) and embeds
every chunk via the OpenAI API into a per-trace cache:
    .dkps_cache_lb/<system>/<instance>.<cfg>.npz   with array 'chunks'

Usage:
    python scripts/leaderboard_chunks.py [--max-queries 150]
"""
import argparse
import hashlib
import json
import os
import sys
from glob import glob
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dkps.traces import load_leaderboard_submission, make_openai_embed_fn
from dkps.traces.rubric import chunk_text

CHUNK_CHARS = 4000
_labels_g = None


def _load_one(args):
    sub_dir, wanted = args
    traces = load_leaderboard_submission(sub_dir, labels=_labels_g)
    out = {}
    for tr in traces:
        if tr.query_id in wanted:
            out[tr.query_id] = chunk_text(tr.steps[0].assistant_text, CHUNK_CHARS)
    return os.path.basename(sub_dir), out


def main():
    global _labels_g
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/leaderboard/verified')
    ap.add_argument('--labels', default='data/leaderboard/verified_labels.json')
    ap.add_argument('--cache-dir', default='.dkps_cache_lb')
    ap.add_argument('--embed-model', default='text-embedding-3-small')
    ap.add_argument('--min-trajs', type=int, default=480)
    ap.add_argument('--max-queries', type=int, default=150)
    ap.add_argument('--load-workers', type=int, default=16)
    ap.add_argument('--api-workers', type=int, default=8)
    args = ap.parse_args()

    labels = json.load(open(args.labels))
    _labels_g = labels
    systems = [s for s in sorted(os.listdir(args.root))
               if len(glob(os.path.join(args.root, s, 'trajs', '*'))) >= args.min_trajs
               and 'resolved' in labels.get(s, {})]

    # same instance subset as leaderboard_baselines.py: intersection, then
    # seeded subsample -- keep the seed and ordering identical
    from dkps.traces.leaderboard import _extract_query_id
    per_sys_ids = []
    for s in systems:
        ids = {_extract_query_id(os.path.basename(os.path.normpath(p)))
               for p in glob(os.path.join(args.root, s, 'trajs', '*'))}
        per_sys_ids.append(ids)
    queries = sorted(set.intersection(*per_sys_ids))
    if args.max_queries and args.max_queries < len(queries):
        rng = np.random.default_rng(0)
        queries = sorted(rng.choice(queries, args.max_queries, replace=False))
    wanted = set(queries)
    print(f'{len(systems)} systems x {len(queries)} instances')

    cfg = hashlib.sha1(f'openai/{args.embed_model}|chunks{CHUNK_CHARS}'.encode()
                       ).hexdigest()[:8]

    def cache_path(s, q):
        return os.path.join(args.cache_dir, s, f'{q}.{cfg}.npz')

    todo_sys = [s for s in systems
                if any(not os.path.exists(cache_path(s, q)) for q in queries)]
    print(f'{len(todo_sys)} systems need embedding')
    embed_fn = make_openai_embed_fn(model=args.embed_model,
                                    max_workers=args.api_workers)

    with Pool(args.load_workers) as pool:
        it = pool.imap(_load_one, [(os.path.join(args.root, s), wanted)
                                   for s in todo_sys])
        for s, chunks_by_q in tqdm(it, total=len(todo_sys), desc='systems'):
            missing = [q for q in queries if not os.path.exists(cache_path(s, q))]
            flat, spans = [], []
            for q in missing:
                ch = chunks_by_q.get(q, [])
                spans.append((len(flat), len(flat) + len(ch)))
                flat.extend(ch)
            if not flat:
                continue
            E = embed_fn(flat)
            for q, (a, b) in zip(missing, spans):
                path = cache_path(s, q)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                np.savez_compressed(path, chunks=E[a:b])
    print('done')


if __name__ == '__main__':
    main()
