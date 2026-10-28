#!/usr/bin/env python
"""Embed the EEE suite pool with a LOCAL open-source embedding model (CPU-friendly),
for the embedding-sensitivity study.

Reuses embed_eee.build_pool(), so the (model, task, query) rows are IDENTICAL to the
gemini-embedding-001 reference; only the vectors change. Writes
    exports/eee_query_embeddings__<tag>.parquet
    exports/eee_response_embeddings__<tag>.parquet
    exports/embed_timing__<tag>.json      (wall time, texts/s, dim, params)
Run:  python data/embed_eee_local.py --model minilm
      python data/embed_eee_local.py --bench-only     # throughput benchmark, no parquets
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed_eee import build_pool, OUT, TRUNC   # noqa: E402

# tag -> (hf id, backend, dim override or None, {query/response prefixes})
MODELS = {
    'minilm':    ('sentence-transformers/all-MiniLM-L6-v2', 'st', None, {}),
    'bge-small': ('BAAI/bge-small-en-v1.5', 'st', None, {}),
    'nomic15':   ('nomic-ai/nomic-embed-text-v1.5', 'st', None,
                  {'query': 'search_query: ', 'response': 'search_document: '}),
    'gemma300m': ('google/embeddinggemma-300m', 'st', None, {}),
    'qwen06b':   ('Qwen/Qwen3-Embedding-0.6B', 'st', None, {}),
    'potion8m':  ('minishlab/potion-base-8M', 'model2vec', None, {}),
    # jina v5 family: both text sizes, base + task-distilled variants, + omni
    'jina5-nano':        ('jinaai/jina-embeddings-v5-text-nano', 'st-task', None, {}),
    'jina5-nano-retr':   ('jinaai/jina-embeddings-v5-text-nano-retrieval', 'st', None, {}),
    'jina5-nano-match':  ('jinaai/jina-embeddings-v5-text-nano-text-matching', 'st', None, {}),
    'jina5-nano-clust':  ('jinaai/jina-embeddings-v5-text-nano-clustering', 'st', None, {}),
    'jina5-nano-class':  ('jinaai/jina-embeddings-v5-text-nano-classification', 'st', None, {}),
    'jina5-small':       ('jinaai/jina-embeddings-v5-text-small', 'st-task', None, {}),
    'jina5-small-retr':  ('jinaai/jina-embeddings-v5-text-small-retrieval', 'st', None, {}),
    'jina5-small-match': ('jinaai/jina-embeddings-v5-text-small-text-matching', 'st', None, {}),
    'jina5-small-clust': ('jinaai/jina-embeddings-v5-text-small-clustering', 'st', None, {}),
    'jina5-small-class': ('jinaai/jina-embeddings-v5-text-small-classification', 'st', None, {}),
    'jina5-omni-small':  ('jinaai/jina-embeddings-v5-omni-small', 'st', None, {}),
}


def load_model(tag):
    hf_id, backend, _, prefixes = MODELS[tag]
    if backend == 'model2vec':
        from model2vec import StaticModel
        m = StaticModel.from_pretrained(hf_id)
        n_params = sum(int(np.prod(t.shape)) for t in [m.embedding]) if hasattr(m, 'embedding') else None
        enc = lambda texts, bs: m.encode(texts, batch_size=bs)
    else:
        import torch
        from sentence_transformers import SentenceTransformer
        kw = dict(model_kwargs={'default_task': 'text-matching'}) if backend == 'st-task' else {}
        m = SentenceTransformer(hf_id, device='cpu', trust_remote_code=True, **kw)
        n_params = sum(p.numel() for p in m.parameters())
        enc = lambda texts, bs: m.encode(texts, batch_size=bs, convert_to_numpy=True,
                                         show_progress_bar=True, normalize_embeddings=False)
    return enc, prefixes, n_params


def embed(enc, texts, prefix, bs=64):
    texts = [prefix + str(t)[:TRUNC] for t in texts]
    t0 = time.time()
    E = np.asarray(enc(texts, bs), dtype=np.float32)
    return E, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=sorted(MODELS), default='minilm')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--bench-only', action='store_true',
                    help='time a fixed 1000-response sample; write no parquets')
    args = ap.parse_args()
    tag = args.model

    _, pool, _ = build_pool()
    uq = pool.drop_duplicates('query_id')[['query_id', 'query']].sort_values('query_id')
    enc, prefixes, n_params = load_model(tag)
    print(f'[{tag}] {len(pool)} responses, {len(uq)} queries, params={n_params}')

    if args.bench_only:
        sample = pool['response'].head(1000).tolist()
        E, dt = embed(enc, sample, prefixes.get('response', ''), args.batch_size)
        print(f'[{tag}] bench: 1000 responses in {dt:.1f}s = {1000/dt:.0f} texts/s, dim {E.shape[1]}')
        return

    qE, qt = embed(enc, uq['query'].tolist(), prefixes.get('query', ''), args.batch_size)
    rE, rt = embed(enc, pool['response'].tolist(), prefixes.get('response', ''), args.batch_size)
    print(f'[{tag}] queries {qE.shape} in {qt:.0f}s | responses {rE.shape} in {rt:.0f}s')

    pd.DataFrame({'query_id': uq.query_id.values, 'emb': [e for e in qE]}
                 ).to_parquet(OUT / f'eee_query_embeddings__{tag}.parquet')
    pd.DataFrame({'model': pool.model.values, 'bench': pool.bench.values,
                  'task': pool.task.values, 'query_id': pool.query_id.values,
                  'score': pool.score.values, 'emb': [e for e in rE]}
                 ).to_parquet(OUT / f'eee_response_embeddings__{tag}.parquet')
    json.dump(dict(tag=tag, hf_id=MODELS[tag][0], dim=int(rE.shape[1]), params=n_params,
                   n_queries=len(uq), n_responses=len(pool),
                   query_seconds=round(qt, 1), response_seconds=round(rt, 1),
                   texts_per_second=round((len(uq) + len(pool)) / (qt + rt), 1)),
              open(OUT / f'embed_timing__{tag}.json', 'w'), indent=1)
    print(f'[{tag}] wrote exports; {(len(uq)+len(pool))/(qt+rt):.0f} texts/s overall')


if __name__ == '__main__':
    main()
