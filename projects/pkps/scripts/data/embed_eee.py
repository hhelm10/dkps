#!/usr/bin/env python
"""Embed the EEE suite (queries + pooled responses) with gemini-embedding-001.

Builds a fixed per-(model, task) query POOL of up to CAP=32 queries (seeded per
model+task, so different models draw different queries -- the suite is unpaired by
construction) and embeds (a) each pooled response and (b) every query that appears
in any pool. The full-depth per-cell score mean (over ALL queries, 42-1288 per cell)
is stored alongside as the leak-free prediction target; only observed (pooled)
responses are ever embedded.

Requires GEMINI_API_KEY. Chunk-level caching via dkps.embed; safe to re-run.

Outputs:
    exports/eee_pool.parquet                 pooled rows: model, bench, task, query_id, score
    exports/eee_cell_targets.parquet         full-depth cell means: model, bench, task, y_full, depth
    exports/eee_response_embeddings.parquet  model, task, query_id, emb (float32[3072])
    exports/eee_query_embeddings.parquet     query_id, emb (float32[3072])
"""
import hashlib
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from rich import print as rprint

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dkps.embed import embed_api

DATA = Path(os.environ.get('DKPS_DATA') or Path(__file__).resolve().parents[4] / 'data')
D = DATA / 'eee'
OUT = DATA / 'exports'
CAP = 32                      # per-cell query pool (observation budget)
TRUNC = 8000                  # chars per text sent to the embedder
MODEL = 'gemini-embedding-001'
BENCHMARKS = ['math-mc', 'gsm-mc', 'gpqa-diamond', 'judgebench', 'reward-bench-2']


def _seed(model, task):
    return int(hashlib.sha256(f'{model}|{task}'.encode()).hexdigest()[:8], 16)


def build_pool():
    frames = []
    for b in BENCHMARKS:
        df = pd.read_parquet(D / f'eee_{b}.parquet').assign(bench=b)
        frames.append(df)
    allb = pd.concat(frames, ignore_index=True)

    # full-depth cell targets (uses every query; never embedded)
    targets = (allb.groupby(['model', 'bench', 'task'])['score']
               .agg(y_full='mean', depth='size').reset_index())

    # per-cell pools: up to CAP queries, model-dependent seed -> unpaired by design
    pools = []
    for (model, task), g in allb.groupby(['model', 'task']):
        rng = np.random.default_rng(_seed(model, task))
        take = g.iloc[rng.permutation(len(g))[:CAP]]
        pools.append(take)
    pool = pd.concat(pools, ignore_index=True)
    return allb, pool, targets


def main():
    allb, pool, targets = build_pool()
    uq = pool.drop_duplicates('query_id')[['query_id', 'query']].sort_values('query_id')
    n_resp, n_q = len(pool), len(uq)
    toks = (pool.response.str.len().clip(upper=TRUNC).sum()
            + uq['query'].str.len().clip(upper=TRUNC).sum()) / 4
    rprint(f'[blue]pool: {n_resp} responses, {n_q} unique queries '
           f'(~{toks/1e6:.1f}M tokens to {MODEL})[/blue]')

    OUT.mkdir(exist_ok=True)
    pool.drop(columns=['query', 'response']).to_parquet(OUT / 'eee_pool.parquet')
    targets.to_parquet(OUT / 'eee_cell_targets.parquet')

    rprint('[blue]embedding queries ...[/blue]')
    qe = embed_api('google', [str(x)[:TRUNC] for x in uq['query']], model=MODEL)
    pd.DataFrame({'query_id': uq.query_id.values,
                  'emb': [e.astype(np.float32) for e in qe]}
                 ).to_parquet(OUT / 'eee_query_embeddings.parquet')
    rprint(f'[green]queries done: {qe.shape}[/green]')

    rprint('[blue]embedding pooled responses ...[/blue]')
    re_ = embed_api('google', [str(x)[:TRUNC] for x in pool['response']], model=MODEL)
    pd.DataFrame({'model': pool.model.values, 'bench': pool.bench.values,
                  'task': pool.task.values, 'query_id': pool.query_id.values,
                  'score': pool.score.values,
                  'emb': [e.astype(np.float32) for e in re_]}
                 ).to_parquet(OUT / 'eee_response_embeddings.parquet')
    rprint(f'[green]responses done: {re_.shape}[/green]')


if __name__ == '__main__':
    main()
