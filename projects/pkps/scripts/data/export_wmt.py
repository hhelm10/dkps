#!/usr/bin/env python
"""
export_wmt.py

Build WMT_14 google-embedding parquets, mirroring the MATH exports.

WMT was run via run.sh with `--sample 0.2 --score_col meteor`, so only a seeded
20% instance subset (np.random.default_rng(123)) was embedded. We reconstruct
that exact subset, pull the cached google response embeddings (free), and embed
the unique query (source-sentence) texts via google (needs GEMINI_API_KEY; ~14
chunks). Outputs:
  exports/wmt_14_google_embeddings.parquet        (dataset, model, instance_id, response, embedding)
  exports/wmt_14_query_google_embeddings.parquet  (query_id, dataset, instance_id, query, query_embedding)
"""
import asyncio
import os
from pathlib import Path

import numpy as np
import pandas as pd
from rich import print as rprint

from dkps.embed import _aembed_google_chunk, embed_api

DATA = Path(os.environ.get('DKPS_DATA') or Path(__file__).resolve().parents[4] / 'data')
TSV = DATA / 'helm/wmt_14.tsv'
OUTDIR = DATA / 'exports'
MODEL = 'gemini-embedding-001'
CHUNK_SIZE = 50
SAMPLE = 0.2
SEED = 123


async def _embed_cached(input_strs, max_concurrency=8):
    chunks = [input_strs[i:i + CHUNK_SIZE] for i in range(0, len(input_strs), CHUNK_SIZE)]
    sem = asyncio.Semaphore(max_concurrency)

    async def _fn(cid, chunk):
        async with sem:
            return await _aembed_google_chunk(cid, None, chunk, MODEL)

    out = [None] * len(chunks)
    for coro in asyncio.as_completed([_fn(i, c) for i, c in enumerate(chunks)]):
        cid, emb = await coro
        out[cid] = emb
    return np.concatenate(out)


def sampled_subset(df_ds):
    """Reconstruct run_dkps --sample 0.2 (seed 123) for one language pair."""
    rng = np.random.default_rng(SEED)
    uids = df_ds.instance_id.unique()
    keep = rng.choice(uids, int(len(uids) * SAMPLE), replace=False)
    return df_ds[df_ds.instance_id.isin(keep)].sort_values(['model', 'instance_id']).reset_index(drop=True)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df_all = pd.read_csv(TSV, sep='\t')
    pairs = sorted(df_all.dataset.unique())

    resp_parts, q_rows = [], []
    for ds in pairs:
        sub = sampled_subset(df_all[df_all.dataset == ds])
        rprint(f'[green]{ds}[/green] rows={len(sub)} instances={sub.instance_id.nunique()}')
        emb = asyncio.run(_embed_cached([str(x) for x in sub.response.values]))
        assert emb.shape[0] == len(sub), (emb.shape, len(sub))
        sub = sub.copy()
        sub['embedding'] = list(emb.astype(np.float32))
        sub['query_id'] = sub['dataset'] + '::' + sub['instance_id'].astype(str)
        resp_parts.append(sub[['dataset', 'model', 'instance_id', 'response', 'embedding']])
        q_rows.append(sub.drop_duplicates('query_id')[['query_id', 'dataset', 'instance_id', 'query']])

    resp = pd.concat(resp_parts, ignore_index=True)
    resp.to_parquet(OUTDIR / 'wmt_14_google_embeddings.parquet')
    rprint(f'[blue]wrote response parquet: {len(resp)} rows[/blue]')

    # Query embeddings (source-sentence text) via google.
    q = pd.concat(q_rows, ignore_index=True).drop_duplicates('query_id').reset_index(drop=True)
    rprint(f'[blue]embedding {len(q)} unique WMT queries via google ...[/blue]')
    qemb = embed_api('google', [str(x) for x in q['query'].values], model=MODEL)
    assert qemb.shape[0] == len(q), (qemb.shape, len(q))
    q['query_embedding'] = list(qemb.astype(np.float32))
    q.to_parquet(OUTDIR / 'wmt_14_query_google_embeddings.parquet')
    rprint(f'[green]wrote query parquet: {len(q)} rows, dim {qemb.shape[1]}[/green]')


if __name__ == '__main__':
    main()
