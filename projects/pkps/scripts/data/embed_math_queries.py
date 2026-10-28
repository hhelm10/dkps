#!/usr/bin/env python
"""
embed_math_queries.py

Embed the 437 unique MATH *instances* (the bare problem text, with model-specific
few-shot presentation stripped) with google gemini-embedding-001 — the same space
as the cached response embeddings — for the DoubleKernelDKPS query kernel.

The `query` column is a full few-shot prompt whose formatting varies by model
family (~3665 unique strings). The underlying target problem is the final
"Problem: ... Answer:" block, which is identical across all models for a given
instance. Extracting it recovers exactly 437 unique problems.

Requires GEMINI_API_KEY (visible to `pixi run`, e.g. exported in your shell
profile or a sourced .env). 437 strings -> 9 chunks. Cached after first run.
"""
import re
import os
from pathlib import Path

import numpy as np
import pandas as pd
from rich import print as rprint

from dkps.embed import embed_api

DATA = Path(os.environ.get('DKPS_DATA') or Path(__file__).resolve().parents[4] / 'data')
TSV = DATA / 'helm/math.tsv'
OUT = DATA / 'exports/math_query_google_embeddings.parquet'
MODEL = 'gemini-embedding-001'


def extract_problem(query):
    """Bare target problem = last 'Problem:' block, minus any trailing 'Answer:'."""
    last = re.split(r'Problem:', str(query))[-1]
    return re.split(r'Answer:', last)[0].strip()


def main():
    df = pd.read_csv(TSV, sep='\t')
    df['query_id'] = df['dataset'] + '::' + df['instance_id'].astype(str)
    df['problem'] = df['query'].apply(extract_problem)

    # One bare problem per instance (verified identical across model variants).
    per_inst = df.groupby('query_id')['problem'].nunique()
    assert (per_inst == 1).all(), \
        f'{(per_inst != 1).sum()} instances have non-constant extracted problem'

    uq = (df.drop_duplicates('query_id')
            .sort_values('query_id')
            .reset_index(drop=True))
    rprint(f'[blue]embedding {len(uq)} unique MATH problems via google/{MODEL} ...[/blue]')

    emb = embed_api('google', [str(x) for x in uq['problem'].values], model=MODEL)
    assert emb.shape[0] == len(uq), (emb.shape, len(uq))
    rprint(f'[green]embeddings: {emb.shape}[/green]')

    out = pd.DataFrame({
        'query_id': uq['query_id'].values,
        'dataset': uq['dataset'].values,
        'instance_id': uq['instance_id'].values,
        'problem': uq['problem'].values,
        'query_embedding': list(emb.astype(np.float32)),
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT)
    rprint(f'[green]wrote {OUT} ({len(out)} rows, dim {emb.shape[1]})[/green]')


if __name__ == '__main__':
    main()
