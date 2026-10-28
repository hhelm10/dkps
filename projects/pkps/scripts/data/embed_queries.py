#!/usr/bin/env python
"""
embed_queries.py  --  Google-embed the bare question text of a HELM dataset's
instances (for the PKPS query kernel), mirroring embed_math_queries.py.

The `query` column is a full few-shot prompt whose formatting varies by model.
The instance-specific content is the final '\\n\\n'-separated block (the actual
question / description). A few models use a non-standard template, so we take the
MODAL bare query per instance. Writes exports/{dataset}_query_google_embeddings.parquet.

Loads GEMINI_API_KEY from the repo-root .env.
"""
import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from rich import print as rprint

load_dotenv(Path(__file__).resolve().parents[2] / '.env')
from dkps.embed import embed_api  # noqa: E402

csv.field_size_limit(sys.maxsize)
MODEL = 'gemini-embedding-001'


def bare(query):
    return str(query).split('\n\n')[-1].strip()


def modal_queries(tsv):
    by_inst = defaultdict(Counter)
    meta = {}
    with open(tsv) as f:
        for row in csv.DictReader(f, delimiter='\t'):
            iid = row['instance_id']
            by_inst[iid][bare(row['query'])] += 1
            meta.setdefault(iid, row['dataset'])
    inst = sorted(by_inst)
    return inst, [meta[i] for i in inst], [by_inst[i].most_common(1)[0][0] for i in inst]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset', help='med_qa | legalbench | ...')
    ap.add_argument('--tsv', default=None)
    args = ap.parse_args()
    tsv = args.tsv or DATA / f'helm/{args.dataset}.tsv'
    out = DATA / f'exports/{args.dataset}_query_google_embeddings.parquet'

    inst, dsets, queries = modal_queries(tsv)
    rprint(f'[blue]{args.dataset}: embedding {len(inst)} unique instances via google/{MODEL} ...[/blue]')
    emb = embed_api('google', [str(q) for q in queries], model=MODEL)
    assert emb.shape[0] == len(inst), (emb.shape, len(inst))
    rprint(f'[green]embeddings: {emb.shape}[/green]')

    df = pd.DataFrame({
        'query_id': [f'{d}::{i}' for d, i in zip(dsets, inst)],
        'dataset': dsets,
        'instance_id': inst,
        'problem': queries,
        'query_embedding': list(emb.astype(np.float32)),
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    rprint(f'[green]wrote {out} ({len(df)} rows, dim {emb.shape[1]})[/green]')


if __name__ == '__main__':
    main()
