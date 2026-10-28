#!/usr/bin/env python
"""Adapter for the Every Eval Ever (EEE) datastore (evaleval/EEE_datastore on HF).

Downloads the item-level `_samples.jsonl` runs for a set of benchmarks and normalizes
them into the suite's long format: one row per (model, task, query) with the query
text, the model's raw response text, and the per-item score. Tasks are derived from
the `sample_id` prefix (e.g. `math_mc_level1`, `judge_judgebench_knowledge`), which
subdivides each benchmark the way MATH subjects subdivide MATH.

Where several runs exist for the same (benchmark, model), the newest (manifest
`added_at`) is kept. Output: data/eee/eee_<benchmark>.parquet.

Usage (from projects/pkps/scripts/):
    python data/eee.py --download          # fetch manifest + sample files (cached)
    python data/eee.py --extract           # normalize to parquet
"""
import argparse
import json
import re
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path

import pandas as pd

BASE = 'https://huggingface.co/datasets/evaleval/EEE_datastore/resolve/main/'
DATA = Path(os.environ.get('DKPS_DATA') or Path(__file__).resolve().parents[4] / 'data')
DEST = DATA / 'eee'
BENCHMARKS = ['math-mc', 'gsm-mc', 'gpqa-diamond', 'judgebench', 'reward-bench-2']


def _fetch(path, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return 'cached'
    tmp = dest.with_suffix(dest.suffix + '.part')
    urllib.request.urlretrieve(BASE + path, tmp)
    tmp.rename(dest)
    return 'ok'


def newest_runs(manifest, benchmarks):
    """Newest `_samples.jsonl` per (benchmark, developer/model)."""
    best = {}
    for path, meta in manifest['files'].items():
        if not path.endswith('_samples.jsonl'):
            continue
        parts = path.split('/')
        if len(parts) < 5 or parts[1] not in benchmarks:
            continue
        key = (parts[1], f'{parts[2]}/{parts[3]}')
        if key not in best or meta['added_at'] > best[key][1]:
            best[key] = (path, meta['added_at'])
    return {k: v[0] for k, v in best.items()}


def download(benchmarks):
    _fetch('manifest.json', DEST / 'manifest.json')
    manifest = json.load(open(DEST / 'manifest.json'))
    runs = newest_runs(manifest, benchmarks)
    print(f'{len(runs)} runs across {len(benchmarks)} benchmarks')
    jobs = {path: DEST / 'raw' / path.replace('data/', '', 1) for path in runs.values()}
    done = fail = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_fetch, p, d): p for p, d in jobs.items()}
        for f in as_completed(futs):
            try:
                f.result(); done += 1
            except Exception as e:
                fail += 1
                print(f'FAIL {futs[f]}: {e}', file=sys.stderr)
            if done % 50 == 0:
                print(f'  {done}/{len(jobs)}')
    print(f'downloaded {done}, failed {fail}')


def _task_of(sample_id, benchmark):
    """Task = the sample_id prefix before ':' (falls back to the benchmark name)."""
    head = sample_id.split(':', 1)[0] if ':' in sample_id else benchmark
    return re.sub(r'[^0-9a-zA-Z_]+', '_', head)


def extract(benchmarks):
    for bench in benchmarks:
        rows = []
        files = sorted((DEST / 'raw' / bench).rglob('*_samples.jsonl'))
        for fp in files:
            dev, model = fp.parts[-3], fp.parts[-2]
            with open(fp) as fh:
                for line in fh:
                    r = json.loads(line)
                    out = r.get('output', {}).get('raw')
                    if isinstance(out, list):
                        out = out[0] if out else None
                    ev = r.get('evaluation', {})
                    score = ev.get('score', float(ev.get('is_correct', float('nan'))))
                    rows.append(dict(
                        model=f'{dev}/{model}',
                        task=_task_of(r.get('sample_id', ''), bench),
                        query_id=r.get('sample_id'),
                        query=r.get('input', {}).get('raw'),
                        response=out,
                        score=score,
                    ))
        df = pd.DataFrame(rows).dropna(subset=['query', 'response', 'score'])
        out_path = DEST / f'eee_{bench}.parquet'
        df.to_parquet(out_path)
        print(f'{bench}: {df.model.nunique()} models, {df.task.nunique()} tasks, '
              f'{df.query_id.nunique()} queries, {len(df)} rows -> {out_path.name}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--download', action='store_true')
    ap.add_argument('--extract', action='store_true')
    ap.add_argument('--benchmarks', nargs='+', default=BENCHMARKS)
    args = ap.parse_args()
    if args.download:
        download(args.benchmarks)
    if args.extract:
        extract(args.benchmarks)
    if not (args.download or args.extract):
        ap.print_help()
