"""Download a pinned EEE cohort and prepare a deterministic bounded response pool.

Raw files stay in --source; derived inputs stay in --out. Only latest runs for the
shared five-benchmark cohort are needed, not every historical datastore object.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from dkps.eee import record_to_row, _parse_score, _task_of

BENCHMARKS = ('math-mc', 'gsm-mc', 'gpqa-diamond', 'judgebench', 'reward-bench-2')
REVISION = '8b9632422dd2b3806431efd758c7cdc84044169f'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def selected_runs(manifest, benchmarks=BENCHMARKS):
    latest = {}
    for path, meta in manifest['files'].items():
        parts = path.split('/')
        if (len(parts) != 5 or parts[0] != 'data' or parts[1] not in benchmarks
                or not path.endswith('_samples.jsonl')):
            continue
        key = (parts[1], '/'.join(parts[2:4]))
        if key not in latest or meta['added_at'] > latest[key][1]['added_at']:
            latest[key] = (path, meta)
    shared = set.intersection(*(set(m for b, m in latest if b == bench) for bench in benchmarks))
    if not shared:
        raise ValueError('no models shared by the requested benchmark collections')
    return {key: value for key, value in sorted(latest.items()) if key[1] in shared}


def fetch(source, revision, path, expected=None):
    dest = source / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (expected is not None and digest(dest) == expected):
        return dest
    temp = dest.with_suffix(dest.suffix + '.part')
    url = f'https://huggingface.co/datasets/evaleval/EEE_datastore/resolve/{revision}/{path}'
    subprocess.run(['curl', '-fLsS', '--retry', '3', '--connect-timeout', '20',
                    '--max-time', '300', '--output', str(temp), url], check=True)
    if expected is not None and digest(temp) != expected:
        raise ValueError(f'SHA256 mismatch: {path}')
    temp.replace(dest)
    return dest


def preserve_metadata(source, out, revision, workers=4):
    """Retain the exact aggregate companions, including generation configurations."""
    manifest = json.loads((source / 'manifest.json').read_text())
    marker = json.loads((source / 'pkps_snapshot.json').read_text())
    if marker['revision'] != revision:
        raise ValueError('metadata revision differs from downloaded samples')
    runs = selected_runs(manifest)
    paths = [path.replace('_samples.jsonl', '.json') for path, _ in runs.values()]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        jobs = [executor.submit(fetch, source, revision, p, manifest['files'][p].get('sha256')) for p in paths]
        for future in as_completed(jobs):
            future.result()
    result = []
    for (bench, model), (path, _) in runs.items():
        companion = path.replace('_samples.jsonl', '.json')
        record = json.loads((source / companion).read_text())
        result.append(dict(path=companion, sha256=digest(source / companion), suite=bench,
                           model_id=model, schema_version=record.get('schema_version'),
                           evaluation_timestamp=record.get('evaluation_timestamp'),
                           source_metadata=record.get('source_metadata'),
                           eval_library=record.get('eval_library'),
                           evaluations=[{k: ev.get(k) for k in ['evaluation_name', 'source_data',
                               'metric_config', 'generation_config', 'evaluation_timestamp']}
                                        for ev in record.get('evaluation_results', [])]))
    (out / 'run_metadata.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'preserved metadata for {len(result)} pinned runs', flush=True)


def prepare(source, out, revision, cap=32, download=False, workers=4):
    if cap < 1:
        raise ValueError('pool cap must be positive')
    out.mkdir(parents=True, exist_ok=True)
    # A revision-specific marker prevents mixing an old manifest with new URLs.
    marker = source / 'pkps_snapshot.json'
    if marker.exists() and json.loads(marker.read_text())['revision'] != revision:
        raise ValueError('source contains a different snapshot; use a new source directory')
    if download:
        fetch(source, revision, 'manifest.json')
        marker.write_text(json.dumps({'revision': revision}, indent=2) + '\n')
    manifest = json.loads((source / 'manifest.json').read_text())
    if not marker.exists():
        raise ValueError('snapshot provenance missing; run once with --download')
    runs = selected_runs(manifest)
    print(f'{len(runs)} runs, {len(set(m for _, m in runs))} shared models', flush=True)
    if download:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            jobs = {executor.submit(fetch, source, revision, path, meta.get('sha256')): path
                    for path, meta in runs.values()}
            for n, future in enumerate(as_completed(jobs), 1):
                future.result()
                if n % 10 == 0 or n == len(jobs):
                    print(f'downloaded/verified {n}/{len(jobs)}', flush=True)

    pools, targets, provenance = [], [], []
    for (bench, model), (path, meta) in runs.items():
        file = source / path
        sha = digest(file)
        if meta.get('sha256') and sha != meta['sha256']:
            raise ValueError(f'local file checksum mismatch: {path}')
        rows, scored_rows, rejected = [], [], 0
        with file.open() as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                score = _parse_score(record.get('evaluation') or {})
                # A scored empty response still contributes to benchmark truth.
                # Response eligibility only controls the embedding pool.
                if np.isfinite(score):
                    scored_rows.append(dict(task_id=f'{bench}:{_task_of(record, bench)}',
                                            query_id=f"{bench}:{record.get('sample_id')}", score=score))
                row = record_to_row(record, model_id=model, suite=bench)
                if row is None or not np.isfinite(row['score']):
                    rejected += 1
                    continue
                row['task_id'] = f"{bench}:{row['task_id']}"
                row['query_id'] = f"{bench}:{row['query_id']}"
                rows.append(row)
        frame = pd.DataFrame(rows)
        if frame.empty:
            raise ValueError(f'no usable rows in {path}')
        if frame.duplicated(['task_id', 'query_id']).any():
            raise ValueError(f'duplicate query IDs within run: {path}')
        full = pd.DataFrame(scored_rows)
        if full.duplicated(['task_id', 'query_id']).any():
            raise ValueError(f'duplicate scored query IDs within run: {path}')
        for task, group in full.groupby('task_id', sort=True):
            targets.append(dict(model_id=model, task_id=task, suite=bench,
                                y_full=float(group.score.mean()), depth=len(group)))
        for task, group in frame.groupby('task_id', sort=True):
            seed = int(hashlib.sha256(f'{model}|{task}'.encode()).hexdigest()[:8], 16)
            group = group.sort_values('query_id')
            take = np.random.default_rng(seed).permutation(len(group))[:cap]
            pools.append(group.iloc[take])
        provenance.append(dict(path=path, model_id=model, suite=bench, sha256=sha,
                               added_at=meta['added_at'], n_rows=len(frame), n_scored_rows=len(full),
                               rejected=rejected))
    pool = pd.concat(pools, ignore_index=True).sort_values(
        ['model_id', 'task_id', 'query_id']).reset_index(drop=True)
    inconsistent = pool.groupby('query_id')['query'].nunique()
    if (inconsistent > 1).any():
        raise ValueError('a pooled query ID maps to different prompt text across models')
    pool.to_parquet(out / 'pool.parquet', index=False)
    pd.DataFrame(targets).to_parquet(out / 'targets.parquet', index=False)
    metadata = dict(dataset='evaleval/EEE_datastore', revision=revision,
                    source=str(source.resolve()), manifest_sha256=digest(source / 'manifest.json'),
                    cap=cap, sampler='sorted-query-id/per-model-task-sha256-seed-v1',
                    n_models=pool.model_id.nunique(), n_tasks=pool.task_id.nunique(),
                    n_rows=len(pool), n_unique_queries=pool.query_id.nunique(), runs=provenance)
    (out / 'dataset_manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print({k: v for k, v in metadata.items() if k != 'runs'}, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=Path('data/eee-followups'))
    ap.add_argument('--revision', default=REVISION)
    ap.add_argument('--cap', type=int, default=32)
    ap.add_argument('--download', action='store_true')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--metadata-only', action='store_true', help='fetch companions after sample preparation')
    args = ap.parse_args()
    if args.metadata_only:
        preserve_metadata(args.source, args.out, args.revision, args.workers)
    else:
        prepare(args.source, args.out, args.revision, args.cap, args.download, args.workers)
        if args.download:
            preserve_metadata(args.source, args.out, args.revision, args.workers)


if __name__ == '__main__':
    main()
