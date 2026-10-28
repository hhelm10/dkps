"""Adapter for the Every Eval Ever (EEE) datastore.

Turns EEE instance-level ``*_samples.jsonl`` records (schema ``instance_level_eval_*``,
huggingface.co/datasets/evaleval/EEE_datastore) into the record table every dkps
estimator consumes, so store data plugs straight into PKPS:

    from dkps import PKPS
    from dkps.eee import fetch_samples, load_records

    paths = fetch_samples(['gsm-mc'], dest='eee_cache')      # or your own run files
    records = load_records(paths)
    est = PKPS(embedding_kwargs=dict(api_key=...)).fit(records)   # embeds the raw text
    est.predict([{'model_id': 'openai/gpt-oss-20b'}])

Normalization applied per record: model from ``model_id`` (or the run path), task from
``metadata.source_task`` (falling back to the ``sample_id`` prefix), query text from
``input.raw``, response text from ``output.raw`` (first element when a list), and the
score from ``evaluation.score`` with ``evaluation.is_correct`` as fallback -- both
parsed from the strings the store uses ("0.0", "False") -- and the benchmark name from
``evaluation_id`` as a ``suite`` column, which makes PKPS.fit build the paper's
block-diagonal multi-benchmark response space automatically. Rows without a query,
response, or score are dropped and counted. When several runs cover the same
(model, task, query), the newest file wins.
"""

import json
import re
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

EEE_BASE = 'https://huggingface.co/datasets/evaleval/EEE_datastore/resolve/main/'


def _parse_score(ev):
    """EEE stores numbers and booleans as strings; prefer 'score', else 'is_correct'."""
    for key in ('score', 'is_correct'):
        v = ev.get(key)
        if v is None:
            continue
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ('true', 'false'):
                return float(s == 'true')
            try:
                return float(s)
            except ValueError:
                continue
        if isinstance(v, bool):
            return float(v)
        if isinstance(v, (int, float)):
            return float(v)
    return np.nan


def _task_of(record, suite=None):
    src = (record.get('metadata') or {}).get('source_task')
    if not src:
        sid = record.get('sample_id') or ''
        src = sid.split(':', 1)[0] if ':' in sid else (suite or 'task')
    return re.sub(r'[^0-9a-zA-Z_]+', '_', str(src))


def _suite_of(record):
    """Benchmark name from the record itself. evaluation_id is 'benchmark/dev_model/ts'
    on some benchmarks and a bare run UUID on others; evaluation_name is the fallback."""
    eid = record.get('evaluation_id') or ''
    if '/' in eid:
        return eid.split('/')[0]
    return record.get('evaluation_name') or None


def _response_of(record):
    """Response text: output.raw (string or list) for single-turn records; for
    multi-turn records (arenas, agent benchmarks) output is empty and the transcript
    lives in 'messages' -- the model's own turns, joined, are the response."""
    out = record.get('output')
    raw = out.get('raw') if isinstance(out, dict) else None
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    if isinstance(raw, str) and raw:
        return raw
    msgs = record.get('messages')
    if isinstance(msgs, list):
        turns = [m.get('content') for m in msgs
                 if isinstance(m, dict) and m.get('role') == 'assistant'
                 and isinstance(m.get('content'), str) and m.get('content')]
        if turns:
            return '\n'.join(turns)
    return None


def record_to_row(record, model_id=None, suite=None):
    """One EEE instance-level dict -> one dkps record row (or None if unusable).
    model_id / suite override what the record carries (the store's directory layout
    is the more reliable source for both)."""
    out = _response_of(record)
    query = (record.get('input') or {}).get('raw')
    score = _parse_score(record.get('evaluation') or {})
    model = model_id or record.get('model_id')
    qid = record.get('sample_id')
    if not model or qid is None or not query or not out:
        return None
    suite = suite or _suite_of(record)
    row = {'model_id': str(model), 'task_id': _task_of(record, suite), 'query_id': str(qid),
           'query': str(query), 'response': out,
           'score': score if np.isfinite(score) else np.nan}
    if suite:
        row['suite'] = suite
    return row


def load_records(paths, max_queries_per_cell=32, seed=0, model_from='auto'):
    """Read EEE ``*_samples.jsonl`` files (paths or directories) into a dkps record
    DataFrame.

    max_queries_per_cell : cap each (model, task) cell at this many queries (seeded
        subsample; the paper's pools use 32). PKPS cost grows with the product of two
        models' per-cell counts, so fitting full-depth cells (often 1000+ items) is
        slow -- pass None to keep everything.
    model_from : where the model id comes from. 'path' takes ``<developer>/<model>``
        from the store's directory layout (``.../<benchmark>/<developer>/<model>/
        <run>_samples.jsonl``) -- the store's normalized naming, shared by the
        manifest. 'record' takes the in-file ``model_id`` (the raw repo name, which
        can differ in casing and developer label). 'auto' (default) uses the path
        when the file sits at least two directories deep, else the record.
    """
    files = []
    for p in (paths if isinstance(paths, (list, tuple)) else [paths]):
        p = Path(p)
        files += sorted(p.rglob('*_samples.jsonl')) if p.is_dir() else [p]
    if not files:
        raise ValueError('no *_samples.jsonl files found')
    rows, dropped = [], 0
    for order, fp in enumerate(sorted(files, key=lambda f: f.stat().st_mtime)):
        use_path = model_from == 'path' or (model_from == 'auto' and len(fp.parts) >= 3)
        path_id = f'{fp.parts[-3]}/{fp.parts[-2]}' if use_path else None
        path_suite = fp.parts[-4] if (use_path and len(fp.parts) >= 4) else None
        with open(fp) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = record_to_row(json.loads(line), model_id=path_id, suite=path_suite)
                if row is None or not np.isfinite(row['score']):
                    dropped += 1
                    continue
                row['_order'] = order
                rows.append(row)
    if not rows:
        raise ValueError('no usable records (every row lacked a query, response, or score)')
    df = pd.DataFrame(rows).sort_values('_order', kind='stable')
    n0 = len(df)
    df = df.drop_duplicates(['model_id', 'task_id', 'query_id'], keep='last')
    n_dup = n0 - len(df)
    df = df.drop(columns='_order').reset_index(drop=True)
    if dropped or n_dup:
        warnings.warn(f'load_records: dropped {dropped} unusable rows; '
                      f'{n_dup} duplicates resolved to the newest run', stacklevel=2)
    if max_queries_per_cell is not None:
        rng = np.random.default_rng(seed)
        keep_idx = []
        for _, idx in df.groupby(['model_id', 'task_id'], sort=False).indices.items():
            if len(idx) > max_queries_per_cell:
                idx = np.sort(rng.choice(idx, max_queries_per_cell, replace=False))
            keep_idx.append(idx)
        df = df.iloc[np.concatenate(keep_idx)].sort_index().reset_index(drop=True)
    return df


def fetch_samples(benchmarks='all', dest='eee_cache', newest_only=True):
    """Download the ``*_samples.jsonl`` runs for the given benchmarks from the EEE
    datastore on Hugging Face (cached in ``dest``). Returns the local file paths.

    benchmarks : list of benchmark names, or 'all' (default) for every benchmark in
        the manifest.
    newest_only : keep only the newest run per (benchmark, model) by the manifest's
        ``added_at`` (the store often holds several runs of the same model).
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    mpath = dest / 'manifest.json'
    if not mpath.exists():
        urllib.request.urlretrieve(EEE_BASE + 'manifest.json', mpath)
    manifest = json.load(open(mpath))
    best = {}
    for path, meta in manifest['files'].items():
        if not path.endswith('_samples.jsonl'):
            continue
        parts = path.split('/')
        if len(parts) < 5 or (benchmarks != 'all' and parts[1] not in benchmarks):
            continue
        key = (parts[1], f'{parts[2]}/{parts[3]}') if newest_only else path
        if key not in best or meta['added_at'] > best[key][1]:
            best[key] = (path, meta['added_at'])
    out = []
    for path, _ in best.values():
        local = dest / path.replace('data/', '', 1)
        local.parent.mkdir(parents=True, exist_ok=True)
        if not (local.exists() and local.stat().st_size > 0):
            tmp = local.with_suffix('.part')
            urllib.request.urlretrieve(EEE_BASE + path, tmp)
            tmp.rename(local)
        out.append(local)
    if not out:
        raise ValueError(f'no runs found for benchmarks {benchmarks!r} in the manifest')
    return sorted(out)
