"""Record parsing shared by all estimators.

Records are JSON-friendly tables: a DataFrame, a list of dicts, a dict of lists, or a JSON
string of either. Fit/update rows carry (model_id, task_id, query_id, score, ...); predict
rows carry (model_id[, task_id]).
"""

import json

import numpy as np
import pandas as pd


def parse_records(records, required=(), aliases=()):
    """Normalize records into a DataFrame with the required columns present."""
    if isinstance(records, str):
        records = json.loads(records)
    if isinstance(records, pd.DataFrame):
        df = records.copy()
    elif isinstance(records, dict):
        df = pd.DataFrame(records)
    elif isinstance(records, (list, tuple)):
        df = pd.DataFrame(list(records))
    else:
        raise TypeError('records must be a DataFrame, list of dicts, dict of lists, '
                        'or a JSON string of either')
    for old, new in aliases:
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
    for col in required:
        if col not in df.columns:
            raise ValueError(f'records missing required column: {col}')
    return df


def merge_records(old, new, keys=('model_id', 'task_id', 'query_id')):
    """Append rows; a re-sent key is replaced by its newer version. Returns (merged, n_replaced)."""
    merged = pd.concat([old, new], ignore_index=True)
    dup = merged.duplicated(list(keys), keep='last')
    return merged[~dup].reset_index(drop=True), int(dup.sum())


class ScoreTable:
    """Sample-score matrix from response-level records.

    sample_ : (n_models x n_tasks) DataFrame of per-cell mean scores (NaN = unobserved).
              A cell-level 'sample_score' column, when present and non-null, overrides the
              mean (for cells scored on more responses than were embedded).
    counts_ : matching DataFrame of queries per cell
    """

    def __init__(self, df, models=None, tasks=None):
        scored = df.dropna(subset=['score'])
        g = scored.groupby(['model_id', 'task_id'])['score']
        models = sorted(df['model_id'].unique()) if models is None else list(models)
        tasks = sorted(df['task_id'].unique()) if tasks is None else list(tasks)
        self.sample_ = g.mean().unstack().reindex(index=models, columns=tasks)
        if 'sample_score' in df.columns:
            ov = df.dropna(subset=['sample_score']).groupby(['model_id', 'task_id'])[
                'sample_score'].first().unstack().reindex(index=models, columns=tasks)
            self.sample_ = self.sample_.where(ov.isna(), ov)
        self.counts_ = g.size().unstack().reindex(index=models, columns=tasks).fillna(0).astype(int)
        self.models = models
        self.tasks = tasks

    @property
    def values(self):
        return self.sample_.to_numpy(dtype=float)

    @property
    def observed(self):
        return np.isfinite(self.values)

    def index(self, m):
        return self.models.index(m)


def parse_pairs(records, models, tasks, default_pairs):
    """Predict-side input: None -> default_pairs; otherwise rows with model_id and optional
    task_id (a model without task_id requests every task)."""
    if records is None:
        return list(default_pairs)
    df = parse_records(records, required=('model_id',))
    known = set(models)
    pairs = []
    for _, r in df.iterrows():
        m = r['model_id']
        if m not in known:
            raise KeyError(f'unknown model_id: {m!r} (fit() or update() with its rows first)')
        t = r.get('task_id') if 'task_id' in df.columns else None
        if t is not None and pd.notna(t):
            pairs.append((m, t))
        else:
            pairs.extend((m, tt) for tt in tasks)
    return pairs


def pairs_to_records(pairs, preds):
    return [{'model_id': m, 'task_id': t, 'score_hat': float(preds[(m, t)])} for m, t in pairs]
