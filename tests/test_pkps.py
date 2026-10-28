"""Tests for the high-level PKPS / DKPS fit / predict / update API."""

import json

import numpy as np
import pandas as pd
import pytest

from dkps import PKPS, DKPS, DataKernelPerspectiveSpace
from dkps.synthetic import generate_benchmark_data


def make_records(seed=0, n_models=40, n_tasks=8, obs_prob=0.6):
    """Synthetic benchmark records: one row per (model, task, query) response with
    the cell's true score attached to every response row."""
    data, scores, observed, _, _ = generate_benchmark_data(
        d_latent=5, d_obs=20, n_models=n_models, n_tasks=n_tasks,
        n_queries_per_task=16, obs_prob=obs_prob, score_noise=0.1,
        random_state=seed,
    )
    mi = data['model_id'].str.slice(6).astype(int)
    ti = data['task_id'].str.slice(5).astype(int)
    data = data.copy()
    data['score'] = scores[mi, ti]
    return data, scores, observed


def test_fit_predict_beats_task_mean():
    records, scores, observed = make_records()
    est = PKPS().fit(records)
    preds = est.predict()  # all cells without an observed sample score

    task_mean = {t: np.nanmean(scores[observed[:, k], k])
                 for k, t in enumerate(est.task_names_)}
    err_pkps, err_mean = [], []
    for r in preds:
        i = int(r['model_id'][6:])
        k = int(r['task_id'][5:])
        assert not observed[i, k]
        err_pkps.append(abs(r['score_hat'] - scores[i, k]))
        err_mean.append(abs(task_mean[r['task_id']] - scores[i, k]))
    assert len(preds) > 20
    assert np.mean(err_pkps) < 0.8 * np.mean(err_mean)


def test_predict_records_input_and_json_round_trip():
    records, scores, observed = make_records(n_models=20, n_tasks=4)
    # JSON string input for fit
    payload = json.dumps([
        {**r, 'embedding': list(r['embedding']),
         'query_embedding': list(r['query_embedding'])}
        for r in records.to_dict('records')
    ])
    est = PKPS().fit(payload)
    # dict-of-lists input for predict; model without task_id requests every task
    out = est.predict({'model_id': ['model_00']})
    assert {r['task_id'] for r in out} == set(est.task_names_)
    out = est.predict([{'model_id': 'model_01', 'task_id': 'task_000'}])
    assert len(out) == 1 and np.isfinite(out[0]['score_hat'])
    with pytest.raises(KeyError):
        est.predict([{'model_id': 'not_a_model'}])


def test_dkps_delta_matches_legacy_paired_distance():
    records, _, _ = make_records(n_models=12, n_tasks=3, obs_prob=1.0)
    est = DKPS(response_kwargs=dict(pca_dim=None)).fit(records)

    legacy = DataKernelPerspectiveSpace()
    legacy.fit_transform(records[['model_id', 'query_id', 'embedding']])
    # legacy: ||X_i - X_k||_F / sqrt(M); delta-kernel PKPS: sqrt(mean_j ||x_ij - x_kj||^2)
    np.testing.assert_allclose(est.dist_matrix_, legacy.dist_matrix_, atol=1e-8)


def test_update_equals_fresh_fit():
    # with data-independent reducers (no PCA, fixed bandwidth), incremental update
    # must reproduce a fresh fit on the union exactly
    records, _, _ = make_records(n_models=25, n_tasks=4)
    kw = dict(
        query_kwargs=dict(kernel='rbf', bandwidth=1.0, pca_dim=None),
        response_kwargs=dict(kernel='linear', pca_dim=None),
    )
    first = records[records['model_id'] < 'model_20']
    rest = records[records['model_id'] >= 'model_20']

    inc = PKPS(**kw).fit(first).update(rest)
    fresh = PKPS(**kw).fit(records)

    assert inc.model_names_ == fresh.model_names_
    np.testing.assert_allclose(inc.dist_matrix_, fresh.dist_matrix_, atol=1e-10)
    pd.testing.assert_frame_equal(inc.sample_scores_, fresh.sample_scores_)
    assert inc.predict() == fresh.predict()


def test_update_replaces_duplicate_rows():
    records, _, _ = make_records(n_models=10, n_tasks=3)
    kw = dict(query_kwargs=dict(bandwidth=1.0, pca_dim=None),
              response_kwargs=dict(pca_dim=None))
    est = PKPS(**kw).fit(records)
    # re-send one model's rows with shifted scores; the newer rows must win
    redo = records[records['model_id'] == 'model_00'].copy()
    redo['score'] = redo['score'] + 1.0
    with pytest.warns(UserWarning, match='replaced'):
        est.update(redo)
    old = records[records['model_id'] == 'model_00'].groupby('task_id')['score'].mean()
    new = est.sample_scores_.loc['model_00'].dropna()
    np.testing.assert_allclose(new.values, (old + 1.0).reindex(new.index).values)
