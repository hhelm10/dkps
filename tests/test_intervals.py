"""Confidence intervals for PKPS/DKPS predictions (predict(interval=...)).

Conformal intervals are calibrated on leave-one-reference-out residuals, so their
marginal coverage over exchangeable models should track the nominal level; the knn
method is a normal approximation with no guarantee but should behave sanely.
"""
import numpy as np
import pytest

from dkps import PKPS, DKPS, generate_benchmark_data


def _records(seed=0, n_models=40, obs_prob=1.0):
    data, scores, observed, _, _ = generate_benchmark_data(
        d_latent=5, d_obs=20, n_models=n_models, n_tasks=6, n_queries_per_task=16,
        obs_prob=obs_prob, random_state=seed)
    mi = data.model_id.str[6:].astype(int)
    ti = data.task_id.str[5:].astype(int)
    data['score'] = 1 / (1 + np.exp(-scores[mi, ti]))
    return data, scores


@pytest.fixture(scope='module')
def fitted():
    data, scores = _records()
    est = PKPS().fit(data)
    truth = 1 / (1 + np.exp(-scores))
    return est, truth


def test_interval_keys_and_ordering(fitted):
    est, _ = fitted
    pairs = [{'model_id': m, 'task_id': t} for m in est.model_names_[:10]
             for t in est.task_names_]
    out = est.predict(pairs, interval=0.9)
    assert all({'score_hat', 'score_lo', 'score_hi'} <= set(r) for r in out)
    for r in out:
        if np.isfinite(r['score_hat']):
            assert r['score_lo'] <= r['score_hat'] <= r['score_hi']


def test_no_interval_backcompat(fitted):
    est, _ = fitted
    out = est.predict([{'model_id': est.model_names_[0], 'task_id': est.task_names_[0]}])
    assert 'score_lo' not in out[0] and 'score_hi' not in out[0]


def test_interval_monotone_in_level(fitted):
    est, _ = fitted
    pairs = [{'model_id': m, 'task_id': est.task_names_[0]} for m in est.model_names_[:12]]
    narrow = est.predict(pairs, interval=0.5)
    wide = est.predict(pairs, interval=0.95)
    for a, b in zip(narrow, wide):
        if np.isfinite(a['score_lo']) and np.isfinite(b['score_lo']):
            assert (b['score_hi'] - b['score_lo']) >= (a['score_hi'] - a['score_lo']) - 1e-12


def test_coverage_tracks_level(fitted):
    """Leave-one-model-out conformal intervals should cover the true scores at
    roughly the nominal level (guaranteed marginally over exchangeable models)."""
    est, truth = fitted
    pairs = [{'model_id': m, 'task_id': t} for m in est.model_names_
             for t in est.task_names_]
    out = est.predict(pairs, interval=0.8)
    hits, n = 0, 0
    for r in out:
        if not np.isfinite(r['score_lo']):
            continue
        mi = int(r['model_id'][6:]); ti = int(r['task_id'][5:])
        n += 1
        hits += r['score_lo'] <= truth[mi, ti] <= r['score_hi']
    assert n > 100
    cover = hits / n
    assert cover >= 0.65, f'coverage {cover:.2f} far below nominal 0.8'


def test_conformal_infinite_when_too_few_references():
    data, _ = _records(n_models=6)
    est = PKPS().fit(data)
    out = est.predict([{'model_id': est.model_names_[0], 'task_id': est.task_names_[0]}],
                      interval=0.99)  # needs ~99 calibration points; only ~5 exist
    assert np.isinf(out[0]['score_lo']) and np.isinf(out[0]['score_hi'])


def test_whiten_intervals_in_unit_range(fitted):
    est, _ = fitted
    pairs = [{'model_id': m, 'task_id': est.task_names_[1]} for m in est.model_names_[:12]]
    out = est.predict(pairs, interval=0.9, whiten=True)
    for r in out:
        if np.isfinite(r['score_lo']):
            assert 0.0 <= r['score_lo'] <= r['score_hi'] <= 1.0


def test_family_holdout_and_dkps(fitted):
    est, _ = fitted
    out = est.predict([{'model_id': est.model_names_[0], 'task_id': est.task_names_[0]}],
                      interval=0.8, holdout='family',
                      family_fn=lambda m: int(m[6:]) % 4)   # synthetic ids share one prefix
    assert np.isfinite(out[0]['score_hat'])
    assert out[0]['score_lo'] <= out[0]['score_hat'] <= out[0]['score_hi']

    data, _ = _records(seed=1)
    dk = DKPS().fit(data)
    out = dk.predict([{'model_id': dk.model_names_[0], 'task_id': dk.task_names_[0]}],
                     interval=0.8)
    assert out[0]['score_lo'] <= out[0]['score_hat'] <= out[0]['score_hi']


def test_score_table_carries_bounds():
    data, _ = _records(seed=2, obs_prob=0.6)
    est = PKPS().fit(data)
    tbl = est.score_table(interval=0.9)
    pred = tbl[tbl.source == 'predicted']
    assert {'score_lo', 'score_hi'} <= set(tbl.columns)
    assert pred['score_lo'].notna().any()
