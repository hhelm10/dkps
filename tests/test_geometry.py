"""Geometry of the PKPS distances. The pair normalization can give negative squared
distances; the local normalization (default) cannot, and keeps the same paired and
flat-kernel limits; the model normalization and the pooling option give Euclidean
distances by construction."""

import numpy as np
import pandas as pd
import pytest

from dkps import PKPS
from dkps.synthetic import generate_benchmark_data
from dkps.unpaired_dkps import ProductKernelPerspectiveSpace, local_dist_sq

KW = dict(response_kwargs=dict(kernel='linear', pca_dim=None))


def _records(seed=0, n_models=24, obs_prob=0.5):
    data, scores, _, _, _ = generate_benchmark_data(
        d_latent=5, d_obs=20, n_models=n_models, n_tasks=6, n_queries_per_task=12,
        obs_prob=obs_prob, score_noise=0.1, random_state=seed)
    mi = data['model_id'].str.slice(6).astype(int)
    ti = data['task_id'].str.slice(5).astype(int)
    data = data.copy()
    data['score'] = scores[mi, ti]
    return data


def _min_centered_eig(D):
    n = len(D)
    J = np.eye(n) - 1.0 / n
    return float(np.linalg.eigvalsh(-0.5 * J @ (D ** 2) @ J).min())


def _counterexample():
    """Two models, two responses each: the pair normalization gives D^2 < 0."""
    rows = []
    for m, qs in (('a', [-1.0, 1.0]), ('b', [-2.0, 2.0])):
        for j, q in enumerate(qs):
            rows.append(dict(model_id=m, query_id=f'{m}{j}', query_embedding=np.array([q]),
                             embedding=np.eye(2)[j]))
    return pd.DataFrame(rows)


def test_pair_normalization_can_be_negative_model_cannot():
    df = _counterexample()
    kw = dict(query_kernel='rbf', response_kernel='linear', query_bandwidth=1.0)
    pair = ProductKernelPerspectiveSpace(normalization='pair', **kw)
    pair.query_bandwidth_ = 1.0
    md, names = pair._prepare_model_data(df)
    sz = {(a, b): pair._compute_SZ(md[a], md[b]) for a in names for b in names}
    d2_pair = pair._d2(sz['a', 'a'], sz['b', 'b'], sz['a', 'b'])
    assert d2_pair == pytest.approx(-0.0835658522284, abs=1e-9)

    model = ProductKernelPerspectiveSpace(normalization='model', **kw)
    model.query_bandwidth_ = 1.0
    d2_model = model._d2(sz['a', 'a'], sz['b', 'b'], sz['a', 'b'])
    assert d2_model > 0
    assert model.fit(df).dist_matrix_[0, 1] == pytest.approx(np.sqrt(d2_model))


@pytest.mark.parametrize('bandwidth', [0.05, 0.5, 5.0])
def test_model_normalization_is_euclidean(bandwidth):
    est = PKPS(query_kwargs=dict(bandwidth=bandwidth, pca_dim=None, normalization='model'),
               **KW).fit(_records())
    D = est.dist_matrix_
    assert np.isfinite(D).all() and (D >= 0).all()
    assert _min_centered_eig(D) > -1e-8 * (D ** 2).max()


def test_normalizations_agree_on_paired_data():
    data = _records(obs_prob=1.0)
    D = {norm: PKPS(query_kwargs=dict(bandwidth=0.5, pca_dim=None, normalization=norm),
                    **KW).fit(data).dist_matrix_ for norm in ('model', 'pair')}
    np.testing.assert_allclose(D['model'], D['pair'], atol=1e-10)
    # under the delta kernel all three are the paired DKPS distance
    D = {norm: PKPS(query_kwargs=dict(kernel='delta', pca_dim=None, normalization=norm),
                    **KW).fit(data).dist_matrix_ for norm in ('local', 'model', 'pair')}
    np.testing.assert_allclose(D['local'], D['pair'], atol=1e-10)
    np.testing.assert_allclose(D['model'], D['pair'], atol=1e-10)


def test_local_is_default_and_never_negative():
    assert PKPS().query_kwargs['normalization'] == 'local'
    assert PKPS(response_kwargs=dict(kernel='rbf')).query_kwargs['normalization'] == 'model'
    # the counterexample that breaks the pair normalization
    df = _counterexample()
    est = ProductKernelPerspectiveSpace(query_kernel='rbf', response_kernel='linear',
                                        query_bandwidth=1.0).fit(df)
    assert est.normalization == 'local' and est.dist_matrix_[0, 1] > 0
    # raw squared distances on sparse unpaired data, before any square root
    data = _records(obs_prob=0.3)
    X = np.stack(data['embedding'].to_numpy())
    Qall = np.stack(data['query_embedding'].to_numpy())
    mid = data['model_id'].astype('category').cat.codes.to_numpy()
    keys = (data['task_id'] + '|' + data['query_id']).to_numpy()
    _, first, qidx = np.unique(keys, return_index=True, return_inverse=True)
    sigmas = [0.01, 0.1, 1.0, 10.0, 1e4]
    D2 = local_dist_sq(X, mid, mid.max() + 1, qidx, Q=Qall[first], sigmas=sigmas)
    for s in sigmas:
        fin = D2[s][np.isfinite(D2[s])]
        assert fin.min() > -1e-10


def test_local_flat_kernel_is_mean_response_distance():
    data = _records(obs_prob=0.5)
    est = PKPS(query_kwargs=dict(bandwidth=1e6, pca_dim=None), **KW).fit(data)
    X = np.stack(data['embedding'].to_numpy())
    means = np.stack([X[(data['model_id'] == m).to_numpy()].mean(0) for m in est.model_names_])
    ref = np.linalg.norm(means[:, None] - means[None], axis=-1)
    np.testing.assert_allclose(est.dist_matrix_, ref, rtol=1e-5, atol=1e-7)


def test_local_update_matches_fresh_fit():
    data = _records()
    kw = dict(query_kwargs=dict(bandwidth=0.5, pca_dim=None), **KW)
    first = data[data['model_id'] < 'model_12']
    rest = data[data['model_id'] >= 'model_12']
    inc = PKPS(**kw).fit(first).update(rest)
    fresh = PKPS(**kw).fit(data)
    np.testing.assert_allclose(inc.dist_matrix_, fresh.dist_matrix_, atol=1e-10)


def test_unknown_normalization_rejected():
    with pytest.raises(ValueError):
        PKPS(query_kwargs=dict(normalization='nope'))
    with pytest.raises(ValueError):
        ProductKernelPerspectiveSpace(normalization='nope')


@pytest.mark.parametrize('qk', [dict(kernel='rbf', bandwidth=0.5, pca_dim=None),
                                dict(kernel='rbf', bandwidth='cv', pca_dim=None),
                                dict(kernel='delta', pca_dim=None)])
def test_pooling_is_euclidean_and_predicts(qk):
    data = _records()
    est = PKPS(query_kwargs=dict(pooling=True, **qk), **KW).fit(data)
    D = est.dist_matrix_
    assert np.isfinite(D).all()
    assert _min_centered_eig(D) > -1e-8 * (D ** 2).max()
    out = est.predict()
    assert len(out) and all(np.isfinite(r['score_hat']) for r in out)


def test_pooling_flat_kernel_is_mean_response():
    """A very wide query kernel removes the same pooled mean from every response up to
    the leave-own-model-out correction, so distances approach those between mean
    responses as the number of models grows."""
    data = _records(n_models=60, obs_prob=1.0)
    est = PKPS(query_kwargs=dict(pooling=True, bandwidth=1e6, pca_dim=None), **KW).fit(data)
    X = np.stack(data['embedding'].to_numpy())
    means = np.stack([X[(data['model_id'] == m).to_numpy()].mean(0) for m in est.model_names_])
    ref = np.linalg.norm(means[:, None] - means[None], axis=-1)
    n = len(means)
    np.testing.assert_allclose(est.dist_matrix_, ref * n / (n - 1), rtol=1e-6, atol=1e-8)


def test_pooling_update_matches_fresh_fit():
    data = _records()
    kw = dict(query_kwargs=dict(pooling=True, bandwidth=0.5, pca_dim=None), **KW)
    first = data[data['model_id'] < 'model_12']
    rest = data[data['model_id'] >= 'model_12']
    inc = PKPS(**kw).fit(first).update(rest)
    fresh = PKPS(**kw).fit(data)
    np.testing.assert_allclose(inc.dist_matrix_, fresh.dist_matrix_, atol=1e-10)


def test_pooling_requires_linear_response_kernel():
    with pytest.raises(ValueError):
        PKPS(query_kwargs=dict(pooling=True), response_kwargs=dict(kernel='rbf'))
