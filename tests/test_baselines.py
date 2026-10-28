"""Baseline / preprocessing / ensemble classes agree with the functions the paper's
experiment scripts call."""
import sys, pathlib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'projects' / 'pkps' / 'scripts'))
from pipeline import loaders as H                       # noqa: E402
from pipeline.query_select import _lofo_regress, family as fam_fn, max_dense_block as mdb_ref  # noqa: E402

from dkps import PKPS, SampleScore, IRT, LRMC, Ensemble, Whitener   # noqa: E402
from dkps.baselines import matrix_completion_predict, max_dense_block, \
    irt_fit_difficulties, irt_estimate_ability, irt_predict            # noqa: E402
from dkps.synthetic import generate_benchmark_data                    # noqa: E402


def _records(seed=0, n_models=30, n_tasks=6, obs_prob=0.6, binary=False):
    data, scores, observed, _, _ = generate_benchmark_data(
        d_latent=5, d_obs=20, n_models=n_models, n_tasks=n_tasks, n_queries_per_task=12,
        obs_prob=obs_prob, score_noise=0.1, random_state=seed)
    rng = np.random.default_rng(seed)
    mi = data['model_id'].str.slice(6).astype(int); ti = data['task_id'].str.slice(5).astype(int)
    p = 1 / (1 + np.exp(-scores))                          # cell probabilities in (0, 1)
    data = data.copy()
    data['score'] = (rng.random(len(data)) < p[mi, ti]).astype(float) if binary else p[mi, ti]
    data['model_id'] = data['model_id'].map(lambda m: f'dev{int(m[6:]) % 5}_{m}')   # 5 families
    return data, p, observed


def test_whitener_matches_pipeline():
    rng = np.random.default_rng(0)
    M = rng.random((12, 5)); O = rng.random((12, 5)) < 0.7
    tgt, bias, mu, sd = H._whiten(M, O)
    w = Whitener().fit(M, O)
    np.testing.assert_allclose(w.transform(M), tgt)
    np.testing.assert_allclose(w.bias_, bias); np.testing.assert_allclose(w.mu_, mu)
    np.testing.assert_allclose(w.sd_, sd)
    R = rng.normal(size=M.shape)
    np.testing.assert_allclose(w.inverse_transform(R),
                               np.clip(1 / (1 + np.exp(-((bias + R) * sd + mu))), 0, 1))


def test_sample_score_and_lrmc_match_functions():
    recs, p, observed = _records()
    ss = SampleScore().fit(recs)
    M = ss.sample_scores_.to_numpy(); O = np.isfinite(M)
    out = ss.predict()
    assert len(out) == O.sum() and all(np.isfinite(r['score_hat']) for r in out)

    lr = LRMC(crossfit_k=0, n_init=2).fit(recs)
    ref = matrix_completion_predict(M, O, n_init=2)
    np.testing.assert_allclose(lr.pred_matrix_[~O], ref[~O])
    assert np.isnan(lr.pred_matrix_[O]).all()
    lr3 = LRMC(crossfit_k=3, n_init=2).fit(recs)
    assert np.isfinite(lr3.pred_matrix_[O]).all()            # observed cells cross-fit
    np.testing.assert_allclose(lr3.pred_matrix_[~O], ref[~O])


def test_irt_matches_functions():
    recs, p, observed = _records(binary=True)
    est = IRT().fit(recs)
    t = est.task_names_[0]
    sub = recs[recs['task_id'] == t]
    S = sub.pivot(index='model_id', columns='query_id', values='score')
    A = S.to_numpy(); amask = np.isfinite(A)
    F, Q = max_dense_block(amask)
    F2, Q2 = mdb_ref(amask)
    np.testing.assert_array_equal(F, F2); np.testing.assert_array_equal(Q, Q2)
    beta, _ = irt_fit_difficulties(np.nan_to_num(A[np.ix_(F, Q)], nan=0.0))
    m = S.index[F[0]]
    want = irt_predict(irt_estimate_ability(A[F[0], Q], beta), beta)
    got = est.predict([{'model_id': m, 'task_id': t}])[0]['score_hat']
    assert got == pytest.approx(want)


def test_irt_warns_on_non_binary():
    recs, _, _ = _records(binary=False)
    with pytest.warns(UserWarning, match='non-binary'):
        IRT().fit(recs)


def test_pkps_family_holdout_matches_lofo_regress():
    recs, p, observed = _records()
    est = PKPS(query_kwargs=dict(bandwidth=1.0, pca_dim=None), response_kwargs=dict(pca_dim=None),
               mds_kwargs=dict(dim=8)).fit(recs)
    models = est.model_names_
    Z = np.stack([est.embedding_[m] for m in models]); n2r = {m: i for i, m in enumerate(models)}
    fams = {m: fam_fn(m) for m in models}
    t = est.task_names_[1]
    ymap = est.sample_scores_[t].to_dict()
    ref = _lofo_regress(Z, n2r, models, {m: (v if np.isfinite(v) else np.nan) for m, v in ymap.items()},
                        fams, predictor='knn')
    got = est.predict([{'model_id': m, 'task_id': t} for m in models], holdout='family')
    got = np.array([r['score_hat'] for r in got])
    np.testing.assert_allclose(np.where(np.isfinite(got), np.clip(got, 0, 1), np.nan), ref, equal_nan=True)


def test_pkps_whiten_matches_predict_from_embedding_all():
    recs, p, observed = _records()
    est = PKPS(query_kwargs=dict(bandwidth=1.0, pca_dim=None), response_kwargs=dict(pca_dim=None),
               mds_kwargs=dict(dim=8)).fit(recs)
    models = est.model_names_
    Z = np.stack([est.embedding_[m] for m in models])
    M = est.sample_scores_.to_numpy(); O = np.isfinite(M)
    ref = H.predict_from_embedding_all(Z, np.nan_to_num(M, nan=0.5), O, predictor='knn', k=5)
    pairs = [{'model_id': m, 'task_id': t} for m in models for t in est.task_names_]
    got = np.array([r['score_hat'] for r in est.predict(pairs, whiten=True)]).reshape(M.shape)
    np.testing.assert_allclose(got, ref, equal_nan=True, atol=1e-10)


def test_pkps_cv_bandwidth_selects_from_grid():
    recs, p, observed = _records()
    est = PKPS(query_kwargs=dict(bandwidth='cv', bandwidth_ref=1.0, pca_dim=None),
               response_kwargs=dict(pca_dim=None), mds_kwargs=dict(dim=8)).fit(recs)
    assert est.query_bandwidth_ in [0.03, 0.1, 0.3, 1.0, 3.0]
    assert set(est.embeddings_by_bandwidth_) == {0.03, 0.1, 0.3, 1.0, 3.0}
    # incremental update still equals a fresh fit (reducers data-independent)
    first = recs[recs['model_id'] < 'dev2']; rest = recs[recs['model_id'] >= 'dev2']
    kw = dict(query_kwargs=dict(bandwidth='cv', bandwidth_ref=1.0, pca_dim=None),
              response_kwargs=dict(pca_dim=None), mds_kwargs=dict(dim=8))
    inc = PKPS(**kw).fit(first).update(rest)
    np.testing.assert_allclose(inc.dist_matrix_, est.dist_matrix_, atol=1e-10)
    assert inc.query_bandwidth_ == est.query_bandwidth_


def test_ensemble_fixed_and_cv():
    recs, p, observed = _records()
    kw = dict(query_kwargs=dict(bandwidth=1.0, pca_dim=None), response_kwargs=dict(pca_dim=None),
              mds_kwargs=dict(dim=8))
    ens = Ensemble([PKPS(**kw), LRMC(n_init=2)], mode='fixed', alpha=0.3, holdout=None,
                   predict_kwargs=[{'whiten': True}, {}]).fit(recs)
    out = ens.predict()
    pk = {(r['model_id'], r['task_id']): r['score_hat'] for r in ens.members[0].predict(whiten=True)}
    mc = {(r['model_id'], r['task_id']): r['score_hat'] for r in ens.members[1].predict()}
    for r in out:
        key = (r['model_id'], r['task_id'])
        assert r['score_hat'] == pytest.approx(np.clip(0.3 * pk[key] + 0.7 * mc[key], 0, 1))
    ens_cv = Ensemble([PKPS(**kw), LRMC(n_init=2)], mode='cv', holdout=None,
                      predict_kwargs=[{'whiten': True}, {}]).fit(recs)
    assert 0.0 <= ens_cv.alpha_global_ <= 1.0
    # QE-style: per-family alpha against reference scores
    recs2 = recs.copy()
    mi = recs2['model_id'].str.slice(-2).astype(int); ti = recs2['task_id'].str.slice(5).astype(int)
    recs2['reference_score'] = p[mi, ti]
    qe = Ensemble([SampleScore(), PKPS(**kw)], mode='cv', holdout='family',
                  predict_kwargs=[{}, {'holdout': 'family'}]).fit(recs2)
    assert set(qe.alpha_) == {f'dev{i}' for i in range(5)}
    assert all(np.isfinite(r['score_hat']) for r in qe.predict())


def test_leakage_own_reference_score_never_used():
    """Corrupt one model's reference scores: its own predictions (family holdout, and the
    QE ensemble built on them) must not change, while another model's do."""
    recs, p, observed = _records()
    mi = recs['model_id'].str.slice(-2).astype(int); ti = recs['task_id'].str.slice(5).astype(int)
    recs['reference_score'] = p[mi, ti]
    kw = dict(query_kwargs=dict(bandwidth=1.0, pca_dim=None), response_kwargs=dict(pca_dim=None),
              mds_kwargs=dict(dim=8))
    victim = sorted(recs['model_id'].unique())[0]
    bad = recs.copy()
    bad.loc[bad['model_id'] == victim, 'reference_score'] = 1 - bad.loc[bad['model_id'] == victim, 'reference_score']

    def run(r):
        ens = Ensemble([SampleScore(), PKPS(**kw)], mode='cv', holdout='family',
                       predict_kwargs=[{}, {'holdout': 'family'}]).fit(r)
        models = ens.model_names_
        q = [{'model_id': m, 'task_id': t} for m in models for t in ens.task_names_]
        pk = {(x['model_id'], x['task_id']): x['score_hat'] for x in ens.members[1].predict(q, holdout='family')}
        en = {(x['model_id'], x['task_id']): x['score_hat'] for x in ens.predict(q)}
        return pk, en, models

    pk0, en0, models = run(recs); pk1, en1, _ = run(bad)
    other = [m for m in models if fam_fn(m) != fam_fn(victim)][0]
    for t in {k[1] for k in pk0}:
        for d0, d1 in ((pk0, pk1), (en0, en1)):
            np.testing.assert_allclose(d0[(victim, t)], d1[(victim, t)], equal_nan=True)
    assert any(not np.isclose(pk0[(other, t)], pk1[(other, t)], equal_nan=True) or
               not np.isclose(en0[(other, t)], en1[(other, t)], equal_nan=True) for t in {k[1] for k in pk0})
