"""Mathematical identities and forbidden-information perturbation checks."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'projects/pkps/scripts'))
from pipeline.stress_geometry import (affinity_components, affinity_grid, normalize_affinity,
    spectral_embedding, shuffle_queries)
from experiments.stress_followups import (observe, geometry_bank, matrix, nested_predict,
                                         query_efficiency, family)
from experiments import stress_completion as completion
from experiments.stress_cache_design import matched_mask
from data.prepare_eee_followups import prepare, BENCHMARKS, REVISION, digest


def test_counterexample_and_psd_repair():
    q = np.array([-1, 1, -2, 2])[:, None]
    r = np.tile(np.eye(2), (2, 1))
    _, s, mass = affinity_components(q, r, ['a', 'a', 'b', 'b'], list('1234'), ['t']*4)
    _, original = spectral_embedding(normalize_affinity(s, mass, 'pairwise'))
    repaired = normalize_affinity(s, mass, 'model')
    _, fixed = spectral_embedding(repaired)
    assert original['min_squared_distance'] == pytest.approx(-.0835658522284)
    assert original['min_eigenvalue'] < -.04
    assert np.linalg.eigvalsh(repaired).min() >= -1e-12
    assert fixed['min_squared_distance'] >= -1e-12
    assert fixed['negative_eigenvalue_mass'] == 0


@pytest.mark.parametrize('kernel', ['rbf', 'constant', 'delta', 'task'])
def test_psd_on_unequal_disjoint_caches(kernel):
    rng = np.random.default_rng(71)
    q, r = rng.normal(size=(14, 3)), rng.normal(size=(14, 5))
    mid = np.repeat(['a', 'b', 'c', 'd'], [2, 3, 4, 5])
    _, s, mass = affinity_components(q, r, mid, np.arange(14), np.arange(14)%3, kernel=kernel)
    a = normalize_affinity(s, mass)
    assert np.linalg.eigvalsh(a).min() > -1e-10
    if kernel == 'constant':
        means = np.stack([r[mid == m].mean(0) for m in sorted(set(mid))])
        np.testing.assert_allclose(a, means @ means.T, atol=1e-12)
    if kernel == 'delta':
        with pytest.raises(ValueError, match='zero cross'):
            normalize_affinity(s, mass, 'pairwise')


def test_paired_delta_recovers_response_matrix_distance():
    rng = np.random.default_rng(9)
    r = rng.normal(size=(3, 4, 5))
    q = np.tile(np.eye(4), (3, 1))
    _, s, mass = affinity_components(q, r.reshape(12, 5), np.repeat(list('abc'), 4),
                                     np.tile(np.arange(4), 3), ['t']*12, kernel='delta')
    a = normalize_affinity(s, mass)
    np.testing.assert_allclose(a, normalize_affinity(s, mass, 'pairwise'))
    d2 = np.diag(a)[:, None] + np.diag(a)[None, :] - 2*a
    expected = np.mean(np.sum((r[:, None] - r[None, :])**2, axis=-1), axis=-1)
    np.testing.assert_allclose(d2, expected, atol=1e-12)


def test_zero_rank_and_shared_within_task_shuffle():
    z, info = spectral_embedding(np.ones((4, 4)))
    assert info['retained_rank'] == 0
    assert np.isfinite(z).all() and np.all(z == 0)
    q = np.array([[1, 0], [2, 0], [1, 0], [2, 0], [0, 1], [0, 2]])
    out = shuffle_queries(q, ['a']*4+['b']*2, [1, 2, 1, 2, 3, 4], seed=1)
    np.testing.assert_equal(out[0], out[2]); np.testing.assert_equal(out[1], out[3])
    assert set(map(tuple, out[:4])) == {(1, 0), (2, 0)}
    assert set(map(tuple, out[4:])) == {(0, 1), (0, 2)}


def test_optimized_grid_matches_separate_affinities():
    rng = np.random.default_rng(73)
    q, r = rng.normal(size=(12, 3)), rng.normal(size=(12, 5))
    mid, qid, tid = np.repeat(list('abc'), 4), np.tile(np.arange(4), 3), np.arange(12)%2
    shuffled = shuffle_queries(q, tid, qid)
    grid = affinity_grid(q, r, mid, qid, tid, {.1: .1, 1.: 1.}, shuffled, list('abc'))
    for key, (s, mass) in grid.items():
        kind, *bw = key.split('_')
        query = shuffled if kind == 'shuffled' else q
        kernel = 'rbf' if kind == 'shuffled' else kind
        _, ss, zz = affinity_components(query, r, mid, qid, tid, kernel=kernel,
                                        bandwidth=float(bw[0]) if bw else 1.)
        np.testing.assert_allclose(s, ss, atol=1e-12)
        np.testing.assert_allclose(mass, zz, atol=1e-12)


def fixture():
    rng = np.random.default_rng(33)
    models = [f'dev{i//3}/model{i}' for i in range(9)]
    tasks = ['b:t0', 'b:t1']
    vectors = rng.normal(size=(100, 6)); vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    rows, truth = [], []
    for i, model in enumerate(models):
        for t, task in enumerate(tasks):
            truth.append(dict(model_id=model, task_id=task, suite='b', y_full=rng.random()))
            for k in range(3):
                rows.append(dict(model_id=model, task_id=task, suite='b', query_id=f'{task}:{k}',
                                 q_index=t*3+k, r_index=10+i*6+t*3+k, score=float(rng.random() < .6)))
    return pd.DataFrame(rows), vectors, pd.DataFrame(truth)


def test_outer_family_full_scores_cannot_change_its_predictions():
    pool, vectors, truth = fixture()
    bad = truth.copy()
    bad.loc[bad.model_id.map(family) == 'dev0', 'y_full'] = [0, 1, 0, 1, 0, 1]
    p0, _, s0, _ = query_efficiency(pool, vectors, truth, 1, 1)
    p1, _, s1, _ = query_efficiency(pool, vectors, bad, 1, 1)
    np.testing.assert_allclose(p0.loc[p0.family == 'dev0', 'prediction'],
                               p1.loc[p1.family == 'dev0', 'prediction'])
    pd.testing.assert_frame_equal(s0[s0.outer_family == 'dev0'], s1[s1.outer_family == 'dev0'])
    # Every cache-only output is independent of *all* evaluation truth.
    np.testing.assert_allclose(p0.loc[p0.regime == 'cache-only', 'prediction'],
                               p1.loc[p1.regime == 'cache-only', 'prediction'])
    other = (p0.regime == 'full-reference') & (p0.family != 'dev0')
    assert np.any(abs(p0.loc[other, 'prediction'].to_numpy()-p1.loc[other, 'prediction'].to_numpy()) > 1e-5)


def test_hidden_rows_do_not_affect_observed_geometry():
    pool, vectors, _ = fixture()
    obs = observe(pool, 1, 8)
    models = sorted(pool.model_id.unique())
    before, _ = geometry_bank(obs, vectors, models)
    hidden = pool.index.difference(obs.index)
    corrupted = pool.copy()
    corrupted.loc[hidden, ['q_index', 'r_index']] = 99
    corrupted.loc[hidden, 'score'] = 999
    after, _ = geometry_bank(observe(corrupted, 1, 8), vectors, models)
    for key in before:
        np.testing.assert_allclose(before[key] @ before[key].T, after[key] @ after[key].T)


def test_predictor_rejects_forbidden_label_tables():
    pool, vectors, truth = fixture()
    models, tasks = sorted(pool.model_id.unique()), sorted(pool.task_id.unique())
    with pytest.raises(ValueError, match='outer-family'):
        nested_predict({}, models, tasks, matrix(pool, models, tasks), truth, 'dev0', 'full-reference')
    with pytest.raises(ValueError, match='cannot receive full'):
        nested_predict({}, models, tasks, matrix(pool, models, tasks), truth, 'dev0', 'cache-only')


def test_budget_subsets_are_nested_and_no_silent_reuse():
    pool, vectors, truth = fixture()
    assert set(observe(pool, 1, 0).index).issubset(observe(pool, 2, 0).index)
    with pytest.raises(ValueError, match='exceeds available'):
        observe(pool, 4, 0)
    with pytest.raises(ValueError, match='cohort exceeds available'):
        completion.run(pool, vectors, truth, 20, .5, 1, 0)


def test_completion_removes_validation_responses_and_forbidden_truth(monkeypatch):
    pool, vectors, truth = fixture()
    seen = []
    original = completion.fit_predictions

    def record_fit(rows, *args, **kwargs):
        seen.append(set(zip(rows.model_id, rows.task_id)))
        return original(rows, *args, **kwargs)

    monkeypatch.setattr(completion, 'fit_predictions', record_fit)
    p0, _, _, counts = completion.run(pool, vectors, truth, 9, .7, 1, 0)
    assert len(seen) == 2
    assert seen[0] < seen[1]
    assert len(seen[1]-seen[0]) == counts['n_validation_cells']
    hidden = {(m, t) for m in pool.model_id.unique() for t in pool.task_id.unique()}-seen[1]
    changed = pool.copy()
    mask = np.array([key in hidden for key in zip(changed.model_id, changed.task_id)])
    changed.loc[mask, ['q_index', 'r_index']] = 99
    changed.loc[mask, 'score'] = 999
    corrupted_truth = truth.assign(y_full=1-truth.y_full)
    p1, _, _, _ = completion.run(changed, vectors, corrupted_truth, 9, .7, 1, 0)
    np.testing.assert_allclose(p0.prediction, p1.prediction)


def test_full_targets_include_scored_empty_responses(tmp_path):
    source, out = tmp_path/'source', tmp_path/'out'
    source.mkdir()
    manifest = {'files': {}}
    for bench in BENCHMARKS:
        path = f'data/{bench}/developer/model/run_samples.jsonl'
        file = source/path
        file.parent.mkdir(parents=True)
        rows = [dict(sample_id=f'q{i}', input={'raw': f'Question {i}'},
                     output={'raw': 'answer' if i == 0 else ''},
                     evaluation={'score': 1-i}, metadata={'source_task': 'task'}) for i in [0, 1]]
        file.write_text('\n'.join(map(json.dumps, rows))+'\n')
        manifest['files'][path] = dict(added_at='2026-01-01', sha256=digest(file))
    (source/'manifest.json').write_text(json.dumps(manifest))
    (source/'pkps_snapshot.json').write_text(json.dumps({'revision': REVISION}))
    prepare(source, out, REVISION)
    targets = pd.read_parquet(out/'targets.parquet')
    pool = pd.read_parquet(out/'pool.parquet')
    assert len(targets) == len(pool) == 5
    assert (targets.y_full == .5).all() and (targets.depth == 2).all()
    assert (pool.score == 1).all()


def test_missingness_probe_preserves_budgets_and_uniform_ignores_scores():
    rng = np.random.default_rng(17)
    scores = rng.random((45, 16))
    uniform = matched_mask(scores, 8, 'uniform', 0)
    np.testing.assert_array_equal(uniform, matched_mask(1-scores, 8, 'uniform', 0))
    hidden = matched_mask(scores, 8, 'hide-poor', 0)
    for mask in [uniform, hidden]:
        assert np.all(mask.sum(1) == 8)
        assert mask.any(0).all()
    assert scores[hidden].mean() > scores[uniform].mean()
    # Even extreme selection into the same easy tasks preserves task coverage.
    extreme = matched_mask(np.tile(np.arange(16), (45, 1)), 8, 'hide-poor', 0)
    assert extreme.any(0).all() and np.all(extreme.sum(1) == 8)
