import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.neighbors import KNeighborsRegressor

from dkps.synthetic import generate_benchmark_data
from dkps.unpaired_dkps import DoubleKernelDKPS, pca_reduce_elbow


def _reduce_df(data, n_elbows=2, max_components=128):
    """Per-run PCA reduction of response and (separately) query embeddings,
    learned only from the rows in `data` (the observed/sampled responses)."""
    R = np.stack(data['embedding'].values)
    Rr = pca_reduce_elbow(R, n_elbows=n_elbows, max_components=max_components)
    out = data.copy()
    out['embedding'] = list(Rr)

    qsub = data[['query_id', 'query_embedding']].drop_duplicates('query_id')
    Q = np.stack(qsub['query_embedding'].values)
    Qr = pca_reduce_elbow(Q, n_elbows=n_elbows, max_components=max_components)
    qmap = {qid: v for qid, v in zip(qsub['query_id'].values, Qr)}
    out['query_embedding'] = [qmap[q] for q in data['query_id'].values]
    return out


import os

# Kernel types are configurable via env (default: both RBF). Set
# DKPS_QUERY_KERNEL / DKPS_RESPONSE_KERNEL=linear for the both-linear supplement.
QUERY_KERNEL = os.environ.get('DKPS_QUERY_KERNEL', 'rbf')
RESPONSE_KERNEL = os.environ.get('DKPS_RESPONSE_KERNEL', 'rbf')

_KERNELS = dict(query_kernel=QUERY_KERNEL, response_kernel=RESPONSE_KERNEL)
# Every panel compares DKPS (delta query kernel: exact query matches only) against
# PKPS (rbf: bridges similar queries). Both use all observed data (no task filter).
ESTIMATORS = {
    'dkps': dict(query_kernel='delta', response_kernel=RESPONSE_KERNEL),
    'pkps': dict(query_kernel='rbf', response_kernel=RESPONSE_KERNEL),
}


def predict_scores_knn(embeddings, scores, observed, k=5):
    """Predict held-out scores using KNN in embedding space."""
    n_models, n_tasks = scores.shape
    predictions = np.full_like(scores, np.nan)

    for task_idx in range(n_tasks):
        train_mask = observed[:, task_idx]
        test_mask = ~observed[:, task_idx]
        n_train = train_mask.sum()
        if n_train < 1 or test_mask.sum() < 1:
            continue

        k_eff = min(k, n_train)
        knn = KNeighborsRegressor(n_neighbors=k_eff, weights='distance')
        knn.fit(embeddings[train_mask], scores[train_mask, task_idx])
        predictions[test_mask, task_idx] = knn.predict(embeddings[test_mask])

    return predictions


def evaluate_predictions(predictions, scores, observed):
    """Compute MAE on held-out (model, task) pairs."""
    held_out = ~observed & np.isfinite(predictions)
    if held_out.sum() == 0:
        return np.nan

    y_true = scores[held_out]
    y_pred = predictions[held_out]
    return float(np.mean(np.abs(y_true - y_pred)))


def _sample_baseline(scores, observed):
    """Score-only floor: predict each held-out cell by the mean observed score of its task
    (no item-level embedding, no model-specific signal beyond the task average)."""
    preds = np.full_like(scores, np.nan)
    for k in range(scores.shape[1]):
        obs = observed[:, k]
        if obs.any() and (~obs).any():
            preds[~obs, k] = scores[obs, k].mean()
    return evaluate_predictions(preds, scores, observed)


def _fit_and_predict(data, scores, observed, k_neighbors=5, **dkps_kwargs):
    """Embed models in the perspective space and predict held-out scores by kNN.
    PKPS (rbf) selects its query bandwidth by CV; DKPS (delta) is the fixed limit.
    `data` is assumed already PCA-reduced (see _reduce_df).
    """
    if dkps_kwargs.get('query_kernel') == 'rbf':
        embeddings = _embed_cv_rbf(data, scores, observed,
                                   dkps_kwargs.get('response_kernel', RESPONSE_KERNEL), k=k_neighbors)
    else:
        embeddings = _embed(data, **dkps_kwargs)
    if embeddings is None:
        return np.nan
    predictions = predict_scores_knn(embeddings, scores, observed, k=k_neighbors)
    return evaluate_predictions(predictions, scores, observed)


def _embed(data, **dkps_kwargs):
    """Fit DoubleKernelDKPS and return model coordinates via ClassicalMDS (a collapsed
    single point if the perspective is degenerate, e.g. DKPS with no shared queries)."""
    # archived synthetic results use the original pair normalization
    est = DoubleKernelDKPS(**{'normalization': 'pair', **dkps_kwargs})
    est.fit(data)
    dist = est.dist_matrix_.copy()
    if np.any(np.isnan(dist)):
        md = np.nanmax(dist)
        if not np.isfinite(md):
            return None
        dist = np.nan_to_num(dist, nan=md)
    from graspologic.embed import ClassicalMDS
    try:
        if not np.any(dist > 0):
            raise ValueError('degenerate')
        return ClassicalMDS(n_components=None, n_elbows=2,
                            dissimilarity='precomputed').fit_transform(dist)
    except Exception:
        return np.zeros((dist.shape[0], 1))


def _cv_loo_obs(Z, target, mask, k=8):
    """Mean leave-one-model-out kNN error predicting the OBSERVED target from the other models
    in the perspective Z (per task). Bandwidth-selection criterion -- uses only
    observed scores, never the held-out cells we report."""
    import numpy as np
    errs = []
    for t in range(target.shape[1]):
        idx = np.where(mask[:, t] & np.isfinite(target[:, t]))[0]
        if len(idx) < k + 1:
            continue
        Zt, yt = Z[idx], target[idx, t]
        D2 = ((Zt[:, None] - Zt[None]) ** 2).sum(-1)
        np.fill_diagonal(D2, np.inf)
        for a in range(len(idx)):
            nbr = np.argsort(D2[a])[:k]
            w = 1.0 / (np.sqrt(D2[a][nbr]) + 1e-9)
            errs.append(abs(np.average(yt[nbr], weights=w) - yt[a]))
    return float(np.mean(errs)) if errs else None


def _embed_cv_rbf(data, target, mask, response_kernel, k=8):
    """PKPS at full strength: select the RBF query bandwidth by cross-validation over a
    median-scaled grid (the bandwidth whose perspective best predicts the observed target under
    leave-one-model-out kNN). CV recovers the delta limit when overlap is high and bridges when
    it is low -- the same selection used on real data."""
    from scipy.spatial.distance import pdist
    qsub = data.drop_duplicates('query_id')
    Q = np.stack(qsub['query_embedding'].values)
    med = float(np.median(pdist(Q))) if len(Q) > 1 else 1.0
    best, best_err = None, np.inf
    for c in (0.03, 0.1, 0.3, 1.0, 3.0):
        Z = _embed(data, query_kernel='rbf', response_kernel=response_kernel, query_bandwidth=med * c)
        if Z is None or Z.shape[1] < 1:
            continue
        e = _cv_loo_obs(Z, target, mask, k)
        if e is not None and e < best_err:
            best_err, best = e, Z
    return best if best is not None else _embed(data, query_kernel='rbf', response_kernel=response_kernel)


def _denoise_mae(embeddings, noisy, true, k=8, observed=None):
    """Predict each cell's true score by a distance-weighted average of the noisy scores of its
    model's k nearest perspective neighbours (leave-one-out). MAE vs the true scores.
    With an `observed` mask, only observed neighbour cells enter each average and MAE is
    evaluated on observed cells only."""
    from sklearn.neighbors import NearestNeighbors
    n = embeddings.shape[0]
    if observed is None:
        observed = np.ones_like(true, dtype=bool)
    dist, idx = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(embeddings).kneighbors(embeddings)
    pred = np.full_like(true, np.nan)
    for i in range(n):
        nb, d = idx[i][1:], dist[i][1:]
        w = 1.0 / (d + 1e-9)
        for t in np.where(observed[i])[0]:
            m = observed[nb, t]
            if m.any():
                wt = w[m] / w[m].sum()
                pred[i, t] = (wt * noisy[nb[m], t]).sum()
    ok = observed & ~np.isnan(pred)
    return float(np.mean(np.abs(pred[ok] - true[ok])))


def _run_estimators(data, scores, observed, experiment, k_neighbors, n_components, extra_fields):
    """Run all estimators on one dataset, return list of result rows.

    Reduces embeddings once per run (observed data only), then runs estimators.
    """
    sample_mae = _sample_baseline(scores, observed)
    data = _reduce_df(data)
    rows = [{'experiment': experiment, 'estimator': 'sample', 'mae': sample_mae, **extra_fields}]
    for est_name, est_kwargs in ESTIMATORS.items():
        mae = _fit_and_predict(
            data, scores, observed,
            k_neighbors=k_neighbors,
            **est_kwargs,
        )
        row = {'experiment': experiment, 'estimator': est_name, 'mae': mae}
        row.update(extra_fields)
        rows.append(row)
    return rows


_DEFAULTS = dict(
    d_latent=5, d_obs=20, score_noise=0.1, response_noise=0.5,
    task_spread=1.0, query_spread=0.5,
)


def _make_runner(experiment_name, sweep_param, default_gen_kwargs):
    """Factory for experiment functions that sweep a single parameter."""

    def run_experiment(k_neighbors=5, n_seeds=50, **kwargs):
        sweep_values = kwargs.pop(sweep_param)
        gen_kwargs = {**_DEFAULTS, **default_gen_kwargs, **kwargs}

        def _run(value, seed):
            gen_kwargs[sweep_param] = value
            n_components = gen_kwargs.get('d_latent', _DEFAULTS['d_latent'])
            data, scores, observed, _, _ = generate_benchmark_data(
                random_state=seed, **gen_kwargs,
            )
            return _run_estimators(
                data, scores, observed, experiment_name, k_neighbors,
                n_components,
                {'seed': seed, sweep_param: value},
            )

        params = [(v, s) for v in sweep_values for s in range(n_seeds)]
        nested = Parallel(n_jobs=-1, verbose=10)(
            delayed(_run)(v, s) for v, s in params)
        return pd.DataFrame([r for sub in nested for r in sub])

    run_experiment.__name__ = f'run_exp_{experiment_name}'
    run_experiment.__doc__ = f'Sweep {sweep_param} for experiment {experiment_name}.'
    return run_experiment


# The six experiments
run_exp_n_models = _make_runner('n_models', 'n_models', dict(
    n_tasks=20, n_queries_per_task=10, obs_prob=0.3, query_obs_prob=0.8))

run_exp_n_tasks = _make_runner('n_tasks', 'n_tasks', dict(
    n_models=100, n_queries_per_task=10, obs_prob=0.3, query_obs_prob=0.8))

run_exp_task_parity = _make_runner('task_parity', 'obs_prob', dict(
    n_models=100, n_tasks=20, n_queries_per_task=10, query_obs_prob=0.8))

run_exp_query_sparsity = _make_runner('query_sparsity', 'query_obs_prob', dict(
    n_models=100, n_tasks=20, n_queries_per_task=10, obs_prob=0.3))

run_exp_task_spread = _make_runner('task_spread', 'task_spread', dict(
    n_models=100, n_tasks=20, n_queries_per_task=10, obs_prob=0.3, query_obs_prob=1.0))


def run_exp_rho(rhos=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), budget=10, n_models=100, n_tasks=20,
                obs_prob=0.3, k_neighbors=5, n_seeds=50, **gen_kwargs):
    """Cross-model query overlap rho at a FIXED per-cell budget. Compare DKPS (delta query
    kernel, exact matches only) vs PKPS (rbf, bridges similar queries)."""
    kw = {k: v for k, v in {**_DEFAULTS, **gen_kwargs}.items() if k != 'n_queries_per_task'}
    ests = {'dkps': dict(query_kernel='delta', response_kernel=RESPONSE_KERNEL),
            'pkps': dict(query_kernel='rbf', response_kernel=RESPONSE_KERNEL)}

    def _run(rho, seed):
        n_sh = int(round(rho * budget))
        data, scores, observed, _, _ = generate_benchmark_data(
            n_models=n_models, n_tasks=n_tasks, n_queries_per_task=budget, obs_prob=obs_prob,
            n_shared_queries=n_sh, n_unique_queries=budget - n_sh, random_state=seed, **kw)
        out = [{'experiment': 'rho', 'estimator': 'sample', 'seed': seed, 'rho': rho,
                'mae': _sample_baseline(scores, observed)}]
        d = _reduce_df(data)
        out += [{'experiment': 'rho', 'estimator': name, 'seed': seed, 'rho': rho,
                 'mae': _fit_and_predict(d, scores, observed, k_neighbors=k_neighbors, **ek)}
                for name, ek in ests.items()]
        return out

    params = [(r, s) for r in rhos for s in range(n_seeds)]
    nested = Parallel(n_jobs=-1, verbose=10)(delayed(_run)(r, s) for r, s in params)
    return pd.DataFrame([r for sub in nested for r in sub])


def run_exp_query_efficiency(budgets=(1, 2, 4, 8, 16, 32), rhos=(0.0, 1.0), n_models=60,
                             n_tasks=12, k_neighbors=8, n_seeds=50, meas_noise=4.0, **gen_kwargs):
    """Query efficiency (denoising face): every cell is observed with M queries, giving a noisy
    sample score (measurement noise ~1/sqrt(M)); we predict its TRUE score. The sample baseline
    is the M-query mean; DKPS/PKPS denoise by smoothing over the perspective. At paired (rho=1)
    and unpaired (rho=0): PKPS denoises in both, DKPS only when paired."""
    kw = {k: v for k, v in {**_DEFAULTS, **gen_kwargs}.items()
          if k not in ('n_queries_per_task', 'score_noise')}
    ests = {'dkps': dict(query_kernel='delta', response_kernel=RESPONSE_KERNEL),
            'pkps': dict(query_kernel='rbf', response_kernel=RESPONSE_KERNEL)}

    def _run(budget, rho, seed):
        n_sh = int(round(rho * budget))
        data, true, _, _, _ = generate_benchmark_data(
            n_models=n_models, n_tasks=n_tasks, n_queries_per_task=budget, obs_prob=1.0,
            n_shared_queries=n_sh, n_unique_queries=budget - n_sh, score_noise=0.0,
            random_state=seed, **kw)
        rng = np.random.default_rng(10_000 + seed)
        noisy = true + meas_noise * rng.normal(size=true.shape) / np.sqrt(budget)
        out = [{'experiment': 'query_efficiency', 'estimator': 'sample', 'seed': seed,
                'budget': budget, 'rho': rho, 'mae': float(np.mean(np.abs(noisy - true)))}]
        d = _reduce_df(data)
        for name, ek in ests.items():
            if ek['query_kernel'] == 'rbf':                    # PKPS: leak-free CV bandwidth
                emb = _embed_cv_rbf(d, noisy, np.ones_like(noisy, dtype=bool),
                                    ek['response_kernel'], k=k_neighbors)
            else:
                emb = _embed(d, **ek)
            mae = _denoise_mae(emb, noisy, true, k=k_neighbors) if emb is not None else np.nan
            out.append({'experiment': 'query_efficiency', 'estimator': name, 'seed': seed,
                        'budget': budget, 'rho': rho, 'mae': mae})
        return out

    params = [(b, r, s) for b in budgets for r in rhos for s in range(n_seeds)]
    nested = Parallel(n_jobs=-1, verbose=10)(delayed(_run)(b, r, s) for b, r, s in params)
    return pd.DataFrame([r for sub in nested for r in sub])


def run_exp_noise_x_queries(
        response_noises=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
        n_queries_values=(3, 10, 50),
        n_models=100,
        n_tasks=20,
        obs_prob=0.3,
        query_obs_prob=1.0,
        k_neighbors=5,
        n_seeds=50,
        **gen_kwargs,
    ):
    """Interaction: response noise x queries per task (combined estimator only)."""
    kw = {**_DEFAULTS, **gen_kwargs}

    def _run(response_noise, n_queries_per_task, seed):
        data, scores, observed, _, _ = generate_benchmark_data(
            n_models=n_models, n_tasks=n_tasks,
            n_queries_per_task=n_queries_per_task,
            obs_prob=obs_prob, query_obs_prob=query_obs_prob,
            response_noise=response_noise,
            random_state=seed, **{k: v for k, v in kw.items() if k != 'response_noise'},
        )
        mae = _fit_and_predict(
            _reduce_df(data), scores, observed,
            k_neighbors=k_neighbors,
            **_KERNELS,
        )
        return {
            'experiment': 'noise_x_queries',
            'estimator': f'nq={n_queries_per_task}',
            'seed': seed,
            'response_noise': response_noise,
            'n_queries_per_task': n_queries_per_task,
            'mae': mae,
        }

    params = [(rn, nq, s)
              for rn in response_noises
              for nq in n_queries_values
              for s in range(n_seeds)]
    results = Parallel(n_jobs=8, verbose=10)(
        delayed(_run)(rn, nq, s) for rn, nq, s in params)
    return pd.DataFrame(results)


def run_exp_qe_lever(experiment_name, sweep_param, sweep_values, budget=4, rho=0.0,
                     n_models=60, n_tasks=12, obs_prob=1.0, k_neighbors=8, n_seeds=50,
                     meas_noise=4.0, **gen_kwargs):
    """Query-efficiency (denoising) protocol with one lever swept: every observed cell has a
    budget of M queries and a noisy sample score; we predict its TRUE score. Levers beyond the
    budget: n_models, n_tasks, obs_prob (task coverage), rho (pairing of the M queries)."""
    kw = {k: v for k, v in {**_DEFAULTS, **gen_kwargs}.items()
          if k not in ('n_queries_per_task', 'score_noise')}
    ests = {'dkps': dict(query_kernel='delta', response_kernel=RESPONSE_KERNEL),
            'pkps': dict(query_kernel='rbf', response_kernel=RESPONSE_KERNEL)}
    base = dict(budget=budget, rho=rho, n_models=n_models, n_tasks=n_tasks, obs_prob=obs_prob)

    def _run(value, seed):
        cfg = dict(base); cfg[sweep_param] = value
        n_sh = int(round(cfg['rho'] * cfg['budget']))
        data, true, observed, _, _ = generate_benchmark_data(
            n_models=cfg['n_models'], n_tasks=cfg['n_tasks'], n_queries_per_task=cfg['budget'],
            obs_prob=cfg['obs_prob'], n_shared_queries=n_sh,
            n_unique_queries=cfg['budget'] - n_sh, score_noise=0.0, random_state=seed, **kw)
        rng = np.random.default_rng(10_000 + seed)
        noisy = true + meas_noise * rng.normal(size=true.shape) / np.sqrt(cfg['budget'])
        fields = {'seed': seed, sweep_param: value}
        out = [{'experiment': experiment_name, 'estimator': 'sample',
                'mae': float(np.mean(np.abs((noisy - true)[observed]))), **fields}]
        d = _reduce_df(data)
        for name, ek in ests.items():
            if ek['query_kernel'] == 'rbf':
                emb = _embed_cv_rbf(d, noisy, observed, ek['response_kernel'], k=k_neighbors)
            else:
                emb = _embed(d, **ek)
            mae = (_denoise_mae(emb, noisy, true, k=k_neighbors, observed=observed)
                   if emb is not None else np.nan)
            out.append({'experiment': experiment_name, 'estimator': name, 'mae': mae, **fields})
        return out

    params = [(v, s_) for v in sweep_values for s_ in range(n_seeds)]
    nested = Parallel(n_jobs=-1, verbose=10)(delayed(_run)(v, s_) for v, s_ in params)
    return pd.DataFrame([r for sub in nested for r in sub])
