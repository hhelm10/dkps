import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


def generate_benchmark_data(
        d_latent,
        d_obs,
        n_models,
        n_tasks,
        n_queries_per_task,
        obs_prob,
        query_obs_prob=1.0,
        score_noise=0.1,
        response_noise=0.5,
        task_spread=1.0,
        query_spread=0.5,
        n_shared_queries=None,
        n_unique_queries=None,
        random_state=None,
    ):
    """
    Generate synthetic benchmark data mimicking partial model evaluation.

    Models have latent vectors v_i, tasks have latent vectors t_k.
    Score(i, k) = v_i . t_k + noise.
    Each task has queries drawn near t_k. Response embeddings are
    x_{i,k,j} = [q_{k,j} + v_i + noise, extra_noise_dims].

    Observation is at the task level: each (model, task) pair is observed
    with probability obs_prob. When observed, the model answers all queries
    in that task.

    Parameters
    ----------
    d_latent : int
        Dimension of model/task latent space.
    d_obs : int
        Dimension of response embeddings. Must be >= d_latent.
    n_models : int
        Number of models.
    n_tasks : int
        Number of tasks.
    n_queries_per_task : int
        Number of queries per task.
    obs_prob : float
        Probability each (model, task) pair is observed.
    query_obs_prob : float
        Probability each query is answered given the model sees the task.
        Default 1.0 means all queries are answered.
    score_noise : float
        Std of noise added to scores.
    response_noise : float
        Std of noise added to response embeddings (active dims).
    task_spread : float
        Std of task vectors in latent space. Controls how spread out
        tasks are from each other.
    query_spread : float
        Std of query vectors around the task center.
    random_state : int or None
        Random seed.

    Returns
    -------
    data : pd.DataFrame
        Long-form table with columns: model_id, task_id, query_id,
        embedding, query_embedding. One row per observed (model, query).
    scores : np.ndarray, shape (n_models, n_tasks)
        Full ground-truth score matrix (including unobserved pairs).
    observed : np.ndarray, shape (n_models, n_tasks), dtype bool
        Which (model, task) pairs are observed.
    model_offsets : np.ndarray, shape (n_models, d_latent)
        Latent model vectors.
    task_vectors : np.ndarray, shape (n_tasks, d_latent)
        Latent task vectors.
    """
    assert d_obs >= d_latent
    rng = np.random.default_rng(random_state)

    model_offsets = rng.normal(0, 1, (n_models, d_latent))
    task_vectors = rng.normal(0, task_spread, (n_tasks, d_latent))

    # Scores: v_i . t_k + noise
    scores = model_offsets @ task_vectors.T + score_noise * rng.normal(size=(n_models, n_tasks))

    # Observation mask (ensure each model has >= 1 task, each task has >= 1 model)
    observed = rng.random((n_models, n_tasks)) < obs_prob
    for i in range(n_models):
        if not observed[i].any():
            observed[i, rng.integers(n_tasks)] = True
    for k in range(n_tasks):
        if not observed[:, k].any():
            observed[rng.integers(n_models), k] = True

    # Query-level pairing mode: each model answers a fixed budget per task, split into
    # n_shared_queries that ALL models answer (paired) and n_unique_queries drawn fresh per
    # model near the same task center (unpaired). Overlap rho = n_shared/(n_shared+n_unique)
    # isolates cross-model overlap from query quantity.
    paired_split = n_shared_queries is not None and n_unique_queries is not None
    shared_queries = {}
    if paired_split:
        for k in range(n_tasks):
            q_embs = task_vectors[k] + query_spread * rng.normal(size=(n_shared_queries, d_latent))
            q_ids = [f'task{k:03d}_s{j:04d}' for j in range(n_shared_queries)]
            shared_queries[k] = (q_ids, q_embs)
    else:
        # Queries per task, shared across all models evaluated on that task.
        task_queries = {}  # task_idx -> (query_ids, query_embeddings)
        for k in range(n_tasks):
            q_embs = task_vectors[k] + query_spread * rng.normal(size=(n_queries_per_task, d_latent))
            q_ids = [f'task{k:03d}_q{j:04d}' for j in range(n_queries_per_task)]
            task_queries[k] = (q_ids, q_embs)

    def _emit(rows, i, k, q_id, q_emb):
        active = q_emb + model_offsets[i] + response_noise * rng.normal(size=d_latent)
        extra = rng.normal(size=d_obs - d_latent)
        rows.append({
            'model_id': f'model_{i:02d}', 'task_id': f'task_{k:03d}', 'query_id': q_id,
            'embedding': np.concatenate([active, extra]), 'query_embedding': q_emb,
        })

    # Generate response embeddings for observed (model, task) pairs
    rows = []
    for i in range(n_models):
        for k in range(n_tasks):
            if not observed[i, k]:
                continue
            if paired_split:
                for q_id, q_emb in zip(*shared_queries[k]):       # shared (paired) queries
                    _emit(rows, i, k, q_id, q_emb)
                for j in range(n_unique_queries):                  # unique (unpaired) queries
                    q_emb = task_vectors[k] + query_spread * rng.normal(size=d_latent)
                    _emit(rows, i, k, f'task{k:03d}_m{i:03d}_u{j:04d}', q_emb)
                continue
            q_ids, q_embs = task_queries[k]
            if query_obs_prob < 1.0:                               # each model a random subset
                q_mask = rng.random(n_queries_per_task) < query_obs_prob
                if not q_mask.any():
                    q_mask[rng.integers(n_queries_per_task)] = True
            else:
                q_mask = np.ones(n_queries_per_task, dtype=bool)
            for j in range(n_queries_per_task):
                if q_mask[j]:
                    _emit(rows, i, k, q_ids[j], q_embs[j])

    data = pd.DataFrame(rows)
    return data, scores, observed, model_offsets, task_vectors
