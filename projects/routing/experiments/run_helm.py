"""Held-out-query routing on the HELM suite's fully-paired core.

Builds the (n_models, m_queries, d) response tensor over the queries answered
by every model (93% of the suite), then runs the standard routing evaluation
(categories = tasks, so the anchor/eval split is stratified by task).

Run from repo root: pixi run python -m projects.routing.experiments.run_helm
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'pkps', 'helm'))
from pipeline import loaders as H  # noqa: E402

from .evaluate import run_experiment  # noqa: E402
from .geometry import pairwise_query_dist_tensor  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def load_paired_core(data=None, with_scores=False):
    if data is None:
        helm_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'pkps', 'helm')
        cwd = os.getcwd()
        os.chdir(helm_dir)
        try:
            data = H.load_suite()
        finally:
            os.chdir(cwd)
    (resp_X, Qu, qid_code, model_id, task_id, query_id,
     score_mat, models, tasks, groups, row_score, qmed) = data
    n = len(models)
    m2i = {mm: i for i, mm in enumerate(models)}

    # index rows by (task, query); keep queries answered by every model
    key = pd.DataFrame({'task': task_id, 'query': query_id,
                        'model': [m2i[mm] for mm in model_id],
                        'row': np.arange(len(task_id)), 'code': qid_code})
    counts = key.groupby(['task', 'query'])['model'].nunique()
    paired = counts[counts == n].index
    key = key.set_index(['task', 'query']).loc[paired].reset_index()

    qkeys = sorted(set(zip(key['task'], key['query'])))
    q2j = {k: j for j, k in enumerate(qkeys)}
    m_q = len(qkeys)

    X = np.zeros((n, m_q, resp_X.shape[1]), dtype=np.float32)
    S = np.zeros((n, m_q), dtype=np.float32)
    code_of = np.zeros(m_q, dtype=int)
    for task, query, mi, row, code in key.itertuples(index=False):
        j = q2j[(task, query)]
        X[mi, j] = resp_X[row]
        S[mi, j] = row_score[row]
        code_of[j] = code
    query_emb = Qu[code_of].astype(np.float32)
    categories = np.array([t for (t, _) in qkeys])
    if with_scores:
        return X, query_emb, categories, models, S
    return X, query_emb, categories, models


def main():
    os.makedirs(RESULTS, exist_ok=True)
    X, query_emb, categories, models = load_paired_core()
    print(f'paired core: X {X.shape}, query_emb {query_emb.shape}, '
          f'{len(set(categories))} tasks')

    P = pairwise_query_dist_tensor(X)
    print(f'P {P.shape} ({P.nbytes / 1e6:.0f} MB)')

    df = run_experiment(P, query_emb, categories)
    df.to_parquet(os.path.join(RESULTS, 'helm_routing_errors.parquet'))

    per_seed = df.groupby(['method', 'seed'])['error'].mean().unstack()
    summary = pd.DataFrame({
        'mean_error': per_seed.mean(axis=1),
        'se_over_seeds': per_seed.std(axis=1) / np.sqrt(per_seed.shape[1]),
    })
    wide = df.pivot_table(index=['seed', 'query', 'target'], columns='method',
                          values='error')
    for meth in wide.columns:
        if meth not in ('static', 'random', 'oracle'):
            summary.loc[meth, 'win_vs_static'] = float(
                (wide[meth] < wide['static']).mean())
    summary = summary.sort_values('mean_error')
    summary.to_csv(os.path.join(RESULTS, 'helm_summary.csv'))
    print(summary.round(4).to_string())

    # per-task view of the best query-aware bandwidth vs static
    best_qa = summary.index[summary.index.str.startswith('qa-')][0]
    by_cat = (df[df.method.isin([best_qa, 'static', 'oracle', 'random'])]
              .groupby(['category', 'method'])['error'].mean().unstack())
    by_cat.round(4).to_csv(os.path.join(RESULTS, 'helm_by_task.csv'))
    print('\nby task (best qa vs static):')
    print(by_cat[[best_qa, 'static', 'oracle', 'random']].round(4).to_string())


if __name__ == '__main__':
    main()
