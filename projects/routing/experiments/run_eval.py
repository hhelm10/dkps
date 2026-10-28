"""Run the held-out-query routing experiment and write results + summary.

Run: python -m projects.routing.experiments.run_eval
"""

import os

import numpy as np
import pandas as pd

from .data import load_jailbreak_suite, load_dist_tensor
from .evaluate import run_experiment

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def main():
    os.makedirs(RESULTS, exist_ok=True)

    suite = load_jailbreak_suite()
    assert suite['query_emb'] is not None, 'run embed_queries first'
    P = load_dist_tensor()
    print(f'P: {P.shape}, query_emb: {suite["query_emb"].shape}')

    df = run_experiment(P, suite['query_emb'], suite['categories'])
    df.to_parquet(os.path.join(RESULTS, 'routing_errors.parquet'))

    # Headline summary: mean mimicry error per method (mean over seeds of the
    # per-seed mean over queries x targets), plus win rate vs the static baseline.
    per_seed = df.groupby(['method', 'seed'])['error'].mean().unstack()
    summary = pd.DataFrame({
        'mean_error': per_seed.mean(axis=1),
        'se_over_seeds': per_seed.std(axis=1) / np.sqrt(per_seed.shape[1]),
    })

    wide = df.pivot_table(index=['seed', 'query', 'target'], columns='method', values='error')
    for m in wide.columns:
        if m not in ('static', 'random', 'oracle'):
            summary.loc[m, 'win_vs_static'] = (wide[m] < wide['static']).mean()
            summary.loc[m, 'tie_vs_static'] = (wide[m] == wide['static']).mean()

    summary = summary.sort_values('mean_error')
    summary.to_csv(os.path.join(RESULTS, 'summary.csv'))
    print(summary.to_string(float_format=lambda v: f'{v:.4f}'))


if __name__ == '__main__':
    main()
