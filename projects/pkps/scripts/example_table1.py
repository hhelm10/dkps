#!/usr/bin/env python
"""Reproduce one cell of the paper's Table 1 with the dkps package classes.

Setting: HELM suite (18 tasks x 93 models), query-efficient evaluation at budget m=1 --
every model answers ONE query per task and we predict each (model, task) full-benchmark
score. Table 1 (16-seed means): sample 0.292 | IRT 0.348 | DKPS 0.168 | PKPS 0.136 |
Ensemble 0.125. This script runs one seed (default 0) and prints the same five MAEs;
the seed-0 values are sample 0.302 | IRT 0.360 | DKPS 0.161 | PKPS 0.137 | Ens 0.128.

Run from projects/pkps/scripts (the loaders read exports/ relative to this directory):

    pixi run python example_table1.py --seed 0 --m 1
"""
import argparse
import warnings

import numpy as np
import pandas as pd

from pipeline import loaders as H
from dkps import PKPS, DKPS, SampleScore, IRT, Ensemble


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--m', type=int, default=1, help='queries per (model, task) cell')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    warnings.filterwarnings('ignore', message='model pairs with zero')   # expected for DKPS's delta limit

    # ---- 1. the suite: response / query embeddings, per-response scores, full-benchmark scores
    (resp_X, Qu, qid_code, model_id, task_id, query_id,
     full_scores, models, tasks, groups, row_score, qmed) = H.load_suite()
    models = list(models)
    print(f'HELM suite: {len(models)} models x {len(tasks)} tasks, {len(model_id)} cached responses')

    # ---- 2. sample m responses per cell (the query-efficiency protocol)
    rng = np.random.default_rng(args.seed)
    keep = []
    for mm in models:
        for tt in tasks:
            idx = groups.get((mm, tt))
            if idx is None:                      # a few (model, task) cells were never evaluated
                continue
            keep.append(idx if len(idx) <= args.m else rng.choice(idx, args.m, replace=False))
    keep = np.concatenate(keep)

    # ---- 3. records: one row per sampled response. 'reference_score' is the cell's
    #         full-benchmark score; it is only ever used as a regression target for OTHER
    #         models (leave-one-family-out), never for the model being predicted.
    mi = {m: i for i, m in enumerate(models)}; ti = {t: j for j, t in enumerate(tasks)}
    records = pd.DataFrame({
        'model_id': model_id[keep], 'task_id': task_id[keep], 'query_id': query_id[keep],
        'response_embedding': list(resp_X[keep]), 'query_embedding': list(Qu[qid_code[keep]]),
        'score': row_score[keep],
    })
    records['reference_score'] = full_scores[records['model_id'].map(mi), records['task_id'].map(ti)]

    # ---- 4. the estimators. The suite loaders already PCA-reduced both channels, so
    #         pca_dim=None; the query bandwidth is chosen by leave-one-model-out CV on a
    #         grid around the within-domain median query distance (qmed); MDS dim 8.
    common = dict(response_kwargs=dict(kernel='linear', pca_dim=None), mds_kwargs=dict(dim=8))
    pkps = PKPS(query_kwargs=dict(kernel='rbf', bandwidth='cv', bandwidth_ref=qmed, pca_dim=None,
                                  normalization=H.NORMALIZATION),
                **common).fit(records)
    dkps = DKPS(sigma_ratio=0.01, bandwidth_ref=qmed, normalization=H.NORMALIZATION, **common).fit(records)   # paired DKPS (delta limit)
    sample = SampleScore().fit(records)
    binary = [t for t in tasks if np.all(np.isin(np.unique(row_score[task_id == t]), [0, 1]))]
    irt = IRT(binary_tasks=binary).fit(records)                                  # WMT tasks are BLEU-scored
    ens = Ensemble([sample, pkps], mode='cv', holdout='family',
                   predict_kwargs=[{}, {'holdout': 'family'}]).fit(records)
    print(f'PKPS query bandwidth selected by CV: {pkps.query_bandwidth_ / qmed:.2f} x qmed')

    # ---- 5. predict every cell and score against the full-benchmark scores
    #         (per-task MAE averaged over tasks, as in the paper)
    cells = [{'model_id': mm, 'task_id': tt} for mm in models for tt in tasks]
    truth = np.array([full_scores[mi[c['model_id']], ti[c['task_id']]] for c in cells])

    tcol = np.array([ti[c['task_id']] for c in cells])

    def mae(preds):                               # Table 1 convention: mean over tasks of per-task MAE
        p = np.array([r['score_hat'] for r in preds])
        ok = np.isfinite(p) & np.isfinite(truth)
        per_task = [np.mean(np.abs(p[ok & (tcol == j)] - truth[ok & (tcol == j)]))
                    for j in range(len(tasks)) if (ok & (tcol == j)).any()]
        return float(np.mean(per_task))

    print(f'MAE vs full-benchmark score, m={args.m}, seed={args.seed}:')
    print(f"  sample score  {mae(sample.predict(cells)):.3f}")
    print(f"  IRT           {mae(irt.predict(cells)):.3f}   (binary tasks only)")
    print(f"  DKPS          {mae(dkps.predict(cells, holdout='family')):.3f}")
    print(f"  PKPS          {mae(pkps.predict(cells, holdout='family')):.3f}")
    print(f"  Ensemble      {mae(ens.predict(cells)):.3f}")


if __name__ == '__main__':
    main()
