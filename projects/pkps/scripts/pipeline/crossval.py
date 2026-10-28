#!/usr/bin/env python
"""
pipeline/crossval.py  --  RD2 missing tasks under the joint observation model.

Three levers: task observation probability (task_parity), cohort size (n_models),
and queries-per-observed-cell (query_obs, i.e. evaluation noise). When query_obs is
set, each observed (model, task) cell is estimated from only q sampled queries -- so
matrix completion sees a NOISY scalar score while PKPS reads the q responses. Emits
combined / matcomplete / ensemble held-out MAE (always vs the true full score).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from dkps.unpaired_dkps import pca_reduce_elbow
from pipeline import loaders as H


def one(resp_X, Qu, qid_code, model_id, task_id, query_id, groups, row_score, score_mat,
        models, tasks, obs_prob, n_models_use, seed, suite=False, query_med=None, q=None):
    rng = np.random.default_rng(seed)
    if n_models_use and n_models_use < len(models):
        sel = np.sort(rng.choice(len(models), n_models_use, replace=False))
        models = [models[i] for i in sel]
        score_mat = score_mat[sel]
    observed = H.sample_observed(len(models), len(tasks), obs_prob, rng) & np.isfinite(score_mat)

    # gather responses for observed cells; if q is set, sample q queries/cell and form a
    # NOISY sample-score matrix used for training (eval is always vs the true full score)
    keep, samp = [], np.full_like(score_mat, np.nan)
    for i in range(len(models)):
        for t in range(len(tasks)):
            if observed[i, t] and (models[i], tasks[t]) in groups:
                idx = groups[(models[i], tasks[t])]
                take = idx if (q is None or len(idx) <= q) else rng.choice(idx, q, replace=False)
                keep.append(take)
                samp[i, t] = row_score[take].mean()
    if not keep:
        return []
    keep = np.concatenate(keep)
    codes = qid_code[keep]
    train_score = samp if q is not None else None

    if suite:
        df = pd.DataFrame({'model_id': model_id[keep], 'task_id': task_id[keep],
                           'query_id': query_id[keep], 'embedding': list(resp_X[keep]),
                           'query_embedding': list(Qu[codes])})
        out = H.cv_predict_all(df, score_mat, observed, models, seed=seed, train_score=train_score,
                               response_kernel='linear', query_med=query_med)
    else:
        rr = pca_reduce_elbow(resp_X[keep])
        uq = np.unique(codes)
        qr = pca_reduce_elbow(Qu[uq])
        c2v = {c: qr[j] for j, c in enumerate(uq)}
        df = pd.DataFrame({'model_id': model_id[keep], 'task_id': task_id[keep],
                           'query_id': query_id[keep], 'embedding': list(rr),
                           'query_embedding': [c2v[c] for c in codes]})
        out = H.cv_predict_all(df, score_mat, observed, models, seed=seed, train_score=train_score)
    held = (~observed) & np.isfinite(score_mat)
    rows = []
    for meth in ('combined', 'matcomplete', 'ensemble'):
        p = out[meth]
        h = held & np.isfinite(p)
        mae = float(np.mean(np.abs(p[h] - score_mat[h]))) if h.any() else np.nan
        rows.append(dict(obs_prob=obs_prob, n_models=len(models), query_obs=(q or 0), seed=seed,
                         method=meth, mae=mae, predictor=out['predictor'], alpha=out['alpha']))
    return rows


def load(dataset, emb_tag=None, response_space='joint', resp_mode='option-text'):
    if dataset == 'suite':
        d = H.load_suite(resp_mode=resp_mode, response_space=response_space)
        return d[:11], d[11], True
    if dataset == 'eee':
        d = H.load_eee(emb_tag=emb_tag, response_space=response_space)
        return d[:11], d[11], True
    if dataset == 'pooled':
        return H.load_pooled(('math', 'wmt_14')), None, False
    cfg = H.DATASETS[dataset]
    return H.load_helm_math(cfg['parquet'], cfg['tsv'], query_source='google',
                            query_parquet=cfg['query_parquet'], score_col=cfg['score_col']), None, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='math')
    ap.add_argument('--sweep', choices=['task_parity', 'n_models', 'query_obs'], default='task_parity')
    ap.add_argument('--obs_probs', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
    ap.add_argument('--n_models_values', type=int, nargs='+', default=[10, 20, 40, 60, 80, 95])
    ap.add_argument('--query_obs_values', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32])
    ap.add_argument('--fixed_obs_prob', type=float, default=0.4)
    ap.add_argument('--n_seeds', type=int, default=20)
    ap.add_argument('--n_jobs', type=int, default=-1)
    ap.add_argument('--outdir', default='results-pkps-rd2cv')
    args = ap.parse_args()

    data, query_med, suite = load(args.dataset)
    resp_X, Qu, qid_code, model_id, task_id, query_id, score_mat, models, tasks, groups, row_score = data

    if args.sweep == 'task_parity':
        specs = [(p, None, None) for p in args.obs_probs]; xcol = 'obs_prob'
    elif args.sweep == 'n_models':
        specs = [(args.fixed_obs_prob, n, None) for n in args.n_models_values]; xcol = 'n_models'
    else:  # query_obs: fix task coverage, sweep queries-per-cell (evaluation noise)
        specs = [(args.fixed_obs_prob, None, q) for q in args.query_obs_values]; xcol = 'query_obs'
    jobs = [delayed(one)(resp_X, Qu, qid_code, model_id, task_id, query_id, groups, row_score,
                         score_mat, models, tasks, p, nm, s, suite, query_med, q)
            for (p, nm, q) in specs for s in range(args.n_seeds)]
    res = pd.DataFrame([r for sub in Parallel(n_jobs=args.n_jobs, verbose=5)(jobs) for r in sub])
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    res.to_csv(Path(args.outdir) / f'rd2_{args.dataset}_{args.sweep}.csv', index=False)
    print(res.pivot_table(index=xcol, columns='method', values='mae').round(4).to_string())


if __name__ == '__main__':
    main()
