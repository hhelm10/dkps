#!/usr/bin/env python
"""Matrix completion on the heterogeneous suite (Result 2): a fraction p_task of (model,task)
cells is evaluated in full (true scores); predict the MISSING cells. Baseline = low-rank
matrix completion; the PKPS embedding completes from the joint perspective; ensemble blends
them. MAE over the MISSING cells only -- no sample line (a missing cell was never run).
Levers: cohort n, task coverage p_task, number of tasks T."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from pipeline import loaders as H
from pipeline.crossval import load
from pipeline.perspective import _perspective, mc_crossfit


def _mae(P, mask, full):
    m = mask & np.isfinite(P)
    return float(np.mean(np.abs(P[m] - full[m]))) if m.any() else np.nan


def _perspective_cv(rX, Qu, qc, mid, tid, qid, keep, mods, qmed, samp, O, k=5):
    """Suite perspective with a CV-selected RBF query bandwidth: pick the bandwidth whose
    embedding best predicts the OBSERVED scores under leave-one-model-out kNN (leakage-free --
    never the held-out missing cells we report)."""
    codes = qc[keep]
    df = pd.DataFrame({'model_id': mid[keep], 'task_id': tid[keep], 'query_id': qid[keep],
                       'embedding': list(rX[keep]), 'query_embedding': list(Qu[codes])})
    bw = H.SubsampleMedianBandwidth()
    est = H.ProductKernelPerspectiveSpace(query_kernel='rbf', response_kernel='linear',
                                          query_bandwidth=bw, response_bandwidth=bw,
                                          normalization=H.NORMALIZATION)
    grid = list(qmed * np.array([0.03, 0.1, 0.3, 1.0, 3.0]))
    names, Ds = est.dist_matrices(df, grid)
    Zs = {sg: H._mds_full(Ds[sg], names, mods, 8) for sg in grid}
    best_sg, best_err = None, np.inf
    for sg, Z in Zs.items():
        if Z is None:
            continue
        errs = []
        for t in range(samp.shape[1]):
            idx = np.where(O[:, t] & np.isfinite(samp[:, t]))[0]
            if len(idx) < k + 1:
                continue
            Zt, yt = Z[idx], samp[idx, t]
            D2 = ((Zt[:, None] - Zt[None]) ** 2).sum(-1)
            np.fill_diagonal(D2, np.inf)
            for a in range(len(idx)):
                nbr = np.argsort(D2[a])[:k]
                w = 1.0 / (np.sqrt(D2[a][nbr]) + 1e-9)
                errs.append(abs(np.average(yt[nbr], weights=w) - yt[a]))
        e = float(np.mean(errs)) if errs else np.inf
        if e < best_err:
            best_err, best_sg = e, sg
    return Zs.get(best_sg) if best_sg is not None and Zs.get(best_sg) is not None \
        else H._mds_full(Ds[grid[2]], names, mods, 8)


def trial(data, qmed, n_models, n_tasks, p_task, seed, p_query=0.8, dump_cells=False):
    rX, Qu, qc, mid, tid, qid, full, mods, tasks, grp, rs = data[:11]
    rng = np.random.default_rng(seed)
    msel = (np.sort(rng.choice(len(mods), min(n_models, len(mods)), replace=False))
            if n_models else np.arange(len(mods)))
    tsel = (np.sort(rng.choice(len(tasks), min(n_tasks, len(tasks)), replace=False))
            if n_tasks else np.arange(len(tasks)))
    mods_s = [mods[i] for i in msel]
    full_s = full[np.ix_(msel, tsel)]
    fin0 = np.isfinite(full_s)
    # observe p_task of the (existing) cells in full
    O = (rng.random(full_s.shape) < p_task) & fin0
    for i in range(O.shape[0]):
        if not O[i].any() and fin0[i].any(): O[i, np.where(fin0[i])[0][0]] = True
    for t in range(O.shape[1]):
        if not O[:, t].any() and fin0[:, t].any(): O[np.where(fin0[:, t])[0][0], t] = True

    # an observed cell is measured with a p_query fraction of its queries, giving a NOISY
    # observed score (full_s holds the true full-eval score, used only for evaluation). The
    # perspective is capped at K_PERSP responses per cell for tractability.
    K_PERSP = 16
    keep, samp = [], np.full(full_s.shape, np.nan)
    for ii, mi in enumerate(msel):
        for tt, ti in enumerate(tsel):
            if O[ii, tt] and (mods[mi], tasks[ti]) in grp:
                idx = grp[(mods[mi], tasks[ti])]
                nq = max(1, int(round(p_query * len(idx))))
                take = idx if nq >= len(idx) else rng.choice(idx, nq, replace=False)
                samp[ii, tt] = rs[take].mean()                       # noisy observed score
                keep.append(take if len(take) <= K_PERSP else rng.choice(take, K_PERSP, replace=False))
    if not keep:
        return None
    keep = np.concatenate(keep)
    Z = _perspective_cv(rX, Qu, qc, mid, tid, qid, keep, mods_s, qmed, samp, O)
    if Z is None:
        return None
    pk = H.predict_from_embedding_all(Z, samp, O, predictor='knn', k=5)
    mc = mc_crossfit(samp, O, rng)
    # ensemble weight fit on observed cells (both channels cross-fit), applied to the missing
    oo = O & np.isfinite(pk) & np.isfinite(mc) & np.isfinite(samp)
    grid = np.linspace(0, 1, 21)
    a = grid[int(np.argmin([np.mean(np.abs(np.clip(g * pk[oo] + (1 - g) * mc[oo], 0, 1) - samp[oo]))
                            for g in grid]))] if oo.sum() >= 5 else 0.5
    ens = np.clip(a * pk + (1 - a) * mc, 0, 1)
    mis = (~O) & fin0
    if dump_cells:                                         # per-missing-cell errors for ridgelines
        cells = []
        for i, t in zip(*np.where(mis)):
            tn = tasks[tsel[t]]
            err = lambda P: float(abs(P[i, t] - full_s[i, t])) if np.isfinite(P[i, t]) else np.nan
            cells.append(dict(n_models=len(mods_s), p_task=p_task, p_query=p_query, seed=seed,
                              model=mods_s[i], task=tn, dataset=tn.split(':')[0],
                              mc_err=err(mc), pkps_err=err(pk), ens_err=err(ens)))
        return cells
    return dict(n_models=len(mods_s), n_tasks=len(tsel), p_task=p_task, p_query=p_query, seed=seed,
                mc=_mae(mc, mis, full_s), pkps=_mae(pk, mis, full_s), ens=_mae(ens, mis, full_s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', choices=['n_models', 'coverage', 'n_tasks', 'p_query'], default='coverage')
    ap.add_argument('--n_models_values', type=int, nargs='+', default=[10, 20, 40, 60, 93])
    ap.add_argument('--coverages', type=float, nargs='+', default=[0.2, 0.35, 0.5, 0.7, 0.9])
    ap.add_argument('--n_tasks_values', type=int, nargs='+', default=[4, 8, 12, 18])
    ap.add_argument('--p_query_values', type=float, nargs='+', default=[0.1, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument('--line_models', type=int, nargs='+', default=[10, 93])  # n_models line family
    ap.add_argument('--fixed_p', type=float, default=0.5)
    ap.add_argument('--fixed_pq', type=float, default=0.5)
    ap.add_argument('--n_seeds', type=int, default=16)
    ap.add_argument('--n_jobs', type=int, default=-1)
    ap.add_argument('--suite', choices=['helm', 'eee'], default='helm')
    ap.add_argument('--emb_tag', default=None)
    ap.add_argument('--resp_mode', choices=['native', 'text', 'option-text'], default='option-text',
                    help='HELM only: response representation (native = published one-hot)')
    ap.add_argument('--response_space', choices=['blocked', 'joint'], default='joint')
    ap.add_argument('--outdir', default=None)
    args = ap.parse_args()
    if args.suite == 'helm' and args.resp_mode == 'native':
        args.response_space = 'blocked'   # one-hot answers cannot share a space
    if args.outdir is None:
        args.outdir = 'results-pkps-unified' if args.suite == 'helm' else 'results-eee-unified'
    if args.emb_tag:
        args.outdir = f'{args.outdir}-{args.emb_tag}'
    if args.suite == 'helm' and args.resp_mode != 'option-text':
        args.outdir += {'native': '-onehot', 'text': '-anstext'}[args.resp_mode]
    if args.response_space != 'joint':
        args.outdir = f'{args.outdir}-{args.response_space}'
    data, qmed, suite = load('suite' if args.suite == 'helm' else 'eee',
                             emb_tag=args.emb_tag, response_space=args.response_space,
                             resp_mode=args.resp_mode)
    # each spec is (n_models, n_tasks, p_task, p_query). The lever panels show one line per
    # n_models in --line_models; the n_models panel sweeps n at fixed levers.
    if args.sweep == 'n_models':
        specs = [(n, None, args.fixed_p, args.fixed_pq) for n in args.n_models_values]
    elif args.sweep == 'coverage':
        specs = [(n, None, p, args.fixed_pq) for n in args.line_models for p in args.coverages]
    elif args.sweep == 'n_tasks':
        specs = [(n, T, args.fixed_p, args.fixed_pq) for n in args.line_models for T in args.n_tasks_values]
    else:  # p_query
        specs = [(n, None, args.fixed_p, pq) for n in args.line_models for pq in args.p_query_values]
    jobs = [delayed(trial)(data, qmed, n, T, p, s, p_query=pq)
            for (n, T, p, pq) in specs for s in range(args.n_seeds)]
    res = pd.DataFrame([r for r in Parallel(n_jobs=args.n_jobs, verbose=5)(jobs) if r])
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    res.to_csv(Path(args.outdir) / f'completion_suite_{args.sweep}.csv', index=False)
    xcol = {'n_models': 'n_models', 'coverage': 'p_task', 'n_tasks': 'n_tasks', 'p_query': 'p_query'}[args.sweep]
    keys = [xcol] if args.sweep == 'n_models' else ['n_models', xcol]
    print(res.groupby(keys)[['mc', 'pkps', 'ens']].mean().round(4).to_string())


if __name__ == '__main__':
    main()
