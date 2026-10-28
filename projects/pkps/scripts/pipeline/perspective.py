#!/usr/bin/env python
"""
pipeline/perspective.py  --  one problem: estimate the full (model x task) score matrix from
observations where each cell has m >= 0 queries. RD1 is the m>0 (denoise) face, RD2
the m=0 (complete) face; here they are unified. Per cell the score channel is the
own noisy sample (observed) or matrix completion (missing); PKPS adds the item-level
embedding. Two blend weights are tuned honestly: the observed (denoising) weight by
splitting each cell's queries in half, the missing (completion) weight by held-out
observed cells. MAE is over EVERY cell vs the true full score.

Levers: task coverage (p), queries per observed cell (q, =0 noiseless/full), cohort.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from dkps.unpaired_dkps import pca_reduce_elbow
from dkps.baselines import matrix_completion_predict
from pipeline import loaders as H
from pipeline.crossval import load

GRID = np.linspace(0, 1, 21)


def mc_crossfit(M, O, rng, k=3, n_init=2):
    """Matrix completion that is clean at EVERY cell: missing cells by a full fit, and
    observed cells by k-fold cross-fitting (each predicted by a fit that excludes it), so
    the completion never echoes a cell's own noisy sample. Mirrors PKPS's leave-one-out."""
    pred = matrix_completion_predict(M, O, n_init=n_init)
    oi = np.argwhere(O & np.isfinite(M))
    if len(oi) >= k:
        for fold in np.array_split(rng.permutation(len(oi)), k):
            cells = oi[fold]
            cvo = O.copy(); cvo[cells[:, 0], cells[:, 1]] = False
            p = matrix_completion_predict(M, cvo, n_init=n_init)
            pred[cells[:, 0], cells[:, 1]] = p[cells[:, 0], cells[:, 1]]
    return pred


def _perspective(rX, Qu, qc, mid, tid, qid, keep, mods, suite, qmed):
    codes = qc[keep]
    if suite:
        df = pd.DataFrame({'model_id': mid[keep], 'task_id': tid[keep], 'query_id': qid[keep],
                           'embedding': list(rX[keep]), 'query_embedding': list(Qu[codes])})
        rk, med = 'linear', qmed
    else:
        rr = pca_reduce_elbow(rX[keep]); uq = np.unique(codes); qr = pca_reduce_elbow(Qu[uq])
        from scipy.spatial.distance import pdist
        c2v = {c: qr[j] for j, c in enumerate(uq)}
        df = pd.DataFrame({'model_id': mid[keep], 'task_id': tid[keep], 'query_id': qid[keep],
                           'embedding': list(rr), 'query_embedding': [c2v[c] for c in codes]})
        rk, med = 'rbf', float(np.median(pdist(qr if len(qr) <= 400 else qr[:400])))
    bw = H.SubsampleMedianBandwidth()
    est = H.ProductKernelPerspectiveSpace(query_kernel='rbf', response_kernel=rk,
                                          query_bandwidth=bw, response_bandwidth=bw,
                                          normalization=H.NORMALIZATION)
    names, Ds = est.dist_matrices(df, [med])
    return H._mds_full(Ds[med], names, mods, 8)


def trial(data, suite, qmed, p, q, n_models, seed, q_ref=None):
    """Estimate the full score matrix. Two modes:
      q_ref is None : symmetric -- coverage p picks observed cells (depth q), eval all cells.
      q_ref set     : reference/target -- a fraction p of cells is a fixed-depth (q_ref)
                      REFERENCE scaffold; the rest are TARGET cells given q queries (q>=0,
                      q=0 = missing/pure completion). Evaluate on the target cells, so the
                      x-axis can include q=0 and unifies completion (q=0) with denoising (q>0).
    """
    rX, Qu, qc, mid, tid, qid, full, mods, tasks, grp, rs = data[:11]
    rng = np.random.default_rng(seed)
    if n_models and n_models < len(mods):
        sel = np.sort(rng.choice(len(mods), n_models, replace=False))
        mods = [mods[i] for i in sel]; full = full[sel]
    fin0 = np.isfinite(full)
    if q_ref is None:
        O = H.sample_observed(len(mods), len(tasks), p, rng) & fin0
        ref = np.zeros_like(O); eval_mask = np.ones_like(O)
    else:
        ref = H.sample_observed(len(mods), len(tasks), p, rng) & fin0
        tgt = (~ref) & fin0
        eval_mask = tgt
        O = ref | tgt if (q and q > 0) else ref.copy()
    keep, samp = [], np.full_like(full, np.nan)
    ncnt = np.zeros_like(full)                                          # queries per observed cell
    s1 = np.full_like(full, np.nan); s2 = np.full_like(full, np.nan)    # half-splits (observed)
    for i in range(len(mods)):
        for t in range(len(tasks)):
            if O[i, t] and (mods[i], tasks[t]) in grp:
                d = q_ref if (q_ref is not None and ref[i, t]) else q
                idx = grp[(mods[i], tasks[t])]
                take = idx if (d is None or len(idx) <= d) else rng.choice(idx, d, replace=False)
                keep.append(take); samp[i, t] = rs[take].mean(); ncnt[i, t] = len(take)
                if len(take) >= 2:
                    h = rng.permutation(len(take))
                    s1[i, t] = rs[take[h[:len(take) // 2]]].mean()
                    s2[i, t] = rs[take[h[len(take) // 2:]]].mean()
    keep = np.concatenate(keep)
    Z = _perspective(rX, Qu, qc, mid, tid, qid, keep, mods, suite, qmed)
    if Z is None:
        return None
    # the completion prior is fit on O_score: all observed cells (symmetric), or only the
    # reference cells (reference/target -- the noisy target samples are what we denoise, so
    # they must not pollute the prior; targets are predicted, then blended with their own
    # sample). Both channels are cross-fit/LOO so the prior never echoes a cell's own sample.
    O_score = ref if q_ref is not None else O
    pk = H.predict_from_embedding_all(Z, samp, O_score, predictor='knn', k=5)
    mc = mc_crossfit(samp, O_score, rng)
    obs = O & np.isfinite(samp)
    nqc = np.maximum(ncnt, 1)

    # completion prior = blend(PKPS, MC). Both are cross-fit, so the prior never echoes a
    # cell's own (noisy) sample; the blend weight and the prior's error are read directly
    # off the now-clean residuals against the observed samples, depth-weighted so low-noise
    # (deeply-sampled) cells dominate.
    oo = obs & np.isfinite(pk) & np.isfinite(mc)
    a_mis = 0.5
    if oo.sum() >= 5:
        so, po, mo, wo = samp[oo], pk[oo], mc[oo], nqc[oo]
        a_mis = GRID[int(np.argmin([np.sum(wo * np.abs(np.clip(a * po + (1 - a) * mo, 0, 1) - so)) / np.sum(wo)
                                    for a in GRID]))]
    completion = a_mis * pk + (1 - a_mis) * mc
    completion = np.clip(np.where(np.isfinite(completion), completion,
                                  np.where(np.isfinite(mc), mc, pk)), 0, 1)

    # observed cells: precision blend of the own sample against the completion prior, PER
    # cell by its query count -- weight on the sample = comp_err/(comp_err + p(1-p)/m). A
    # shallow cell leans on the prior (never doing worse than completing it, the q=0 point);
    # a deeply-sampled cell trusts its own mean. Essential when cells differ in depth.
    pcl = np.clip(np.where(np.isfinite(pk), pk, 0.5), 1e-3, 1 - 1e-3)
    sig_cell = pcl * (1 - pcl) / nqc
    ce = obs & np.isfinite(completion)
    if ce.sum() >= 5:
        err = (completion[ce] - samp[ce]) ** 2 - sig_cell[ce]
        comp_err = max(float(np.sum(nqc[ce] * err) / np.sum(nqc[ce])), 1e-6)
    else:
        comp_err = 1e-6
    w_samp = comp_err / (comp_err + sig_cell)
    score = np.where(obs, samp, mc)
    ens = np.where(obs, np.clip(w_samp * samp + (1 - w_samp) * completion, 0, 1), completion)
    w_pk = float(np.nanmean(np.where(obs, w_samp, np.nan)))
    fin = np.isfinite(full)
    ev = eval_mask & fin
    def mae(P, m): mm = m & np.isfinite(P); return float(np.mean(np.abs(P[mm] - full[mm]))) if mm.any() else np.nan
    return dict(p=p, q=(q if q is not None else 0), n_models=len(mods), seed=seed,
                score_all=mae(score, ev), pkps_all=mae(pk, ev),
                mc_all=mae(mc, ev), ens_all=mae(ens, ev),
                score_obs=mae(score, ev & obs), ens_obs=mae(ens, ev & obs),
                score_mis=mae(score, ev & ~O), ens_mis=mae(ens, ev & ~O),
                w_pk=w_pk, a_mis=a_mis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='math')
    ap.add_argument('--sweep', choices=['queries', 'coverage', 'n_models'], default='queries')
    ap.add_argument('--queries', type=int, nargs='+', default=[0, 1, 2, 4, 8, 16, 32])
    ap.add_argument('--coverages', type=float, nargs='+', default=[0.2, 0.35, 0.5, 0.7, 0.9])
    ap.add_argument('--n_models_values', type=int, nargs='+', default=[10, 20, 40, 60, 93])
    ap.add_argument('--fixed_p', type=float, default=0.5)
    ap.add_argument('--fixed_q', type=int, default=8)
    ap.add_argument('--q_ref', type=int, default=32, help='reference-cell depth for the queries sweep (enables q=0)')
    ap.add_argument('--n_seeds', type=int, default=12)
    ap.add_argument('--n_jobs', type=int, default=-1)
    ap.add_argument('--outdir', default='results-pkps-unified')
    args = ap.parse_args()
    data, qmed, suite = load(args.dataset)

    # the queries sweep uses a reference/target split so the x-axis can include q=0
    # (pure completion); coverage and n_models keep the symmetric eval-all framing.
    if args.sweep == 'queries':
        specs = [(args.fixed_p, q, None, args.q_ref) for q in args.queries]; xcol = 'q'
    elif args.sweep == 'coverage':
        specs = [(p, args.fixed_q, None, None) for p in args.coverages]; xcol = 'p'
    else:
        specs = [(args.fixed_p, args.fixed_q, n, None) for n in args.n_models_values]; xcol = 'n_models'
    jobs = [delayed(trial)(data, suite, qmed, p, q, nm, s, qr)
            for (p, q, nm, qr) in specs for s in range(args.n_seeds)]
    res = pd.DataFrame([r for r in Parallel(n_jobs=args.n_jobs, verbose=5)(jobs) if r])
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    res.to_csv(Path(args.outdir) / f'unified_{args.dataset}_{args.sweep}.csv', index=False)
    print(res.groupby(xcol)[['mc_all', 'pkps_all', 'score_all', 'ens_all', 'score_obs', 'ens_obs',
                             'score_mis', 'ens_mis', 'w_pk', 'a_mis']].mean().round(4).to_string())


if __name__ == '__main__':
    main()
