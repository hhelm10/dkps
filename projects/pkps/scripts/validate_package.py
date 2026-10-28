#!/usr/bin/env python
"""Side-by-side validation of the dkps package classes against the paper's Table 1.

For each operating point the experiment scripts' *sampling* is replayed line-for-line
(same seeds, same RNG call order) -- see experiments/query_efficiency.run_seed and
experiments/completion.trial -- then the sampled rows are handed to the package classes
(PKPS / DKPS / SampleScore / IRT / LRMC / Ensemble) and the per-(seed, task) or per-seed
MAEs are compared with the result CSVs those scripts wrote. The experiment scripts are
not modified or called.

  python validate_package.py --suite helm --protocol qe --m 1 8 --n_seeds 16
  python validate_package.py --suite eee --protocol completion --n_seeds 16
"""
import argparse
import warnings
import sys, pathlib
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pipeline import loaders as H                                   # noqa: E402
from dkps import PKPS, DKPS, SampleScore, IRT, LRMC, Ensemble      # noqa: E402

MDS_DIM = 8


def _records(data, keep, score_mat=None, samp=None, models=None, tasks=None):
    resp_X, Qu, qid_code, model_id, task_id, query_id, _, mods_all, tasks_all, _, row_score = data[:11]
    df = pd.DataFrame({'model_id': model_id[keep], 'task_id': task_id[keep], 'query_id': query_id[keep],
                       'embedding': list(resp_X[keep]), 'query_embedding': list(Qu[qid_code[keep]]),
                       'score': row_score[keep]})
    models = list(mods_all if models is None else models); tasks = list(tasks_all if tasks is None else tasks)
    mi = {m: i for i, m in enumerate(models)}; ti = {t: j for j, t in enumerate(tasks)}
    ii = df['model_id'].map(mi).to_numpy(); jj = df['task_id'].map(ti).to_numpy()
    if score_mat is not None:
        df['reference_score'] = score_mat[ii, jj]
    if samp is not None:
        df['sample_score'] = samp[ii, jj]
    return df


def _pkps(qmed, bandwidth='cv', cv_k=8):
    # the QE script selects the bandwidth with k=8 LOMO neighbours, the completion script with k=5
    return PKPS(query_kwargs=dict(kernel='rbf', bandwidth=bandwidth, bandwidth_ref=qmed, pca_dim=None,
                                  cv_k=cv_k, normalization=H.NORMALIZATION),
                response_kwargs=dict(kernel='linear', pca_dim=None), mds_kwargs=dict(dim=MDS_DIM))


# ---------------------------------------------------------------- query efficiency
def qe_seed(data, m, seed):
    """Replays experiments/query_efficiency.run_seed(data, m, seed) with the default budget-sweep
    arguments (full cohort, p_task=1, no pairing control, ens_mode='cv', knn)."""
    (resp_X, Qu, qid_code, model_id, task_id, query_id,
     score_mat, models, tasks, groups, row_score, qmed) = data
    models = list(models)
    rng = np.random.default_rng(seed)
    keep, sample_mat = [], np.full((len(models), len(tasks)), np.nan)
    for i, mm in enumerate(models):
        for t, tt in enumerate(tasks):
            idx = groups.get((mm, tt))
            if idx is None or not len(idx):
                continue
            take = idx if len(idx) <= m else rng.choice(idx, m, replace=False)
            keep.append(take); sample_mat[i, t] = row_score[take].mean()
    keep = np.concatenate(keep)
    recs = _records(data, keep, score_mat=score_mat)

    pk = _pkps(qmed).fit(recs)
    dk = DKPS(normalization=H.NORMALIZATION,
              response_kwargs=dict(kernel='linear', pca_dim=None), mds_kwargs=dict(dim=MDS_DIM),
              sigma_ratio=1e-2, bandwidth_ref=qmed).fit(recs)
    ss = SampleScore().fit(recs)
    binary = [tt for tt in tasks if np.all(np.isin(np.unique(row_score[task_id == tt]), [0.0, 1.0]))]
    irt = IRT(binary_tasks=binary).fit(recs)          # decided on the full task pool, as in run_seed
    ens = Ensemble([ss, pk], mode='cv', holdout='family', predict_kwargs=[{}, {'holdout': 'family'}]).fit(recs)

    pairs = [{'model_id': mm, 'task_id': tt} for mm in models for tt in tasks]
    def mat(out):
        return np.array([r['score_hat'] for r in out]).reshape(len(models), len(tasks))
    preds = {'sample': mat(ss.predict(pairs)), 'dkps': mat(dk.predict(pairs, holdout='family')),
             'pkps': mat(pk.predict(pairs, holdout='family')), 'ens': mat(ens.predict(pairs)),
             'irt': mat(irt.predict(pairs))}
    rows = []
    for t, tt in enumerate(tasks):
        y = score_mat[:, t]
        for meth, P in preds.items():
            h = np.isfinite(P[:, t]) & np.isfinite(y)
            if meth == 'irt' and not h.any():
                continue
            rows.append(dict(m=m, seed=seed, task=tt, method=meth,
                             mae=float(np.mean(np.abs(P[h, t] - y[h]))) if h.any() else np.nan))
    return rows


# ---------------------------------------------------------------- completion
def completion_seed(data, qmed, n_models, p_task, seed, p_query=0.5):
    """Replays experiments/completion.trial(data, qmed, n_models, None, p_task, seed, p_query)."""
    rX, Qu, qc, mid, tid, qid, full, mods, tasks, grp, rs = data[:11]
    rng = np.random.default_rng(seed)
    msel = (np.sort(rng.choice(len(mods), min(n_models, len(mods)), replace=False))
            if n_models else np.arange(len(mods)))
    tsel = np.arange(len(tasks))
    mods_s = [mods[i] for i in msel]
    full_s = full[np.ix_(msel, tsel)]
    fin0 = np.isfinite(full_s)
    O = (rng.random(full_s.shape) < p_task) & fin0
    for i in range(O.shape[0]):
        if not O[i].any() and fin0[i].any(): O[i, np.where(fin0[i])[0][0]] = True
    for t in range(O.shape[1]):
        if not O[:, t].any() and fin0[:, t].any(): O[np.where(fin0[:, t])[0][0], t] = True
    K_PERSP = 16
    keep, samp = [], np.full(full_s.shape, np.nan)
    for ii, mi in enumerate(msel):
        for tt, ti in enumerate(tsel):
            if O[ii, tt] and (mods[mi], tasks[ti]) in grp:
                idx = grp[(mods[mi], tasks[ti])]
                nq = max(1, int(round(p_query * len(idx))))
                take = idx if nq >= len(idx) else rng.choice(idx, nq, replace=False)
                samp[ii, tt] = rs[take].mean()
                keep.append(take if len(take) <= K_PERSP else rng.choice(take, K_PERSP, replace=False))
    keep = np.concatenate(keep)
    recs = _records(data, keep, samp=samp, models=mods_s, tasks=tasks)

    pk = _pkps(qmed, cv_k=5).fit(recs)
    mc = LRMC(n_init=2, crossfit_k=3, crossfit_random_state=rng).fit(recs)   # rng state continues, as in trial()
    ens = Ensemble([pk, mc], mode='cv', holdout=None, predict_kwargs=[{'whiten': True}, {}],
                   fallback=False).fit(recs)
    pairs = [{'model_id': mm, 'task_id': tt} for mm in mods_s for tt in tasks]
    def mat(out):
        return np.array([r['score_hat'] for r in out]).reshape(len(mods_s), len(tasks))
    P = {'pkps': mat(pk.predict(pairs, whiten=True)), 'mc': mat(mc.predict(pairs)), 'ens': mat(ens.predict(pairs))}
    mis = (~O) & fin0
    def mae(M):
        h = mis & np.isfinite(M)
        return float(np.mean(np.abs(M[h] - full_s[h]))) if h.any() else np.nan
    return dict(n_models=len(mods_s), p_task=p_task, p_query=p_query, seed=seed,
                **{k: mae(v) for k, v in P.items()})


# ---------------------------------------------------------------- driver
def main():
    warnings.filterwarnings('ignore', message='IRT: task')        # WMT tasks are non-binary by design
    warnings.filterwarnings('ignore', message='model pairs with zero')  # delta-limit DKPS bandwidth
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', choices=['helm', 'eee'], default='helm')
    ap.add_argument('--protocol', choices=['qe', 'completion'], default='qe')
    ap.add_argument('--m', type=int, nargs='+', default=[1, 8])
    ap.add_argument('--n_seeds', type=int, default=16)
    ap.add_argument('--n_jobs', type=int, default=-1)
    ap.add_argument('--tol', type=float, default=5e-4, help='per-row |diff| tolerance')
    ap.add_argument('--response_space', choices=['joint', 'blocked'], default='joint',
                    help='joint = base method (default result dirs); blocked = the '
                         'archived per-dataset protocol (HELM: the published one-hot '
                         'construction in the -onehot-blocked dirs; EEE: -blocked dirs)')
    args = ap.parse_args()
    if args.suite == 'helm':
        data = H.load_suite() if args.response_space == 'joint' else H.load_suite(
            resp_mode='native', response_space='blocked')
        sfx = '' if args.response_space == 'joint' else '-onehot-blocked'
    else:
        data = H.load_eee(response_space=args.response_space)
        sfx = '' if args.response_space == 'joint' else '-blocked'
    nmax = len(data[7])
    print(f'suite {args.suite}: {nmax} models, {len(data[8])} tasks')

    if args.protocol == 'qe':
        ref = pd.read_csv(f"results-{'pkps' if args.suite == 'helm' else 'eee'}-rd1{sfx}/rd1_suite_budget.csv")
        ref = ref[(ref['n_paired'] == -1) & ref['m'].isin(args.m) & (ref['seed'] < args.n_seeds)]
        jobs = [delayed(qe_seed)(data, m, s) for m in args.m for s in range(args.n_seeds)]
        got = pd.DataFrame([r for sub in Parallel(n_jobs=args.n_jobs, verbose=2)(jobs) for r in sub])
        keys = ['m', 'seed', 'task', 'method']
        cmp = ref[keys + ['mae']].merge(got, on=keys, how='outer', suffixes=('_paper', '_pkg'))
        summary = cmp.groupby(['m', 'method'])[['mae_paper', 'mae_pkg']].mean()
    else:
        ref = pd.read_csv(f"results-{'pkps' if args.suite == 'helm' else 'eee'}-unified{sfx}/completion_suite_coverage.csv")
        specs = [(10, 0.5), (nmax, 0.5), (10, 0.2)]
        ref = ref[ref.apply(lambda r: (r['n_models'], r['p_task']) in specs, axis=1) & (ref['seed'] < args.n_seeds)]
        jobs = [delayed(completion_seed)(data, data[11], n, p, s) for n, p in specs for s in range(args.n_seeds)]
        got = pd.DataFrame(Parallel(n_jobs=args.n_jobs, verbose=2)(jobs))
        keys = ['n_models', 'p_task', 'seed']
        long = lambda d, tag: d.melt(id_vars=keys, value_vars=['mc', 'pkps', 'ens'], var_name='method',
                                     value_name=f'mae_{tag}')
        cmp = long(ref, 'paper').merge(long(got, 'pkg'), on=keys + ['method'], how='outer')
        summary = cmp.groupby(['n_models', 'p_task', 'method'])[['mae_paper', 'mae_pkg']].mean()

    cmp['diff'] = cmp['mae_pkg'] - cmp['mae_paper']
    summary['diff'] = summary['mae_pkg'] - summary['mae_paper']
    summary['max_abs_row_diff'] = cmp.groupby(summary.index.names)['diff'].apply(lambda d: d.abs().max())
    summary['n_rows'] = cmp.groupby(summary.index.names).size()
    pd.set_option('display.width', 200)
    print(summary.round(4).to_string())
    bad = cmp[cmp['diff'].abs() > args.tol]
    print(f'\nrows differing by more than {args.tol}: {len(bad)} / {len(cmp)}')
    if len(bad):
        print(bad.head(20).to_string())
    out = Path('results-validation'); out.mkdir(exist_ok=True)
    cmp.to_csv(out / f'{args.suite}_{args.protocol}{sfx}.csv', index=False)


if __name__ == '__main__':
    main()
