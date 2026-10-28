"""Score-profile routing baseline (no embeddings anywhere).

Route to the candidate whose per-task ANCHOR score profile is nearest the
target's (L2 over task means). This is the legitimate no-embedding
baseline -- what a leaderboard alone can do -- unlike the tabular
outcome-regression ablation, whose labels derive from response embeddings.

Result: loses badly on both suites (HELM 0.402 vs qa 0.295 / static 0.342;
EEE 0.617 vs qa 0.532 / random 0.670): models with similar score profiles
do not behave similarly.

Run from repo root: pixi run python -m projects.routing.experiments.run_profile_baseline
"""

import numpy as np
import pandas as pd

from .evaluate import stratified_split
from .geometry import pairwise_query_dist_tensor, weighted_dist_matrix
from .run_eee import load_eee_rows
from .run_helm import load_paired_core


def helm(seeds=5):
    X, U, cats, models, S = load_paired_core(with_scores=True)
    P = pairwise_query_dist_tensor(X)
    E = np.sqrt(P)
    n = len(models)
    eye = np.eye(n, dtype=bool)
    res = []
    for seed in range(seeds):
        a_idx, e_idx = stratified_split(cats, 0.2, np.random.default_rng(seed))
        prof = np.stack([S[:, a_idx[cats[a_idx] == t]].mean(axis=1)
                         for t in sorted(set(cats))], axis=1)
        Dp = np.linalg.norm(prof[:, None, :] - prof[None, :, :], axis=2)
        Dst = weighted_dist_matrix(P[a_idx], np.ones(len(a_idx)))
        for name, D in (('profile', Dp), ('static', Dst)):
            Dm = D.copy(); Dm[eye] = np.inf
            picks = Dm.argmin(axis=1)
            res.append((seed, name, float(np.mean(
                [E[q][np.arange(n), picks].mean() for q in e_idx]))))
    return pd.DataFrame(res, columns=['seed', 'method', 'error'])


def eee(seeds=5, n_eval=300):
    X, Qu, rows, models, qmed = load_eee_rows()
    n = len(models)
    res = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        resp = rows.groupby(['task', 'query'])['model'].nunique()
        elig = resp[resp >= 3].reset_index()[['task', 'query']]
        ekeys = set()
        for t, g in elig.groupby('task'):
            k = min(len(g), max(1, int(round(n_eval * len(g) / len(elig)))))
            take = g.iloc[rng.choice(len(g), size=k, replace=False)]
            ekeys.update(zip(take['task'], take['query']))
        isev = np.array([(t, q) in ekeys
                         for t, q in zip(rows['task'], rows['query'])])
        prof = (rows[~isev].pivot_table(index='model', columns='task',
                                        values='score')
                .reindex(range(n)).to_numpy())
        Dp = np.linalg.norm(prof[:, None, :] - prof[None, :, :], axis=2)
        errs = []
        for (t, q), g in rows[isev].groupby(['task', 'query']):
            ms = g['model'].to_numpy()
            Xq = X[g.index.to_numpy()]
            Eq = np.sqrt(((Xq[:, None] - Xq[None, :]) ** 2).sum(-1))
            for ti in range(len(ms)):
                cand = [c for c in range(len(ms)) if c != ti]
                d = Dp[ms[ti], ms[cand]]
                errs.append(Eq[ti, cand][int(np.argmin(d))])
        res.append((seed, 'profile', float(np.mean(errs))))
    return pd.DataFrame(res, columns=['seed', 'method', 'error'])


def main():
    h = helm()
    print('HELM:'); print(h.groupby('method')['error'].mean().round(4).to_string())
    e = eee()
    print('EEE:'); print(e.groupby('method')['error'].mean().round(4).to_string())


if __name__ == '__main__':
    main()
