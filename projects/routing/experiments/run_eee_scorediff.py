"""Score-difference routing on the (unpaired) EEE suite.

Outcome among an eval query's responders: err(t, r, q*) = |s_{r,q*} - s_{t,q*}|.

Routers:
    shat-*   : |s_hat_c(q*) - s_hat_t(q*)| from each model's OWN localized
               score history (kernel weights k_Q(., q*)) -- factorized,
               needs no pairwise overlap (the PKPS move applied to scores);
               plus task/static weightings.
    qa-*     : nearest to target in the localized response geometry.
    rfS-*    : direct kernel regression of realized |score diff| over the
               anchor queries the PAIR shared (overlap-starved baseline).
random/oracle bracket. LOQO: eval-query rows excluded from all histories.

Run from repo root: pixi run python -m projects.routing.experiments.run_eee_scorediff
"""

import os

import numpy as np
import pandas as pd

from .run_eee import (RESULTS, batched_localized_stats, load_eee_rows,
                      localized_stats)

FRACTIONS = (0.25, 0.5, 1.0, 2.0)


def build_pair_score_overlap(rows, anchor_mask, Qu):
    pair_u, pair_e = {}, {}
    a = rows[anchor_mask]
    for (_, _), g in a.groupby(['task', 'query']):
        ms = g['model'].to_numpy()
        if len(ms) < 2:
            continue
        s = g['score'].to_numpy()
        u = Qu[g['code'].iloc[0]]
        for i in range(len(ms)):
            for k in range(i + 1, len(ms)):
                key = (min(ms[i], ms[k]), max(ms[i], ms[k]))
                pair_u.setdefault(key, []).append(u)
                pair_e.setdefault(key, []).append(abs(float(s[i] - s[k])))
    return ({k: np.stack(v) for k, v in pair_u.items()},
            {k: np.asarray(v) for k, v in pair_e.items()})


def rf_predict(pair_u, pair_e, grand_mean, a, b, u_star, sigma):
    key = (min(a, b), max(a, b))
    if key not in pair_e:
        return grand_mean
    e = pair_e[key]
    if sigma is None:
        return float(e.mean())
    d2 = ((pair_u[key] - u_star) ** 2).sum(axis=1)
    w = np.exp(-(d2 - d2.min()) / (2.0 * sigma ** 2))
    return float((w * e).sum() / w.sum())


def run_seed(X, Qu, rows, n, qmed, seed, n_eval=300):
    rng = np.random.default_rng(seed)
    resp = rows.groupby(['task', 'query'])['model'].nunique()
    eligible = resp[resp >= 3].reset_index()[['task', 'query']]
    eval_keys = set()
    for t_name, g in eligible.groupby('task'):
        k = min(len(g), max(1, int(round(n_eval * len(g) / len(eligible)))))
        take = g.iloc[rng.choice(len(g), size=k, replace=False)]
        eval_keys.update(zip(take['task'], take['query']))
    is_eval_row = np.array([(t, q) in eval_keys
                            for t, q in zip(rows['task'], rows['query'])])
    anchor = ~is_eval_row
    assert all((t, q) not in eval_keys
               for t, q in zip(rows['task'][anchor], rows['query'][anchor]))

    Xa = X[anchor]
    sa = rows['score'].to_numpy()[anchor]
    model_a = rows['model'].to_numpy()[anchor]
    task_a = rows['task'].to_numpy()[anchor]
    ua = Qu[rows['code'].to_numpy()[anchor]]

    pair_u, pair_e = build_pair_score_overlap(rows, anchor, Qu)
    grand = float(np.concatenate(list(pair_e.values())).mean())
    sigmas = {f'{f:g}x': f * qmed for f in FRACTIONS}

    model_groups = [np.flatnonzero(model_a == m) for m in range(n)]
    eval_groups = list(rows[is_eval_row].groupby(['task', 'query']))
    Ue = np.stack([Qu[g['code'].iloc[0]] for _, g in eval_groups])
    D2 = ((Ue[:, None, :] - ua[None, :, :]) ** 2).sum(-1)
    D2 = D2 - D2.min(axis=1, keepdims=True)
    batched = {name: batched_localized_stats(
                   Xa, sa, model_groups, n,
                   np.exp(-D2 / (2.0 * s ** 2)).astype(np.float32))
               for name, s in sigmas.items()}
    task_stats = {tn: localized_stats(Xa, sa, (task_a == tn).astype(float), model_a, n)
                  for tn in np.unique(task_a)}
    static_stats = localized_stats(Xa, sa, np.ones(len(Xa)), model_a, n)

    out = []
    for gi, ((t_name, q), g) in enumerate(eval_groups):
        resp_models = g['model'].to_numpy()
        u_star = Ue[gi]
        s_real = g['score'].to_numpy()
        r = len(resp_models)
        E = np.abs(s_real[:, None] - s_real[None, :])

        stats = {name: (B[0][gi], B[1][gi], B[2][gi]) for name, B in batched.items()}
        stats['task'] = task_stats[t_name]
        stats['static'] = static_stats

        for ti in range(r):
            cand = [c for c in range(r) if c != ti]
            errs = E[ti, cand]
            for name, (phi, shat, ok) in stats.items():
                pref = 'shat-' if name in sigmas or name in ('task', 'static') else ''
                # score router
                d = np.abs(shat[resp_models[cand]] - shat[resp_models[ti]])
                d[~ok[resp_models[cand]]] = np.inf
                out.append((seed, t_name, q, f'shat-{name}', float(errs[int(np.argmin(d))])))
                # response-geometry router
                dg = np.linalg.norm(phi[resp_models[cand]] - phi[resp_models[ti]], axis=1)
                dg[~ok[resp_models[cand]]] = np.inf
                out.append((seed, t_name, q, f'qa-{name}', float(errs[int(np.argmin(dg))])))
            for name, s in list(sigmas.items())[:2] + [('static', None)]:
                pred = np.array([rf_predict(pair_u, pair_e, grand,
                                            resp_models[ti], resp_models[c], u_star,
                                            s if isinstance(s, float) else None)
                                 for c in cand])
                out.append((seed, t_name, q, f'rfS-{name}', float(errs[int(np.argmin(pred))])))
            out.append((seed, t_name, q, 'random', float(errs.mean())))
            out.append((seed, t_name, q, 'oracle', float(errs.min())))
    return pd.DataFrame(out, columns=['seed', 'task', 'query', 'method', 'error'])


def main():
    os.makedirs(RESULTS, exist_ok=True)
    X, Qu, rows, models, qmed = load_eee_rows()
    n = len(models)
    df = pd.concat([run_seed(X, Qu, rows, n, qmed, s) for s in range(5)],
                   ignore_index=True)
    df.to_parquet(os.path.join(RESULTS, 'eee_scorediff.parquet'))
    per_seed = df.groupby(['method', 'seed'])['error'].mean().unstack()
    summary = pd.DataFrame({
        'mean_error': per_seed.mean(axis=1),
        'se': per_seed.std(axis=1) / np.sqrt(per_seed.shape[1]),
    }).sort_values('mean_error')
    summary.to_csv(os.path.join(RESULTS, 'eee_scorediff_summary.csv'))
    print(summary.round(4).to_string())


if __name__ == '__main__':
    main()
