"""Cost-aware routing: offload flagship queries to cheap substitutes.

Reframing of mimicry routing with an expensive target: the flagship f* is
the suite's best model by anchor mean score. For each evaluation query, a
router proposes a substitute r and a confidence d (predicted substitution
quality); sweeping an offload budget (fraction of queries served by the
substitute, lowest-d first) traces realized-score-retention vs. savings.
Savings per offloaded query = 1 - (substitute price / flagship price),
reported parametrically as the offload fraction.

Routers (substitute choice + query ordering):
    qa       : nearest to flagship in the query-localized geometry; order by
               that distance (label-free, response-based)
    static   : same with uniform weights (query ordering constant per pair
               structure -> ordering by distance of the chosen substitute)
    task*    : nearest in the task-indicator geometry (hidden labels)
    cascade  : leaderboard cascade -- per task, substitute = anchor-best
               non-flagship model; order tasks by historical score gap
               (uses task labels; what practitioners do today)
    profile  : substitute = nearest anchor score profile to the flagship;
               constant ordering (query-independent)
    random   : random substitute, random order (expectation over draws)

Metric: mean realized score of the policy at each offload fraction,
relative to all-flagship. LOQO: eval-query rows never enter geometries,
profiles, or anchor score means.

Run from repo root: pixi run python -m projects.routing.experiments.run_cost_routing
"""

import os

import numpy as np
import pandas as pd

from .evaluate import stratified_split
from .geometry import pairwise_query_dist_tensor, rbf_weights, weighted_dist_matrix
from .run_helm import RESULTS, load_paired_core

FRACS = np.linspace(0.0, 1.0, 11)
SIGMA_FRACTION = 0.25


def parse_params(name):
    """Parameter count from the model name (compute-cost proxy for open
    models); None for closed/unsized models (the expensive tier)."""
    import re
    n = name.lower()
    m = re.findall(r'(\d+)x(\d+(?:\.\d+)?)b', n)
    if m:
        return float(m[0][0]) * float(m[0][1])
    m = re.findall(r'(\d+(?:\.\d+)?)b(?![a-z0-9])', n)
    if m:
        return max(float(x) for x in m)
    return None


def policy_curve(order, sub_scores, flag_scores):
    """Mean realized score at each offload fraction (lowest-order first)."""
    q = len(order)
    rank = np.argsort(order)
    out = []
    for f in FRACS:
        k = int(round(f * q))
        take = np.zeros(q, dtype=bool)
        take[rank[:k]] = True
        out.append(float(np.where(take, sub_scores, flag_scores).mean()))
    return out


def run_helm(seeds=5, pools=(('le13b', 13.0), ('le40b', 40.0), ('all', None))):
    X, U, cats, models, S = load_paired_core(with_scores=True)
    P = pairwise_query_dist_tensor(X)
    n, m = S.shape
    sizes = [parse_params(mm) for mm in models]
    rows = []
    for pool_name, cap in pools:
      for seed in range(seeds):
        rng = np.random.default_rng(seed)
        a_idx, e_idx = stratified_split(cats, 0.2, rng)
        flag = int(np.argmax(S[:, a_idx].mean(axis=1)))
        if cap is None:
            cand = np.array([i for i in range(n) if i != flag])
        else:
            cand = np.array([i for i in range(n)
                             if i != flag and sizes[i] is not None
                             and sizes[i] <= cap])
        flag_scores = S[flag, e_idx]

        # anchor statistics
        med_i = rng.integers(0, len(a_idx), 20000)
        med_k = rng.integers(0, len(a_idx), 20000)
        keep = med_i != med_k
        med = float(np.median(np.linalg.norm(
            U[a_idx][med_i[keep]] - U[a_idx][med_k[keep]], axis=1)))
        tasks = sorted(set(cats))
        t_of = {t: np.flatnonzero(cats[a_idx] == t) for t in tasks}
        task_mean = {t: S[:, a_idx[t_of[t]]].mean(axis=1) for t in tasks}
        prof = np.stack([task_mean[t] for t in tasks], axis=1)     # (n, T)

        # --- qa: localized distance flagship <-> candidates, per eval query
        W = rbf_weights(U[a_idx], U[e_idx], SIGMA_FRACTION * med)
        Wn = W / W.sum(axis=1, keepdims=True)
        D_qa = weighted_dist_matrix(P[a_idx], W)                   # (q, n, n)
        d_qa = D_qa[:, flag, :][:, cand]                           # (q, n-1)
        pick_qa = cand[d_qa.argmin(axis=1)]
        conf_qa = d_qa.min(axis=1)
        # qa-cal: scale-free ordering (min / mean distance) so one-hot and
        # text domains are comparable under a single threshold (label-free)
        conf_qacal = d_qa.min(axis=1) / d_qa.mean(axis=1)

        # --- qa-score: label-free analog of the cascade -- localized score
        # regression per model; substitute = argmax predicted score in pool,
        # ordering = predicted flagship - substitute gap
        shat = Wn @ S[:, a_idx].T                                  # (q, n)
        pick_qs = cand[shat[:, cand].argmax(axis=1)]
        conf_qs = shat[np.arange(len(e_idx)), flag] \
            - shat[np.arange(len(e_idx)), pick_qs]

        # --- static
        D_st = weighted_dist_matrix(P[a_idx], np.ones(len(a_idx)))
        d_st = D_st[flag, cand]
        pick_st = np.full(len(e_idx), cand[int(np.argmin(d_st))])
        # order queries by the qa-agnostic proxy: constant -> random tiebreak
        conf_st = rng.random(len(e_idx))

        # --- task* (hidden labels): per-task geometry
        pick_tk = np.empty(len(e_idx), dtype=int)
        conf_tk = np.empty(len(e_idx))
        for t in tasks:
            Dt = weighted_dist_matrix(P[a_idx],
                                      (cats[a_idx] == t).astype(float))
            dt = Dt[flag, cand]
            sel = cats[e_idx] == t
            pick_tk[sel] = cand[int(np.argmin(dt))]
            conf_tk[sel] = dt.min()

        # --- cascade: per-task best substitute by anchor score; order by gap
        pick_cs = np.empty(len(e_idx), dtype=int)
        conf_cs = np.empty(len(e_idx))
        for t in tasks:
            tm = task_mean[t]
            best = cand[int(np.argmax(tm[cand]))]
            gap = float(tm[flag] - tm[best])
            sel = cats[e_idx] == t
            pick_cs[sel] = best
            conf_cs[sel] = gap

        # --- profile: nearest score profile to flagship (constant)
        dp = np.linalg.norm(prof[cand] - prof[flag], axis=1)
        pick_pf = np.full(len(e_idx), cand[int(np.argmin(dp))])
        conf_pf = rng.random(len(e_idx))

        # --- random
        pick_rd = cand[rng.integers(0, len(cand), len(e_idx))]
        conf_rd = rng.random(len(e_idx))

        for name, pick, conf in (('qa', pick_qa, conf_qa),
                                 ('qa-cal', pick_qa, conf_qacal),
                                 ('qa-score', pick_qs, conf_qs),
                                 ('task*', pick_tk, conf_tk),
                                 ('cascade', pick_cs, conf_cs),
                                 ('static', pick_st, conf_st),
                                 ('profile', pick_pf, conf_pf),
                                 ('random', pick_rd, conf_rd)):
            sub_scores = S[pick, e_idx]
            curve = policy_curve(conf, sub_scores, flag_scores)
            for f, v in zip(FRACS, curve):
                rows.append((pool_name, seed, name, float(f), v,
                             float(flag_scores.mean())))
    return pd.DataFrame(rows, columns=['pool', 'seed', 'method',
                                       'offload_frac', 'policy_score',
                                       'flagship_score'])


def run_eee_cost(seeds=5, pools=(('le5b', 5.0), ('le30b', 30.0)), n_eval=300):
    """Cost-aware offloading on the unpaired EEE suite. Candidates per eval
    query = the query's responders within the size-capped pool; flagship must
    be a responder (its pools define the eval set)."""
    from .run_eee import (batched_localized_stats, load_eee_rows,
                          localized_stats)
    X, Qu, rows, models, qmed = load_eee_rows()
    n = len(models)
    sizes = [parse_params(mm) for mm in models]
    out = []
    for pool_name, cap in pools:
        pool = {i for i in range(n) if sizes[i] is not None and sizes[i] <= cap}
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            # flagship by anchor-agnostic overall mean (fixed across seeds)
            gmean = rows.groupby('model')['score'].mean()
            flag = int(gmean.idxmax())
            pool_c = pool - {flag}

            # eval queries: flagship responded AND >=1 pool responder
            resp = rows.groupby(['task', 'query'])['model'].agg(set)
            elig = [(tq, ms) for tq, ms in resp.items()
                    if flag in ms and len(ms & pool_c) >= 1]
            take = rng.choice(len(elig), size=min(n_eval, len(elig)),
                              replace=False)
            eval_keys = {elig[i][0] for i in take}
            is_eval = np.array([(t, q) in eval_keys
                                for t, q in zip(rows['task'], rows['query'])])
            anchor_m = ~is_eval
            Xa = X[anchor_m]
            sa = rows['score'].to_numpy()[anchor_m]
            model_a = rows['model'].to_numpy()[anchor_m]
            task_a = rows['task'].to_numpy()[anchor_m]
            ua = Qu[rows['code'].to_numpy()[anchor_m]]
            model_groups = [np.flatnonzero(model_a == m) for m in range(n)]
            tasks = sorted(set(task_a))
            tmean = {tn: pd.Series(sa[task_a == tn]).groupby(
                        pd.Series(model_a[task_a == tn])).mean()
                     for tn in tasks}

            eval_groups = list(rows[is_eval].groupby(['task', 'query']))
            Ue = np.stack([Qu[g['code'].iloc[0]] for _, g in eval_groups])
            D2 = ((Ue[:, None, :] - ua[None, :, :]) ** 2).sum(-1)
            D2 = D2 - D2.min(axis=1, keepdims=True)
            W = np.exp(-D2 / (2.0 * (SIGMA_FRACTION * 2 * qmed) ** 2)).astype(np.float32)
            phi, shat, okm = batched_localized_stats(Xa, sa, model_groups, n, W)

            recs = {m: ([], [], []) for m in
                    ('qa', 'qa-score', 'cascade', 'random')}
            for gi, ((t_name, q), g) in enumerate(eval_groups):
                ms = set(g['model'].to_numpy())
                avail = sorted((ms & pool_c))
                srow = dict(zip(g['model'].to_numpy(), g['score'].to_numpy()))
                f_sc = srow[flag]
                # qa: nearest available in localized geometry
                d = np.linalg.norm(phi[gi][avail] - phi[gi][flag], axis=1)
                d[~okm[gi][avail]] = np.inf
                r_qa = avail[int(np.argmin(d))]
                # qa-score: argmax predicted score, gap ordering
                sv = shat[gi][avail].copy(); sv[~okm[gi][avail]] = -np.inf
                r_qs = avail[int(np.argmax(sv))]
                g_qs = float(shat[gi][flag] - shat[gi][r_qs])
                # cascade: best available by anchor task mean, gap ordering
                tm = tmean[t_name]
                av_means = [float(tm.get(a, -np.inf)) for a in avail]
                r_cs = avail[int(np.argmax(av_means))]
                g_cs = float(tm.get(flag, 0.0) - max(av_means))
                # random
                r_rd = avail[rng.integers(0, len(avail))]
                for name, r, conf in (('qa', r_qa, float(np.min(d))),
                                      ('qa-score', r_qs, g_qs),
                                      ('cascade', r_cs, g_cs),
                                      ('random', r_rd, float(rng.random()))):
                    recs[name][0].append(conf)
                    recs[name][1].append(srow[r])
                    recs[name][2].append(f_sc)
            for name, (conf, sub_s, fl_s) in recs.items():
                conf, sub_s, fl_s = map(np.asarray, (conf, sub_s, fl_s))
                curve = policy_curve(conf, sub_s, fl_s)
                for f, v in zip(FRACS, curve):
                    out.append((pool_name, seed, name, float(f), v,
                                float(fl_s.mean())))
    return (pd.DataFrame(out, columns=['pool', 'seed', 'method',
                                       'offload_frac', 'policy_score',
                                       'flagship_score']),
            models[flag], {models[i]: sizes[i] for i in sorted(pool)})


def run_combined_cost(seeds=5, pools=(('le13b', 13.0), ('le40b', 40.0)),
                      n_eval=400):
    """Cost-aware offloading on the COMBINED unlabeled pool. One shared
    router (joint geometry, no task/suite labels); per-suite flagships (the
    expensive tier serving that traffic) since the suites share no queries.
    Candidates per eval query = its responders within the size-capped pool.
    Orderings are score-denominated (comparable across suites)."""
    from .run_combined import load_combined
    from .run_eee import batched_localized_stats
    X, Qu, rows, n, names = load_combined()
    sizes = [parse_params(nm.split(':', 1)[1]) for nm in names]
    suite_of_model = [nm.split(':', 1)[0] for nm in names]

    out = []
    for pool_name, cap in pools:
        pool = {i for i in range(n) if sizes[i] is not None and sizes[i] <= cap}
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            gmean = rows.groupby('model')['score'].mean()
            flags = {}
            for su in ('helm', 'eee'):
                idx = [i for i in range(n) if suite_of_model[i] == su]
                flags[su] = int(gmean.loc[idx].idxmax())
            pool_c = pool - set(flags.values())

            resp = rows.groupby('query')['model'].agg(set)
            suite_q = rows.drop_duplicates('query').set_index('query')['suite']
            elig = [q for q, ms in resp.items()
                    if flags[suite_q[q]] in ms and len(ms & pool_c) >= 1]
            take = rng.choice(len(elig), size=min(n_eval, len(elig)),
                              replace=False)
            eval_q = {elig[i] for i in take}
            is_eval = rows['query'].isin(eval_q).to_numpy()
            anchor_m = ~is_eval
            assert not rows['query'][anchor_m].isin(eval_q).any()

            Xa = X[anchor_m]
            sa = rows['score'].to_numpy()[anchor_m]
            model_a = rows['model'].to_numpy()[anchor_m]
            task_a = rows['task'].to_numpy()[anchor_m]
            ua = Qu[rows['code'].to_numpy()[anchor_m]]
            model_groups = [np.flatnonzero(model_a == m) for m in range(n)]
            i = rng.integers(0, len(ua), 20000)
            k = rng.integers(0, len(ua), 20000)
            keep = i != k
            med = float(np.median(np.linalg.norm(
                ua[i[keep]] - ua[k[keep]], axis=1)))
            tmean = {tn: pd.Series(sa[task_a == tn]).groupby(
                        pd.Series(model_a[task_a == tn])).mean()
                     for tn in np.unique(task_a)}

            eval_groups = list(rows[is_eval].groupby('query'))
            Ue = np.stack([Qu[g['code'].iloc[0]] for _, g in eval_groups])
            D2 = ((Ue[:, None, :] - ua[None, :, :]) ** 2).sum(-1)
            D2 = D2 - D2.min(axis=1, keepdims=True)
            W = np.exp(-D2 / (2.0 * (0.25 * med) ** 2)).astype(np.float32)
            phi, shat, okm = batched_localized_stats(Xa, sa, model_groups, n, W)
            # static (uniform-weight) behavioral geometry, and the anchor
            # global score mean (the "pick the leaderboard-best cheap model"
            # baseline -- the one score practitioners actually have)
            phi_st, _, ok_st = batched_localized_stats(
                Xa, sa, model_groups, n, np.ones((1, len(Xa)), np.float32))
            gmean_a = pd.Series(sa).groupby(pd.Series(model_a)).mean()

            recs = {m: ([], [], [], [], []) for m in
                    ('qa', 'qa-score', 'cascade', 'static', 'lead', 'random')}
            for gi, (q, g) in enumerate(eval_groups):
                suite = g['suite'].iloc[0]
                t_name = g['task'].iloc[0]
                flag = flags[suite]
                gm = g['model'].to_numpy()
                ms = set(gm)
                avail = sorted(ms & pool_c)
                srow = dict(zip(gm, g['score'].to_numpy()))
                f_sc = srow[flag]
                # realized behavioral deviation of each responder vs flagship
                Xq = X[g.index.to_numpy()]
                x_flag = Xq[list(gm).index(flag)]
                devrow = {m: float(np.linalg.norm(Xq[i] - x_flag))
                          for i, m in enumerate(gm)}
                d = np.linalg.norm(phi[gi][avail] - phi[gi][flag], axis=1)
                d[~okm[gi][avail]] = np.inf
                r_qa = avail[int(np.argmin(d))]
                sv = shat[gi][avail].copy()
                sv[~okm[gi][avail]] = -np.inf
                r_qs = avail[int(np.argmax(sv))]
                g_qs = float(shat[gi][flag] - shat[gi][r_qs])
                tm = tmean[t_name]
                av_means = [float(tm.get(a, -np.inf)) for a in avail]
                r_cs = avail[int(np.argmax(av_means))]
                g_cs = float(tm.get(flag, 0.0) - max(av_means))
                d_st = np.linalg.norm(phi_st[0][avail] - phi_st[0][flag],
                                      axis=1)
                d_st[~ok_st[0][avail]] = np.inf
                r_st = avail[int(np.argmin(d_st))]
                r_ld = avail[int(np.argmax(
                    [float(gmean_a.get(a, -np.inf)) for a in avail]))]
                r_rd = avail[rng.integers(0, len(avail))]
                for name, r, conf in (('qa', r_qa, float(np.min(d))),
                                      ('qa-score', r_qs, g_qs),
                                      ('cascade', r_cs, g_cs),
                                      ('static', r_st, float(rng.random())),
                                      ('lead', r_ld, float(rng.random())),
                                      ('random', r_rd, float(rng.random()))):
                    recs[name][0].append(conf)
                    recs[name][1].append(srow[r])
                    recs[name][2].append(f_sc)
                    recs[name][3].append(devrow[r])
                    recs[name][4].append(suite)
            for name, (conf, sub_s, fl_s, dev, sus) in recs.items():
                conf, sub_s, fl_s, dev = map(np.asarray,
                                             (conf, sub_s, fl_s, dev))
                curve = policy_curve(conf, sub_s, fl_s)
                dcurve = policy_curve(conf, dev, np.zeros_like(dev))
                for f, v, dv in zip(FRACS, curve, dcurve):
                    out.append((pool_name, seed, name, float(f), v,
                                float(fl_s.mean()), dv))
    return pd.DataFrame(out, columns=['pool', 'seed', 'method',
                                      'offload_frac', 'policy_score',
                                      'flagship_score', 'policy_dev'])


def run_scarcity(seeds=5, pools=(('le13b', 13.0), ('le40b', 40.0)),
                 coverages=(0.0, 0.01, 0.03, 0.1, 0.3, 1.0), n_eval=400):
    """Score-scarcity sweep on the combined pool: candidates' anchor rows are
    scored only at coverage c (the flagship's own history stays scored -- the
    operator serves and audits it). Real caches are raw responses; per-query
    scores are the expensive benchmark artifact. qa never sees a score, so it
    is flat in c; qa-score and cascade degrade toward it. c = 0 is the
    cold-start setting: a new candidate with responses but no evaluations.
    When a router has no scored anchors for any available candidate it
    abstains (random pick, ordered last)."""
    from .run_combined import load_combined
    from .run_eee import batched_localized_stats
    X, Qu, rows, n, names = load_combined()
    sizes = [parse_params(nm.split(':', 1)[1]) for nm in names]
    suite_of_model = [nm.split(':', 1)[0] for nm in names]

    out = []
    for pool_name, cap in pools:
        pool = {i for i in range(n) if sizes[i] is not None and sizes[i] <= cap}
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            gmean = rows.groupby('model')['score'].mean()
            flags = {}
            for su in ('helm', 'eee'):
                idx = [i for i in range(n) if suite_of_model[i] == su]
                flags[su] = int(gmean.loc[idx].idxmax())
            pool_c = pool - set(flags.values())

            resp = rows.groupby('query')['model'].agg(set)
            suite_q = rows.drop_duplicates('query').set_index('query')['suite']
            elig = [q for q, ms in resp.items()
                    if flags[suite_q[q]] in ms and len(ms & pool_c) >= 1]
            take = rng.choice(len(elig), size=min(n_eval, len(elig)),
                              replace=False)
            eval_q = {elig[i] for i in take}
            is_eval = rows['query'].isin(eval_q).to_numpy()
            anchor_m = ~is_eval
            assert not rows['query'][anchor_m].isin(eval_q).any()

            Xa = X[anchor_m]
            sa = rows['score'].to_numpy()[anchor_m]
            model_a = rows['model'].to_numpy()[anchor_m]
            task_a = rows['task'].to_numpy()[anchor_m]
            ua = Qu[rows['code'].to_numpy()[anchor_m]]
            model_groups = [np.flatnonzero(model_a == m) for m in range(n)]
            i = rng.integers(0, len(ua), 20000)
            k = rng.integers(0, len(ua), 20000)
            keep = i != k
            med = float(np.median(np.linalg.norm(
                ua[i[keep]] - ua[k[keep]], axis=1)))

            eval_groups = list(rows[is_eval].groupby('query'))
            Ue = np.stack([Qu[g['code'].iloc[0]] for _, g in eval_groups])
            D2 = ((Ue[:, None, :] - ua[None, :, :]) ** 2).sum(-1)
            D2 = D2 - D2.min(axis=1, keepdims=True)
            W = np.exp(-D2 / (2.0 * (0.25 * med) ** 2)).astype(np.float32)
            phi, _, okm = batched_localized_stats(Xa, sa, model_groups, n, W)

            flag_set = set(flags.values())
            for cov in coverages:
                # per-candidate scored subsample; flagship history fully scored
                groups_c = []
                for m, idx_m in enumerate(model_groups):
                    if m in flag_set:
                        groups_c.append(idx_m)
                    else:
                        k_m = int(round(cov * len(idx_m)))
                        groups_c.append(rng.choice(idx_m, size=k_m,
                                                   replace=False)
                                        if k_m else idx_m[:0])
                _, shat_c, ok_c = batched_localized_stats(
                    Xa, sa, groups_c, n, W)
                sc_mask = np.zeros(len(Xa), dtype=bool)
                sc_mask[np.concatenate([g for g in groups_c
                                        if len(g)]).astype(int)] = True
                tmean_c = {tn: pd.Series(sa[sc_mask & (task_a == tn)]).groupby(
                              pd.Series(model_a[sc_mask & (task_a == tn)])
                           ).mean() for tn in np.unique(task_a)}

                recs = {m: ([], [], []) for m in
                        ('qa', 'qa-score', 'cascade', 'random')}
                for gi, (q, g) in enumerate(eval_groups):
                    suite = g['suite'].iloc[0]
                    t_name = g['task'].iloc[0]
                    flag = flags[suite]
                    gm = g['model'].to_numpy()
                    avail = sorted(set(gm) & pool_c)
                    srow = dict(zip(gm, g['score'].to_numpy()))
                    f_sc = srow[flag]
                    d = np.linalg.norm(phi[gi][avail] - phi[gi][flag], axis=1)
                    d[~okm[gi][avail]] = np.inf
                    r_qa = avail[int(np.argmin(d))]
                    sv = shat_c[gi][avail].copy()
                    sv[~ok_c[gi][avail]] = -np.inf
                    if np.isfinite(sv).any():
                        r_qs = avail[int(np.argmax(sv))]
                        g_qs = float(shat_c[gi][flag] - sv.max())
                    else:                        # abstain: keep on flagship
                        r_qs, g_qs = avail[rng.integers(0, len(avail))], np.inf
                    tm = tmean_c[t_name]
                    av = [float(tm.get(a, -np.inf)) for a in avail]
                    if np.isfinite(av).any():
                        r_cs = avail[int(np.argmax(av))]
                        g_cs = float(tm.get(flag, 0.0) - max(av))
                    else:
                        r_cs, g_cs = avail[rng.integers(0, len(avail))], np.inf
                    r_rd = avail[rng.integers(0, len(avail))]
                    for name, r, conf in (('qa', r_qa, float(np.min(d))),
                                          ('qa-score', r_qs, g_qs),
                                          ('cascade', r_cs, g_cs),
                                          ('random', r_rd,
                                           float(rng.random()))):
                        recs[name][0].append(conf)
                        recs[name][1].append(srow[r])
                        recs[name][2].append(f_sc)
                for name, (conf, sub_s, fl_s) in recs.items():
                    conf, sub_s, fl_s = map(np.asarray, (conf, sub_s, fl_s))
                    curve = policy_curve(conf, sub_s, fl_s)
                    for f, v in zip(FRACS, curve):
                        out.append((pool_name, seed, float(cov), name,
                                    float(f), v, float(fl_s.mean())))
    return pd.DataFrame(out, columns=['pool', 'seed', 'coverage', 'method',
                                      'offload_frac', 'policy_score',
                                      'flagship_score'])


def run_tolerance(seeds=5, pools=(('le13b', 13.0), ('le40b', 40.0)),
                  n_eval=400, n_cal=400):
    """Tolerance-gated offloading: serve the substitute only when PREDICTED
    behavioral deviation is within a tolerance eps. The qa confidence
    (localized flagship-candidate distance) is mapped to expected realized
    deviation by isotonic regression on a disjoint calibration split,
    held out of the geometry exactly like eval queries (leak-free). The
    contract is per-query; baselines have no per-query error signal, so at a
    matched offload fraction they serve an arbitrary subset. Returns
    per-decision records; policy curves are computed by the caller."""
    from sklearn.isotonic import IsotonicRegression  # noqa: F401 (caller)
    from .run_combined import load_combined
    from .run_eee import batched_localized_stats
    X, Qu, rows, n, names = load_combined()
    sizes = [parse_params(nm.split(':', 1)[1]) for nm in names]
    suite_of_model = [nm.split(':', 1)[0] for nm in names]

    out = []
    for pool_name, cap in pools:
        pool = {i for i in range(n) if sizes[i] is not None and sizes[i] <= cap}
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            gmean = rows.groupby('model')['score'].mean()
            flags = {}
            for su in ('helm', 'eee'):
                idx = [i for i in range(n) if suite_of_model[i] == su]
                flags[su] = int(gmean.loc[idx].idxmax())
            pool_c = pool - set(flags.values())

            resp = rows.groupby('query')['model'].agg(set)
            suite_q = rows.drop_duplicates('query').set_index('query')['suite']
            elig = [q for q, ms in resp.items()
                    if flags[suite_q[q]] in ms and len(ms & pool_c) >= 1]
            k_hold = min(n_eval + n_cal, len(elig))
            take = rng.choice(len(elig), size=k_hold, replace=False)
            eval_q = {elig[i] for i in take[:k_hold // 2]}
            cal_q = {elig[i] for i in take[k_hold // 2:]}
            hold_q = eval_q | cal_q
            is_hold = rows['query'].isin(hold_q).to_numpy()
            anchor_m = ~is_hold
            assert not rows['query'][anchor_m].isin(hold_q).any()

            Xa = X[anchor_m]
            sa = rows['score'].to_numpy()[anchor_m]
            model_a = rows['model'].to_numpy()[anchor_m]
            ua = Qu[rows['code'].to_numpy()[anchor_m]]
            model_groups = [np.flatnonzero(model_a == m) for m in range(n)]
            i = rng.integers(0, len(ua), 20000)
            k = rng.integers(0, len(ua), 20000)
            keep = i != k
            med = float(np.median(np.linalg.norm(
                ua[i[keep]] - ua[k[keep]], axis=1)))

            hold_groups = list(rows[is_hold].groupby('query'))
            Ue = np.stack([Qu[g['code'].iloc[0]] for _, g in hold_groups])
            D2 = ((Ue[:, None, :] - ua[None, :, :]) ** 2).sum(-1)
            D2 = D2 - D2.min(axis=1, keepdims=True)
            W = np.exp(-D2 / (2.0 * (0.25 * med) ** 2)).astype(np.float32)
            phi, _, okm = batched_localized_stats(Xa, sa, model_groups, n, W)
            phi_st, _, ok_st = batched_localized_stats(
                Xa, sa, model_groups, n, np.ones((1, len(Xa)), np.float32))
            gmean_a = pd.Series(sa).groupby(pd.Series(model_a)).mean()

            for gi, (q, g) in enumerate(hold_groups):
                suite = g['suite'].iloc[0]
                flag = flags[suite]
                gm = g['model'].to_numpy()
                avail = sorted(set(gm) & pool_c)
                srow = dict(zip(gm, g['score'].to_numpy()))
                Xq = X[g.index.to_numpy()]
                x_flag = Xq[list(gm).index(flag)]
                devrow = {m: float(np.linalg.norm(Xq[i] - x_flag))
                          for i, m in enumerate(gm)}
                d = np.linalg.norm(phi[gi][avail] - phi[gi][flag], axis=1)
                d[~okm[gi][avail]] = np.inf
                r_qa = avail[int(np.argmin(d))]
                d_st = np.linalg.norm(phi_st[0][avail] - phi_st[0][flag],
                                      axis=1)
                d_st[~ok_st[0][avail]] = np.inf
                r_st = avail[int(np.argmin(d_st))]
                r_ld = avail[int(np.argmax(
                    [float(gmean_a.get(a, -np.inf)) for a in avail]))]
                r_rd = avail[rng.integers(0, len(avail))]
                out.append((pool_name, seed,
                            'eval' if q in eval_q else 'cal', suite, q,
                            float(np.min(d)),
                            devrow[r_qa], srow[r_qa],
                            devrow[r_ld], srow[r_ld],
                            devrow[r_st], srow[r_st],
                            devrow[r_rd], srow[r_rd],
                            srow[flag]))
    return pd.DataFrame(out, columns=[
        'pool', 'seed', 'split', 'suite', 'query', 'conf',
        'dev_qa', 's_qa', 'dev_ld', 's_ld', 'dev_st', 's_st',
        'dev_rd', 's_rd', 's_flag'])


def tolerance_curves(df, eps_grid=(0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)):
    """Per (pool, seed): calibrate conf -> expected deviation on the cal
    split, gate eval offloading at each eps. Baselines offload a random
    subset of matched size (expectations computed in closed form)."""
    from sklearn.isotonic import IsotonicRegression
    rows = []
    for (pool, seed), sub in df.groupby(['pool', 'seed']):
        cal = sub[sub.split == 'cal']
        ev = sub[sub.split == 'eval'].reset_index(drop=True)
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(cal['conf'].to_numpy(), cal['dev_qa'].to_numpy())
        pred = iso.predict(ev['conf'].to_numpy())
        flag = ev['s_flag'].to_numpy()
        for eps in eps_grid:
            off = pred <= eps
            f = float(off.mean())
            rec = {'pool': pool, 'seed': seed, 'eps': eps, 'frac': f}
            dv, sc = ev['dev_qa'].to_numpy(), ev['s_qa'].to_numpy()
            rec['qa_dev'] = float(dv[off].mean()) if off.any() else 0.0
            rec['qa_viol'] = float((dv[off] > eps).mean()) if off.any() else 0.0
            rec['qa_ret'] = float(np.where(off, sc, flag).mean()
                                  / flag.mean())
            for b in ('ld', 'st', 'rd'):
                dv, sc = ev[f'dev_{b}'].to_numpy(), ev[f's_{b}'].to_numpy()
                # random subset of matched size: per-query offload prob = f
                rec[f'{b}_dev'] = float(dv.mean())
                rec[f'{b}_viol'] = float((dv > eps).mean())
                rec[f'{b}_ret'] = float((f * sc + (1 - f) * flag).mean()
                                        / flag.mean())
            rows.append(rec)
    return pd.DataFrame(rows)


def conformal_gate(df, eps_grid=(0.1, 0.2, 0.3, 0.5),
                   alphas=(0.05, 0.10, 0.20), min_prefix=20):
    """Conformal-style gate on run_tolerance decisions: per (pool, seed,
    eps, alpha), the largest confidence cutoff whose calibration-split
    violation rate P(dev > eps | conf <= t) stays <= alpha (requiring a
    minimally stable prefix; abstain otherwise). Baselines cannot target
    (eps, alpha) at all -- their violation rate is a fixed population
    property of the pick, reported at matched volume."""
    rows = []
    for (pool, seed), sub in df.groupby(['pool', 'seed']):
        cal = sub[sub.split == 'cal'].sort_values('conf')
        ev = sub[sub.split == 'eval']
        conf_sorted = cal['conf'].to_numpy()
        for eps in eps_grid:
            bad = (cal['dev_qa'].to_numpy() > eps).astype(float)
            cum_viol = np.cumsum(bad) / np.arange(1, len(bad) + 1)
            for alpha in alphas:
                ok = np.flatnonzero(cum_viol <= alpha)
                ok = ok[ok >= min_prefix - 1]
                t = conf_sorted[ok.max()] if len(ok) else -np.inf
                off = ev['conf'].to_numpy() <= t
                dv, sc = ev['dev_qa'].to_numpy(), ev['s_qa'].to_numpy()
                fl = ev['s_flag'].to_numpy()
                rows.append({
                    'pool': pool, 'seed': seed, 'eps': eps, 'alpha': alpha,
                    'frac': float(off.mean()),
                    'viol': (float((dv[off] > eps).mean())
                             if off.any() else 0.0),
                    'ret': float(np.where(off, sc, fl).mean() / fl.mean()),
                    'ld_viol': float((ev['dev_ld'].to_numpy() > eps).mean()),
                    'rd_viol': float((ev['dev_rd'].to_numpy() > eps).mean()),
                })
    return pd.DataFrame(rows)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    cdf = run_combined_cost()
    cdf.to_parquet(os.path.join(RESULTS, 'cost_routing_combined.parquet'))
    cdf = cdf.assign(retention=cdf.policy_score / cdf.flagship_score)
    for pool, sub in cdf.groupby('pool'):
        piv = sub.groupby(['method', 'offload_frac'])['retention'].mean().unstack()
        piv.to_csv(os.path.join(RESULTS, f'cost_routing_combined_{pool}.csv'))
        print(f'\n== COMBINED pool {pool}: retention (%) ==')
        print((100 * piv).round(2).to_string())
        print('max offload at >= 99% retention:')
        for meth in piv.index:
            okm = piv.loc[meth][piv.loc[meth] >= 0.99]
            print(f'  {meth:9s} {(okm.index.max() if len(okm) else 0.0):.0%}')

    eee_df, eee_flag, eee_pool = run_eee_cost()
    eee_df.to_parquet(os.path.join(RESULTS, 'cost_routing_eee.parquet'))
    eee_df = eee_df.assign(retention=eee_df.policy_score / eee_df.flagship_score)
    print(f'EEE flagship: {eee_flag}; example pool: {eee_pool}')
    for pool, sub in eee_df.groupby('pool'):
        piv = sub.groupby(['method', 'offload_frac'])['retention'].mean().unstack()
        piv.to_csv(os.path.join(RESULTS, f'cost_routing_eee_{pool}.csv'))
        print(f'\n== EEE pool {pool}: retention (%) ==')
        print((100 * piv).round(2).to_string())

    df = run_helm()
    df.to_parquet(os.path.join(RESULTS, 'cost_routing_helm.parquet'))
    df = df.assign(retention=df.policy_score / df.flagship_score)
    for pool, sub in df.groupby('pool'):
        piv = sub.groupby(['method', 'offload_frac'])['retention'].mean().unstack()
        piv.to_csv(os.path.join(RESULTS, f'cost_routing_helm_{pool}.csv'))
        print(f'\n== pool {pool}: score retention (%) by offload fraction ==')
        print((100 * piv).round(2).to_string())
        print('max offload at >= 99% retention, and bill reduction at')
        print('flagship/substitute price ratios R in {10, 25, 50}:')
        for meth in piv.index:
            okm = piv.loc[meth][piv.loc[meth] >= 0.99]
            f = float(okm.index.max()) if len(okm) else 0.0
            save = {R: f * (1 - 1 / R) for R in (10, 25, 50)}
            print(f'  {meth:9s} offload {f:4.0%} | save '
                  + ' '.join(f'{save[R]:4.0%}@{R}x' for R in (10, 25, 50)))


if __name__ == '__main__':
    main()
