"""Cutoff-free evaluation: rank all candidate groups by p-value and report
recall of known coordinated agents down the ranking.

    python rank_eval.py --res data/derived/platform_jan31_feb17/res_<tag>.pkl [...more]

Known coordinated agents (never used by the pipeline):
  siblings  agents whose names differ only by a 1-2 digit suffix
            (FloClaw9..FloClaw14, xmolt02..06, ...), families of >= 3 system
            agents, generic stems (agent, claw, clawd, bot) excluded
  operator  agents claimed by the same X account (hashed operator key)
  ring      the planted ring, when the run had one

A known pair counts as recovered at rank k if both agents sit together in
one of the top-k groups. Recall is plotted against the number of distinct
agents covered by the top-k groups, so large groups get no free credit.

Only groups of at most --max-size agents are ranked (default 100): z-based
p-values grow with group size, and very large weakly aligned groups would
otherwise fill the top of the list without isolating any specific group.

Rankings (normal-approximation p-values from z-scores, which avoid the
permutation p-value floor and its ties):
  comove    z_C of the group (co-movement of displacements)
  intensity mean pairwise cosine of displacements (size-free tightness)
  coact     z of mean pairwise co-activity vs activity-matched random sets
  combined  Stouffer (z_C + z_coact) / sqrt 2 (dilutes whichever signal is absent)
  either    min of the two p-values (max of z_C, z_coact): a group counts as
            coordinated if it co-moves in content OR acts together in time
  either_rank  groups interleaved by the better of their intensity rank and
            co-activity rank
  reply     z of mean pairwise reply strength (post-t0 replies between the two
            agents, scaled by sqrt of their reply strengths) vs activity-matched
            random sets
  any3_rank groups interleaved by the best of their intensity, co-activity and
            reply ranks (content, timing, reply graph)
  random    the same groups in random order (mean of 200 shuffles)
"""
import argparse
import json
import os
import pickle
import re
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm

from pipeline import candidates as cands
from pipeline.gate import _pair_sums
from pipeline.ingest_moltnet import ORIGIN

HERE = os.path.dirname(os.path.abspath(__file__))
from pipeline.paths import DERIVED, RESULTS  # noqa: E402
GENERIC = {'agent', 'claw', 'clawd', 'bot', 'ai', 'molt'}


def sibling_families(names):
    fam = {}
    for n in names:
        m = re.match(r'^(.*?[A-Za-z])[_\-]?(\d{1,2})$', n)
        if m and len(m.group(1)) >= 4 and m.group(1).lower().strip('_-') not in GENERIC:
            fam.setdefault(m.group(1).lower().strip('_-'), []).append(n)
    return {k: v for k, v in fam.items() if len(v) >= 3}


def coact_z(W, S, bins, pools, rng, B=1999, budget=5e8):
    """z of the mean pairwise weight of W within S vs activity-matched random sets."""
    S = np.asarray(S)
    s = len(S)
    obs = W[np.ix_(S, S)].sum() / (s * (s - 1))
    Bc = int(max(199, min(B, budget // s ** 2)))
    cols = [pools[b][rng.integers(0, len(pools[b]), (Bc, int((bins[S] == b).sum())))]
            for b in range(4) if (bins[S] == b).any()]
    R = np.concatenate(cols, 1)
    null = _pair_sums(W, R) / (s * (s - 1))
    return (obs - null.mean()) / max(null.std(), 1e-12)


def reply_weights(G, n):
    """Dense symmetric reply-strength matrix: count / sqrt(strength_i * strength_j)."""
    W = np.zeros((n, n), np.float32)
    if G.number_of_edges():
        e = np.array([(u, v, d['weight']) for u, v, d in G.edges(data=True)])
        u, v, w = e[:, 0].astype(int), e[:, 1].astype(int), e[:, 2]
        st = np.bincount(u, w, n) + np.bincount(v, w, n)
        W[u, v] = W[v, u] = w / np.sqrt(st[u] * st[v])
    return W


def recall_curve(order, sets, pairs, n_cover_max):
    """Pair recall and agents covered after each of the top-k groups."""
    covered, found = set(), set()
    pair_index = {}
    for p in pairs:
        pair_index.setdefault(p[0], []).append(p)
    cov, rec = [], []
    for j in order:
        S = sets[j]
        covered |= S
        for a in S:
            for p in pair_index.get(a, ()):
                if p not in found and p[1] in S:
                    found.add(p)
        cov.append(len(covered))
        rec.append(len(found) / max(len(pairs), 1))
        if len(covered) >= n_cover_max:
            break
    return np.array(cov), np.array(rec)


def at(cov, rec, c):
    i = np.searchsorted(cov, c, side='right') - 1
    return float(rec[i]) if i >= 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--res', nargs='+', required=True)
    ap.add_argument('--scope', default='platform_jan31_feb17')
    ap.add_argument('--out', default='rank_eval')
    ap.add_argument('--max-size', type=int, default=100)
    a = ap.parse_args()
    d = os.path.join(DERIVED, a.scope)
    meta = json.load(open(os.path.join(d, 'meta.json')))
    lo = meta['window_hours'][0]
    acts0 = pd.read_parquet(os.path.join(d, 'actions.parquet'))
    acts0 = acts0.assign(t=acts0.t - lo)
    agents = pd.read_parquet(os.path.join(d, 'agents.parquet'))
    rows_out, curves, saved = [], {}, {}
    for path in a.res:
        o = pickle.load(open(path, 'rb'))
        res, truth = o['res'], o['truth']
        system = res['system']
        idx = {x: i for i, x in enumerate(system)}
        el = res['eligible']
        acts = acts0
        if truth.get('ring'):  # rebuild the planted burst comments for co-activity
            from run_platform import hours
            from pipeline.plant import plant, CAMPAIGN
            from pipeline import embed
            E = np.load(os.path.join(d, 'emb_actions.npy'))
            ids = json.load(open(os.path.join(d, 'emb_actions_ids.json')))
            ctx = pd.read_parquet(os.path.join(d, 'contexts.parquet'))
            ctx = ctx.assign(t=ctx.t - lo)
            data = dict(actions=acts0, contexts=ctx, emb=dict(action_ids=ids, E=E),
                        score_events=pd.DataFrame(columns=['agent_id', 't', 'value']),
                        meta=dict(T=meta['window_hours'][1] - lo), system=system)
            seed = int(re.search(r'_s(\d+)', path).group(1))
            planted, tr2 = plant(data, embed.embed([CAMPAIGN])[0], m=truth['m'],
                                 alpha=truth['alpha'], t_star=truth['t_star'], seed=seed)
            assert set(tr2['ring']) == set(truth['ring'])
            acts = planted['actions']
        _, W = cands.coactivity(acts, idx, res['t0'], return_weights=True)

        # known coordinated pairs
        labels = {}
        for k, v in sibling_families([x for x in system if el[idx[x]]]).items():
            labels[f'siblings:{k}'] = [idx[x] for x in v]
        op = agents[agents.name.isin(idx) & (agents.operator_key != '')]
        for k, g in op.groupby('operator_key').name:
            if len(g) > 1:
                labels[f'operator:{k[:6]}'] = [idx[x] for x in g]
        if truth.get('ring'):
            labels['ring'] = [idx[x] for x in truth['ring']]
        pairs_by = {lab: {(min(i, j), max(i, j)) for i in m for j in m if i < j}
                    for lab, m in labels.items()}

        rows = [r for r in res['scan']['rows'] if r['size'] <= a.max_size]
        sets = [set(r['members']) for r in rows]
        bins = res['scan']['activity_bins']
        pools = [np.flatnonzero(el & (bins == b)) for b in range(4)]
        rng = np.random.default_rng(0)
        zc = np.array([r['zC'] for r in rows])
        zco = np.array([coact_z(W, r['members'], bins, pools, rng) for r in rows])
        Wr = reply_weights(res['G_int'], len(system))
        zrep = np.array([coact_z(Wr, r['members'], bins, pools, rng) for r in rows])
        inten = np.array([r['comove'] for r in rows])
        rk = lambda x: np.argsort(np.argsort(-x, kind='stable'), kind='stable')
        scores = {'comove': zc, 'intensity': inten, 'coact': zco,
                  'combined': (zc + zco) / np.sqrt(2), 'either': np.maximum(zc, zco),
                  'either_rank': -np.minimum(rk(inten), rk(zco)) - 1e-6 * np.maximum(rk(inten), rk(zco)),
                  'reply': zrep}
        r3 = np.stack([rk(inten), rk(zco), rk(zrep)])
        scores['any3_rank'] = -r3.min(0) - 1e-6 * np.sort(r3, 0)[1]
        saved[os.path.basename(path)] = dict(members=[r['members'] for r in rows], sizes=[r['size'] for r in rows],
                                             sources=[r['source'] for r in rows], labels=labels,
                                             **{k: v for k, v in scores.items()})
        tag = os.path.basename(path)[4:-4]
        n_el = int(el.sum())
        grid = [25, 50, 100, 250, 500, 1000]
        kgrid = [5, 10, 25, 50, 100]
        for group, pairs in [('siblings', set().union(*[p for k, p in pairs_by.items()
                                                       if k.startswith('siblings')] or [set()])),
                             ('ring', pairs_by.get('ring', set()))]:
            if not pairs:
                continue
            for meth, z in scores.items():
                order = np.argsort(-z, kind='stable')
                cov, rec = recall_curve(order, sets, pairs, n_el)
                curves[(tag, group, meth)] = (cov, rec)
                rows_out.append(dict(run=tag, labels=group, n_pairs=len(pairs), ranking=meth,
                                     **{f'rec@{c}ag': at(cov, rec, c) for c in grid},
                                     **{f'rec@{k}gr': float(rec[min(k, len(rec)) - 1]) for k in kgrid}))
            rr, rk = [], []
            for s_ in range(200):
                order = np.random.default_rng(s_).permutation(len(rows))
                cov, rec = recall_curve(order, sets, pairs, n_el)
                rr.append([at(cov, rec, c) for c in grid])
                rk.append([rec[min(k, len(rec)) - 1] for k in kgrid])
            rows_out.append(dict(run=tag, labels=group, n_pairs=len(pairs), ranking='random',
                                 **{f'rec@{c}ag': float(v) for c, v in zip(grid, np.mean(rr, 0))},
                                 **{f'rec@{k}gr': float(v) for k, v in zip(kgrid, np.mean(rk, 0))}))
        print(f'[{tag}] {len(rows)} groups of <= {a.max_size} agents ranked; labels: '
              + ', '.join(f'{k} ({len(v)})' for k, v in labels.items()), flush=True)

    df = pd.DataFrame(rows_out)
    df.to_csv(os.path.join(RESULTS, f'{a.out}.csv'), index=False)
    pickle.dump(dict(curves=curves, scores=saved), open(os.path.join(d, f'{a.out}_curves.pkl'), 'wb'))
    print(df.to_string(index=False, float_format=lambda x: f'{x:.2f}'))


if __name__ == '__main__':
    main()
