"""Moltbook: member-level investigation of the groups found by real_tstep.py.

    python real_investigate.py [--tag 01-30_02-09_T8_ms6] [--samples 4] [--null-sets 500]

For every peeling round's membership set (the subnetwork, S_hat), the strength of
each kind of evidence, against random agents of the same cohort:
  co-movement  the method's own statistic: the pair majority z of per-step rank scores
               (tstep.pair_scores), and each member's majority z against the rest of the set
               (tstep.member_z)
  content      cosine of the two agents' mean positions over the window (what they write
               about, after conditioning on the threads they answer), and template overlap
               (Jaccard of the sets of texts with digits and the agent's own name masked)
  timing       correlation of hourly action counts
  replies      direct replies between the two agents (either direction)
  threads      threads both agents posted or commented in
Per set: the median over member pairs of each measure, and its p-value against the same median
over random sets of the same size drawn from the cohort (--null-sets draws). Per pair and per
member: the raw values. Per member also: actions, median gap between actions, mint share, most
common template, and the number of steps with an observed increment.

Writes data/results/real_investigate_<tag>.json. --samples prints that many texts per member
(terminal only). No operator information is used.
"""
import argparse
import json
import os
import re
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd

from pipeline import tstep
from pipeline.ingest_moltnet import RAW
from pipeline.paths import DERIVED, RESULTS
from real_tstep import load_window

MINT = re.compile(r'"p"\s*:\s*"mbc-20".*?"tick"\s*:\s*"([^"]+)"', re.S)
MEASURES = ['comove_z', 'content_cos', 'template_jaccard', 'hourly_corr', 'replies', 'shared_threads']


def template(text, name):
    t = text.replace(name, '<self>')
    t = re.sub(r'\d+', '#', t)
    return re.sub(r'\s+', ' ', t).strip()[:120]


class Evidence:
    """Pairwise evidence for any two cohort agents (indices into the cohort)."""

    def __init__(self, Zp, pos, acts, names, cauth, cpar):
        N = len(names)
        self.Zp = Zp
        Z = pos['Z']
        cnt = (pos['nA'] + pos['nB'])[:, :, None]
        m = (Z * (cnt > 0)).sum(1) / np.maximum((cnt > 0).sum(1), 1)    # mean position over observed steps
        self.C = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)
        ix = pd.Series(np.arange(N), index=names)
        a = acts[acts.agent.isin(ix.index)]
        ai = ix[a.agent].values
        self.tpl = [set() for _ in range(N)]
        self.thr = [set() for _ in range(N)]
        for i, tx, nm, th in zip(ai, a.text.values, a.agent.values, a.thread.values):
            self.tpl[i].add(template(tx, nm)); self.thr[i].add(th)
        h = (a.t.values - a.t.values.min()).astype(int)
        H = np.zeros((N, h.max() + 1)); np.add.at(H, (ai, h), 1)
        H -= H.mean(1, keepdims=True)
        self.H = H / np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-12)
        pauth = dict(zip(a.action_id[a.kind == 'post'], a.agent[a.kind == 'post']))
        rep = Counter()
        for i, k, aid, th, nm in zip(ai, a.kind.values, a.action_id.values, a.thread.values, a.agent.values):
            if k != 'comment':
                continue
            par = cpar.get(aid)
            to = cauth.get(par) if isinstance(par, str) and par else pauth.get(th)
            if to is not None and to != nm and to in ix.index:
                rep[frozenset((i, int(ix[to])))] += 1
        self.rep = rep

    def pair(self, i, j):
        ti, tj = self.tpl[i], self.tpl[j]
        return dict(comove_z=float(self.Zp[i, j]), content_cos=float(self.C[i] @ self.C[j]),
                    template_jaccard=len(ti & tj) / max(len(ti | tj), 1),
                    hourly_corr=float(self.H[i] @ self.H[j]), replies=self.rep.get(frozenset((i, j)), 0),
                    shared_threads=len(self.thr[i] & self.thr[j]))

    def set_medians(self, S):
        P = [self.pair(i, j) for i, j in combinations(S, 2)]
        return {k: float(np.median([p[k] for p in P])) for k in MEASURES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='01-30_02-09_T8_ms6')
    ap.add_argument('--samples', type=int, default=0)
    ap.add_argument('--null-sets', type=int, default=500)
    a = ap.parse_args()
    summ = json.load(open(os.path.join(RESULTS, f'real_tstep_{a.tag}.json')))
    d = os.path.join(DERIVED, 'real_tstep')
    R = np.load(os.path.join(d, f'{a.tag}_R.npy')).astype(np.float32)
    rows = pd.read_parquet(os.path.join(d, f'{a.tag}_rows.parquet'), columns=['agent', 'agent_ix', 'step'])
    names = np.array(sorted(rows.agent.unique()))
    N, T = len(names), summ['T']
    idx = {n: i for i, n in enumerate(names)}

    # the scan's residual increments and rank scores, rebuilt exactly as real_tstep.py builds them
    pos = tstep.positions(R, rows.agent_ix.values, rows.step.values, N, T, seed=summ['seed'])
    w = summ.get('weight')
    mm = summ.get('min_move')
    D = tstep.increments(pos, weight=None if w in (None, 'none') else w,
                         min_move=mm if mm is not None and mm >= 0 else None)
    alpha = summ.get('two_signed_alpha')
    D, kf = tstep.remove_factors(D, two_signed_alpha=alpha if alpha is not None and alpha >= 0 else None)
    Q = tstep.step_scores(D)
    Zp = tstep.pair_scores(Q)
    obs = np.linalg.norm(D, axis=2) > 0

    acts, _ = load_window(*summ['window'])
    com = pd.read_parquet(os.path.join(RAW, 'comments.parquet'), columns=['id', 'parent_id', 'author_name'])
    cauth = dict(zip(com.id, com.author_name)); cpar = dict(zip(com.id, com.parent_id)); com = None
    ev = Evidence(Zp, pos, acts, names, cauth, cpar)
    rng = np.random.default_rng(summ['seed'])
    rand_pairs = [ev.pair(*rng.choice(N, 2, replace=False)) for _ in range(20000)]
    base = {k: float(np.median([p[k] for p in rand_pairs])) for k in MEASURES}
    null_cache = {}

    def null_medians(s):
        if s not in null_cache:
            null_cache[s] = pd.DataFrame([ev.set_medians(rng.choice(N, s, replace=False))
                                          for _ in range(a.null_sets)])
        return null_cache[s]

    sets = [dict(kind='round', index=gi, p=g['p'], members=g['members']) for gi, g in enumerate(summ['groups'])
            if g.get('subnetwork', len(g['members']) >= 2) and g['p'] <= 0.05]
    out = dict(tag=a.tag, n_factors=kf, random_pair_median=base, sets=[])
    print('random cohort pairs, median:', {k: round(v, 3) for k, v in base.items()})
    for st in sets:
        M = st['members']
        if len(M) < 2:
            continue
        m_ix = np.array([idx[x] for x in M])
        med = ev.set_medians(m_ix)
        nm = null_medians(len(M))
        pv = {k: float((1 + (nm[k] >= med[k]).sum()) / (len(nm) + 1)) for k in MEASURES}
        tz, _ = tstep.member_z(Q, m_ix)
        A = acts[acts.agent.isin(M)].copy()
        A['tpl'] = [template(t, n) for t, n in zip(A.text, A.agent)]
        A['tick'] = A.text.map(lambda s: (MINT.search(s) or [None, None])[1])
        members = []
        for n, i in zip(M, m_ix):
            x = A[A.agent == n]
            tpl = x.tpl.value_counts()
            members.append(dict(
                agent=n, member_z=round(float(tz[i]), 2), steps_observed=int(obs[i].sum()),
                actions=len(x), posts=int((x.kind == 'post').sum()), comments=int((x.kind == 'comment').sum()),
                top_submolts={str(k): int(c) for k, c in x.submolt.value_counts().head(3).items()},
                mint_share=round(float(x.tick.notna().mean()), 3),
                top_template=tpl.index[0][:90] if len(tpl) else '',
                top_template_share=round(float(tpl.iloc[0] / len(x)), 3) if len(tpl) else 0.0,
                median_gap_min=round(float(np.median(np.diff(np.sort(x.t.values))) * 60), 1) if len(x) > 1 else None))
        pairs = [dict(a=M[u], b=M[v], **{k: (round(x, 3) if isinstance(x, float) else x)
                                         for k, x in ev.pair(m_ix[u], m_ix[v]).items()},
                      step_scores=[None if np.isnan(q) else round(float(q), 2) for q in Q[:, m_ix[u], m_ix[v]]])
                 for u, v in combinations(range(len(M)), 2)]
        rec = dict(kind=st['kind'], index=st['index'], p=st['p'], size=len(M), median=med,
                   p_vs_random_sets=pv, members=members, pairs=pairs)
        out['sets'].append(rec)

        hdr = f"{st['kind']} {st['index']}" + (f", p = {st['p']:.3f}" if st['p'] is not None else '')
        print(f'\n===== {hdr}, {len(M)} agents')
        print('  ' + '  '.join(f'{k} {med[k]:.2f} (p {pv[k]:.3f})' for k in MEASURES))
        for r in sorted(members, key=lambda r: -r['member_z']):
            print(f"  {r['agent'][:22]:22s} z {r['member_z']:5.2f} steps {r['steps_observed']}  n {r['actions']:5d} "
                  f"({r['posts']}p/{r['comments']}c)  gap {r['median_gap_min']}m  mint {r['mint_share']:.2f}  "
                  f"tpl ({r['top_template_share']:.2f}) {r['top_template'][:60]!r}")
        for p in sorted(pairs, key=lambda r: -r['comove_z'])[:8]:
            print(f"  pair {p['a'][:16]:16s} {p['b'][:16]:16s} comove z {p['comove_z']:5.2f}  content {p['content_cos']:.2f}  "
                  f"tpl J {p['template_jaccard']:.2f}  hourly r {p['hourly_corr']:.2f}  replies {p['replies']}  "
                  f"threads {p['shared_threads']}")
        if a.samples:
            for n in M:
                for t in A[A.agent == n].sample(min(a.samples, (A.agent == n).sum()), random_state=0).text:
                    print(f'     [{n}] {t[:200]!r}')
    with open(os.path.join(RESULTS, f'real_investigate_{a.tag}.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f'\nwrote data/results/real_investigate_{a.tag}.json')


if __name__ == '__main__':
    main()
