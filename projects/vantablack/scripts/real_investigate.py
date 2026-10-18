"""Moltbook: member-level investigation of the groups found by real_tstep.py.

    python real_investigate.py [--tag 01-30_02-18_T8] [--samples 4]

Per member: actions (posts / comments), top submolts, share of actions carrying an mbc-20
mint payload (and its tickers), the most common text template (digits and the agent's own
name masked), median gap between actions, account creation time, and the membership score
t_i against the rest of the group (sqrt(d_eff) <u_i, v_-i>, ~ N(0, 1) for an agent
independent of the group).
Per pair of members: increment cosine (the evidence used by the scan), template overlap
(Jaccard of masked texts), correlation of hourly action counts, shared threads, and replies.
Per group: each step's contribution to the mean within-group increment cosine.

Writes data/results/real_investigate_<tag>.json. --samples prints that many texts per member
(terminal only). No operator information is used.
"""
import argparse
import json
import os
import re
from itertools import combinations

import numpy as np
import pandas as pd

from pipeline import tstep
from pipeline.ingest_moltnet import RAW
from pipeline.paths import DERIVED, RESULTS
from real_tstep import load_window

MINT = re.compile(r'"p"\s*:\s*"mbc-20".*?"tick"\s*:\s*"([^"]+)"', re.S)


def template(text, name):
    t = text.replace(name, '<self>')
    t = re.sub(r'\d+', '#', t)
    return re.sub(r'\s+', ' ', t).strip()[:120]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='01-30_02-18_T8')
    ap.add_argument('--samples', type=int, default=0)
    a = ap.parse_args()
    summ = json.load(open(os.path.join(RESULTS, f'real_tstep_{a.tag}.json')))
    d = os.path.join(DERIVED, 'real_tstep')
    R = np.load(os.path.join(d, f'{a.tag}_R.npy')).astype(np.float32)
    rows = pd.read_parquet(os.path.join(d, f'{a.tag}_rows.parquet'), columns=['agent', 'agent_ix', 'step'])
    names = np.array(sorted(rows.agent.unique()))
    N, T = len(names), summ['T']
    idx = {n: i for i, n in enumerate(names)}

    # the scan's residual increments, rebuilt exactly as real_tstep.py builds them
    pos = tstep.positions(R, rows.agent_ix.values, rows.step.values, N, T, seed=summ['seed'])
    D = tstep.increments(pos)
    alpha = summ.get('two_signed_alpha')
    D, kf = tstep.remove_factors(D, two_signed_alpha=alpha if alpha is not None and alpha >= 0 else None)
    d_eff = D.shape[1] * D.shape[2] - kf
    Uu = tstep._units(D)

    acts, _ = load_window(*summ['window'])
    com = pd.read_parquet(os.path.join(RAW, 'comments.parquet'), columns=['id', 'parent_id', 'author_name'])
    cauth = dict(zip(com.id, com.author_name)); cpar = dict(zip(com.id, com.parent_id)); com = None
    pauth = dict(zip(acts[acts.kind == 'post'].action_id, acts[acts.kind == 'post'].agent))
    created = pd.read_parquet(os.path.join(RAW, 'agents.parquet'), columns=['name', 'created_at']) \
        .drop_duplicates('name').set_index('name').created_at
    hours = np.arange(int(acts.t.min()), int(acts.t.max()) + 1)

    out = dict(tag=a.tag, n_factors=kf, groups=[])
    for gi, g in enumerate(summ['groups']):
        M = g['members']
        if len(M) < 2:
            continue
        m_ix = np.array([idx[x] for x in M])
        tot = Uu[m_ix].sum(0)
        A = acts[acts.agent.isin(M)].copy()
        A['tpl'] = [template(t, n) for t, n in zip(A.text, A.agent)]
        A['tick'] = A.text.map(lambda s: (MINT.search(s) or [None, None])[1])
        A['reply_to'] = [None if k == 'post' else
                         (cauth.get(cpar.get(i)) if isinstance(cpar.get(i), str) and cpar.get(i) else pauth.get(th))
                         for i, k, th in zip(A.action_id, A.kind, A.thread)]
        members = []
        for n, i in zip(M, m_ix):
            x = A[A.agent == n]
            v = tot - Uu[i]
            t_i = float(np.sqrt(d_eff) * Uu[i] @ v / max(np.linalg.norm(v), 1e-12))
            tpl = x.tpl.value_counts()
            members.append(dict(
                agent=n, t_i=round(t_i, 2), actions=len(x), posts=int((x.kind == 'post').sum()),
                comments=int((x.kind == 'comment').sum()),
                top_submolts={str(k): int(c) for k, c in x.submolt.value_counts().head(3).items()},
                mint_share=round(float(x.tick.notna().mean()), 3),
                ticks={str(k): int(c) for k, c in x.tick.value_counts().head(3).items()},
                top_template=tpl.index[0][:90], top_template_share=round(float(tpl.iloc[0] / len(x)), 3),
                median_gap_min=round(float(np.median(np.diff(np.sort(x.t.values))) * 60), 1) if len(x) > 1 else None,
                created=str(created.get(n)) if isinstance(created.get(n), str) else None))
        pairs = []
        hc = {n: np.bincount((A.t[A.agent == n].astype(int) - hours[0]).values, minlength=len(hours)) for n in M}
        for x, y in combinations(M, 2):
            tx, ty = set(A.tpl[A.agent == x]), set(A.tpl[A.agent == y])
            thx, thy = set(A.thread[(A.agent == x)]), set(A.thread[(A.agent == y)])
            rep = int(((A.agent == x) & (A.reply_to == y)).sum() + ((A.agent == y) & (A.reply_to == x)).sum())
            pairs.append(dict(a=x, b=y, inc_cos=round(float(Uu[idx[x]] @ Uu[idx[y]]), 3),
                              inc_z=round(float(Uu[idx[x]] @ Uu[idx[y]] * np.sqrt(d_eff)), 2),
                              template_jaccard=round(len(tx & ty) / max(len(tx | ty), 1), 3),
                              hourly_corr=round(float(np.corrcoef(hc[x], hc[y])[0, 1]), 3),
                              shared_threads=len(thx & thy), replies=rep))
        L = D.shape[1]
        step_c = []
        for s in range(L):
            Us = D[m_ix, s]; nrm = np.linalg.norm(D[m_ix], axis=(1, 2))[:, None]
            G = (Us / nrm) @ (Us / nrm).T
            step_c.append(round(float(G[np.triu_indices(len(M), 1)].mean()), 4))
        grp = dict(index=gi, p=g['p'], size=len(M), seed_band=g['band'], members=members, pairs=pairs,
                   step_contribution=step_c, mint_share=round(float(A.tick.notna().mean()), 3))
        out['groups'].append(grp)

        print(f'\n===== group {gi}: p = {g["p"]:.3f}, {len(M)} agents, seed band {g["band"]}, '
              f'mint share {grp["mint_share"]}')
        print('  step contributions to mean pair cosine:', step_c)
        for m in sorted(members, key=lambda r: -r['t_i']):
            print(f'  {m["agent"][:22]:22s} t {m["t_i"]:6.2f}  n {m["actions"]:5d} ({m["posts"]}p/{m["comments"]}c)  '
                  f'mint {m["mint_share"]:.2f} {m["ticks"]}  gap {m["median_gap_min"]}m  created {str(m["created"])[:16]}  '
                  f'subs {m["top_submolts"]}')
            print(f'      template ({m["top_template_share"]:.2f}): {m["top_template"]!r}')
        top = sorted(pairs, key=lambda r: -r['inc_z'])[:8]
        for p in top:
            print(f'  pair {p["a"][:18]:18s} {p["b"][:18]:18s} inc z {p["inc_z"]:5.2f}  tpl J {p["template_jaccard"]:.2f}  '
                  f'hourly r {p["hourly_corr"]:.2f}  threads {p["shared_threads"]}  replies {p["replies"]}')
        if a.samples:
            for n in M:
                for t in A[A.agent == n].sample(min(a.samples, (A.agent == n).sum()), random_state=0).text:
                    print(f'     [{n}] {t[:200]!r}')
    with open(os.path.join(RESULTS, f'real_investigate_{a.tag}.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote data/results/real_investigate_{a.tag}.json')


if __name__ == '__main__':
    main()
