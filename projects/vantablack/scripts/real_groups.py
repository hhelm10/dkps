"""Moltbook: describe the groups found by real_tstep.py (peeled groups), against random
cohort sets of the same size.

    python real_groups.py [--tag 01-30_02-18_T8] [--samples 3]

Per group: activity (actions per agent, comment share), submolt concentration, account
age, thread overlap (share of member pairs that commented in a common thread, and in a
common thread within the same step), reply links (share of members' comments that reply to
another member, directly or to a member's post), and the number of member pairs with the same
owner (owners are compared by a salted hash in memory; only the pair count is reported).
Each statistic gets a reference: its mean over 200 random cohort sets of the same size.

Writes data/results/real_groups_<tag>.json. --samples prints that many random texts per
member for reading in the terminal (not written to disk).
"""
import argparse
import hashlib
import json
import os
import secrets
import warnings
from itertools import combinations

import numpy as np
import pandas as pd

from pipeline.ingest_moltnet import RAW
from pipeline.paths import DERIVED, RESULTS
from real_tstep import load_window


def describe(members, ctx, rng, n_ref=200):
    """Statistics for one member set, and their means over random cohort sets of equal size."""
    def stats(M):
        M = list(M)
        a = ctx['acts'][ctx['acts'].agent.isin(M)]
        sub = a.submolt.value_counts(normalize=True)
        pairs = list(combinations(M, 2))
        th, ths = ctx['threads'], ctx['thread_steps']
        share_th = np.mean([bool(th[x] & th[y]) for x, y in pairs])
        share_ths = np.mean([bool(ths[x] & ths[y]) for x, y in pairs])
        c = a[a.kind == 'comment']
        Ms = set(M)
        to = c.reply_to.map(lambda x: x in Ms) & (c.reply_to != c.agent)
        same_owner = sum(ctx['owner'].get(x) is not None and ctx['owner'].get(x) == ctx['owner'].get(y)
                         for x, y in pairs)
        age = (ctx['created'].reindex(M) - pd.Timestamp('2026-01-30', tz='UTC')).dt.total_seconds() / 86400
        return dict(actions_median=float(a.groupby('agent').size().median()),
                    comment_share=float((a.kind == 'comment').mean()),
                    top_submolt_share=float(sub.iloc[0]), general_share=float(sub.get('general', 0.0)),
                    submolt_hhi=float((sub ** 2).sum()),
                    created_days_rel_jan30_median=float(age.median()),
                    pair_share_common_thread=float(share_th), pair_share_common_thread_same_step=float(share_ths),
                    reply_to_member_share=float(to.mean()) if len(c) else float('nan'),
                    same_owner_pairs=int(same_owner))
    obs = stats(members)
    refs = [stats(rng.choice(ctx['cohort'], len(members), replace=False)) for _ in range(n_ref)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)               # all-NaN references (no comments)
        ref = {k: float(np.nanmean([r[k] for r in refs])) for k in obs}
    sub = ctx['acts'][ctx['acts'].agent.isin(members)].submolt.value_counts(normalize=True).head(5)
    return dict(stats=obs, random_mean=ref, top_submolts={str(k): round(float(v), 3) for k, v in sub.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='01-30_02-18_T8')
    ap.add_argument('--samples', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    summ = json.load(open(os.path.join(RESULTS, f'real_tstep_{a.tag}.json')))
    rows = pd.read_parquet(os.path.join(DERIVED, 'real_tstep', f'{a.tag}_rows.parquet'))
    cohort = np.array(sorted(rows.agent.unique()))

    # reply targets: a comment's parent comment author, else the thread's author
    com = pd.read_parquet(os.path.join(RAW, 'comments.parquet'), columns=['id', 'parent_id', 'author_name'])
    posts = pd.read_parquet(os.path.join(RAW, 'posts.parquet'), columns=['id', 'author_name'])
    auth = dict(zip(com.id, com.author_name)); pauth = dict(zip(posts.id, posts.author_name))
    par = dict(zip(com.id, com.parent_id))
    rows['reply_to'] = [auth.get(par.get(i)) if isinstance(par.get(i), str) and par.get(i) else pauth.get(th)
                        for i, th in zip(rows.action_id, rows.thread)]
    rows.loc[rows.kind == 'post', 'reply_to'] = None
    com = posts = auth = pauth = par = None

    ag = pd.read_parquet(os.path.join(RAW, 'agents.parquet'), columns=['name', 'created_at', 'owner_x_handle'])
    ag = ag[ag.name.isin(cohort)].drop_duplicates('name').set_index('name')
    salt = secrets.token_bytes(16)                                   # in memory only, never stored
    owner = {n: hashlib.sha256(salt + h.lower().encode()).hexdigest()
             for n, h in ag.owner_x_handle.items() if isinstance(h, str) and h}
    created = pd.to_datetime(ag.created_at, utc=True, errors='coerce')

    c = rows[rows.kind == 'comment']
    threads = c.groupby('agent').thread.agg(set).reindex(cohort).apply(lambda s: s if isinstance(s, set) else set())
    ts = (c.thread + '|' + c.step.astype(str)).groupby(c.agent).agg(set).reindex(cohort) \
        .apply(lambda s: s if isinstance(s, set) else set())
    ctx = dict(acts=rows, cohort=cohort, threads=threads.to_dict(), thread_steps=ts.to_dict(),
               owner=owner, created=created)
    rng = np.random.default_rng(a.seed)

    out = dict(tag=a.tag, groups=[])
    for i, g in enumerate(summ['groups']):
        if not g['members'] or len(g['members']) < 2:
            continue
        d = describe(g['members'], ctx, rng)
        d.update(index=i, p=g['p'], size=g['size'], members=g['members'])
        out['groups'].append(d)
        print(f'\n== group {i}: p = {g["p"]:.3f}, {g["size"]} agents')
        for k, v in d['stats'].items():
            print(f'  {k:38s} {v:9.3f}   random {d["random_mean"][k]:9.3f}')
        print('  top submolts', d['top_submolts'])
        print('  members', ', '.join(g['members'][:40]) + (' ...' if len(g['members']) > 40 else ''))
        if a.samples:
            acts, _ = load_window(*summ['window'])
            for m in g['members'][:12]:
                s = acts[acts.agent == m].sample(min(a.samples, (acts.agent == m).sum()), random_state=0)
                for _, r in s.iterrows():
                    print(f'   [{m} | {r.kind} | {r.submolt}] {r.text[:220]!r}')
    with open(os.path.join(RESULTS, f'real_groups_{a.tag}.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote data/results/real_groups_{a.tag}.json')


if __name__ == '__main__':
    main()
