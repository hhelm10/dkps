"""MoltNet (iNLP-Lab, v2026-02-28) -> canonical tables (pipeline/schema.py).

    python -m pipeline.ingest_moltnet --start 2026-01-29 --end 2026-02-04 \
        --submolts general --top-n 300 --name general_hackerclaw

Raw files are read-only and never modified. Output goes to
data/derived/<name>/{actions,contexts,score_events,agents}.parquet + meta.json.

Mapping:
  post     -> action_type 'post', thread_id = post id, context row (title +
              content)
  comment  -> action_type 'comment', thread_id = post_id, parent_id =
              parent comment if present else the post, target_agent = its
              author
  votes    -> not attributable to voters in any public source. Score
              events are increments of (upvotes - downvotes) between
              consecutive daily snapshots in votes_history, credited to
              the author at the snapshot time (first increment from 0).
  operator -> owner_x_handle is replaced by a salted hash (operator_key);
              no handle is written to disk.
"""
import argparse
import hashlib
import json
import os
import secrets
import numpy as np
import pandas as pd
from . import paths

RAW = os.path.join(paths.RAW,
                   'iNLP-Lab_Moltbook-MoltNet', 'data', 'v2026-02-28')
OUT = paths.DERIVED
ORIGIN = pd.Timestamp('2026-01-27T00:00:00Z')


def _hours(s):
    return (pd.to_datetime(s, utc=True, format='mixed', errors='coerce') - ORIGIN) \
        .dt.total_seconds() / 3600.0


def load(start, end):
    lo, hi = pd.Timestamp(start, tz='UTC'), pd.Timestamp(end, tz='UTC')
    lo_h, hi_h = (lo - ORIGIN).total_seconds() / 3600, (hi - ORIGIN).total_seconds() / 3600
    posts = pd.read_parquet(os.path.join(RAW, 'posts.parquet'),
                            columns=['id', 'title', 'content', 'created_at', 'author_id',
                                     'author_name', 'submolt_name', 'votes_history'])
    posts['t'] = _hours(posts.created_at)
    posts = posts[(posts.t >= lo_h) & (posts.t < hi_h)]
    comments = pd.read_parquet(os.path.join(RAW, 'comments.parquet'),
                               columns=['id', 'content', 'created_at', 'post_id', 'parent_id',
                                        'author_id', 'author_name', 'votes_history'])
    comments['t'] = _hours(comments.created_at)
    comments = comments[(comments.t >= lo_h) & (comments.t < hi_h)]
    return posts, comments, lo_h, hi_h


def _snap_events(df, author_col='author_name'):
    rows = []
    for author, vh in zip(df[author_col].values, df.votes_history.values):
        if not isinstance(vh, str):
            continue
        try:
            snaps = json.loads(vh)
        except ValueError:
            continue
        pts = []
        for s in snaps:
            up, dn = s.get('upvotes'), s.get('downvotes')
            if up is None and s.get('score') is None:
                continue
            v = (up or 0) - (dn or 0) if up is not None else s['score']
            pts.append((s['timestamp'], v))
        pts.sort()
        prev = 0
        for ts, v in pts:
            if v != prev:
                rows.append((author, ts, v - prev))
            prev = v
    ev = pd.DataFrame(rows, columns=['agent_id', 'ts', 'value'])
    ev['t'] = _hours(ev.ts)
    return ev[['agent_id', 't', 'value']]


def build(start, end, submolts=None, top_n=None, salt=None, donor_submolts=None,
          n_donors=200):
    posts, comments, lo_h, hi_h = load(start, end)
    donors, ev_donor = [], None
    if donor_submolts:
        dp = posts[posts.submolt_name.isin(donor_submolts)]
        dc = comments[comments.post_id.isin(dp.id)]
        cnt = pd.concat([dp.author_name, dc.author_name]).value_counts()
        if submolts:
            ins = set(posts[posts.submolt_name.isin(submolts)].author_name) | set(
                comments[comments.post_id.isin(posts[posts.submolt_name.isin(submolts)].id)].author_name)
            cnt = cnt[~cnt.index.isin(ins)]  # donors never acted in the analysed scope
        donors = list(cnt.index[:n_donors])
        ev_donor = pd.concat([_snap_events(dp), _snap_events(dc)], ignore_index=True)
        ev_donor = ev_donor[ev_donor.agent_id.isin(donors)]
    if submolts:
        posts = posts[posts.submolt_name.isin(submolts)]
        comments = comments[comments.post_id.isin(posts.id)]
    sub_of = dict(zip(posts.id, posts.submolt_name))
    p_author = dict(zip(posts.id, posts.author_name))
    c_author = dict(zip(comments.id, comments.author_name))

    act_p = pd.DataFrame(dict(
        action_id=posts.id.values, agent_id=posts.author_name.values, action_type='post',
        t=posts.t.values, submolt=posts.submolt_name.values, thread_id=posts.id.values,
        parent_id='', target_agent='',
        text=(posts.title.fillna('') + '\n\n' + posts.content.fillna('')).values,
        score_at_fetch=np.nan))
    par = comments.parent_id.where(comments.parent_id.notna() & (comments.parent_id != ''),
                                   comments.post_id)
    tgt = [c_author.get(p, p_author.get(p, '')) for p in par.values]
    act_c = pd.DataFrame(dict(
        action_id=comments.id.values, agent_id=comments.author_name.values,
        action_type='comment', t=comments.t.values,
        submolt=[sub_of.get(p, '') for p in comments.post_id.values],
        thread_id=comments.post_id.values, parent_id=par.values, target_agent=tgt,
        text=comments.content.fillna('').values, score_at_fetch=np.nan))
    acts = pd.concat([act_p, act_c], ignore_index=True)
    acts = acts[acts.agent_id.notna() & (acts.agent_id != '')]
    acts = acts[acts.thread_id.isin(posts.id)]  # comments on posts outside window lack context

    if top_n:
        keep = acts.agent_id.value_counts().index[:top_n]
        system = set(keep)
    else:
        system = set(acts.agent_id)
    contexts = pd.DataFrame(dict(
        thread_id=posts.id.values, submolt=posts.submolt_name.values, t=posts.t.values,
        text=(posts.title.fillna('') + '\n\n' + posts.content.fillna('').str[:2000]).values))
    ev = pd.concat([_snap_events(posts), _snap_events(comments)] +
                   ([ev_donor] if ev_donor is not None else []), ignore_index=True)

    agents = pd.read_parquet(os.path.join(RAW, 'agents.parquet'),
                             columns=['name', 'created_at', 'owner_x_handle', 'description_history',
                                      'karma_history'])
    agents = agents[agents.name.isin(set(acts.agent_id))].copy()
    salt = salt or secrets.token_hex(16)
    agents['operator_key'] = [hashlib.sha256((salt + h).encode()).hexdigest()[:16]
                              if isinstance(h, str) and h else '' for h in agents.owner_x_handle]
    agents = agents.drop(columns=['owner_x_handle'])
    agents['in_system'] = agents.name.isin(system)
    meta = dict(source='iNLP-Lab/Moltbook-MoltNet v2026-02-28', start=start, end=end,
                origin=str(ORIGIN), window_hours=[lo_h, hi_h], submolts=submolts, top_n=top_n,
                n_system=len(system), donor_submolts=donor_submolts, n_donors=len(donors),
                caveats=['score events are daily snapshot increments credited at snapshot time',
                         'no per-voter vote records exist; votes absent from the interaction graph',
                         'operator_key is a salted hash; salt not stored'])
    return dict(actions=acts.sort_values('t').reset_index(drop=True), contexts=contexts,
                score_events=ev, agents=agents, meta=meta, system=sorted(system),
                donors=donors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    ap.add_argument('--submolts', nargs='*')
    ap.add_argument('--top-n', type=int)
    ap.add_argument('--donor-submolts', nargs='*')
    ap.add_argument('--name', required=True)
    a = ap.parse_args()
    d = build(a.start, a.end, a.submolts, a.top_n, donor_submolts=a.donor_submolts)
    out = os.path.join(OUT, a.name)
    os.makedirs(out, exist_ok=True)
    for k in ('actions', 'contexts', 'score_events', 'agents'):
        d[k].to_parquet(os.path.join(out, f'{k}.parquet'))
    d['meta']['system'] = d['system']
    d['meta']['donors'] = d['donors']
    with open(os.path.join(out, 'meta.json'), 'w') as fh:
        json.dump(d['meta'], fh, indent=1)
    from .schema import coverage
    print(json.dumps(coverage(d['actions']), indent=1, default=str))


if __name__ == '__main__':
    main()


def build_platform(start, end, t0, min_actions=3, cap_per_day=40, lookback_days=3,
                   n_donors=300, seed=0, salt=None):
    """Whole-platform scope: every agent with >= min_actions content actions on
    each side of t0 in [start, end), all submolts.

    Actions are capped at cap_per_day per agent per UTC day (seeded uniform
    sample; keeps timing structure, bounds the heavy tail, e.g. agents with
    300K+ comments). Threads opened up to lookback_days before start are kept
    as contexts so early comments are not dropped. Donors for synthetic
    control: the n_donors most active agents in the window that are NOT in
    the system (they lack burn-in or post-t0 activity).
    """
    rng = np.random.default_rng(seed)
    lo, hi = pd.Timestamp(start, tz='UTC'), pd.Timestamp(end, tz='UTC')
    lo_h, hi_h = (lo - ORIGIN).total_seconds() / 3600, (hi - ORIGIN).total_seconds() / 3600
    t0_h = (pd.Timestamp(t0, tz='UTC') - ORIGIN).total_seconds() / 3600
    posts = pd.read_parquet(os.path.join(RAW, 'posts.parquet'),
                            columns=['id', 'title', 'content', 'created_at', 'author_name',
                                     'submolt_name', 'votes_history'])
    posts['t'] = _hours(posts.created_at)
    posts = posts[(posts.t >= lo_h - 24 * lookback_days) & (posts.t < hi_h)]
    comments = pd.read_parquet(os.path.join(RAW, 'comments.parquet'),
                               columns=['id', 'content', 'created_at', 'post_id', 'parent_id',
                                        'author_name', 'votes_history'])
    comments['t'] = _hours(comments.created_at)
    comments = comments[(comments.t >= lo_h) & (comments.t < hi_h) & comments.post_id.isin(posts.id)]
    pw = posts[posts.t >= lo_h]

    both = pd.concat([pw[['author_name', 't']], comments[['author_name', 't']]])
    both = both[both.author_name.notna() & (both.author_name != '')]
    pre = both[both.t <= t0_h].groupby('author_name').size()
    post = both[both.t > t0_h].groupby('author_name').size()
    b = pd.concat([pre.rename('pre'), post.rename('post')], axis=1).fillna(0)
    system = set(b.index[(b.pre >= min_actions) & (b.post >= min_actions)])
    act_cnt = both.author_name.value_counts()
    donors = [a for a in act_cnt.index if a not in system][:n_donors]

    sub_of = dict(zip(posts.id, posts.submolt_name))
    p_author = dict(zip(posts.id, posts.author_name))
    c_author = dict(zip(comments.id, comments.author_name))
    act_p = pd.DataFrame(dict(
        action_id=pw.id.values, agent_id=pw.author_name.values, action_type='post',
        t=pw.t.values, submolt=pw.submolt_name.values, thread_id=pw.id.values,
        parent_id='', target_agent='',
        text=(pw.title.fillna('') + '\n\n' + pw.content.fillna('')).values,
        score_at_fetch=np.nan))
    par = comments.parent_id.where(comments.parent_id.notna() & (comments.parent_id != ''),
                                   comments.post_id)
    act_c = pd.DataFrame(dict(
        action_id=comments.id.values, agent_id=comments.author_name.values,
        action_type='comment', t=comments.t.values,
        submolt=comments.post_id.map(sub_of).values,
        thread_id=comments.post_id.values, parent_id=par.values,
        target_agent=[c_author.get(p, p_author.get(p, '')) for p in par.values],
        text=comments.content.fillna('').values, score_at_fetch=np.nan))
    acts = pd.concat([act_p, act_c], ignore_index=True)
    acts = acts[acts.agent_id.notna() & (acts.agent_id != '')]
    n_raw_sys = int(acts.agent_id.isin(system).sum())
    # cap system agents' content per agent-day; others are kept only as
    # interaction-graph endpoints (no text needed)
    sysm = acts.agent_id.isin(system)
    s = acts[sysm].copy()
    s['day'] = (s.t // 24).astype(int)
    s['r'] = rng.uniform(size=len(s))
    s['rank'] = s.groupby(['agent_id', 'day']).r.rank(method='first')
    s = s[s['rank'] <= cap_per_day].drop(columns=['day', 'r', 'rank'])
    others = acts[~sysm].assign(text='')
    acts = pd.concat([s, others], ignore_index=True).sort_values('t').reset_index(drop=True)

    used = set(s.thread_id)
    ctx = posts[posts.id.isin(used)]
    contexts = pd.DataFrame(dict(
        thread_id=ctx.id.values, submolt=ctx.submolt_name.values, t=ctx.t.values,
        text=(ctx.title.fillna('') + '\n\n' + ctx.content.fillna('').str[:2000]).values))
    keep_auth = system | set(donors)
    ev = pd.concat([_snap_events(posts[posts.author_name.isin(keep_auth)]),
                    _snap_events(comments[comments.author_name.isin(keep_auth)])],
                   ignore_index=True)
    agents = pd.read_parquet(os.path.join(RAW, 'agents.parquet'),
                             columns=['name', 'created_at', 'owner_x_handle'])
    agents = agents[agents.name.isin(keep_auth)].copy()
    salt = salt or secrets.token_hex(16)
    agents['operator_key'] = [hashlib.sha256((salt + h).encode()).hexdigest()[:16]
                              if isinstance(h, str) and h else '' for h in agents.owner_x_handle]
    agents = agents.drop(columns=['owner_x_handle'])
    agents['in_system'] = agents.name.isin(system)
    meta = dict(source='iNLP-Lab/Moltbook-MoltNet v2026-02-28', start=start, end=end, t0=t0,
                origin=str(ORIGIN), window_hours=[lo_h, hi_h], submolts='all',
                min_actions=min_actions, cap_per_day=cap_per_day,
                n_system=len(system), n_donors=len(donors),
                n_system_actions_raw=n_raw_sys, n_system_actions_capped=int(len(s)),
                caveats=['score events are daily snapshot increments credited at snapshot time',
                         'no per-voter vote records exist; votes absent from the interaction graph',
                         f'system actions capped at {cap_per_day}/agent/day by seeded sampling',
                         'operator_key is a salted hash; salt not stored'])
    return dict(actions=acts, contexts=contexts, score_events=ev, agents=agents, meta=meta,
                system=sorted(system), donors=donors)


def save(d, name):
    out = os.path.join(OUT, name)
    os.makedirs(out, exist_ok=True)
    for k in ('actions', 'contexts', 'score_events', 'agents'):
        d[k].to_parquet(os.path.join(out, f'{k}.parquet'))
    d['meta']['system'] = d['system']
    d['meta']['donors'] = d['donors']
    with open(os.path.join(out, 'meta.json'), 'w') as fh:
        json.dump(d['meta'], fh, indent=1)
    return out
