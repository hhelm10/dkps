"""Descriptive facts about MoltNet to inform the section-0 scope decisions.

    python scope_report.py   -> writing/notes/scope_report.md

Raw activity counts only; no detection pipeline output is computed here,
so the weak ground truth can still be pre-registered before any run.
Operator handles are only counted, never printed.
"""
import json
import os
import numpy as np
import pandas as pd

from pipeline.ingest_moltnet import RAW, _hours

HERE = os.path.dirname(os.path.abspath(__file__))
from pipeline.paths import NOTES  # noqa: E402
WATCH = ['Hackerclaw', 'RoyMas', 'Dragon234Sad', 'LamaKor13264']
MARKER = '<molt_instructions>'
SCOPES = {  # candidate scopes from TODO section 0: (start, end, submolts, t0)
    'A_hackerclaw_general': ('2026-01-29', '2026-02-04', ['general'], '2026-01-31T11:00'),
    'B_zenity_general': ('2026-02-08', '2026-02-20', ['general'], '2026-02-12T00:00'),
}


def md_table(df):
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(map(str, cols)) + ' |', '|' + '---|' * len(cols)]
    for _, r in df.iterrows():
        lines.append('| ' + ' | '.join(str(r[c]) for c in cols) + ' |')
    return '\n'.join(lines)


def main():
    P = pd.read_parquet(os.path.join(RAW, 'posts.parquet'),
                        columns=['id', 'title', 'content', 'created_at', 'author_name',
                                 'submolt_name', 'votes_history'])
    C = pd.read_parquet(os.path.join(RAW, 'comments.parquet'),
                        columns=['id', 'content', 'created_at', 'post_id', 'parent_id',
                                 'author_name', 'votes_history'])
    A = pd.read_parquet(os.path.join(RAW, 'agents.parquet'), columns=['name', 'owner_x_handle'])
    P['day'] = pd.to_datetime(P.created_at, utc=True, format='mixed', errors='coerce').dt.date
    C['day'] = pd.to_datetime(C.created_at, utc=True, format='mixed', errors='coerce').dt.date
    C['submolt_name'] = C.post_id.map(dict(zip(P.id, P.submolt_name)))
    out = ['# MoltNet scope report', '',
           'Source: iNLP-Lab/Moltbook-MoltNet v2026-02-28 (see data/raw/PROVENANCE.md). '
           'Descriptive counts only.', '']

    # coverage
    snaps = P.votes_history.dropna().map(lambda s: len(json.loads(s)) if s.startswith('[') else 0)
    multi = A.owner_x_handle.dropna()
    multi = multi[multi != '']
    per_op = multi.value_counts()
    out += ['## Coverage', '',
            f'- Posts {len(P):,}; comments {len(C):,}; agents {len(A):,}.',
            f'- Comments with a parent comment id: {(C.parent_id.notna() & (C.parent_id != "")).mean():.1%} '
            '(the rest are treated as replies to the post).',
            f'- Comments whose post is in the dump: {C.post_id.isin(P.id).mean():.1%}.',
            f'- Vote snapshots per post: median {int(snaps.median())}, '
            f'90th pct {int(snaps.quantile(.9))}. No per-voter records.',
            f'- Agents with an owner handle: {len(multi) / len(A):.1%}. '
            f'Operators with more than one agent: {(per_op > 1).sum():,} '
            f'(covering {per_op[per_op > 1].sum():,} agents); largest operator runs '
            f'{per_op.max()} agents.', '']

    # daily volume
    dp = P.groupby('day').size().rename('posts')
    dc = C.groupby('day').size().rename('comments')
    dg = P[P.submolt_name == 'general'].groupby('day').size().rename('posts_general')
    daily = pd.concat([dp, dg, dc], axis=1).fillna(0).astype(int).reset_index()
    out += ['## Daily volume', '', md_table(daily), '']

    # watch-list accounts
    rows = []
    for n in WATCH:
        p, c = P[P.author_name == n], C[C.author_name == n]
        days = sorted(set(p.day.dropna()) | set(c.day.dropna()))
        rows.append(dict(agent=n, posts=len(p), comments=len(c),
                         first=str(days[0]) if days else '', last=str(days[-1]) if days else '',
                         top_submolt=p.submolt_name.mode().iat[0] if len(p) else ''))
    out += ['## Weak-ground-truth accounts (activity only)', '', md_table(pd.DataFrame(rows)), '']
    mp = P[(P.title.fillna('') + P.content.fillna('')).str.contains(MARKER, regex=False)]
    mc = C[C.content.fillna('').str.contains(MARKER, regex=False)]
    out += [f'Zenity payload marker `{MARKER}`: {len(mp)} posts by {mp.author_name.nunique()} agents, '
            f'{len(mc)} comments by {mc.author_name.nunique()} agents; dates '
            f'{min(list(mp.day.dropna()) + list(mc.day.dropna()), default="")} to '
            f'{max(list(mp.day.dropna()) + list(mc.day.dropna()), default="")}.', '']

    # top submolts in each candidate window + burn-in feasibility
    for name, (s, e, subs, t0) in SCOPES.items():
        lo, hi, t0h = _hours(pd.Series([s]))[0], _hours(pd.Series([e]))[0], _hours(pd.Series([t0]))[0]
        tp, tc = _hours(P.created_at), _hours(C.created_at)
        pw = P[(tp >= lo) & (tp < hi)]
        cw = C[(tc >= lo) & (tc < hi)]
        top = pd.concat([pw.submolt_name, cw.submolt_name]).value_counts().head(8)
        act = pd.concat([
            pd.DataFrame(dict(a=pw.author_name, t=tp[pw.index], s=pw.submolt_name)),
            pd.DataFrame(dict(a=cw.author_name, t=tc[cw.index], s=cw.submolt_name))])
        act = act[act.s.isin(subs)]
        pre = act[act.t <= t0h].groupby('a').size()
        post = act[act.t > t0h].groupby('a').size()
        both = pd.concat([pre.rename('pre'), post.rename('post')], axis=1).fillna(0)
        feas = {K: int(((both.pre >= K) & (both.post >= K)).sum()) for K in (3, 5, 10, 20)}
        out += [f'## Scope {name}: {s} to {e}, submolts {subs}, t0 = {t0}', '',
                f'- Actions in scope: {len(act):,} by {act.a.nunique():,} agents.',
                f'- Agents with at least K actions both before and after t0: '
                + ', '.join(f'K={k}: {v:,}' for k, v in feas.items()) + '.',
                f'- Top-100 agents cover {act.a.value_counts().head(100).sum() / max(len(act), 1):.1%} '
                f'of actions; top-500 cover {act.a.value_counts().head(500).sum() / max(len(act), 1):.1%}.',
                '- Busiest submolts in the window (all submolts): '
                + ', '.join(f'{k} ({v:,})' for k, v in top.items()) + '.', '']
    os.makedirs(os.path.join(NOTES), exist_ok=True)
    with open(os.path.join(NOTES, 'scope_report.md'), 'w') as fh:
        fh.write('\n'.join(out))
    print('\n'.join(out))


if __name__ == '__main__':
    main()
