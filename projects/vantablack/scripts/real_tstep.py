"""Moltbook: T-step scan for colluding subnetworks (the simulation-validated method).

    python real_tstep.py [--start 2026-01-30 --end 2026-02-18 --T 8 --B 199 --workers 20]

Design (writing/notes/decisions.md, "Real-data analysis v2"):
  window    MoltNet posts and comments in [start, end), all submolts
  steps     activity time: T steps with equal shares of platform-wide action volume
            (boundaries at quantiles of action times), so crawl gaps (no comments on
            Feb 1 or after Feb 17, few on Feb 10-12) merge into neighbouring steps
  agents    every agent with at least one action in every step; all of its actions are used
  content   pinned nomic v1.5 embeddings (pipeline/embed.py); existing scope embeddings
            are reused by action id, the rest are embedded here with the same text rules
  queries   a comment's query is its thread (opening post), a post's is its submolt
  condition PKPS pooled conditioning on everyone's responses (pipeline/pkps.py): the
            cohort's actions plus every embedded action of other agents in the window;
            --cond step uses only the responses of the same step (removes stimuli that are
            shared at one time, e.g. topic waves)
  test      pipeline/tstep.run_residuals: split-half increments, factor removal, scan with
            the per-step agent-ID shuffle null, membership estimate and agent ranking

Raw files are read only. Writes data/derived/real_tstep/<tag>.pkl (full result) and
data/results/real_tstep_<tag>.json (summary: steps, cohort, p-values, best sets, top of the
agent ranking). Agent names are public Moltbook handles; no operator information is used.
"""
import argparse
import glob
import json
import os
import pickle
import time

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np
import pandas as pd

from pipeline import embed, pkps, tstep
from pipeline.ingest_moltnet import RAW, ORIGIN, _hours
from pipeline.paths import DERIVED, RESULTS

SCOPE = os.path.join(DERIVED, 'platform_jan31_feb17')   # existing embeddings, reused by id


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def load_window(start, end, lookback_days=3):
    lo = (pd.Timestamp(start, tz='UTC') - ORIGIN).total_seconds() / 3600
    hi = (pd.Timestamp(end, tz='UTC') - ORIGIN).total_seconds() / 3600
    posts = pd.read_parquet(os.path.join(RAW, 'posts.parquet'),
                            columns=['id', 'title', 'content', 'created_at', 'author_name', 'submolt_name'])
    posts['t'] = _hours(posts.created_at)
    posts = posts[(posts.t >= lo - 24 * lookback_days) & (posts.t < hi)]
    comments = pd.read_parquet(os.path.join(RAW, 'comments.parquet'),
                               columns=['id', 'content', 'created_at', 'post_id', 'author_name'])
    comments['t'] = _hours(comments.created_at)
    comments = comments[(comments.t >= lo) & (comments.t < hi) & comments.post_id.isin(posts.id)]
    pw = posts[posts.t >= lo]
    sub_of = dict(zip(posts.id, posts.submolt_name))
    acts = pd.concat([
        pd.DataFrame(dict(action_id=pw.id.values, agent=pw.author_name.values, kind='post', t=pw.t.values,
                          thread=pw.id.values, submolt=pw.submolt_name.values,
                          text=(pw.title.fillna('') + '\n\n' + pw.content.fillna('')).values)),
        pd.DataFrame(dict(action_id=comments.id.values, agent=comments.author_name.values, kind='comment',
                          t=comments.t.values, thread=comments.post_id.values,
                          submolt=comments.post_id.map(sub_of).values,
                          text=comments.content.fillna('').values))], ignore_index=True)
    acts = acts[acts.agent.notna() & (acts.agent != '')].sort_values('t', kind='stable').reset_index(drop=True)
    return acts, posts


def activity_steps(t, T):
    """Step index per action: T equal-volume steps in time order; boundaries in hours."""
    order = np.argsort(t, kind='stable')
    step = np.empty(len(t), int)
    step[order] = np.arange(len(t)) * T // len(t)
    bounds = [float(t[order[0]])] + [float(t[order[(k * len(t)) // T]]) for k in range(1, T)] \
        + [float(t[order[-1]])]
    return step, bounds


def _lookup(name):
    E = np.load(os.path.join(SCOPE, f'emb_{name}.npy'), mmap_mode='r')
    ids = json.load(open(os.path.join(SCOPE, f'emb_{name}_ids.json')))
    return E, {a: i for i, a in enumerate(ids)}


STORE = os.path.join(DERIVED, 'emb_store')                       # embeddings made by earlier runs, by id


def _store(kind):
    """All chunks of the per-id store for kind ('actions', 'threads', 'submolts')."""
    out = []
    for f in sorted(glob.glob(os.path.join(STORE, f'{kind}_*_ids.json'))):
        out.append((np.load(f[:-len('_ids.json')] + '.npy', mmap_mode='r'),
                    {a: i for i, a in enumerate(json.load(open(f)))}))
    return out


def _add_to_store(kind, ids, E):
    os.makedirs(STORE, exist_ok=True)
    stem = os.path.join(STORE, f'{kind}_{time.strftime("%Y%m%d-%H%M%S")}_{len(ids)}')
    np.save(stem + '.npy', np.asarray(E, np.float32))
    with open(stem + '_ids.json', 'w') as fh:
        json.dump(list(ids), fh)


def gather(ids, texts, sources, cache_name, kind=None):
    """Embeddings for ids: from the existing stores (and, with kind, the store of earlier runs)
    when present, else embedded now and added to that store."""
    sources = list(sources) + (_store(kind) if kind else [])
    out = np.zeros((len(ids), 768), np.float32)
    found = np.zeros(len(ids), bool)
    for E, pos in sources:
        idx = np.array([pos.get(a, -1) for a in ids])
        hit = (idx >= 0) & ~found
        out[hit] = E[idx[hit]]
        found |= hit
    miss = np.flatnonzero(~found)
    log(f'{cache_name}: {len(ids)} needed, {found.sum()} reused, {len(miss)} to embed')
    if len(miss):
        out[miss] = embed.embed([texts[i] for i in miss], cache_name=cache_name)
        if kind:
            _add_to_store(kind, [ids[i] for i in miss], out[miss])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2026-01-30')
    ap.add_argument('--end', default='2026-02-18')
    ap.add_argument('--T', type=int, default=8)
    ap.add_argument('--B', type=int, default=199)
    ap.add_argument('--k', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=20)
    ap.add_argument('--cond', choices=['all', 'step'], default='all',
                    help="PKPS reference: all responses in the window, or only those of the same step")
    ap.add_argument('--peel', type=int, default=10, help='peel up to this many groups (0: one scan only)')
    ap.add_argument('--two-signed', type=float, default=0.01,
                    help='alpha of the two-signed factor rule (design note 1b); negative = off')
    ap.add_argument('--min-steps', type=int, default=None,
                    help='cohort: agents with at least one action in this many steps (default: all T)')
    ap.add_argument('--min-move', type=float, default=0.01,
                    help='increments shorter than this times the median raw increment are missing '
                         '(static agent-steps; negative: off)')
    ap.add_argument('--weight', choices=['reliability', 'none'], default='reliability',
                    help='increment weighting (tstep.increments)')
    ap.add_argument('--reuse', action='store_true',
                    help='reuse the saved conditioned responses of the same tag and rerun only the scan')
    a = ap.parse_args()
    a.min_steps = a.min_steps or a.T
    tag = f'{a.start[5:]}_{a.end[5:]}_T{a.T}' + ('' if a.cond == 'all' else f'_cond{a.cond}') \
        + ('' if a.min_steps == a.T else f'_ms{a.min_steps}')

    if a.reuse:
        return scan(a, tag, **reuse(tag))
    acts, posts = load_window(a.start, a.end)
    acts['step'], bounds = activity_steps(acts.t.values, a.T)
    per = acts.groupby(['agent', 'step']).size().unstack(fill_value=0)
    cohort = sorted(per.index[(per >= 1).sum(1) >= a.min_steps])
    log(f'{len(acts)} actions from {acts.agent.nunique()} agents; {a.T} activity-time steps; '
        f'cohort (>= 1 action in >= {a.min_steps} steps): {len(cohort)} agents')
    sysm = acts.agent.isin(cohort).values
    sys = acts[sysm].reset_index(drop=True)
    cnt = sys.groupby(['agent', 'step']).size()
    log(f'cohort: {len(sys)} actions; per agent-step median {cnt.median():.0f}, '
        f'share with one action {(cnt == 1).mean():.3f}, max {cnt.max()}')

    # ---- embeddings: cohort actions (all), reference actions (already embedded only)
    stores = [_lookup('actions'), _lookup('ref_actions')]
    E_sys = gather(sys.action_id.tolist(), sys.text.tolist(), stores, f'real_tstep_sys_{tag}', kind='actions')
    have = set().union(*[set(p) for _, p in stores])
    ref = acts[~sysm & acts.action_id.isin(have)].reset_index(drop=True)
    E_ref = gather(ref.action_id.tolist(), [''] * len(ref), stores, f'real_tstep_ref_{tag}')
    log(f'reference: {len(ref)} embedded actions of {ref.agent.nunique()} other agents '
        f'({len(ref) / max((~sysm).sum(), 1):.2f} of their actions in the window)')

    # ---- queries: threads (comments) and submolts (posts)
    threads = sorted(set(sys.thread[sys.kind == 'comment']) | set(ref.thread[ref.kind == 'comment']))
    p = posts.drop_duplicates('id').set_index('id').reindex(threads)
    th_text = (p.title.fillna('') + '\n\n' + p.content.fillna('').str[:2000]).tolist()
    Q_thread = gather(threads, th_text, [_lookup('contexts'), _lookup('threads_extra')],
                      f'real_tstep_threads_{tag}', kind='threads')
    subs_used = sorted(set(sys.submolt[sys.kind == 'post'].dropna()) | set(ref.submolt[ref.kind == 'post'].dropna()))
    sm = pd.read_parquet(os.path.join(RAW, 'submolts.parquet'), columns=['name', 'display_name', 'description'])
    sm = sm.drop_duplicates('name').set_index('name').reindex(subs_used)
    sub_text = [f'{n if not isinstance(dn, str) or not dn else dn}\n\n{ds if isinstance(ds, str) else ""}'
                for n, dn, ds in zip(subs_used, sm.display_name, sm.description)]
    Q_sub = gather(subs_used, sub_text, [_lookup('submolts')], f'real_tstep_submolts_{tag}',
                   kind='submolts')

    tpos = {x: i for i, x in enumerate(threads)}
    spos = {x: i for i, x in enumerate(subs_used)}
    qidx = lambda df: np.array([tpos[th] if k == 'comment' else spos.get(sb, -1)
                                for k, th, sb in zip(df.kind, df.thread, df.submolt)])
    q_sys, q_ref = qidx(sys), qidx(ref)
    okr = q_ref >= 0
    if (q_sys < 0).any():
        log(f'dropping {(q_sys < 0).sum()} cohort posts without a submolt')
    oks = q_sys >= 0
    agent_ix = {g: i for i, g in enumerate(cohort)}
    ag = np.array([agent_ix[g] for g in sys.agent])

    # ---- PKPS conditioning on everyone's responses, then the T-step scan
    R, share, info = pkps.condition_all(E_sys[oks], ag[oks], sys.t.values[oks], sys.kind.values[oks], q_sys[oks],
                                        E_ref[okr], ref.kind.values[okr], q_ref[okr], Q_thread, Q_sub,
                                        t0=None, k=a.k, time=a.cond, log=log,
                                        period_sys=sys.step.values[oks], period_ref=ref.step.values[okr])
    os.makedirs(os.path.join(DERIVED, 'real_tstep'), exist_ok=True)
    R = R.astype(np.float16)                      # stored at half precision; scan the stored values, so
    np.save(os.path.join(DERIVED, 'real_tstep', f'{tag}_R.npy'), R)   # --reuse reproduces a fresh run
    R = R.astype(np.float32)
    sys[oks].drop(columns=['text']).assign(agent_ix=ag[oks]).to_parquet(
        os.path.join(DERIVED, 'real_tstep', f'{tag}_rows.parquet'))
    base = dict(window=[a.start, a.end], T=a.T, k=a.k, cond=a.cond,
                step_bounds=[str(ORIGIN + pd.Timedelta(hours=h)) for h in bounds],
                n_actions_window=int(len(acts)), n_agents_window=int(acts.agent.nunique()),
                cohort=len(cohort), cohort_actions=int(oks.sum()),
                per_agent_step=dict(median=float(cnt.median()), share_one=float((cnt == 1).mean())),
                reference_actions=int(okr.sum()), pkps=info)
    scan(a, tag, R, ag[oks], sys.step.values[oks], cohort, base, share)


def reuse(tag):
    """Saved conditioned responses and the run description of an earlier run of this tag."""
    d = os.path.join(DERIVED, 'real_tstep')
    R = np.load(os.path.join(d, f'{tag}_R.npy')).astype(np.float32)
    rows = pd.read_parquet(os.path.join(d, f'{tag}_rows.parquet'), columns=['agent', 'agent_ix', 'step'])
    old = pickle.load(open(os.path.join(d, f'{tag}.pkl'), 'rb'))
    keys = ['window', 'T', 'k', 'cond', 'step_bounds', 'n_actions_window', 'n_agents_window', 'cohort',
            'cohort_actions', 'per_agent_step', 'reference_actions', 'pkps']
    log(f'reusing {tag}: {len(R)} conditioned responses')
    return dict(R=R, agent=rows.agent_ix.values, step=rows.step.values, cohort=old['cohort'],
                base={k: old['summary'].get(k, 'all' if k == 'cond' else None) for k in keys}, share=old['self_share'])


def scan(a, tag, R, agent, step, cohort, base, share):
    log('scan ...')
    t = time.time()
    res = tstep.run_residuals(R, agent, step, len(cohort), a.T, B=a.B, seed=a.seed,
                              workers=a.workers, peel=a.peel,
                              two_signed_alpha=a.two_signed if a.two_signed >= 0 else None,
                              weight=None if a.weight == 'none' else a.weight,
                              min_move=a.min_move if a.min_move >= 0 else None)
    log(f'scan done in {time.time() - t:.0f}s: p = {res["p"]:.3f}, bands {np.round(res["p_band"], 3).tolist()}, '
        f'd = {res["d"]}, factors removed {res["n_factors"]}, missing increments {res["n_missing"]} '
        f'(static {res["n_static"]})')

    # ---- outputs
    names = np.array(cohort)
    order = np.argsort(-res['rank_score'])
    groups = [dict(p=g['p'], p_band=g['p_band'].tolist(), obs=g['obs'].tolist(), band=g['band'],
                   n_tested=g['n_tested'], size=len(g['members']), members=sorted(names[g['members']].tolist()),
                   components=[sorted(names[c].tolist()) for c in g.get('components', [])],
                   best=[None if x is None else sorted(names[x].tolist()) for x in g['best']])
              for g in res.get('groups', [])]
    for i, g in enumerate(groups):
        log(f'group {i}: p = {g["p"]:.3f}, bands {np.round(g["p_band"], 3).tolist()}, '
            f'{g["size"]} of {g["n_tested"]} agents; seed band {g["band"]}; '
            f'components {[len(c) for c in g["components"]]}')
        for c in g['components']:
            log('    ' + ', '.join(c))
    summary = dict(
        **base, B=a.B, seed=a.seed,
        p=res['p'], p_band=np.asarray(res['p_band']).tolist(), obs=np.asarray(res['obs']).tolist(),
        band_edges=tstep.BANDS.tolist(),
        best=[sorted(names[sorted(b)].tolist()) if b is not None else None for b in res['best']],
        S_hat=sorted(names[res['S_hat']].tolist()), band_hat=res['band_hat'],
        ranking_top50=[dict(agent=str(names[i]), score=float(res['rank_score'][i])) for i in order[:50]],
        d=res['d'], d_eff=res['d_eff'], n_factors=res['n_factors'], n_missing=res['n_missing'],
        explained=res['explained'], centered=True, two_signed_alpha=a.two_signed, weight=a.weight,
        pair_thr=res.get('pair_thr'),
        cohort_components=[sorted(names[c].tolist()) for c in res.get('cohort_components', [])],
        reliability_fit=res.get('reliability_fit'), min_move=a.min_move, n_static=res.get('n_static'),
        min_steps=a.min_steps, groups=groups)
    os.makedirs(os.path.join(DERIVED, 'real_tstep'), exist_ok=True)
    with open(os.path.join(DERIVED, 'real_tstep', f'{tag}.pkl'), 'wb') as fh:
        pickle.dump(dict(summary=summary, res=res, cohort=cohort, self_share=share), fh)
    with open(os.path.join(RESULTS, f'real_tstep_{tag}.json'), 'w') as fh:
        json.dump(summary, fh, indent=1)
    log(f'wrote data/results/real_tstep_{tag}.json')


if __name__ == '__main__':
    main()
