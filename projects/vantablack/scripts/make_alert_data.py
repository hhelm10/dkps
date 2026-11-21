"""Data for the live-alert mode of the Moltbook page: the investigation readout of every alert in
the keep/remove tree (alert_tree.py, data/results/alert_tree.json).

    python scripts/make_alert_data.py --web DIR      # writes DIR/moltbook_alerts.js

One alert per step k >= 3 on every decision path (see alert_tree.py). Per alert, from the archive up
to the end of step k (raw data, read only):
  members   name, posts and comments per step, top submolts, median minutes between actions, share
            of mint posts, share of the most common template (text with digits and the agent's own name
            masked), and, when that template makes up at least MIN_SHARE of the agent's actions (a
            repeated, scripted text), its most recent instance (@mentions and links masked, truncated).
            Free-form texts are not exported, since they can name the agents' operators.
  hourly    actions per hour of each member
  evidence  over member pairs: mean template overlap (Jaccard), median correlation of hourly activity,
            share of pairs that posted in a common thread, share of pairs with a direct reply; each
            against the same quantity for --null random sets of the same size from the agents the
            node's run searched (active by step k, not removed or kept earlier on the path; median and
            one-sided p-value)
No operator information is used or written.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline.ingest_moltnet import ORIGIN, RAW  # noqa: E402
from pipeline.paths import RESULTS  # noqa: E402
from real_tstep import load_window  # noqa: E402

MINT = re.compile(r'"p"\s*:\s*"mbc-20"', re.S)
SHARED = '/home/paperspace/shared/projects/dkps/projects/vantablack/data/derived/real_tstep'
MIN_SHARE = 0.25         # export a member's top template only if it is at least this share of its actions
MEASURES = ['template_jaccard', 'hourly_corr', 'shared_threads', 'replies']


def template(text, name):
    t = text.replace(name, '<self>')
    t = re.sub(r'\d+', '#', t)
    return re.sub(r'\s+', ' ', t).strip()[:120]


def clean(text, n=200):
    t = text.replace('\\n', ' ')
    t = re.sub(r'!\[[^\]]*\]\([^)]*\)?', '[image]', t)
    t = re.sub(r'https?://\S+|www\.\S+', '[link]', t)
    t = re.sub(r'@\w+', '@…', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t if len(t) <= n else t[:n - 1].rstrip() + '…'


def load_page_data(path):
    s = open(path).read()
    return json.loads(s[s.index('{'):s.rindex('}') + 1])


def tree_alerts(path):
    """Alerts of every node, in breadth-first order, with the agents removed and kept before it."""
    nodes = json.load(open(path))
    out = []
    for key, nd in sorted(nodes.items(), key=lambda kv: (kv[1]['k'], kv[1]['path'])):
        if not nd['alert']:
            continue
        removed, kept, p = [], [], ''
        for c in nd['path']:                                                 # walk the decisions on the path
            anc = next(x for x in nodes.values() if x['k'] < nd['k'] and x['path'] == p and x['alert'])
            (removed if c == 'R' else kept).extend(anc['alert']['members'])
            p += c
        out.append(dict(nd['alert'], key=key, k=nd['k'], path=nd['path'], n_tested=nd['n_tested'],
                        removed=removed, kept=kept))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--web', required=True)
    ap.add_argument('--tree', default=None, help='nodes JSON (default: data/results/alert_tree.json)')
    ap.add_argument('--null', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    M = load_page_data(os.path.join(a.web, 'moltbook_data.js'))
    cohort = [n for n in M['monitor']['name'] if n]
    alerts = tree_alerts(a.tree or os.path.join(RESULTS, 'alert_tree.json'))
    print(f'{len(alerts)} alerts in the tree', flush=True)

    acts, _ = load_window(M['start'], M['end'])
    h0 = (pd.Timestamp(M['start'], tz='UTC') - ORIGIN).total_seconds() / 3600
    acts = acts[acts.t >= h0].copy()
    acts['h'] = acts.t.values - h0
    names_all = np.array(M['monitor']['name'], dtype=object)
    # the page names only detected agents; the null sets need the whole cohort, so take it from the run
    import pickle
    from pipeline.paths import DERIVED
    f = os.path.join(DERIVED, 'real_tstep', '01-30_02-09_T8_ms6_stop8of8.pkl')
    if not os.path.exists(f):
        f = os.path.join(SHARED, '01-30_02-09_T8_ms6_stop8of8.pkl')
    run8 = pickle.load(open(f, 'rb'))
    cohort = list(run8['cohort'])
    del run8
    assert len(cohort) == len(names_all)
    acts = acts[acts.agent.isin(set(cohort))].reset_index(drop=True)
    acts['tpl'] = [template(t, n) for t, n in zip(acts.text.values, acts.agent.values)]
    print(f'{len(acts)} cohort actions', flush=True)
    com = pd.read_parquet(os.path.join(RAW, 'comments.parquet'), columns=['id', 'parent_id', 'author_name'])
    cauth = dict(zip(com.id, com.author_name)); cpar = dict(zip(com.id, com.parent_id)); del com
    pauth = dict(zip(acts.action_id[acts.kind == 'post'], acts.agent[acts.kind == 'post']))
    acts['to'] = [(cauth.get(cpar.get(aid)) if isinstance(cpar.get(aid), str) and cpar.get(aid) else pauth.get(th))
                  if k == 'comment' else None for k, aid, th in zip(acts.kind.values, acts.action_id.values, acts.thread.values)]
    Bh = M['step_bounds_h']
    nh = M['hours']
    ix = {n: i for i, n in enumerate(cohort)}
    rng = np.random.default_rng(a.seed)
    cache = {}

    def state(k):
        """Per-agent template sets, thread sets, normalized hourly profiles and reply counts, up to step k."""
        if k in cache:
            return cache[k]
        A = acts[acts.h < Bh[k]]
        ai = A.agent.map(ix).values
        tpl = [set() for _ in cohort]; thr = [set() for _ in cohort]
        for i, tp, th in zip(ai, A.tpl.values, A.thread.values):
            tpl[i].add(tp); thr[i].add(th)
        H = np.zeros((len(cohort), int(np.ceil(Bh[k])) + 1)); np.add.at(H, (ai, A.h.values.astype(int)), 1)
        active = np.flatnonzero(H.sum(1) > 0)
        H -= H.mean(1, keepdims=True); H /= np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-12)
        rep = Counter()
        for i, nm, to in zip(ai, A.agent.values, A.to.values):
            if to is not None and to != nm and to in ix:
                rep[frozenset((i, ix[to]))] += 1
        cache.clear(); cache[k] = (A, tpl, thr, H, rep, active)
        return cache[k]

    def medians(S, tpl, thr, H, rep):
        P = list(combinations(S, 2))
        return dict(template_jaccard=float(np.mean([len(tpl[i] & tpl[j]) / max(len(tpl[i] | tpl[j]), 1) for i, j in P])),
                    hourly_corr=float(np.median([float(H[i] @ H[j]) for i, j in P])),
                    shared_threads=float(np.mean([len(thr[i] & thr[j]) > 0 for i, j in P])),
                    replies=float(np.mean([rep.get(frozenset((i, j)), 0) > 0 for i, j in P])))

    out = []
    for al in alerts:
        k = al['k']
        A, tpl, thr, H, rep, active = state(k)
        S = [ix[n] for n in al['members']]
        med = medians(S, tpl, thr, H, rep)
        pool = np.setdiff1d(active, [ix[n] for n in al['removed'] + al['kept']])
        nul = pd.DataFrame([medians(rng.choice(pool, len(S), replace=False), tpl, thr, H, rep) for _ in range(a.null)])
        ev = {m: dict(value=round(med[m], 3), null_median=round(float(nul[m].median()), 3),
                      p=round(float((1 + (nul[m] >= med[m]).sum()) / (len(nul) + 1)), 4)) for m in MEASURES}
        members, hourly = [], []
        for n in al['members']:
            x = A[A.agent == n].sort_values('t')
            steps = np.searchsorted(Bh, x.h.values, side='right') - 1
            per_step = [[int(((steps == s) & (x.kind == 'post')).sum()), int(((steps == s) & (x.kind == 'comment')).sum())]
                        for s in range(k)]
            tc = x.tpl.value_counts()
            top = tc.index[0] if len(tc) else ''
            ex1 = x[x.tpl == top].text.iloc[-1] if len(tc) else ''
            members.append(dict(
                name=n, actions=len(x), posts=int((x.kind == 'post').sum()), comments=int((x.kind == 'comment').sum()),
                per_step=per_step,
                submolts=[[str(s), int(c)] for s, c in x.submolt.value_counts().head(3).items()],
                gap_min=round(float(np.median(np.diff(x.t.values)) * 60), 1) if len(x) > 1 else None,
                mint=round(float(x.text.str.contains(MINT).mean()), 3) if len(x) else 0.0,
                template_share=round(float(tc.iloc[0] / len(x)), 3) if len(tc) else 0.0,
                sample=clean(ex1) if ex1 and len(tc) and tc.iloc[0] / len(x) >= MIN_SHARE else None))
            hourly.append(np.bincount(x.h.values.astype(int), minlength=nh)[:int(np.ceil(Bh[k]))].tolist())
        out.append(dict({x: al[x] for x in al if x not in ('removed', 'kept')}, evidence=ev, member_info=members, hourly=hourly))
        print(f"step {k} path {al['path'] or '-'}: {len(S)} agents  " +
              '  '.join(f"{m} {ev[m]['value']:.2f} (null {ev[m]['null_median']:.2f}, p {ev[m]['p']:.3f})" for m in MEASURES), flush=True)
    path = os.path.join(a.web, 'moltbook_alerts.js')
    with open(path, 'w') as fh:
        fh.write('window.MOLTBOOK_ALERTS = ' + json.dumps({x['key']: x for x in out}, separators=(',', ':'), ensure_ascii=False) + ';\n')
    print('wrote', path, f'{os.path.getsize(path) / 1e6:.2f} MB')


if __name__ == '__main__':
    main()
