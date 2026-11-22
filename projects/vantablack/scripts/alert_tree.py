"""Live-alert tree for the Moltbook page: the method run after every step k = 3, ..., 8, where each
alert's keep/remove decision changes the later runs.

    python scripts/alert_tree.py [--B 199 --workers 14] [--checks 3,6,8] [--only-root]

One alert per step: after step k the method tests the agents not yet reviewed and, if the test
rejects (p <= 0.05) and the membership estimate has at least two agents, raises one alert for it.
The viewer then decides:
  remove  the agents are taken out of Moltbook: their responses are dropped from every later run
          (perspectives, increments and centering are computed without them);
  keep    the agents stay: they remain in the perspectives, increments and centering of later runs,
          but are left out of the search, as the paper's peeling sets detected agents aside.
Every decision path is run (a binary tree over the steps with an alert; a step without an alert
does not branch). Each node reuses the saved PKPS-conditioned responses of the stop-k run on the
reference run's cohort (real_tstep.py --stop-step k --cohort-tag 01-30_02-09_T8_ms6; read only),
with that run's settings (old candidate family, family='z'), so nothing is re-embedded. With no
decisions (the root), a node reproduces round 1 of the saved run.

--checks: the steps after which the monitor checks (default: every step from 3 to 8). With fewer checks
the decisions at one check carry over to the next.

Writes data/derived/alert_tree[/checks_<list>]/<k>_<path>.json per node (path: one letter per earlier alert,
R = removed, K = kept; resumable) and data/results/alert_tree[_<list>].json (all nodes).
"""
import argparse
import json
import os
import pickle
import sys
import time

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import tstep  # noqa: E402
from pipeline.paths import DERIVED, RESULTS  # noqa: E402
from make_web_data import stem  # noqa: E402

SRC = '/home/paperspace/shared/projects/dkps/projects/vantablack/data/derived/real_tstep'
REF = '01-30_02-09_T8_ms6'
TAGS = {3: f'01-30_02-09_T3_stop3of8_coh-{REF}', 4: f'01-30_02-09_T4_ms3_stop4of8_coh-{REF}',
        5: f'01-30_02-09_T5_ms4_stop5of8_coh-{REF}', 6: f'01-30_02-09_T6_ms5_stop6of8_coh-{REF}',
        7: f'01-30_02-09_T7_ms6_stop7of8_coh-{REF}', 8: '01-30_02-09_T8_ms6_stop8of8'}
OUT = os.path.join(DERIVED, 'alert_tree')
r2 = lambda x, n=2: None if x is None or not np.isfinite(x) else round(float(x), n)


def prepare(k, removed, kept):
    """The node's data: agent names, the searched agents (keep), increments, effective dimension, saved settings."""
    run = pickle.load(open(os.path.join(SRC, f'{TAGS[k]}.pkl'), 'rb'))
    s = run['summary']
    cohort = np.array(run['cohort'], dtype=object)
    del run
    rows = pd.read_parquet(os.path.join(SRC, f'{TAGS[k]}_rows.parquet'), columns=['agent_ix', 'step'])
    stay = np.flatnonzero(~np.isin(cohort, list(removed)))                   # agents still on Moltbook
    new_ix = np.full(len(cohort), -1); new_ix[stay] = np.arange(len(stay))
    m = new_ix[rows.agent_ix.values] >= 0
    R = np.load(os.path.join(SRC, f'{TAGS[k]}_R.npy'), mmap_mode='r')
    R = np.asarray(R[np.flatnonzero(m)], np.float32)
    agent, step = new_ix[rows.agent_ix.values[m]], rows.step.values[m]
    names = cohort[stay]
    N, T = len(names), s['T']
    tstep.FAMILY = 'z'
    # as tstep.run_residuals, with the saved run's settings
    pos = tstep.positions(R, agent, step, N, T, d=s.get('d_arg'), seed=s['seed'])
    del R
    D = tstep.increments(pos, weight=None if s['weight'] == 'none' else s['weight'],
                         min_move=s['min_move'] if s['min_move'] >= 0 else None)
    tsa = s.get('two_signed_alpha')
    D, kf = tstep.remove_factors(D, two_signed_alpha=None if tsa is None or tsa < 0 else tsa)
    d_eff = D.shape[1] * D.shape[2] - kf
    keep = np.flatnonzero(~np.isin(names, list(kept)))                       # kept agents: not searched again
    return names, keep, D, d_eff, pos, kf, s


def outsiders(k, removed, kept, members, n=3):
    """Membership scores of the n searched agents closest to a detected group, from the node's own data
    (no permutation test), and the members' scores recomputed as a check."""
    names, keep, D, *_ = prepare(k, removed, kept)
    Q = tstep.step_scores(D[keep])
    ix = {nm: i for i, nm in enumerate(names[keep])}
    S = np.array([ix[m] for m in members])
    sc = tstep.member_z(Q, S, combine='sum')[0]
    out = np.setdiff1d(np.arange(len(keep)), S)
    top = out[np.argsort(-sc[out])][:n]
    return [[str(names[keep][i]), r2(sc[i], 1)] for i in top], [r2(sc[i], 1) for i in S]


def run_node(k, removed, kept, B, workers):
    names, keep, D, d_eff, pos, kf, s = prepare(k, removed, kept)
    N = len(names)
    Q = tstep.step_scores(D[keep])
    res = tstep.test(D[keep], d_eff, B=B, seed=s['seed'], workers=workers, Q=Q)
    S, b, score = tstep.estimate_rank(Q, res)
    out = dict(k=k, N=int(N), n_tested=int(len(keep)), n_removed=len(removed), n_kept=len(kept),
               p=float(res['p']), d=int(pos['d']), n_factors=int(kf), B=B, alert=None)
    if res['p'] <= 0.05 and len(S) >= 2:
        sc = tstep.member_z(Q, S, combine='sum')[0]
        order = np.argsort(-sc[S])
        Zs = tstep._set_steps(Q[:, S][:, :, S][:, None])[:, 0]              # per-increment set score
        nm = names[keep[S]][order].tolist()
        b_ = int(b) if b is not None else int(np.nanargmax(res['obs']))
        null = res['nulls'][:, b_]; null = null[np.isfinite(null)]
        out['alert'] = dict(members=nm, scores=[r2(x, 1) for x in sc[S][order]], size=len(nm), p=r2(res['p'], 3),
                            band=b_, n_exceed=int((null >= res['obs'][b_]).sum()), n_null=int(len(null)),
                            step_scores=[r2(x) for x in Zs], n_obs=int(np.isfinite(Zs).sum()),
                            n_above=int((Zs > 2).sum()), stem=stem(nm))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=199)
    ap.add_argument('--workers', type=int, default=14)
    ap.add_argument('--only-root', action='store_true')
    ap.add_argument('--checks', default='3,4,5,6,7,8')
    ap.add_argument('--outsiders', action='store_true', help='add the nearest non-members to saved nodes')
    a = ap.parse_args()
    checks = sorted(int(x) for x in a.checks.split(','))
    tag = '' if checks == [3, 4, 5, 6, 7, 8] else '-'.join(map(str, checks))
    out = os.path.join(OUT, f'checks_{tag}') if tag else OUT
    os.makedirs(out, exist_ok=True)
    nxt = dict(zip(checks, checks[1:]))
    frontier = [(checks[0], '', (), ())]                                             # (k, path, removed, kept)
    nodes = {}
    while frontier:
        k, path, removed, kept = frontier.pop(0)
        f = os.path.join(out, f'{k}_{path or "root"}.json')
        if os.path.exists(f):
            node = json.load(open(f))
            if a.outsiders and node['alert'] and 'outsiders' not in node['alert']:
                t = time.time()
                node['alert']['outsiders'], chk = outsiders(k, set(removed), set(kept), node['alert']['members'])
                assert np.allclose(chk, node['alert']['scores'], atol=0.11), (chk, node['alert']['scores'])
                json.dump(node, open(f, 'w'), indent=1)
                print(f'  outsiders {node["alert"]["outsiders"]} ({time.time() - t:.0f}s)', flush=True)
        else:
            t = time.time()
            node = run_node(k, set(removed), set(kept), a.B, a.workers)
            node.update(path=path, seconds=round(time.time() - t))
            json.dump(node, open(f, 'w'), indent=1)
        al = node['alert']
        print(f'step {k} path {path or "-":6s} N={node["N"]} tested={node["n_tested"]} p={node["p"]:.3f} '
              + (f'alert: {al["size"]} agents {al["members"][:3]}' if al else 'no alert') + f' ({node.get("seconds")}s)', flush=True)
        nodes[f'{k}_{path}'] = node
        if a.only_root or k not in nxt:
            continue
        if al:
            frontier.append((nxt[k], path + 'R', removed + tuple(al['members']), kept))
            frontier.append((nxt[k], path + 'K', removed, kept + tuple(al['members'])))
        else:
            frontier.append((nxt[k], path, removed, kept))
    res = os.path.join(RESULTS, f'alert_tree_{tag}.json' if tag else 'alert_tree.json')
    json.dump(nodes, open(res, 'w'), indent=1)
    print(f'{len(nodes)} nodes; wrote {res}', flush=True)


if __name__ == '__main__':
    main()
