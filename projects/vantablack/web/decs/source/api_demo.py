"""Synthetic API demo for the web page: agents calling a public API, with planted colluding groups.

Fully synthetic. Each scenario is the paper's simulation generator (pipeline/simgen.py) with its own
settings; the method is the paper's (pipeline/tstep.py, rank statistic, peeling), run once per
stopping point on the calls made up to it, exactly as for Moltbook (make_web_data.py).

How the generator reads as API traffic. Each call is a pair: what the call is about (the resource
or task it targets; the provider estimates it from the request, with error: the query) and how the
agent makes it (its parameters, phrasing and follow-ups: the response). An agent's perspective is
how it tends to call the API about a given resource. Colluding agents take direction from a shared
coordinator, so their perspectives change together from one time step to the next (the group
process); everyone else changes independently.

Scenarios (seed 0 for the page; seeds 1-3 for the pilot, never the page's):
  scrape    two crews of 15 workers each split a crawl of the catalog; every client keeps to its
            own part of the catalog (pref), so the workers rarely touch the same resources
  distill   40 agents harvest a model API's outputs to train another model; a shared curriculum
            moves them from skill to skill (a larger group, more weakly coordinated)
  incident  12 agents in an evaluation environment open a side channel partway through the run
            and divide labor from then on (collusion starts with the move from step 6 to step 7 of 16, minute 18: onset)

    python api_demo.py --pilot            # seeds 1-3, full data only: detection per scenario
    python api_demo.py                    # seed 0, every stopping point -> web/decs/api_data.js
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline.simgen import SimConfig, generate  # noqa: E402
from pipeline import tstep  # noqa: E402

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, 'web', 'decs', 'api_data.js')
BASE = dict(N=1000, T=16, m=40, m_sd=1.0, m_step_sd=0.7, m_time_sd=0.35)  # bursty, unequal activity; 16 steps
# of 3 minutes. With 8 steps of ~20 calls, membership was data-limited (oracle Jaccard 0.4-0.8 on seeds 1-2): the
# colluders missed were the least active. 16 steps of ~40 calls give the method the evidence to place them.
SCENARIOS = {                   # effect sizes from the paper's power sweeps (N = 1,000: |S| = 20 needs rho ~ 0.5)
    'scrape': dict(n_groups=2, S_size=15, rho=0.5, pref=1.0),
    'distill': dict(n_groups=1, S_size=40, rho=0.35),           # 0.25 missed on the page's seed (oracle p 0.005)
    'incident': dict(n_groups=1, S_size=12, rho=0.85, onset=6),  # minute 18, as before: 10 of 15 increments
}
STOPS = range(4, 17, 2)           # a saved run every 6 minutes, from minute 12
MIN_PER_STEP = 3                  # time is in minutes: 16 steps of 3 minutes
B = 199


def truncate(data, k):
    """The calls made in the first k steps, as a dataset of T = k steps."""
    keep = data['step'] < k
    out = dict(data, agent=data['agent'][keep], step=data['step'][keep], U_obs=data['U_obs'][keep],
               X=data['X'][keep], cfg=dict(data['cfg'], T=k))
    return out


def run(data, seed, workers):
    return tstep.run(data, B=B, seed=seed, workers=workers, peel=10, keep_arrays=True)


def found_sets(res, alpha=0.05):
    return [set(g['members'].tolist()) for g in res['groups'] if g['p'] <= alpha and g['subnetwork']]


def call_times(data, seed, burst=1.5):
    """A time in minutes (float, from the start) for every call, within its step: each 10-second slot
    gets a random intensity (gamma, shape 1 / burst^2: coefficient of variation burst), so traffic
    comes in bursts. Steps are MIN_PER_STEP long."""
    rng = np.random.default_rng(seed + 7)
    T = int(data['step'].max()) + 1
    slots = MIN_PER_STEP * 6                                           # 10-second slots per step
    w = rng.gamma(1 / burst ** 2, burst ** 2, size=(T, slots))
    w /= w.sum(1, keepdims=True)
    cum = w.cumsum(1)
    st = data['step']
    j = np.minimum((cum[st] < rng.random(len(st))[:, None]).sum(1), slots - 1)
    return st * MIN_PER_STEP + (j + rng.random(len(st))) / 6


def export_scenario(name, workers, seed=0):
    import make_web_data as mw
    cfg = SimConfig(seed=seed, **BASE, **SCENARIOS[name])
    data = generate(cfg)
    N, T = cfg.N, cfg.T
    names = np.array([f'agent-{i:05d}' for i in range(N)])
    h = call_times(data, seed)
    H = T * MIN_PER_STEP                                               # page time unit: minutes
    per_hour = np.bincount(np.floor(h).astype(int), minlength=H)
    first = np.full(N, H); np.minimum.at(first, data['agent'], np.floor(h).astype(int))
    start = np.datetime64('2026-01-01T00:00')                        # nominal: the page shows elapsed time
    iso = lambda mn: str(start + np.timedelta64(int(round(mn * 60)), 's')) + '+00:00'
    groups = [names[g].tolist() for g in data['groups']]
    sc = dict(name=name, synthetic=True, unit='minute', start=str(start)[:10],
              hours=H, per_hour=per_hour.tolist(),                     # 'hours', 'per_hour': time bins (minutes here)
              new_agents=np.bincount(first, minlength=H)[:H].tolist(), n_actions=int(len(h)), n_agents=N,
              step_bounds_h=[float(t * MIN_PER_STEP) for t in range(T + 1)],
              config=dict(cfg.__dict__), truth=groups, stops={})
    full = None
    for k in STOPS:
        t = time.time()
        dk = truncate(data, k)
        res = run(dk, seed, workers)
        obs = np.bincount(dk['agent'] * k + dk['step'], minlength=N * k).reshape(N, k) > 0
        st, extra = mw.export_result(res, names, obs, k=k, B=B,
                                     step_bounds=[iso(t * MIN_PER_STEP) for t in range(k + 1)],
                                     actions=int(len(dk['agent'])))
        for sb in st['subnetworks']:                                   # ground truth, for the reveal
            sb['stem'] = None                                          # synthetic names carry no information
            mem = {m['name'] for m in sb['members']}
            hits = [len(mem & set(g)) for g in groups]
            j = int(np.argmax(hits)) if groups else -1
            sb['truth'] = dict(group=j + 1 if j >= 0 and hits[j] else None, hits=int(max(hits, default=0)),
                               false=int(len(mem) - sum(hits)))
        sc['stops'][k] = st
        print(f'{name} stop {k}: subnetworks {[(sb["size"], sb["truth"]) for sb in st["subnetworks"]]}, '
              f'{time.time() - t:.0f}s', flush=True)
        if k == T:
            full = extra
    nm, grp, Zm, obs = full
    sc['monitor'] = mw.monitor(nm, Zm, obs, sc['stops'], show=[n for g in groups for n in g])
    return sc


def pilot(workers, names):
    jac = lambda a, b: len(a & b) / len(a | b)
    for name, sc in ((n, SCENARIOS[n]) for n in names):
        for seed in (1, 2, 3):
            t = time.time()
            data = generate(SimConfig(seed=seed, **BASE, **sc))
            res = run(data, seed, workers)
            F = found_sets(res)
            G = [set(g.tolist()) for g in data['groups']]
            best = [max((jac(f, g) for f in F), default=0.0) for g in G]
            n_false = sum(1 for f in F if not any(f & g for g in G))
            print(f'{name} seed {seed}: found {len(F)} (sizes {[len(f) for f in F]}); best Jaccard per planted '
                  f'group {[round(b, 2) for b in best]}; found sets with no colluder {n_false}; '
                  f'{time.time() - t:.0f}s', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--only', nargs='*', default=list(SCENARIOS))
    a = ap.parse_args()
    if a.pilot:
        return pilot(a.workers, a.only)
    out = {}
    if os.path.exists(OUT):                                            # keep scenarios not rerun
        s = open(OUT).read()
        out = json.loads(s[s.index('{'):s.rstrip().rstrip(';').rindex('}') + 1])['scenarios']
    for name in a.only:
        out[name] = export_scenario(name, a.workers)
        with open(OUT, 'w') as fh:
            fh.write('window.API_DEMO = ' + json.dumps(dict(scenarios=out), separators=(',', ':')) + ';\n')
        print('wrote', OUT, f'{os.path.getsize(OUT) / 1e6:.2f} MB', flush=True)


if __name__ == '__main__':
    main()
