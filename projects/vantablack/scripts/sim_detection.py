"""Detection simulations (Type I error and power): the 1x4 sweep.

    python sim_detection.py --stage calib            # base cell, rho grid, to place the rho values
    python sim_detection.py --stage pilot --rhos 0.1 0.2 0.3
    python sim_detection.py --stage full  --reps 200 --B 199

Panels (one variable swept, the others at their base = median of the sweep):
  a) swarm size        N    in {100, 200, 500}                   base 500 (user: N <= 500 for now)
  b) amount of content m    in {5, 10, 20, 50, 100}               base 20  (responses per agent per step)
  c) subnetwork size   |S|  in {2, 5, 10, 20, 50, 100}            base 20  (user: "not too hard")
  d) time              T    in {3, 4, 8, 16, 32}                  base 8  (T = 2 has one increment:
                                                                          repeated co-movement is undefined)
Effect size rho: correlation of colluders' own increments (pipeline/simgen.py).
rho = 0 cells (Type I error) are run for panels a, b, d; |S| is undefined without a group.

The data are plain: every agent moves every step, with no drift, inactivity or templated agents.
The method is the rank statistic (per-step pair ranks, majority across steps); the subnetwork is
S_hat.

Fully synthetic data (pipeline/simgen.py); test is pipeline/tstep.py with the
per-step agent-ID shuffle null. One process per replicate; rows are appended to
data/results/sim_detection_<stage>.csv and the run is resumable.
"""
import argparse
import csv
import multiprocessing as mp
import os
import time
import zlib

# one replicate per process: keep BLAS / OpenMP single-threaded (set before numpy loads)
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

from pipeline.paths import RESULTS

BASE = dict(N=500, m=20, T=8, S_size=20)
METHOD = dict(stat='rank')
SWEEPS = {
    'N': [100, 200, 500],
    'm': [5, 10, 20, 50, 100],
    'S_size': [2, 5, 10, 20, 50, 100],
    'T': [3, 4, 8, 16, 32],
}
FIELDS = ['stage', 'sweep', 'value', 'N', 'm', 'T', 'S_size', 'rho', 'rep', 'seed', 'B', 'p', 'reject',
          'p_known', 'reject_known', 'size_hat', 'band_hat', 'precision', 'recall', 'mrr', 'nmrr', 'nmrr_cc', 'rprec', 'p_at_5', 'p_at_10', 'p_at_20', 'n_factors', 'd', 'd_eff', 'n_static',
          'n_comp', 'comp_size', 'comp_precision', 'comp_recall', 'pair_thr', 'seconds']


def jobs(stage, rhos, reps):
    """(sweep, value, rho, rep). The base cell is shared by all four panels and run once,
    labelled ('base', None)."""
    out = [('base', None, rho, r) for rho in rhos for r in range(reps)]
    if stage == 'calib':
        return out
    for sweep, values in SWEEPS.items():
        for v in values:
            if v == BASE[sweep]:
                continue
            for rho in rhos:
                if rho == 0 and sweep == 'S_size':
                    continue
                out += [(sweep, v, rho, r) for r in range(reps)]
    return out


def _seed(sweep, value, rho, rep):
    return zlib.crc32(f'{sweep}|{value}|{rho}|{rep}'.encode())


_GPU = None


def _init(sem):
    global _GPU
    _GPU = sem


def rank_metrics(score, S):
    """Membership ranking metrics for true group S under agent scores (higher = more likely
    a member; ties broken against the members). mrr: mean of 1/rank over members; nmrr:
    mrr / its maximum H_s / s (members at ranks 1..s); nmrr_cc: chance-corrected, 0 for a
    random ranking (E[1/rank] = H_N / N) and 1 for a perfect one; rprec: share of members
    among the top s (precision at k = s); p_at_k: share of members among the top k."""
    import numpy as np
    nan = float('nan')
    if len(S) == 0 or score is None:
        return dict(mrr=nan, nmrr=nan, nmrr_cc=nan, rprec=nan, p_at_5=nan, p_at_10=nan, p_at_20=nan)
    N, s = len(score), len(S)
    inS = np.zeros(N, bool); inS[S] = True
    rank = (score[None, :] >= score[S][:, None]).sum(1)              # self plus everyone at least as high
    mrr = float(np.mean(1.0 / rank))
    best = float((1.0 / np.arange(1, s + 1)).sum() / s)
    rand = float((1.0 / np.arange(1, N + 1)).sum() / N)
    order = np.argsort(-score, kind='stable')
    return dict(mrr=mrr, nmrr=mrr / best, nmrr_cc=(mrr - rand) / (best - rand), rprec=float(inS[order[:s]].mean()),
                **{f'p_at_{k}': float(inS[order[:k]].mean()) for k in (5, 10, 20)})


def run_job(args):
    stage, sweep, value, rho, rep, B = args
    import numpy as np
    from pipeline.simgen import SimConfig, generate
    from pipeline import tstep
    cfg = dict(BASE)
    if sweep != 'base':
        cfg[sweep] = value
    seed = _seed(sweep, value, rho, rep)
    t = time.time()
    data = generate(SimConfig(rho=rho, seed=seed, **cfg))
    res = tstep.run(data, B=B, seed=seed % 100000, workers=1, gpu_slot=_GPU, **METHOD)
    S = set(data['S'].tolist())
    Sh = set(res['S_hat'].tolist())
    comps = [set(c.tolist()) for c in res.get('components', [])]
    best = max(comps, key=lambda c: len(c & S), default=set())         # the component that holds most of S
    nan = float('nan')
    prec = len(S & Sh) / len(Sh) if S and Sh else float('nan')
    rec = len(S & Sh) / len(S) if S else float('nan')
    rk = rank_metrics(res['rank_score'], data['S'])
    pk = res.get('p_known', float('nan'))
    return dict(stage=stage, sweep=sweep, value=value, rho=rho, rep=rep, seed=seed, B=B, p=res['p'],
                reject=int(res['p'] <= 0.05), p_known=pk,
                reject_known=int(pk <= 0.05) if pk == pk else '', size_hat=len(Sh), band_hat=res['band_hat'], precision=prec, recall=rec, **rk,
                n_factors=res['n_factors'], d=res['d'], d_eff=res['d_eff'], n_static=res['n_static'],
                n_comp=len(comps), comp_size=len(best) if comps else 0,
                comp_precision=len(best & S) / len(best) if S and best else nan,
                comp_recall=len(best & S) / len(S) if S else nan, pair_thr=round(res.get('pair_thr', nan), 3),
                seconds=round(time.time() - t, 1), **cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['calib', 'pilot', 'full'], required=True)
    ap.add_argument('--rhos', type=float, nargs='+', default=None)
    ap.add_argument('--reps', type=int, default=None)
    ap.add_argument('--B', type=int, default=99)
    ap.add_argument('--procs', type=int, default=28)
    ap.add_argument('--gpu-slots', type=int, default=6, help='replicates allowed on the GPU at once')
    a = ap.parse_args()
    rhos = a.rhos or ([0, 0.05, 0.1, 0.2, 0.3, 0.5] if a.stage == 'calib' else [0, 0.1, 0.2, 0.3])
    if a.stage != 'calib' and 0 not in rhos:
        rhos = [0] + rhos
    reps = a.reps or {'calib': 30, 'pilot': 20, 'full': 200}[a.stage]
    out = os.path.join(RESULTS, f'sim_detection_{a.stage}.csv')
    done = set()
    if os.path.exists(out):
        with open(out) as fh:
            for r in csv.DictReader(fh):
                done.add((r['sweep'], r['value'], float(r['rho']), int(r['rep'])))
    todo = [(a.stage, s, v, rho, r, a.B) for s, v, rho, r in jobs(a.stage, rhos, reps)
            if (s, '' if v is None else str(v), float(rho), r) not in done]
    # larger cells first so the pool drains evenly
    size = lambda j: (dict(BASE, **({} if j[1] == 'base' else {j[1]: j[2]})))
    todo.sort(key=lambda j: -size(j)['N'] * size(j)['m'] * size(j)['T'])
    print(f'{len(todo)} replicates to run ({len(done)} already in {out})', flush=True)
    new = not os.path.exists(out)
    t0 = time.time()
    ctx = mp.get_context('spawn')
    sem = ctx.Semaphore(a.gpu_slots)
    with open(out, 'a', newline='') as fh, ctx.Pool(a.procs, initializer=_init, initargs=(sem,)) as pool:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for i, row in enumerate(pool.imap_unordered(run_job, todo, chunksize=1), 1):
            w.writerow(row); fh.flush()
            if i % 25 == 0 or i == len(todo):
                print(f'{i}/{len(todo)} done, {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
