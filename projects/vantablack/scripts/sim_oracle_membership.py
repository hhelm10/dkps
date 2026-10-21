"""Oracle membership for the detection simulations: the membership rule of the method
(tstep.membership: score against the set, two-groups model, posterior > 1/2) applied with the
true group as the set, i.e. with S known. For the number-of-subnetworks panel, one pass per true
group, precision and recall averaged over the groups.

    python sim_oracle_membership.py [--stage pilot] [--procs 28]

Regenerates each replicate of data/results/sim_detection_<stage>.csv from its seed (the same
data and pipeline as sim_detection.run_job; no shuffles are needed) and writes
data/results/sim_oracle_membership_<stage>.csv (sweep, value, rho, rep, oracle_precision,
oracle_recall). Resumable.
"""
import argparse
import csv
import multiprocessing as mp
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

from pipeline.paths import RESULTS

FIELDS = ['sweep', 'value', 'rho', 'rep', 'oracle_precision', 'oracle_recall', 'oracle_size']
_GPU = None


def _init(sem):
    global _GPU
    _GPU = sem


def run_job(args):
    sweep, value, rho, rep = args
    import contextlib
    import numpy as np
    from pipeline.simgen import SimConfig, generate
    from pipeline import tstep
    import sim_detection as sd
    cfg = dict(sd.BASE)
    v = None if value in ('', None) else int(float(value))
    if sweep != 'base':
        cfg[sweep] = v
    if sweep == 'n_groups':
        cfg['S_size'] = sd.K_TOTAL // v
    seed = sd._seed(sweep, v, rho, rep)
    data = generate(SimConfig(rho=rho, seed=seed, **cfg))
    with (_GPU or contextlib.nullcontext()):
        R, _ = tstep.condition(data['U_obs'], data['X'], k=64)
    c = data['cfg']
    pos = tstep.positions(R, data['agent'], data['step'], c['N'], c['T'], seed=seed % 100000)
    D, _ = tstep.remove_factors(tstep.increments(pos))
    Q = tstep.step_scores(D)
    groups = list(data['groups']) if sweep == 'n_groups' else [data['S']]
    prec, rec, size = [], [], []
    for G in groups:
        Sh = set(tstep.membership(Q, np.asarray(G)).tolist()); G = set(G.tolist())
        prec.append(len(Sh & G) / len(Sh) if Sh else np.nan); rec.append(len(Sh & G) / len(G)); size.append(len(Sh))
    return dict(sweep=sweep, value=value, rho=rho, rep=rep,
                oracle_precision=float(np.nanmean(prec)) if np.isfinite(prec).any() else float('nan'),
                oracle_recall=float(np.mean(rec)), oracle_size=float(np.mean(size)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='pilot')
    ap.add_argument('--procs', type=int, default=28)
    ap.add_argument('--gpu-slots', type=int, default=6)
    a = ap.parse_args()
    src = os.path.join(RESULTS, f'sim_detection_{a.stage}.csv')
    out = os.path.join(RESULTS, f'sim_oracle_membership_{a.stage}.csv')
    with open(src) as fh:
        rows = [r for r in csv.DictReader(fh) if float(r['rho']) > 0]
    done = set()
    if os.path.exists(out):
        with open(out) as fh:
            done = {(r['sweep'], r['value'], float(r['rho']), int(r['rep'])) for r in csv.DictReader(fh)}
    todo = [(r['sweep'], r['value'], float(r['rho']), int(r['rep'])) for r in rows]
    todo = [j for j in todo if j not in done]
    print(f'{len(todo)} replicates to run ({len(done)} already in {out})', flush=True)
    new = not os.path.exists(out)
    ctx = mp.get_context('spawn')
    sem = ctx.Semaphore(a.gpu_slots)
    with open(out, 'a', newline='') as fh, ctx.Pool(a.procs, initializer=_init, initargs=(sem,)) as pool:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for i, row in enumerate(pool.imap_unordered(run_job, todo, chunksize=1), 1):
            w.writerow(row); fh.flush()
            if i % 50 == 0 or i == len(todo):
                print(f'{i}/{len(todo)} done', flush=True)


if __name__ == '__main__':
    main()
