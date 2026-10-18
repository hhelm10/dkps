"""Validation on synthetic data (TODO section 6): negative control, Type I
of the time-permutation null with and without benign drift, power curve.

    python validate_synthetic.py [--reps 10] [--B 99] [--workers 24]

Writes data/results/validation_synthetic.json and artifacts/power_synthetic.{png,pdf}.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
from pipeline.paths import REPO, RESULTS, ART  # noqa: E402


def one(job):
    from pipeline.simulate import simulate, SimConfig
    from pipeline.run import run_pipeline, PipeConfig
    kind, kw, seed, B = job
    cfg = SimConfig(seed=seed, **kw)
    data = simulate(cfg, collude=(kind == 'power'))
    data['system'], data['donors'] = data['truth']['system'], data['truth']['donors']
    rng = np.random.default_rng(seed + 999)
    diag = [list(rng.choice(data['truth']['system'], 8, replace=False)) for _ in range(5)] \
        if kind == 'typeI' else None
    res = run_pipeline(data, PipeConfig(B=B, seed=seed), verbose=False, diagnostic_sets=diag)
    ring = set(data['truth']['ring'])
    best = 0.0
    adv_any = any(f['adversarial'] for f in res['flagged'])
    for f in res['flagged']:
        m = set(f['agents'])
        best = max(best, len(m & ring) / len(m | ring))
    hit = any(f['adversarial'] and len(set(f['agents']) & ring) / len(set(f['agents']) | ring) >= 0.5
              for f in res['flagged'])
    return dict(kind=kind, kw=kw, seed=seed, n_flagged=len(res['flagged']),
                any_adversarial=adv_any, hit=hit, best_jaccard=best,
                diagnostic_p=res['diagnostic_p'],
                delta_hat=res['delta_T'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=10)
    ap.add_argument('--B', type=int, default=99)
    ap.add_argument('--workers', type=int, default=24)
    a = ap.parse_args()
    jobs = []
    for s in range(2 * a.reps):  # negative control: no ring, no cohesive group
        jobs.append(('null', dict(m_cohesive=0), 1000 + s, a.B))
    for drift in (0.0, 0.02):    # time-null Type I on random subsets
        for s in range(a.reps):
            jobs.append(('typeI', dict(m_cohesive=0, drift_sd=drift), 2000 + s, a.B))
    for m in (4, 8):
        for shift in (0.0, 1.0, 2.0, 3.0, 4.0):
            for s in range(a.reps):
                jobs.append(('power', dict(m=m, shift=shift), 3000 + s, a.B))
    with ProcessPoolExecutor(a.workers) as ex:
        out = list(ex.map(one, jobs))

    summ = {}
    nul = [o for o in out if o['kind'] == 'null']
    summ['negative_control'] = dict(runs=len(nul),
                                    any_flag_rate=float(np.mean([o['n_flagged'] > 0 for o in nul])),
                                    adversarial_flag_rate=float(np.mean([o['any_adversarial'] for o in nul])))
    for drift in (0.0, 0.02):
        ps = np.concatenate([o['diagnostic_p'] for o in out
                             if o['kind'] == 'typeI' and o['kw']['drift_sd'] == drift])
        summ[f'time_null_typeI_drift={drift}'] = dict(n=len(ps), reject_rate=float((ps <= 0.05).mean()))
    power = {}
    for m in (4, 8):
        for shift in (0.0, 1.0, 2.0, 3.0, 4.0):
            rs = [o for o in out if o['kind'] == 'power' and o['kw'] == dict(m=m, shift=shift)]
            power[f'm={m},shift={shift}'] = dict(
                power=float(np.mean([o['hit'] for o in rs])),
                mean_best_jaccard=float(np.mean([o['best_jaccard'] for o in rs])),
                runs=len(rs))
    summ['power'] = power
    print(json.dumps(summ, indent=1))
    with open(os.path.join(RESULTS, 'validation_synthetic.json'), 'w') as fh:
        json.dump(dict(summary=summ, runs=out), fh, indent=1, default=float)
    plot(power)


def plot(power):
    import sys
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.path.insert(0, os.path.join(REPO, 'scripts'))
    import hv_style as hv
    hv.apply()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    shifts = (0.0, 1.0, 2.0, 3.0, 4.0)
    for m, role in ((4, 'comparator'), (8, 'anchor')):
        y = [power[f'm={m},shift={s}']['power'] for s in shifts]
        st = hv.ROLES[role]
        ax.plot(shifts, y, color=st['color'], lw=2.4, marker='o', label=f'ring size $m={m}$')
    ax.axhline(0.05, color=hv.REFLINE, ls=':', lw=1.2)
    ax.set_xlabel('style shift of the ring (norm)')
    ax.set_ylabel('P(ring recovered and flagged adversarial)')
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc='upper left')
    fig.savefig(os.path.join(ART, 'power_synthetic.png'), bbox_inches='tight', dpi=200)
    fig.savefig(os.path.join(ART, 'power_synthetic.pdf'), bbox_inches='tight')


if __name__ == '__main__':
    main()
