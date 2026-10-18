"""Simulation experiments, one group per problem (P0 detection, P1 membership).

Usage
  python run_grid.py --list                          # cells with indices and experiment tags
  python run_grid.py --exp E1 --workers 32           # one experiment group -> results_E1.csv
  python run_grid.py --cell 7                        # one cell (array job) -> parts/cell_7.csv; concatenate per experiment afterwards
Env/flags: R replicates per cell (default 60), B permutations (default 100; the moment set is max(B/2,30)).
Results CSVs are resumable (existing (cell, seed) rows are skipped).
"""
import argparse, csv, os, sys, time
from multiprocessing import Pool
from colsim import SimConfig, run_once, run_sequential

base = dict(N=50, S_size=5, m=10.0, sigma_q=1.0, alpha=1.0)   # vantablack base: estimated queries
DELTAS = [0.0, 0.5, 1.0, 1.5]

def cells():
    out = []
    # ---------------- E1: detection (P0). Type I error, power, correct detection vs. one factor at a time.
    for delta in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        out.append(('E1', 'A_delta', dict(base, delta=delta)))
    for N in [20, 100, 200]:
        for delta in DELTAS: out.append(('E1', 'B_N', dict(base, N=N, delta=delta)))
    for m in [3.0, 5.0, 20.0, 40.0]:
        for delta in DELTAS: out.append(('E1', 'C_m', dict(base, m=m, delta=delta)))
    for S_size in [2, 3, 10, 15]:
        for delta in DELTAS[1:]: out.append(('E1', 'D_Ssize', dict(base, S_size=S_size, delta=delta)))
    for sq in [0.0, 0.5, 2.0, 4.0]:                                     # 0 = black-box reference
        for delta in DELTAS: out.append(('E1', 'E_sigma_q', dict(base, sigma_q=sq, delta=delta)))
    for al in [0.1, 0.2, 5.0]:
        for delta in DELTAS[1:]: out.append(('E1', 'F_alpha', dict(base, alpha=al, delta=delta)))
    for dr in [0.5, 1.0]:
        for delta in DELTAS: out.append(('E1', 'G_drift', dict(base, drift=dr, delta=delta)))
    # ---------------- E2: membership (P1). Jaccard / precision / recall / nomination AUC.
    for S_size in [3, 5, 10, 15]:
        for delta in [0.5, 1.0, 1.5, 2.0, 3.0]: out.append(('E2', 'H_Ssize', dict(base, S_size=S_size, delta=delta)))
    for S2 in [3, 5]:
        for delta in [1.0, 1.5, 2.0]: out.append(('E2', 'I_two', dict(base, S2_size=S2, delta=delta)))
    for N, S_size in [(50, 5), (100, 10), (200, 20)]:
        for delta in [1.0, 1.5, 2.0]: out.append(('E2', 'J_scale', dict(base, N=N, S_size=S_size, delta=delta)))
    for m in [5.0, 20.0, 40.0]:
        for delta in [1.0, 1.5]: out.append(('E2', 'K_m', dict(base, m=m, delta=delta)))
    return [c if len(c) == 4 else (c[0], c[1], c[2], None) for c in out]

def key_of(cell, seed):
    exp, grid, cfgd, seqd = cell
    s = seqd or {}
    return (grid, str(cfgd['N']), str(cfgd['S_size']), str(float(cfgd['delta'])), str(float(cfgd['m'])),
            str(float(cfgd['sigma_q'])), str(float(cfgd['alpha'])), str(float(cfgd.get('drift', 0.0))),
            str(cfgd.get('S2_size', 0)), str(s.get('L', '')), str(s.get('ramp', '')), str(seed))

def job(args):
    cell, seed, B, B_seq = args
    exp, grid, cfgd, seqd = cell
    cfg = SimConfig(**cfgd, seed=seed)
    if seqd is None:
        r = run_once(cfg, seed=seed, B=B, B_mom=max(B // 2, 30))
    else:
        r = run_sequential(cfg, seed=seed, L=seqd['L'], t_star=seqd['t_star'], ramp=seqd['ramp'], B=B_seq, B_mom=max(B_seq // 4, 30))
    r['exp'] = exp; r['grid'] = grid
    return r

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', type=int); ap.add_argument('--exp', choices=['E1', 'E2'])
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--out', default=None, help='default: results_<exp>.csv, or parts/cell_<k>.csv with --cell')
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--R', type=int, default=int(os.environ.get('R', 60)))
    ap.add_argument('--B', type=int, default=int(os.environ.get('B', 100)))
    ap.add_argument('--B_seq', type=int, default=int(os.environ.get('B_SEQ', 500)))
    a = ap.parse_args()
    allcells = cells()
    if a.list:
        for k, c in enumerate(allcells): print(k, c[0], c[1], c[2], c[3] or '')
        sys.exit(0)
    if a.cell is None and a.exp is None:
        sys.exit('specify --exp E1|E2 or --cell k (results for different experiments have different columns)')
    todo = [allcells[a.cell]] if a.cell is not None else [c for c in allcells if c[0] == a.exp]
    here = os.path.dirname(os.path.abspath(__file__))
    if a.out is None:
        a.out = os.path.join(here, f'parts/cell_{a.cell}.csv') if a.cell is not None else os.path.join(here, f'results_{a.exp}.csv')
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            for row in csv.DictReader(f):
                done.add((row['grid'], row['N'], row['S_size'], row['delta'], row['m'], row['sigma_q'], row['alpha'],
                          row.get('drift', '0.0'), row.get('S2_size', '0'), row.get('L', ''), row.get('ramp', ''), row['seed']))
    jobs = [(c, s, a.B, a.B_seq) for c in todo for s in range(a.R) if key_of(c, s) not in done]
    print(f"{len(jobs)} jobs to run with {a.workers} workers", flush=True)
    t0 = time.time(); fields = None
    with Pool(a.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(job, jobs, chunksize=2)):
            new = not os.path.exists(a.out)
            with open(a.out, 'a', newline='') as f:
                if fields is None:
                    fields = list(r.keys()) if new else next(csv.reader(open(a.out)))
                w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
                if new: w.writeheader()
                w.writerow({**{k_: '' for k_ in fields}, **r})
            if (k + 1) % 50 == 0: print(f"{k+1}/{len(jobs)} done, {(time.time()-t0)/60:.1f} min", flush=True)
    print("finished", flush=True)
