#!/usr/bin/env python
"""Leakage / leave-one-out verification for the RD1 estimator.

The reported metric is MAE against held-out FULL scores. A method is leak-free iff a model's
own prediction never depends on that model's held-out full score. We test this directly:
corrupt one model's full scores, re-run, and check its own predictions are unchanged (while a
DIFFERENT model's predictions DO change, since the corrupted model is a legitimate reference
for it). Run:  pixi run python tests/test_leakage.py
"""
import sys
import os
import collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pipeline import loaders as H
from experiments.query_efficiency import run_seed

TOL = 1e-9


def _preds(rows, model):
    return {(x['task'], x['method']): x['pred'] for x in rows if x['model'] == model}


def main():
    data = H.load_suite()
    models = data[7]
    r0 = run_seed(data, 4, 0, n_models=None, p_task=1.0, dump_cells=True)
    sm = data[6].copy(); sm[0] = 1.0 - sm[0]                      # corrupt model_0's full scores
    r1 = run_seed(tuple(sm if i == 6 else data[i] for i in range(len(data))),
                  4, 0, n_models=None, p_task=1.0, dump_cells=True)

    # (1) model_0's OWN predictions must be unchanged, per method
    a, b = _preds(r0, models[0]), _preds(r1, models[0])
    by = collections.defaultdict(float)
    for k in a:
        if k in b:
            by[k[1]] = max(by[k[1]], abs(a[k] - b[k]))
    print('self-influence (model_0 predictions when its OWN full scores are corrupted):')
    ok = True
    for meth in ['sample', 'dkps', 'pkps', 'ens', 'irt']:
        d = by.get(meth, 0.0)
        flag = 'OK' if d <= TOL else 'LEAK'
        ok &= d <= TOL
        print(f'  {meth:7s} {d:.2e}  [{flag}]')

    # (2) sanity: the corruption MUST propagate to some other model (model_0 is a kNN reference
    # for its neighbours) -- otherwise the test is vacuous (score_mat ignored entirely)
    ref_change = 0.0
    for mj in models[1:]:
        c, e = _preds(r0, mj), _preds(r1, mj)
        ref_change = max(ref_change, max((abs(c[k] - e[k]) for k in c if k in e), default=0.0))
    print(f'\nmax change across all reference models: {ref_change:.2e}  '
          f'[{"OK" if ref_change > TOL else "VACUOUS (score_mat had no effect anywhere)"}]')

    print('\nRESULT:', 'PASS -- leak-free' if ok and ref_change > TOL else 'FAIL')
    sys.exit(0 if (ok and ref_change > TOL) else 1)


if __name__ == '__main__':
    main()
