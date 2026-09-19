"""Supervised channel weighting: close the gap between DKPS-kNN error (~0.08)
and the oracle kNN floor (~0.02) by learning score-aligned channel weights.

Distance family: d_w(i,j)^2 = sum_c w_c * mean_{q in S} ||c_i^q - c_j^q||^2
over the per-channel (RMS-scaled) embeddings -- uniform w recovers the
concatenated-channel geometry.

Protocol (honest): for each held-out model, weights w and k are selected by
internal leave-one-out kNN MAE among the 13 reference models only (random
Dirichlet candidates + corners + uniform), then applied once to the held-out
model. The in-sample variant (select w with the held-out model's error
visible) is reported as a capacity ceiling for the weight family, not a
legitimate result.

Run scripts/embed_traces.py first. Usage:
    python scripts/learn_channel_weights.py [--n-candidates 2000] [--budget 12 3]
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dkps.traces import (TraceEmbedder, file_localization, load_langfuse_corpus,
                         rms_scale)
from predict_localization import gold_files_cached, per_query_sq_dists

CHANNELS = ['action', 'step_text', 'outcome', 'scalar', 'whole']
KS = (1, 2, 3, 5)


def loo_mae_matrix(D, y, ks=KS):
    """Internal leave-one-out kNN MAE for each k, given distances D among the
    pool. Returns {k: (mae, per-point predictions)}."""
    n = len(y)
    Dm = D + np.eye(n) * 1e9
    order = np.argsort(Dm, axis=1)
    out = {}
    for k in ks:
        nn = order[:, :k]
        w = 1.0 / (np.take_along_axis(Dm, nn, axis=1) + 1e-12)
        pred = (w * y[nn]).sum(1) / w.sum(1)
        out[k] = (np.abs(pred - y).mean(), pred)
    return out


def predict_one(D_row, y_ref, k):
    nn = np.argsort(D_row)[:k]
    w = 1.0 / (D_row[nn] + 1e-12)
    return float(np.dot(w, y_ref[nn]) / w.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traces-root', default='data/traces')
    ap.add_argument('--cache-dir', default='.dkps_cache')
    ap.add_argument('--n-candidates', type=int, default=2000)
    ap.add_argument('--budgets', type=int, nargs='+', default=[12, 3])
    ap.add_argument('--n-subsets', type=int, default=20,
                    help='MC query subsets per budget < full')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    traces = load_langfuse_corpus(args.traces_root)
    models = sorted({t.model_id for t in traces})
    queries = sorted({t.query_id for t in traces})
    M, Q = len(models), len(queries)

    gold = gold_files_cached(queries)
    cell = defaultdict(list)
    for t in traces:
        cell[(t.model_id, t.query_id)].append(file_localization(t, gold))
    y = np.array([[np.mean(cell[(m, q)]) for q in queries] for m in models]).mean(1)

    blocks = TraceEmbedder(cache_dir=args.cache_dir).transform_channels(traces)
    # per-channel per-query squared distances: (C, Q, M, M)
    PDc = np.stack([per_query_sq_dists(traces, rms_scale(blocks[c]), models, queries)
                    for c in CHANNELS])

    # candidate weight vectors: Dirichlet cloud + corners + uniform
    W = np.vstack([rng.dirichlet(np.ones(len(CHANNELS)), args.n_candidates),
                   np.eye(len(CHANNELS)),
                   np.full((1, len(CHANNELS)), 1 / len(CHANNELS))])

    # oracle floor for reference
    floor = []
    for i in range(M):
        nn = sorted((j for j in range(M) if j != i), key=lambda j: abs(y[j] - y[i]))[:3]
        wts = 1 / (np.abs(y[nn] - y[i]) + 1e-3)
        floor.append(abs(np.dot(wts, y[nn]) / wts.sum() - y[i]))
    print(f'oracle kNN floor: {np.mean(floor):.4f}\n')

    for budget in args.budgets:
        subsets = [np.arange(Q)] if budget >= Q else \
                  [rng.choice(Q, budget, replace=False) for _ in range(args.n_subsets)]
        # PDs[s]: (C, M, M) mean over the subset's queries
        PDs = [PDc[:, S].mean(axis=1) for S in subsets]

        errs_honest, errs_cheat, errs_uniform = [], [], []
        chosen_w = []
        for i in range(M):
            ref = np.array([j for j in range(M) if j != i])
            for PD in PDs:
                # select (w, k) by internal LOO among the 13 references
                best, best_score = None, np.inf
                cheat_best, cheat_score = None, np.inf
                for w in W:
                    D = np.sqrt(np.tensordot(w, PD, axes=1))
                    internal = loo_mae_matrix(D[np.ix_(ref, ref)], y[ref])
                    for k, (mae, _) in internal.items():
                        if mae < best_score:
                            best_score, best = mae, (w, k)
                        err_i = abs(predict_one(D[i][ref], y[ref], k) - y[i])
                        if err_i < cheat_score:
                            cheat_score, cheat_best = err_i, (w, k)
                w, k = best
                D = np.sqrt(np.tensordot(w, PD, axes=1))
                errs_honest.append(abs(predict_one(D[i][ref], y[ref], k) - y[i]))
                errs_cheat.append(cheat_score)
                chosen_w.append(w)
                Du = np.sqrt(PD.mean(axis=0))
                errs_uniform.append(abs(predict_one(Du[i][ref], y[ref], 3) - y[i]))

        wbar = np.mean(chosen_w, axis=0)
        print(f'budget {budget:2d}:')
        print(f'  uniform weights, k=3 (baseline)   MAE = {np.mean(errs_uniform):.4f}')
        print(f'  learned weights (honest LOO)      MAE = {np.mean(errs_honest):.4f}')
        print(f'  learned weights (in-sample bound) MAE = {np.mean(errs_cheat):.4f}')
        print(f'  mean selected weights: '
              + ', '.join(f'{c}={v:.2f}' for c, v in zip(CHANNELS, wbar)) + '\n')


if __name__ == '__main__':
    main()
