"""UnpairedDKPS (PKPS-style) on the leaderboard cache.

The head/tail embedding cache is naturally unpaired: ~25 systems were embedded
on up to 418 instances before the standard subset settled at 150. This script
compares paired DKPS (150-instance intersection) against UnpairedDKPS using
every cached (system, instance) cell, with instance problem statements as the
query vectors for the RBF query kernel.

Usage: python scripts/leaderboard_unpaired.py
"""
import argparse
import hashlib
import json
import os
import re
import sys
from glob import glob

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dkps.traces import make_openai_embed_fn
from dkps.traces.leaderboard import _extract_query_id
from dkps.unpaired_dkps import UnpairedDKPS

HCFG = hashlib.sha1('openai/text-embedding-3-small|headtail8000'.encode()).hexdigest()[:8]


def knn_predict(dist_row, y_ref, k):
    nn = np.argsort(dist_row)[:k]
    w = 1.0 / (dist_row[nn] + 1e-12)
    return float(np.dot(w, y_ref[nn]) / w.sum())


def eval_coords(coords, systems, y, families, k=5):
    """kNN in the CMDS coordinate space, LOO and leave-family-out."""
    from scipy.spatial.distance import squareform, pdist
    D = squareform(pdist(coords))
    M = len(systems)
    out = {}
    for proto in ('loo', 'family'):
        preds = []
        for i in range(M):
            if proto == 'loo':
                mask = np.arange(M) != i
            else:
                mask = np.array([j != i and not (
                    (families[i][0] and families[j][0] == families[i][0]) or
                    (families[i][1] and families[j][1] == families[i][1]))
                    for j in range(M)])
            preds.append(knn_predict(D[i][mask], y[mask], k))
        preds = np.array(preds)
        out[proto] = (np.abs(preds - y).mean(), spearmanr(preds, y).statistic)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/leaderboard/verified')
    ap.add_argument('--labels', default='data/leaderboard/verified_labels.json')
    ap.add_argument('--cache-dir', default='.dkps_cache_lb')
    ap.add_argument('--query-vec-dim', type=int, default=64,
                    help='PCA dim for problem-statement query vectors')
    ap.add_argument('--center', action='store_true',
                    help='median-center embeddings per instance (over available systems)')
    args = ap.parse_args()

    labels = json.load(open(args.labels))
    systems = [s for s in sorted(os.listdir(args.root))
               if len(glob(os.path.join(args.root, s, 'trajs', '*'))) >= 480
               and 'resolved' in labels.get(s, {})]
    y = np.array([len(labels[s]['resolved']) / 500 for s in systems])

    def tag(s, key):
        m = re.search(rf'^\s+{key}:\s*(.*)$', labels[s].get('metadata_yaml', ''), re.M)
        return m.group(1).strip().strip('"\'') if m else None
    families = [(tag(s, 'agent'), tag(s, 'model_display')) for s in systems]

    # ---- gather all cached (system, instance) head/tail embeddings ---------
    cells = {}
    for s in systems:
        for p in glob(os.path.join(args.cache_dir, s, f'*.{HCFG}.npz')):
            q = os.path.basename(p).rsplit('.', 2)[0]
            with np.load(p) as z:
                cells[(s, q)] = np.concatenate([z['head'], z['tail']]).astype(np.float32)
    qids = sorted({q for _, q in cells})
    counts = pd.Series([q for _, q in cells]).value_counts()
    print(f'{len(systems)} systems, {len(qids)} distinct instances, '
          f'{len(cells)} cells (density {len(cells)/(len(systems)*len(qids)):.2f})')

    if args.center:
        import collections
        by_q = collections.defaultdict(list)
        for (s, q), v in cells.items():
            by_q[q].append((s, v))
        for q, rows in by_q.items():
            med = np.median(np.stack([v for _, v in rows]), axis=0)
            for s, v in rows:
                r = v - med
                cells[(s, q)] = r / max(np.linalg.norm(r), 1e-9)

    # ---- query vectors: problem statements, embedded then PCA'd ------------
    qv_path = f'data/leaderboard/query_vecs_{args.query_vec_dim}.npz'
    if os.path.exists(qv_path):
        z = np.load(qv_path, allow_pickle=True)
        qvec = dict(zip(z['ids'].tolist(), z['vecs']))
    else:
        from datasets import load_dataset
        ds = load_dataset('princeton-nlp/SWE-bench_Verified', split='test')
        stmts = {r['instance_id']: r['problem_statement'][:24_000] for r in ds}
        embed_fn = make_openai_embed_fn()
        E = embed_fn([stmts[q] for q in qids])
        mu = E.mean(0, keepdims=True)
        _, _, Vt = np.linalg.svd(E - mu, full_matrices=False)
        P = (E - mu) @ Vt[:args.query_vec_dim].T
        qvec = dict(zip(qids, P))
        np.savez_compressed(qv_path, ids=np.array(qids, dtype=object),
                            vecs=np.stack([qvec[q] for q in qids]))
        qvec = dict(zip(qids, np.stack([qvec[q] for q in qids])))
    print(f'query vectors: {len(qvec)} x {args.query_vec_dim}')

    df = pd.DataFrame([
        {'model_id': s, 'query_id': q, 'embedding': v, 'query_vec': qvec[q]}
        for (s, q), v in cells.items()])

    # ---- paired reference: intersection only -------------------------------
    common = set(qids)
    for s in systems:
        common &= {q for (ss, q) in cells if ss == s}
    print(f'paired intersection: {len(common)} instances')
    df_paired = df[df.query_id.isin(common)]

    variants = {
        'paired DKPS (intersection)': (df_paired, dict(mode='paired')),
        'unpaired, constant kernel': (df, dict(mode='combined', query_kernel='constant')),
        'unpaired, RBF query kernel': (df, dict(mode='combined', query_kernel='rbf')),
    }
    print(f'\n{"variant":30s} {"LOO MAE/rho":>16s} {"family MAE/rho":>16s}')
    for name, (dfx, kw) in variants.items():
        model = UnpairedDKPS(n_components_cmds=8, **kw)
        coords_map = model.fit_transform(dfx)
        if isinstance(coords_map, dict):
            coords = np.stack([coords_map[s] for s in systems])
        else:
            coords = np.asarray(coords_map)
        res = eval_coords(coords, systems, y, families)
        (m1, r1), (m2, r2) = res['loo'], res['family']
        print(f'{name:30s} {m1:8.4f}/{r1:.3f} {m2:10.4f}/{r2:.3f}')


if __name__ == '__main__':
    main()
