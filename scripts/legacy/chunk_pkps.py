"""Chunk-level PKPS: compare traces as unpaired bags of chunks via product
kernels (side kernel over chunk context x content kernel), aggregated to a
system-level distance matrix and evaluated on resolve-rate prediction.

Side kernels: constant (= mean-pooling), position RBF, soft/hard rubric-section
membership, and position*section. rubric_hard vector pooling is the indicator
special case of this family.

Usage: python scripts/chunk_pkps.py
"""
import hashlib
import json
import os
import re
import sys
from glob import glob

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dkps.traces import make_openai_embed_fn
from dkps.traces.leaderboard import _extract_query_id
from dkps.traces.rubric import DEFAULT_RUBRIC, embed_rubric

KINDS = ('constant', 'position', 'sect-soft', 'sect-hard', 'pos*sect')
POS_H = 0.15
TAU = 0.05

dev = 'cuda' if torch.cuda.is_available() else 'cpu'

labels = json.load(open('data/leaderboard/verified_labels.json'))
root = 'data/leaderboard/verified'
all_systems = [s for s in sorted(os.listdir(root))
               if len(glob(os.path.join(root, s, 'trajs', '*'))) >= 480
               and 'resolved' in labels.get(s, {})]


def tag(s, key):
    m = re.search(rf'^\s+{key}:\s*(.*)$', labels[s].get('metadata_yaml', ''), re.M)
    return m.group(1).strip().strip('"\'') if m else None


per_sys_ids = [{_extract_query_id(os.path.basename(os.path.normpath(p)))
                for p in glob(os.path.join(root, s, 'trajs', '*'))}
               for s in all_systems]
q418 = sorted(set.intersection(*per_sys_ids))
rng = np.random.default_rng(0)
q150 = sorted(rng.choice(q418, 150, replace=False))
rng2 = np.random.default_rng(1)
queries = sorted(rng2.choice(q150, 20, replace=False))
ccfg = hashlib.sha1('openai/text-embedding-3-small|chunks4000'.encode()).hexdigest()[:8]
systems = [s for s in all_systems
           if all(os.path.exists(f'.dkps_cache_lb/{s}/{q}.{ccfg}.npz') for q in queries)]
y = np.array([len(labels[s]['resolved']) / 500 for s in systems])
model_tag = {s: tag(s, 'model_display') for s in systems}
M = len(systems)
print(f'{M} systems, {len(queries)} instances, device={dev}')

anchors = torch.tensor(embed_rubric(DEFAULT_RUBRIC, make_openai_embed_fn()),
                       dtype=torch.float32, device=dev)

D2 = {(c, k): np.zeros((M, M)) for c in (False, True) for k in KINDS}
for q in queries:
    Xs, seg, pos = [], [], []
    for i, s in enumerate(systems):
        with np.load(f'.dkps_cache_lb/{s}/{q}.{ccfg}.npz') as z:
            C = z['chunks']
        if len(C) == 0:
            C = np.zeros((1, anchors.shape[1]), np.float32)
        Xs.append(C.astype(np.float32))
        seg.extend([i] * len(C))
        pos.extend(((np.arange(len(C)) + 0.5) / len(C)).tolist())
    X = torch.tensor(np.vstack(Xs), device=dev)
    N = X.shape[0]
    seg_t = torch.tensor(seg, device=dev)
    pos_t = torch.tensor(pos, dtype=torch.float32, device=dev)
    B = torch.zeros((M, N), device=dev)
    B[seg_t, torch.arange(N, device=dev)] = 1.0

    Wk = {}
    Xn0 = X / X.norm(dim=1, keepdim=True).clamp_min(1e-9)
    sim = Xn0 @ anchors.T
    S = torch.softmax(sim / TAU, dim=1)
    hard = sim.argmax(1)
    Wk['constant'] = None                       # implicit ones
    Wk['position'] = torch.exp(-(pos_t[:, None] - pos_t[None, :]) ** 2 / (2 * POS_H ** 2))
    Wk['sect-soft'] = S @ S.T
    Wk['sect-hard'] = (hard[:, None] == hard[None, :]).float()
    Wk['pos*sect'] = Wk['position'] * Wk['sect-soft']

    for center in (False, True):
        Xc = X - X.mean(0, keepdim=True) if center else X
        Xn = Xc / Xc.norm(dim=1, keepdim=True).clamp_min(1e-9)
        G = Xn @ Xn.T
        for kind in KINDS:
            W = Wk[kind]
            WG = G if W is None else W * G
            num = B @ WG @ B.T
            den = (B @ (torch.ones_like(G) if W is None else W) @ B.T).clamp_min(1e-9)
            A = (num / den).cpu().numpy()
            d2 = np.diag(A)[:, None] + np.diag(A)[None, :] - 2 * A
            D2[(center, kind)] += np.maximum(d2, 0)
    del X, G, Wk
    torch.cuda.empty_cache()


def llm_mask(i):
    return np.array([j != i and not (model_tag[systems[i]]
                     and model_tag[systems[j]] == model_tag[systems[i]])
                     for j in range(M)])


def knn_predict(dr, yr, k=3):
    nn = np.argsort(dr)[:k]
    w = 1.0 / (dr[nn] + 1e-12)
    return float(np.dot(w, yr[nn]) / w.sum())


for center in (False, True):
    print(f'--- chunk-level PKPS, {"centered" if center else "raw"} content ---')
    for kind in KINDS:
        D = np.sqrt(D2[(center, kind)] / len(queries))
        row = f'{kind:12s}'
        for proto in ('loo', 'llm'):
            preds = np.array([
                knn_predict(D[i][(np.arange(M) != i) if proto == 'loo' else llm_mask(i)],
                            y[(np.arange(M) != i) if proto == 'loo' else llm_mask(i)])
                for i in range(M)])
            row += f'  {proto}: {np.abs(preds - y).mean():.4f}/{spearmanr(preds, y).statistic:.3f}'
        print(row)
