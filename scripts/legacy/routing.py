"""Per-instance routing / solution selection from trace representations.

Task: for each instance q, pick the system whose run most likely resolved it,
WITHOUT any outcome knowledge on q. Prediction is cross-instance outcome
transfer: p(s solves q) = distance-weighted kNN over traces on OTHER
instances with known outcomes (references restricted leave-one-LLM-out wrt
the candidate system). Route to argmax; score = did the routed system
actually resolve q.

Bar (HH): beat the same protocol on raw off-the-shelf embeddings.
Reference points: mean system, best single system (in hindsight), oracle.

Usage: python scripts/routing.py
Writes figures/routing.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from quench import load_graded, load_labels, model_tag  # noqa: E402
from dkps.traces.qubric import consensus_center  # noqa: E402


def main():
    labels = load_labels()
    systems, queries, graded = load_graded('data/judge/structured-qspec', labels)
    M, Q = len(systems), len(queries)
    B = np.array([[q in set(labels[s]['resolved']) for q in queries]
                  for s in systems], float)
    z = np.load('data/judge/pillars_emb_structured-qspec_'
                'nomic-ai_nomic-embed-text-v1.5.npz')
    inst = np.tile(np.arange(Q), M)
    sysid = np.repeat(np.arange(M), Q)
    reps = {
        'qubric': consensus_center(z['Xq'], inst),
        'raw': np.concatenate([z['H'], z['T']], axis=1),
    }
    tags = [model_tag(labels, s) for s in systems]
    outcome = B[sysid, inst]

    def route(X, k=15):
        Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
        hits, choices = [], []
        for qi in range(Q):
            scores = np.full(M, -np.inf)
            tgt = np.where(inst == qi)[0]
            for s in range(M):
                i = tgt[s]
                # references: other instances, leave-one-LLM-out wrt system s
                ok = (inst != qi) & np.array(
                    [sysid[j] != s and not (tags[s] and tags[sysid[j]] == tags[s])
                     for j in range(len(inst))])
                D = np.linalg.norm(Xn[ok] - Xn[i], axis=1)
                nn = np.argsort(D)[:k]
                w = 1 / (D[nn] + 1e-9)
                scores[s] = np.dot(w, outcome[ok][nn]) / w.sum()
            ch = int(scores.argmax())
            choices.append(systems[ch])
            hits.append(B[ch, qi])
        return float(np.mean(hits)), choices

    out = {}
    for name, X in reps.items():
        acc, choices = route(X)
        out[name] = {'routed_resolve_rate': acc,
                     'n_distinct_systems': len(set(choices))}
        print(f'{name:8s} routed resolve rate {acc:.3f} '
              f'({len(set(choices))} distinct systems chosen)')
    out['mean_system'] = float(B.mean())
    out['best_single_system'] = float(B.mean(1).max())
    out['oracle_per_instance'] = float(B.max(0).mean())
    print(f"mean system {out['mean_system']:.3f}  "
          f"best single {out['best_single_system']:.3f}  "
          f"oracle {out['oracle_per_instance']:.3f}")
    json.dump(out, open('figures/routing.json', 'w'), indent=2)
    print('wrote figures/routing.json')


if __name__ == '__main__':
    main()
