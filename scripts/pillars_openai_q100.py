"""pillars metrics for the canonical OpenAI embedder on the q100 era,
computed from the existing embedding caches (no API calls):
  qubric  data/judge/q100_emb_openai_small.npz
  raw     data/judge/q100_raw_emb_openai_small.npz (head+tail 8K tokens)
Metric definitions identical to scripts/pillars.py.
Writes figures/pillars_q100_text-embedding-3-small.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
from run_q100 import load_panel  # noqa: E402
from pillars import harness_tag, vendor_tag  # noqa: E402
from dkps.traces.qubric import consensus_center  # noqa: E402

OUT = 'figures/pillars_q100_text-embedding-3-small.json'


def main():
    labels, systems, q100 = load_panel()
    M, Q = len(systems), len(q100)
    B = np.array([[q in set(labels[s]['resolved']) for q in q100]
                  for s in systems], float)
    Xq = np.load('data/judge/q100_emb_openai_small.npz')['X']
    HT = np.load('data/judge/q100_raw_emb_openai_small.npz')['HT']
    raw = HT.reshape(M * Q, -1)
    inst = np.tile(np.arange(Q), M)
    sysid = np.repeat(np.arange(M), Q)
    reps = {'raw': raw, 'qubric': consensus_center(Xq, inst)}
    vend = [vendor_tag(labels, systems[i]) for i in sysid]
    harn = [harness_tag(systems[i]) for i in sysid]
    outcome = B[sysid, inst]

    out = {'embed_model': 'text-embedding-3-small',
           'judge_dir': 'data/judge/q100-qspec-flash0731',
           'chance': {'task': 1 / Q, 'identity': 1 / M,
                      'family': None, 'harness': None},
           'reps': {}}
    from scipy.spatial.distance import pdist, squareform
    for name, X in reps.items():
        Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
        Xn = Xn.astype(np.float32)
        D = squareform(pdist(Xn)).astype(np.float32)
        np.fill_diagonal(D, np.inf)
        D_task = D.copy()
        D_task[sysid[:, None] == sysid[None, :]] = np.inf
        task = float((inst[D_task.argmin(1)] == inst).mean())
        D_id = D.copy()
        D_id[inst[:, None] == inst[None, :]] = np.inf
        ident = float((sysid[D_id.argmin(1)] == sysid).mean())
        D_who = D.copy()
        D_who[inst[:, None] != inst[None, :]] = np.inf
        nn = D_who.argmin(1)
        v = np.asarray(vend)
        h = np.asarray(harn)
        vok = v != None  # noqa: E711
        hok = h != None  # noqa: E711
        fam = float((v[nn[vok]] == v[vok]).mean())
        har = float((h[nn[hok]] == h[hok]).mean())
        aucs = []
        for qi in range(Q):
            sel = np.where(inst == qi)[0]
            o = outcome[sel]
            if o.min() == o.max():
                continue
            Dq = D[np.ix_(sel, sel)]
            same = o[:, None] == o[None, :]
            iu = np.triu_indices(len(sel), 1)
            ds, dd = Dq[iu][same[iu]], Dq[iu][~same[iu]]
            aucs.append(float((ds[:, None] < dd[None, :]).mean()))
        behavior = float(np.mean(aucs)) - 0.5
        rng = np.random.default_rng(0)
        hits = 0
        for _ in range(20):
            perm = rng.permutation(Q)
            a, b = perm[:Q // 2], perm[Q // 2:]
            Xa = Xn.reshape(M, Q, -1)[:, a].mean(1)
            Xb = Xn.reshape(M, Q, -1)[:, b].mean(1)
            Dab = ((Xa[:, None] - Xb[None]) ** 2).sum(-1)
            hits += (Dab.argmin(1) == np.arange(M)).mean()
        ident_agg = float(hits / 20)
        out['reps'][name] = dict(task=task, behavior=behavior,
                                 identity_trace=ident,
                                 identity_agg=ident_agg,
                                 family=fam, harness=har)
        print(f"{name:8s} task {task:.3f}  behavior {behavior:+.3f}  "
              f"id {ident:.3f}/{ident_agg:.3f}  fam {fam:.3f}  "
              f"harness {har:.3f}")

    def base(gr):
        g = gr[gr != None]  # noqa: E711
        _, c = np.unique(g, return_counts=True)
        return float(((c / c.sum()) ** 2).sum())
    out['chance']['family'] = base(np.asarray(vend))
    out['chance']['harness'] = base(np.asarray(harn))
    json.dump(out, open(OUT, 'w'), indent=2)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
