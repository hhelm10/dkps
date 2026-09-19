"""Cross-model matrix: extraction judge x rubric construction (Fig 3 right panel).

Every cached (judge, construction) cell evaluated under ONE common local
embedder so mini/nano/4o-mini cells are comparable (the nano judge texts were
never embedded with the OpenAI embedder). Protocol identical to
judge_structured.py: per-(instance,section) median centering over systems, L2,
concat; leave-one-LLM-out kNN(k=3) and ridge readouts of the Verified resolve
rate.

Cells cached today (rubric writer fixed = gpt-5.4-mini):
  free-form blob : gpt-4o-mini, gpt-5.4-mini      (data/judge/<model>/<sys>/<q>.txt)
  generic rubric : gpt-5.4-mini                    (structured-fixed)
  verdict quest. : gpt-5.4-mini                    (structured-questions)
  qubric (qspec) : gpt-5.4-mini, gpt-5.4-nano      (structured-qspec[-nano])

Usage: python scripts/judge_matrix.py [--embed-model nomic-ai/nomic-embed-text-v1.5]
Writes figures/judge_matrix.json.
"""
import argparse
import json
import os
import re
import sys

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dkps.traces import make_sentence_transformer_embed_fn

SECTIONS = ('understanding', 'localization', 'reproduction',
            'editing', 'verification', 'final_state')

CELLS = {  # (construction, judge) -> (dir, kind)
    ('blob', 'gpt-4o-mini'): ('data/judge/gpt-4o-mini', 'txt'),
    ('blob', 'gpt-5.4-mini'): ('data/judge/gpt-5.4-mini', 'txt'),
    ('generic', 'gpt-5.4-mini'): ('data/judge/structured-fixed', 'json'),
    ('verdict', 'gpt-5.4-mini'): ('data/judge/structured-questions', 'json'),
    ('qubric', 'gpt-5.4-mini'): ('data/judge/structured-qspec', 'json'),
    ('qubric', 'gpt-5.4-nano'): ('data/judge/structured-qspec-gpt-5.4-nano', 'json'),
    ('qubric', 'gpt-oss-20b'): ('data/judge/structured-qspec-openai_gpt-oss-20b', 'json'),
    ('qubric', 'gpt-oss-120b'): ('data/judge/structured-qspec-openai_gpt-oss-120b', 'json'),
    ('qubric', 'deepseek-v3.1'): ('data/judge/structured-qspec-deepseek_deepseek-chat-v3.1', 'json'),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', default='data/leaderboard/verified_labels.json')
    ap.add_argument('--embed-model', default='nomic-ai/nomic-embed-text-v1.5')
    ap.add_argument('--out', default='figures/judge_matrix.json')
    ap.add_argument('--emb-cache', default='data/judge/matrix_emb')
    args = ap.parse_args()

    labels = json.load(open(args.labels))
    ref = 'data/judge/structured-qspec'
    q20 = sorted(f[:-5] for f in os.listdir(os.path.join(ref, sorted(os.listdir(ref))[0])))
    systems = sorted(s for s in os.listdir(ref)
                     if 'resolved' in labels.get(s, {})
                     and len(os.listdir(os.path.join(ref, s))) == len(q20))
    # every cell must cover the same systems
    for (c, j), (d, kind) in CELLS.items():
        have = {s for s in os.listdir(d) if os.path.isdir(os.path.join(d, s))}
        systems = [s for s in systems if s in have]
    M, Q = len(systems), len(q20)
    y = np.array([len(labels[s]['resolved']) / 500 for s in systems])
    print(f'{M} systems x {Q} instances; {len(CELLS)} cells')

    def tagf(s, k):
        m = re.search(rf'^\s+{k}:\s*(.*)$', labels[s].get('metadata_yaml', ''), re.M)
        return m.group(1).strip().strip('"\'') if m else None
    model_tag = {s: tagf(s, 'model_display') for s in systems}
    allowed = np.array([[jj != i and not (model_tag[systems[i]]
                        and model_tag[systems[jj]] == model_tag[systems[i]])
                        for jj in range(M)] for i in range(M)])

    embed_fn = make_sentence_transformer_embed_fn(model_name=args.embed_model)
    os.makedirs(args.emb_cache, exist_ok=True)
    emb_tag = args.embed_model.replace('/', '_')

    def load_texts(d, kind):
        n_bad = 0
        if kind == 'txt':
            T = np.empty((M, Q, 1), object)
            for i, s in enumerate(systems):
                for j, q in enumerate(q20):
                    p = os.path.join(d, s, f'{q}.txt')
                    T[i, j, 0] = open(p).read() if os.path.exists(p) else ' '
        else:
            T = np.empty((M, Q, len(SECTIONS)), object)
            for i, s in enumerate(systems):
                for j, q in enumerate(q20):
                    try:
                        dd = json.loads(open(os.path.join(d, s, f'{q}.json')).read())
                        if isinstance(dd, list) and dd:
                            dd = dd[0]
                        if not isinstance(dd, dict):
                            dd, n_bad = {}, n_bad + 1
                    except (json.JSONDecodeError, FileNotFoundError):
                        dd, n_bad = {}, n_bad + 1
                    for k, sec in enumerate(SECTIONS):
                        T[i, j, k] = str(dd.get(sec, '') or ' ')
        return T, n_bad

    def embed_cell(name, T):
        p = os.path.join(args.emb_cache, f'{name}.{emb_tag}.npz')
        if os.path.exists(p):
            return np.load(p)['E']
        flat = [T[i, j, k] for i in range(M) for j in range(Q)
                for k in range(T.shape[2])]
        E = embed_fn(flat).reshape(M, Q, T.shape[2], -1).astype(np.float32)
        np.savez_compressed(p, E=E)
        return E

    def knn_eval(Xv):
        D = squareform(pdist(Xv.reshape(M, -1)))
        preds = []
        for i in range(M):
            idx = np.where(allowed[i])[0]
            nn = idx[np.argsort(D[i][idx])[:3]]
            w = 1 / (D[i][nn] + 1e-12)
            preds.append(np.dot(w, y[nn]) / w.sum())
        preds = np.array(preds)
        return float(np.abs(preds - y).mean()), float(spearmanr(preds, y).statistic)

    def ridge_eval(Xv, lams=(1.0, 10.0, 100.0)):
        feats = Xv.reshape(M, -1)
        best = (np.inf, 0.0)
        for lam in lams:
            preds = []
            for i in range(M):
                tr = np.where(allowed[i])[0]
                A = feats[tr]; b = y[tr]
                mu = A.mean(0); Ac = A - mu; bm = b.mean()
                G = Ac @ Ac.T + lam * np.eye(len(tr))
                al = np.linalg.solve(G, b - bm)
                preds.append(bm + (feats[i] - mu) @ (Ac.T @ al))
            preds = np.array(preds)
            mae = float(np.abs(preds - y).mean())
            if mae < best[0]:
                best = (mae, float(spearmanr(preds, y).statistic))
        return best

    results = {}
    for (c, j), (d, kind) in CELLS.items():
        T, n_bad = load_texts(d, kind)
        E = embed_cell(f'{c}-{j}', T)
        Es = E - np.median(E, axis=0, keepdims=True)
        Es = Es / np.maximum(np.linalg.norm(Es, axis=-1, keepdims=True), 1e-9)
        km, kr = knn_eval(Es)
        rm, rr = ridge_eval(Es)
        results[f'{c}|{j}'] = dict(knn_mae=km, knn_rho=kr, ridge_mae=rm,
                                   ridge_rho=rr, parse_failures=n_bad)
        print(f'{c:8s} x {j:12s}  knn {km:.4f}/{kr:.3f}  ridge {rm:.4f}/{rr:.3f}'
              f'  (bad json: {n_bad})')

    out = dict(embedder=args.embed_model, systems=M, instances=Q,
               rubric_writer='gpt-5.4-mini', protocol='leave-one-LLM-out, '
               'per-(instance,section) median centering, L2, concat',
               cells=results)
    json.dump(out, open(args.out, 'w'), indent=2)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
