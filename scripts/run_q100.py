"""q100 era run: DeepSeek V4 Flash 0731 as Model 1 + Model 2, pruned full
traces, OpenAI text-embedding-3-small.

Config (locked 2026-09-04, per HH):
  panel     data/judge/q100.json (q20 UNION rng(2)-seeded 80 from q418)
  Model 1   deepseek/deepseek-v4-flash-0731 (rubrics for ALL 100 -- no
            mini-rubric carryover; clean single-config era)
  Model 2   deepseek/deepseek-v4-flash-0731 (extraction)
  inputs    data/judge/trace_texts_full_pruned/ (render_full_pruned.py
            --panel data/judge/q100.json)
  embedder  openai/text-embedding-3-small (access restored 2026-09-04)

Stages (all resumable, per-file caches):
  rubrics   -> data/judge/q100_rubrics/<q>.json
  extract   -> data/judge/q100-qspec-flash0731/<sys>/<q>.json
  embed     -> data/judge/q100_emb_openai_small.npz
  eval      -> figures/q100_eval.json (kNN + ridge + count-lookup, LLM-out)

Usage: python scripts/run_q100.py --stage rubrics|extract|embed|eval [--limit N]
"""
import argparse
import json
import os
import sys

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from dkps.traces.qubric import (DEFAULT_SECTIONS, consensus_center,
                                embed_graded, grade_traces, write_rubrics)

JUDGE = 'deepseek/deepseek-v4-flash-0731'
PANEL = 'data/judge/q100.json'
TXT = 'data/judge/trace_texts_full_pruned'
RUB = 'data/judge/q100_rubrics'
EXT = 'data/judge/q100-qspec-flash0731'
EMB = 'data/judge/q100_emb_openai_small.npz'


def load_panel():
    labels = json.load(open('data/leaderboard/verified_labels.json'))
    q100 = sorted(json.load(open(PANEL))['instances'])
    systems = sorted(s for s in os.listdir('data/judge/structured-qspec')
                     if 'resolved' in labels.get(s, {}))
    return labels, systems, q100


def stage_rubrics(key, args):
    _, _, q100 = load_panel()
    os.makedirs(RUB, exist_ok=True)
    missing = [q for q in q100 if not os.path.exists(f'{RUB}/{q}.json')]
    if args.limit:
        missing = missing[:args.limit]
    print(f'{len(missing)} rubrics to write')
    if not missing:
        return
    from datasets import load_dataset
    stmts = {r['instance_id']: r['problem_statement']
             for r in load_dataset('princeton-nlp/SWE-bench_Verified', split='test')
             if r['instance_id'] in set(missing)}
    rubs = write_rubrics({q: stmts[q] for q in missing}, key, JUDGE, workers=8)
    for q, rub in rubs.items():
        open(f'{RUB}/{q}.json', 'w').write(json.dumps(rub))
    print(f'wrote {len(rubs)}')


def stage_extract(key, args):
    from concurrent.futures import ThreadPoolExecutor
    _, systems, q100 = load_panel()
    rubrics = {q: json.loads(open(f'{RUB}/{q}.json').read()) for q in q100}

    def one_system(s):
        todo = {q: open(os.path.join(TXT, s, f'{q}.txt')).read()
                for q in q100
                if not os.path.exists(os.path.join(EXT, s, f'{q}.json'))
                and os.path.exists(os.path.join(TXT, s, f'{q}.txt'))}
        if args.limit:
            todo = dict(list(todo.items())[:args.limit])
        if not todo:
            return 0
        graded = grade_traces(rubrics, todo, key, JUDGE,
                              task_ids={q: q for q in todo}, workers=8,
                              max_trace_chars=1_000_000, on_error='skip')
        os.makedirs(os.path.join(EXT, s), exist_ok=True)
        for q, g in graded.items():
            open(os.path.join(EXT, s, f'{q}.json'), 'w').write(json.dumps(g))
        return len(graded)

    with ThreadPoolExecutor(max_workers=4) as ex:
        list(tqdm(ex.map(one_system, systems), total=len(systems),
                  desc='extract'))
    total = sum(os.path.exists(os.path.join(EXT, s, f'{q}.json'))
                for s in systems for q in q100)
    print(f'coverage {total}/{len(systems) * len(q100)}')


def load_graded(systems, q100):
    graded = []
    for s in systems:
        for q in q100:
            try:
                d = json.loads(open(os.path.join(EXT, s, f'{q}.json')).read())
                if isinstance(d, list) and d:
                    d = d[0]
                if not isinstance(d, dict):
                    d = {}
            except (json.JSONDecodeError, FileNotFoundError):
                d = {}
            graded.append(d)
    return graded


def stage_embed(key_openai, args):
    _, systems, q100 = load_panel()
    graded = load_graded(systems, q100)
    n_bad = sum(not g for g in graded)
    print(f'{len(graded)} graded traces ({n_bad} empty)')
    X = embed_graded(graded, key_openai, 'text-embedding-3-small')
    np.savez_compressed(EMB, X=X.astype(np.float32))
    print('wrote', EMB, X.shape)


def stage_eval(args):
    import re
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import spearmanr
    labels, systems, q100 = load_panel()
    M, Q = len(systems), len(q100)
    y = np.array([len(labels[s]['resolved']) / 500 for s in systems])
    B = np.array([[q in set(labels[s]['resolved']) for q in q100]
                  for s in systems], float)

    def tagf(s):
        m = re.search(r'^\s+model_display:\s*(.*)$',
                      labels[s].get('metadata_yaml', ''), re.M)
        return m.group(1).strip().strip('"\'') if m else None
    tags = [tagf(s) for s in systems]
    allowed = np.array([[j != i and not (tags[i] and tags[j] == tags[i])
                         for j in range(M)] for i in range(M)])

    X = np.load(EMB)['X']
    Xc = consensus_center(X, np.tile(np.arange(Q), M)).reshape(M, Q, -1)

    def knn(D, i, k=3):
        idx = np.where(allowed[i])[0]
        nn = idx[np.argsort(D[i][idx])[:k]]
        w = 1 / (D[i][nn] + 1e-12)
        return float(np.dot(w, y[nn]) / w.sum())

    def count_lookup(i, cols):
        k = B[i, cols].sum()
        ks = B[:, cols].sum(1)
        for tol in range(len(cols) + 1):
            ok = allowed[i] & (np.abs(ks - k) <= tol)
            if ok.any():
                return y[ok].mean()

    rng = np.random.default_rng(0)
    out = {'panel': PANEL, 'judge': JUDGE, 'embedder': 'text-embedding-3-small',
           'curves': {}}
    D_full = squareform(pdist(Xc.reshape(M, -1)))
    preds = np.array([knn(D_full, i) for i in range(M)])
    out['full_panel'] = {'knn_mae': float(np.abs(preds - y).mean()),
                         'knn_rho': float(spearmanr(preds, y).statistic)}
    print(f"full q100 kNN: {out['full_panel']['knn_mae']:.4f} / "
          f"{out['full_panel']['knn_rho']:.3f}")
    for m in (1, 3, 5, 10, 20, 50):
        gs, cs = [], []
        for _ in range(30):
            cols = rng.choice(Q, m, replace=False)
            D = squareform(pdist(Xc[:, cols, :].reshape(M, -1)))
            gs.append(np.mean([abs(knn(D, i) - y[i]) for i in range(M)]))
            cs.append(np.mean([abs(count_lookup(i, cols) - y[i])
                               for i in range(M)]))
        out['curves'][m] = {'geometry': float(np.mean(gs)),
                            'count_lookup': float(np.mean(cs))}
        print(f"m={m:2d} geometry {np.mean(gs):.4f}  count-lookup {np.mean(cs):.4f}")
    json.dump(out, open('figures/q100_eval.json', 'w'), indent=2)
    print('wrote figures/q100_eval.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True,
                    choices=('rubrics', 'extract', 'embed', 'eval'))
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
    if args.stage == 'rubrics':
        stage_rubrics(os.environ['OPENROUTER_API_KEY'], args)
    elif args.stage == 'extract':
        stage_extract(os.environ['OPENROUTER_API_KEY'], args)
    elif args.stage == 'embed':
        stage_embed(os.environ['OPENAI_API_KEY'], args)
    else:
        stage_eval(args)


if __name__ == '__main__':
    main()
