"""Judge-describe representation: an LLM reads each trace and writes a short
natural-language description of the agent's *behavior* (not the problem),
which is then embedded. The judge acts as a semantic renormalizer: it can read
execution outcomes (test passes/failures) that embeddings of raw logs cannot
distinguish, and it emits text in a shared register that embeds cleanly.

Cache: data/judge/<model>/<system>/<instance>.txt
Usage: python scripts/judge_describe.py [--judge-model gpt-5.4-mini]
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from glob import glob
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dkps.traces import load_dotenv, make_openai_embed_fn
from dkps.traces.leaderboard import _extract_query_id
import dkps.traces.leaderboard as lb

PROMPT = """Below is the (truncated) execution trace of a software-engineering agent \
working on a repository issue. Describe HOW this agent worked, in 150-250 words of \
plain prose. Cover, concretely and factually:
- exploration strategy (how it searched/navigated the code)
- whether it reproduced the issue before fixing, and how
- how it located the code to change
- what it changed (files, nature of the edit; small targeted patch vs broad rewrite)
- whether it verified the fix afterwards (ran tests or a repro script) and what the \
observed outcome of that verification was (passing, failing, errors, unclear)
- the apparent final state: did it finish cleanly, run out of steps, or submit \
without verification

Do NOT restate the issue/problem itself. Do NOT speculate about correctness beyond \
what the trace shows. Write only the description."""

_labels_g = None
_wanted_g = None


def _load_one(args):
    sub_dir, wanted = args
    ts = lb.load_leaderboard_submission(sub_dir, labels=None)
    out = {}
    for t in ts:
        if t.query_id in wanted:
            txt = t.steps[0].assistant_text or ''
            out[t.query_id] = txt[:40_000] + ('\n...[omitted]...\n' + txt[-20_000:]
                                              if len(txt) > 60_000 else '')
    return os.path.basename(sub_dir), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/leaderboard/verified')
    ap.add_argument('--labels', default='data/leaderboard/verified_labels.json')
    ap.add_argument('--cache-dir', default='.dkps_cache_lb')
    ap.add_argument('--judge-model', default='gpt-5.4-mini')
    ap.add_argument('--workers', type=int, default=12)
    args = ap.parse_args()

    load_dotenv()
    key = os.environ['OPENAI_API_KEY']
    url = (os.environ.get('OPENAI_BASE_URL') or 'https://api.openai.com/v1'
           ).rstrip('/') + '/chat/completions'

    labels = json.load(open(args.labels))
    all_systems = [s for s in sorted(os.listdir(args.root))
                   if len(glob(os.path.join(args.root, s, 'trajs', '*'))) >= 480
                   and 'resolved' in labels.get(s, {})]
    per_sys = [{_extract_query_id(os.path.basename(os.path.normpath(p)))
                for p in glob(os.path.join(args.root, s, 'trajs', '*'))}
               for s in all_systems]
    q418 = sorted(set.intersection(*per_sys))
    rng0 = np.random.default_rng(0)
    q150 = sorted(rng0.choice(q418, 150, replace=False))
    rng2 = np.random.default_rng(1)
    q20 = sorted(rng2.choice(q150, 20, replace=False))
    ccfg = hashlib.sha1('openai/text-embedding-3-small|chunks4000'.encode()).hexdigest()[:8]
    systems = [s for s in all_systems
               if all(os.path.exists(f'{args.cache_dir}/{s}/{q}.{ccfg}.npz') for q in q20)]
    y = np.array([len(labels[s]['resolved']) / 500 for s in systems])
    M = len(systems)
    print(f'{M} systems x {len(q20)} instances, judge={args.judge_model}')

    jdir = os.path.join('data', 'judge', args.judge_model)

    def jpath(s, q):
        return os.path.join(jdir, s, f'{q}.txt')

    todo_pairs = [(s, q) for s in systems for q in q20
                  if not os.path.exists(jpath(s, q))]
    if todo_pairs:
        need_sys = sorted({s for s, _ in todo_pairs})
        with Pool(16) as pool:
            texts = dict(pool.map(_load_one, [(os.path.join(args.root, s), set(q20))
                                              for s in need_sys]))

        def describe(pair):
            s, q = pair
            body = {'model': args.judge_model,
                    'messages': [{'role': 'user',
                                  'content': PROMPT + '\n\n=== TRACE ===\n'
                                             + texts[s].get(q, '(empty trace)')}],
                    'max_completion_tokens': 2000}
            delay = 2.0
            for _ in range(10):
                r = requests.post(url, json=body,
                                  headers={'Authorization': f'Bearer {key}'},
                                  timeout=300)
                if r.status_code == 200:
                    out = r.json()['choices'][0]['message']['content'] or ''
                    os.makedirs(os.path.dirname(jpath(s, q)), exist_ok=True)
                    with open(jpath(s, q), 'w') as f:
                        f.write(out)
                    return True
                if r.status_code in (429, 500, 502, 503, 529):
                    time.sleep(max(delay, float(r.headers.get('retry-after', 0) or 0)))
                    delay = min(delay * 2, 60)
                    continue
                raise RuntimeError(f'{r.status_code}: {r.text[:200]}')
            raise RuntimeError('chat retries exhausted')

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(tqdm(ex.map(describe, todo_pairs), total=len(todo_pairs),
                      desc='judge'))

    descs = {(s, q): open(jpath(s, q)).read() for s in systems for q in q20}
    embed_fn = make_openai_embed_fn()
    flat = [descs[(s, q)] for s in systems for q in q20]
    E = embed_fn(flat).reshape(M, len(q20), -1).astype(np.float32)

    # evaluation vs naive baseline on same panel, both readouts, LLM-out
    def tag(s, k):
        m = re.search(rf'^\s+{k}:\s*(.*)$', labels[s].get('metadata_yaml', ''), re.M)
        return m.group(1).strip().strip('"\'') if m else None
    model_tag = {s: tag(s, 'model_display') for s in systems}
    allowed = np.array([[jj != i and not (model_tag[systems[i]]
                        and model_tag[systems[jj]] == model_tag[systems[i]])
                        for jj in range(M)] for i in range(M)])

    def knn_eval(Xv):
        from scipy.spatial.distance import pdist, squareform
        D = squareform(pdist(Xv.reshape(M, -1)))
        preds = []
        for i in range(M):
            idx = np.where(allowed[i])[0]
            nn = idx[np.argsort(D[i][idx])[:3]]
            w = 1 / (D[i][nn] + 1e-12)
            preds.append(np.dot(w, y[nn]) / w.sum())
        preds = np.array(preds)
        return np.abs(preds - y).mean(), spearmanr(preds, y).statistic

    def ridge_eval(Xv, lams=(1.0, 10.0, 100.0)):
        feats = Xv.reshape(M, -1)
        best = (np.inf, 0)
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
            mae = np.abs(preds - y).mean()
            if mae < best[0]:
                best = (mae, spearmanr(preds, y).statistic)
        return best

    def norm(A):
        return A / np.maximum(np.linalg.norm(A, axis=-1, keepdims=True), 1e-9)

    def report(name, Xv):
        km, kr = knn_eval(Xv)
        rm, rr = ridge_eval(Xv)
        print(f'{name:36s} knn: {km:.4f}/{kr:.3f}  ridge: {rm:.4f}/{rr:.3f}')

    hcfg = hashlib.sha1('openai/text-embedding-3-small|headtail8000'.encode()).hexdigest()[:8]
    X20 = np.zeros((M, len(q20), 3072), np.float32)
    for i, s in enumerate(systems):
        for j, q in enumerate(q20):
            with np.load(f'{args.cache_dir}/{s}/{q}.{hcfg}.npz') as z:
                X20[i, j] = np.concatenate([z['head'], z['tail']])
    REF = norm(X20 - np.median(X20, axis=0, keepdims=True))

    def rms(A):
        r = np.sqrt((A ** 2).mean())
        return A / max(r, 1e-9)

    report('NAIVE head+tail centered', REF)
    report('judge-desc embedding', norm(E))
    report('judge-desc centered', norm(E - np.median(E, axis=0, keepdims=True)))
    Ec = norm(E - np.median(E, axis=0, keepdims=True))
    report('NAIVE + judge-desc', np.concatenate([rms(REF), rms(Ec)], -1))


if __name__ == '__main__':
    main()
