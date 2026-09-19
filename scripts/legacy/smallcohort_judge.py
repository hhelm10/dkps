"""Judge the replicate-rich small cohort (14 models x 12 instances x 5 reps)
for the Stability column of the pillar heatmap.

Produces, via OpenRouter (default judge deepseek/deepseek-chat-v3.1):
  data/judge/smallcohort_texts/<model>/<rep>/<query>.txt      rendered traces
  data/judge/smallcohort_rubrics/<query>.json                 12 qspec rubrics
  data/judge/smallcohort-qspec-<judge>/<model>/<rep>/<query>.json
  data/judge/smallcohort-freeform-<judge>/<model>/<rep>/<query>.txt

Usage: python scripts/smallcohort_judge.py [--limit N] [--workers 16]
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from judge_structured import EXTRACT_PROMPT, QSPEC_PROMPT, SECTIONS  # noqa: E402
from judge_describe import PROMPT as FREEFORM_PROMPT  # noqa: E402
from judge_openrouter import SCHEMA  # noqa: E402
from dkps.traces.langfuse import load_langfuse_corpus  # noqa: E402
from dkps.traces.canonicalize import canonicalize_step  # noqa: E402

TXT = 'data/judge/smallcohort_texts'
RUB = 'data/judge/smallcohort_rubrics'


def render(trace, head=40_000, tail=20_000):
    parts = []
    for st in trace.steps:
        parts.append(canonicalize_step(st))
    txt = '\n'.join(parts)
    if len(txt) > head + tail:
        txt = txt[:head] + '\n...[omitted]...\n' + txt[-tail:]
    return txt or '(empty trace)'


def chat(key, model, content, max_tokens=2500, schema=True):
    body = {'model': model, 'messages': [{'role': 'user', 'content': content}],
            'max_tokens': max_tokens,
            'reasoning': {'effort': 'low'},
            'provider': {'sort': 'price', 'require_parameters': schema},
            'usage': {'include': True}}
    if schema:
        body['response_format'] = SCHEMA
    delay = 2.0
    for _ in range(10):
        r = requests.post('https://openrouter.ai/api/v1/chat/completions', json=body,
                          headers={'Authorization': f'Bearer {key}'}, timeout=300)
        if r.status_code == 200:
            try:
                j = r.json()
            except ValueError:
                time.sleep(delay); delay = min(delay * 2, 60); continue
            if 'choices' not in j:
                time.sleep(delay); delay = min(delay * 2, 60); continue
            if j['choices'][0].get('finish_reason') == 'length':
                body['max_tokens'] = 8000
                body['reasoning'] = {'effort': 'medium'}
                continue
            return j['choices'][0]['message']['content'] or ''
        if r.status_code in (429, 500, 502, 503, 529):
            time.sleep(max(delay, float(r.headers.get('retry-after', 0) or 0)))
            delay = min(delay * 2, 60)
            continue
        raise RuntimeError(f'{r.status_code}: {r.text[:200]}')
    raise RuntimeError('chat retries exhausted')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--judge-model', default='deepseek/deepseek-chat-v3.1')
    ap.add_argument('--root', default='data/traces')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
    key = os.environ['OPENROUTER_API_KEY']
    safe = args.judge_model.split('/')[-1]
    qdir = f'data/judge/smallcohort-qspec-{safe}'
    fdir = f'data/judge/smallcohort-freeform-{safe}'

    print('loading corpus...')
    traces = load_langfuse_corpus(args.root)
    print(f'{len(traces)} traces')

    # render + cache texts
    texts = {}
    for tr in tqdm(traces, desc='render'):
        p = os.path.join(TXT, tr.model_id, str(tr.replicate), f'{tr.query_id}.txt')
        if not os.path.exists(p):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, 'w').write(render(tr))
        texts[(tr.model_id, tr.replicate, tr.query_id)] = open(p).read()

    # rubrics for the 12 instances
    queries = sorted({tr.query_id for tr in traces})
    print(f'{len(queries)} instances')
    os.makedirs(RUB, exist_ok=True)
    missing_rub = [q for q in queries if not os.path.exists(os.path.join(RUB, f'{q}.json'))]
    if missing_rub:
        from datasets import load_dataset
        stmts = {r['instance_id']: r['problem_statement'][:20_000]
                 for r in load_dataset('princeton-nlp/SWE-bench_Verified', split='test')
                 if r['instance_id'] in set(missing_rub)}
        for q in tqdm(missing_rub, desc='rubrics'):
            out = chat(key, args.judge_model, QSPEC_PROMPT.format(problem=stmts[q]),
                       max_tokens=1500)
            json.loads(out)          # validate before caching
            open(os.path.join(RUB, f'{q}.json'), 'w').write(out)
    rubrics = {q: json.loads(open(os.path.join(RUB, f'{q}.json')).read())
               for q in queries}

    def qpath(k):
        m, r, q = k
        return os.path.join(qdir, m, str(r), f'{q}.json')

    def fpath(k):
        m, r, q = k
        return os.path.join(fdir, m, str(r), f'{q}.txt')

    todo = []
    for k in texts:
        if not os.path.exists(qpath(k)):
            todo.append(('q', k))
        if not os.path.exists(fpath(k)):
            todo.append(('f', k))
    if args.limit:
        todo = todo[:args.limit]
    print(f'{len(todo)} judge calls todo')

    def run(job):
        kind, k = job
        _, _, q = k
        try:
            if kind == 'q':
                rub = '\n'.join(f'- {s}: {rubrics[q].get(s, "")}' for s in SECTIONS)
                base = EXTRACT_PROMPT.format(rubric=rub, keys=SECTIONS)
                out = chat(key, args.judge_model,
                           base + '\n\n=== TRACE ===\n' + texts[k])
                path = qpath(k)
            else:
                out = chat(key, args.judge_model,
                           FREEFORM_PROMPT + '\n\n=== TRACE ===\n' + texts[k],
                           schema=False)
                path = fpath(k)
        except RuntimeError as e:
            print(f'FAIL {kind} {k}: {e}', file=sys.stderr)
            return
        if out.strip():
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, 'w').write(out)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(tqdm(ex.map(run, todo), total=len(todo), desc=safe))

    n_q = sum(os.path.exists(qpath(k)) for k in texts)
    n_f = sum(os.path.exists(fpath(k)) for k in texts)
    print(f'qspec {n_q}/{len(texts)}  freeform {n_f}/{len(texts)}')


if __name__ == '__main__':
    main()
