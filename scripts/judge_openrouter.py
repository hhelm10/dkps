"""Extraction-only qubric judging via OpenRouter (open-weight judges).

Migration step 1 (PAPER.md): re-run q20 extraction with open judges on the
IDENTICAL cached gpt-5.4-mini-written rubrics, so only the extractor changes.
Caches to data/judge/structured-qspec-<safe_model>/<sys>/<q>.json -- the same
layout judge_matrix.py consumes.

Efficiency (audited 2026-08-19):
- Rendered trace texts cached once to data/judge/trace_texts/<sys>/<q>.txt;
  every judge run after the first skips the ~10-min 40GB re-parse.
- Provider routing: sort=price + require_parameters=True -- cheapest provider
  that actually honors response_format (gpt-oss-20b: $0.03/M in vs $0.07 at
  the default-routing worst case; Bedrock ignores response_format entirely).
- Strict json_schema output (6 required string sections): malformed shapes
  (list-wrapping, missing keys) become impossible instead of patched-around.
- Reasoning effort low: extraction is transcription, not reasoning.
- Usage accumulated from every response; run prints measured $ cost.

Usage:
  python scripts/judge_openrouter.py --judge-model openai/gpt-oss-20b [--limit N]
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import requests
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from judge_structured import EXTRACT_PROMPT, SECTIONS, _load_one  # noqa: E402

TEXT_CACHE = 'data/judge/trace_texts'

SCHEMA = {
    'type': 'json_schema',
    'json_schema': {
        'name': 'trace_sections', 'strict': True,
        'schema': {
            'type': 'object',
            'properties': {k: {'type': 'string'} for k in SECTIONS},
            'required': list(SECTIONS),
            'additionalProperties': False,
        },
    },
}

usage_lock = threading.Lock()
usage_tot = {'prompt': 0, 'completion': 0, 'cost': 0.0}


def chat(key, model, content, max_tokens=2500):
    body = {'model': model, 'messages': [{'role': 'user', 'content': content}],
            'max_tokens': max_tokens,
            'response_format': SCHEMA,
            'reasoning': {'effort': 'low'},
            'provider': {'sort': 'price', 'require_parameters': True},
            'usage': {'include': True}}
    delay = 2.0
    for _ in range(10):
        r = requests.post('https://openrouter.ai/api/v1/chat/completions', json=body,
                          headers={'Authorization': f'Bearer {key}'}, timeout=300)
        if r.status_code == 200:
            j = r.json()
            if 'choices' not in j:            # provider-level error in 200 body
                time.sleep(delay); delay = min(delay * 2, 60); continue
            u = j.get('usage') or {}
            with usage_lock:
                usage_tot['prompt'] += u.get('prompt_tokens', 0)
                usage_tot['completion'] += u.get('completion_tokens', 0)
                usage_tot['cost'] += u.get('cost', 0) or 0
            if j['choices'][0].get('finish_reason') == 'length':
                return None                    # truncated JSON -> retry next run
            return j['choices'][0]['message']['content'] or '{}'
        if r.status_code in (429, 500, 502, 503, 529):
            time.sleep(max(delay, float(r.headers.get('retry-after', 0) or 0)))
            delay = min(delay * 2, 60)
            continue
        raise RuntimeError(f'{r.status_code}: {r.text[:200]}')
    raise RuntimeError('chat retries exhausted')


def get_texts(root, systems, q20):
    """Rendered trace text per (system, instance), disk-cached once."""
    texts = {}
    missing = []
    for s in systems:
        d = os.path.join(TEXT_CACHE, s)
        if all(os.path.exists(os.path.join(d, f'{q}.txt')) for q in q20):
            texts[s] = {q: open(os.path.join(d, f'{q}.txt')).read() for q in q20}
        else:
            missing.append(s)
    if missing:
        print(f'rendering trace texts for {len(missing)} systems (one-time)')
        with Pool(16) as pool:
            for s, out in tqdm(pool.imap_unordered(
                    _load_one, [(os.path.join(root, s), set(q20)) for s in missing]),
                    total=len(missing), desc='render'):
                os.makedirs(os.path.join(TEXT_CACHE, s), exist_ok=True)
                for q, t in out.items():
                    open(os.path.join(TEXT_CACHE, s, f'{q}.txt'), 'w').write(t)
                texts[s] = out
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--judge-model', required=True)
    ap.add_argument('--root', default='data/leaderboard/verified')
    ap.add_argument('--labels', default='data/leaderboard/verified_labels.json')
    ap.add_argument('--rubrics', default='data/judge/qspec_rubrics')
    ap.add_argument('--ref-cache', default='data/judge/structured-qspec')
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--limit', type=int, default=0,
                    help='only run this many calls (pilot mode)')
    ap.add_argument('--prune', action='store_true',
                    help='prune low-information content before judging '
                         '(dkps.traces.prune); caches to a -pruned dir')
    args = ap.parse_args()

    load_dotenv()
    key = os.environ['OPENROUTER_API_KEY']

    q20 = sorted(f[:-5] for f in os.listdir(
        os.path.join(args.ref_cache, sorted(os.listdir(args.ref_cache))[0])))
    labels = json.load(open(args.labels))
    systems = sorted(s for s in os.listdir(args.ref_cache)
                     if 'resolved' in labels.get(s, {}))
    rubrics = {q: json.loads(open(os.path.join(args.rubrics, f'{q}.json')).read())
               for q in q20}

    safe = args.judge_model.replace('/', '_')
    jdir = f'data/judge/structured-qspec-{safe}' + ('-pruned' if args.prune else '')

    def jpath(s, q):
        return os.path.join(jdir, s, f'{q}.json')

    todo = [(s, q) for s in systems for q in q20 if not os.path.exists(jpath(s, q))]
    if args.limit:
        todo = todo[:args.limit]
    print(f'{len(systems)} systems x {len(q20)} instances; {len(todo)} calls todo')
    if not todo:
        return

    texts = get_texts(args.root, sorted({s for s, _ in todo}), q20)
    if args.prune:
        from dkps.traces.prune import prune_for_judging
        texts = {s: {q: prune_for_judging(t) for q, t in d.items()}
                 for s, d in texts.items()}

    def extract(pair):
        s, q = pair
        rub = '\n'.join(f'- {k}: {rubrics[q].get(k, "")}' for k in SECTIONS)
        base = EXTRACT_PROMPT.format(rubric=rub, keys=SECTIONS)
        content = base + '\n\n=== TRACE ===\n' + texts[s].get(q, '(empty trace)')
        try:
            out = chat(key, args.judge_model, content)
        except RuntimeError as e:
            print(f'FAIL {s}/{q}: {e}', file=sys.stderr)
            return
        if out is None:
            print(f'TRUNCATED {s}/{q}', file=sys.stderr)
            return
        os.makedirs(os.path.dirname(jpath(s, q)), exist_ok=True)
        open(jpath(s, q), 'w').write(out)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(tqdm(ex.map(extract, todo), total=len(todo), desc=safe))

    n_ok = n_parse = 0
    for s, q in todo:
        if os.path.exists(jpath(s, q)):
            n_ok += 1
            try:
                d = json.loads(open(jpath(s, q)).read())
                if isinstance(d, list) and d:
                    d = d[0]
                if isinstance(d, dict) and any(d.get(k) for k in SECTIONS):
                    n_parse += 1
            except json.JSONDecodeError:
                pass
    dt = time.time() - t0
    print(f'written {n_ok}/{len(todo)}; parseable-with-content {n_parse}; '
          f'{dt/60:.1f} min ({len(todo)/max(dt,1)*60:.0f} calls/min)')
    print(f"usage: {usage_tot['prompt']:,} prompt + {usage_tot['completion']:,} "
          f"completion tokens; measured cost ${usage_tot['cost']:.4f}")


if __name__ == '__main__':
    main()
