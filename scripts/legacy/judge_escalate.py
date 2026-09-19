"""Escalated retry for persistently-empty extractions (long traces).

gpt-oss-20b at effort=low returns empty content on some ~60K-char traces
(reasoning consumes the completion budget). Escalation: max_tokens 8000,
effort medium, provider unpinned, parallel with verification-before-write.

Usage: python scripts/judge_escalate.py --judge-model openai/gpt-oss-20b
"""
import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from judge_structured import EXTRACT_PROMPT, SECTIONS  # noqa: E402
from judge_openrouter import SCHEMA  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--judge-model', required=True)
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
    key = os.environ['OPENROUTER_API_KEY']
    safe = args.judge_model.replace('/', '_')

    bad = []
    for p in glob.glob(f'data/judge/structured-qspec-{safe}/*/*.json'):
        try:
            d = json.loads(open(p).read())
            if isinstance(d, list) and d:
                d = d[0]
            if not (isinstance(d, dict) and any(d.get(k) for k in SECTIONS)):
                bad.append(p)
        except json.JSONDecodeError:
            bad.append(p)
    print(len(bad), 'to escalate')
    if not bad:
        return
    rubrics = {q: json.loads(open(f'data/judge/qspec_rubrics/{q}.json').read())
               for q in {p.split('/')[-1][:-5] for p in bad}}

    def fix(p):
        s, q = p.split('/')[-2], p.split('/')[-1][:-5]
        trace = open(f'data/judge/trace_texts/{s}/{q}.txt').read()
        rub = '\n'.join(f'- {k}: {rubrics[q].get(k, "")}' for k in SECTIONS)
        content = (EXTRACT_PROMPT.format(rubric=rub, keys=SECTIONS)
                   + '\n\n=== TRACE ===\n' + trace)
        body = {'model': args.judge_model,
                'messages': [{'role': 'user', 'content': content}],
                'max_tokens': 8000, 'response_format': SCHEMA,
                'reasoning': {'effort': 'medium'},
                'provider': {'require_parameters': True}}
        for _ in range(5):
            try:
                r = requests.post('https://openrouter.ai/api/v1/chat/completions',
                                  json=body, headers={'Authorization': f'Bearer {key}'},
                                  timeout=300)
            except requests.RequestException:
                time.sleep(5)
                continue
            if r.status_code != 200:
                time.sleep(5)
                continue
            j = r.json()
            if 'choices' not in j:
                time.sleep(5)
                continue
            out = j['choices'][0]['message']['content'] or ''
            try:
                d = json.loads(out)
                if isinstance(d, dict) and any(d.get(k) for k in SECTIONS):
                    open(p, 'w').write(out)
                    return True
            except json.JSONDecodeError:
                pass
        return False

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(fix, bad))
    print(f'recovered {sum(results)}/{len(bad)}')


if __name__ == '__main__':
    main()
