"""Generic-rubric (fixed six original criteria, no Model 1) extraction +
embedding on the paper panels, for the hero-table generic-geometry row.

  swe   107 systems x q100, inputs data/judge/trace_texts_full_pruned
        -> data/judge/q100-generic-flash/<sys>/<q>.json
        -> data/judge/q100_generic_emb_openai_small.npz
  tb2   72 kept systems x 89, inputs data/terminal_bench/tb2_txt
        -> data/terminal_bench/tb2-generic-flash/<sys>/<t>.json
        -> data/terminal_bench/tb2_generic_emb_openai_small.npz

Judge deepseek-v4-flash-0731, gate 128, sections = original six verbatim.
Usage: python scripts/generic_run.py --stage extract|embed --bench swe|tb2
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
import dkps.traces.qubric as qubric_mod
from dkps.traces.qubric import embed_graded, grade_traces
from rubric16 import FIELDS

JUDGE = 'deepseek/deepseek-v4-flash-0731'
SIX = {k: FIELDS[k] for k in list(FIELDS)[:6]}
SECTIONS = tuple(SIX)
DROP = {'Droid__GPT-5.3-Codex', 'Droid__Claude-Opus-4.6',
        'just-another-coding-agent__GLM-5'}


def cfg(bench):
    if bench == 'swe':
        from run_q100 import load_panel
        labels, systems, qs = load_panel()
        return systems, qs, 'data/judge/trace_texts_full_pruned', \
            'data/judge/q100-generic-flash', \
            'data/judge/q100_generic_emb_openai_small.npz'
    d = json.load(open('data/terminal_bench/tb2_panel.json'))
    all_sys = [s for s in d['systems'] if s not in DROP]
    return all_sys, d['tasks'], 'data/terminal_bench/tb2_txt', \
        'data/terminal_bench/tb2-generic-flash', \
        'data/terminal_bench/tb2_generic_emb_openai_small.npz'


def stage_extract(key, bench):
    from concurrent.futures import ThreadPoolExecutor
    qubric_mod.set_concurrency_gate(128)
    systems, qs, txt, outd, _ = cfg(bench)

    def one_system(s):
        todo = {q: open(os.path.join(txt, s, f'{q}.txt')).read()
                for q in qs
                if not os.path.exists(os.path.join(outd, s, f'{q}.json'))
                and os.path.exists(os.path.join(txt, s, f'{q}.txt'))}
        if not todo:
            return 0
        graded = grade_traces(dict(SIX), todo, key, JUDGE,
                              sections=SECTIONS, workers=16,
                              max_trace_chars=1_000_000, on_error='skip')
        os.makedirs(os.path.join(outd, s), exist_ok=True)
        for q, g in graded.items():
            open(os.path.join(outd, s, f'{q}.json'), 'w').write(
                json.dumps(g))
        return len(graded)

    with ThreadPoolExecutor(max_workers=12) as ex:
        list(tqdm(ex.map(one_system, systems), total=len(systems),
                  desc=f'generic:{bench}'))
    n = sum(os.path.exists(os.path.join(outd, s, f'{q}.json'))
            for s in systems for q in qs)
    print(f'{bench} coverage {n}/{len(systems) * len(qs)}')


def stage_embed(key_openai, bench):
    systems, qs, _, outd, emb = cfg(bench)
    graded = []
    for s in systems:
        for q in qs:
            try:
                d = json.loads(open(
                    os.path.join(outd, s, f'{q}.json')).read())
                if isinstance(d, list) and d:
                    d = d[0]
                if not isinstance(d, dict):
                    d = {}
            except (json.JSONDecodeError, FileNotFoundError):
                d = {}
            graded.append(d)
    print(f'{bench}: {len(graded)} ({sum(not g for g in graded)} empty)')
    X = embed_graded(graded, key_openai, 'text-embedding-3-small',
                     sections=SECTIONS)
    np.savez_compressed(emb, X=X.astype(np.float32))
    print('wrote', emb, X.shape)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['extract', 'embed'])
    ap.add_argument('--bench', required=True, choices=['swe', 'tb2'])
    a = ap.parse_args()
    load_dotenv()
    key = os.environ['OPENROUTER_API_KEY' if a.stage == 'extract'
                     else 'OPENAI_API_KEY']
    (stage_extract if a.stage == 'extract' else stage_embed)(key, a.bench)
