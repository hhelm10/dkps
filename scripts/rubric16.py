"""16-field rubric sensitivity experiment -- extraction pipeline
(colleague proposal, config frozen 2026-09-10):

  panel      q20 (107 systems x 20 queries; q20 = structured-qspec files)
  writer/judge  deepseek/deepseek-v4-flash-0731 (paper era)
  embedder   text-embedding-3-small
  inputs     data/judge/trace_texts_full_pruned (500K-char cap renders)
  fields     6 existing criteria VERBATIM + 10 additions (proposal text)

Stages (resumable):
  qrubrics -> data/judge/rubric16/qrubrics/<q>.json   (writer sees field
              definitions appended to the problem statement)
  extract  -> data/judge/rubric16/<arm>/<sys>/<q>.json, arm in
              {generic, qspec}
  embed    -> data/judge/rubric16_emb_<arm>.npz

Usage: python scripts/rubric16.py --stage qrubrics|extract|embed
                                  [--arm generic|qspec]
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
from dkps.traces.qubric import embed_graded, grade_traces, write_rubrics

JUDGE = 'deepseek/deepseek-v4-flash-0731'
TXT = 'data/judge/trace_texts_full_pruned'
ROOT = 'data/judge/rubric16'
Q20_DIR = 'data/judge/structured-qspec'

# first six verbatim from scripts/judge_structured.py FIXED_RUBRIC
FIELDS = {
    'understanding': 'how the agent oriented itself before acting',
    'localization': 'how it searched/navigated to find the relevant code',
    'reproduction': 'whether/how it reproduced the issue before fixing',
    'editing': 'what it changed (files, nature and size of the edit)',
    'verification': ('whether it re-ran tests or a repro script after '
                     'editing, and the OBSERVED outcome '
                     '(passing/failing/errors)'),
    'final_state': ('how the run ended (clean finish, step/cost limit, '
                    'submitted unverified)'),
    'requirements': ('which expected behaviors, constraints, and acceptance '
                     'conditions the agent identified'),
    'causal_diagnosis': ('what cause it proposed and what trace evidence '
                         'supported or contradicted that cause'),
    'code_context': ('how it inspected callers, dependencies, interfaces, '
                     'or surrounding implementation'),
    'planning': ('what concrete plan it stated and how subsequent actions '
                 'followed or revised it'),
    'test_design': ('what tests or reproductions it added or modified and '
                    'which behaviors they exercised'),
    'edge_cases': ('which boundary conditions or alternative inputs it '
                   'considered or checked'),
    'regression_scope': ('what related behavior it checked beyond the '
                         'reported failure'),
    'feedback_response': ('how it changed course after test failures, '
                          'errors, or contradictory evidence'),
    'tool_recovery': ('how it handled failed commands, environment '
                      'problems, or unavailable dependencies'),
    'patch_review': ('whether/how it inspected the final diff, removed '
                     'temporary changes, or checked patch scope'),
}
FIELDS_EXTRA = {
    'exploration_breadth': ('how many distinct files, modules, or '
                            'approaches it explored before committing to '
                            'one'),
    'hypothesis_testing': ('what explicit hypotheses it stated and how it '
                           'confirmed or refuted them'),
    'search_strategy': ('the search/grep/navigation patterns it used to '
                        'find information'),
    'dependency_handling': ('how it dealt with imports, environment '
                            'setup, or package versions'),
    'error_interpretation': ('how it read tracebacks or error messages '
                             'and what it concluded from them'),
    'incrementalism': ('whether it worked in small verified steps or '
                       'large unverified jumps'),
    'backtracking': ('which approaches it abandoned or reverted, and '
                     'why'),
    'code_reuse': ('whether it followed existing repository conventions '
                   'and helpers or reinvented functionality'),
    'instruction_adherence': ('how closely it followed the task '
                              'constraints and stated scope'),
    'efficiency': ('wasted actions, repeated commands, or redundant '
                   'reads in the run'),
    'state_tracking': ('whether it kept accurate track of file and '
                       'system state across steps'),
    'assumption_checking': ('which assumptions it made without '
                            'verification versus checked explicitly'),
    'output_quality': ('the clarity and structure of its final summary '
                       'or explanation'),
    'risk_management': ('how it handled potentially destructive commands '
                        'or guarded against breaking working code'),
    'domain_knowledge': ('what external knowledge (APIs, algorithms, '
                         'standards) it applied to the task'),
    'completion_confidence': ('what certainty the agent expressed about '
                              'its solution and on what basis'),
}
FIELDS32 = {**FIELDS, **FIELDS_EXTRA}
SECTIONS = tuple(FIELDS)
SECTIONS32 = tuple(FIELDS32)


def bank(n):
    """(fields dict, sections tuple, cache root, emb prefix) for a bank."""
    if n == 16:
        return FIELDS, SECTIONS, 'data/judge/rubric16', 'rubric16'
    if n == 32:
        return FIELDS32, SECTIONS32, 'data/judge/rubric32', 'rubric32'
    raise ValueError(n)


def panel():
    labels = json.load(open('data/leaderboard/verified_labels.json'))
    systems = sorted(s for s in os.listdir(Q20_DIR)
                     if 'resolved' in labels.get(s, {}))
    any_sys = os.path.join(Q20_DIR, systems[0])
    q20 = sorted(f[:-5] for f in os.listdir(any_sys) if f.endswith('.json'))
    assert len(q20) == 20, len(q20)
    return labels, systems, q20


def stage_qrubrics(key, args):
    _, _, q20 = panel()
    outd = f'{ROOT}/qrubrics'
    os.makedirs(outd, exist_ok=True)
    missing = [q for q in q20 if not os.path.exists(f'{outd}/{q}.json')]
    print(f'{len(missing)} qrubrics to write')
    if not missing:
        return
    from datasets import load_dataset
    stmts = {r['instance_id']: r['problem_statement']
             for r in load_dataset('princeton-nlp/SWE-bench_Verified',
                                   split='test')
             if r['instance_id'] in set(missing)}
    defs = '\n'.join(f'- {k}: {v}' for k, v in FIELDS.items())
    tasks = {q: stmts[q] + '\n\nField definitions (specialize each to '
             'THIS issue):\n' + defs for q in missing}
    rubs = write_rubrics(tasks, key, JUDGE, sections=SECTIONS, workers=8)
    for q, rub in rubs.items():
        open(f'{outd}/{q}.json', 'w').write(json.dumps(rub))
    print(f'wrote {len(rubs)}')


def stage_extract(key, args):
    from concurrent.futures import ThreadPoolExecutor
    qubric_mod.set_concurrency_gate(128)
    _, systems, q20 = panel()
    arm = args.arm
    outd = f'{ROOT}/{arm}'
    if arm == 'generic':
        rubrics = dict(FIELDS)          # one fixed rubric
        task_ids = None
    else:
        rubrics = {q: json.loads(open(f'{ROOT}/qrubrics/{q}.json').read())
                   for q in q20}

    def one_system(s):
        todo = {q: open(os.path.join(TXT, s, f'{q}.txt')).read()
                for q in q20
                if not os.path.exists(os.path.join(outd, s, f'{q}.json'))
                and os.path.exists(os.path.join(TXT, s, f'{q}.txt'))}
        if not todo:
            return 0
        kw = dict(sections=SECTIONS, workers=16,
                  max_trace_chars=1_000_000, on_error='skip')
        if arm == 'qspec':
            kw['task_ids'] = {q: q for q in todo}
        graded = grade_traces(rubrics, todo, key, JUDGE, **kw)
        os.makedirs(os.path.join(outd, s), exist_ok=True)
        for q, g in graded.items():
            open(os.path.join(outd, s, f'{q}.json'), 'w').write(
                json.dumps(g))
        return len(graded)

    with ThreadPoolExecutor(max_workers=12) as ex:
        list(tqdm(ex.map(one_system, systems), total=len(systems),
                  desc=f'extract:{arm}'))
    total = sum(os.path.exists(os.path.join(outd, s, f'{q}.json'))
                for s in systems for q in q20)
    print(f'{arm} coverage {total}/{len(systems) * len(q20)}')


def stage_embed(key_openai, args):
    _, systems, q20 = panel()
    arm = args.arm
    graded = []
    n_bad = 0
    for s in systems:
        for q in q20:
            try:
                d = json.loads(open(f'{ROOT}/{arm}/{s}/{q}.json').read())
                if isinstance(d, list) and d:
                    d = d[0]
                if not isinstance(d, dict):
                    d = {}
            except (json.JSONDecodeError, FileNotFoundError):
                d = {}
            n_bad += not d
            graded.append(d)
    print(f'{arm}: {len(graded)} traces ({n_bad} empty)')
    X = embed_graded(graded, key_openai, 'text-embedding-3-small',
                     sections=SECTIONS)
    out = f'data/judge/{PREFIX}_emb_{arm}.npz'
    np.savez_compressed(out, X=X.astype(np.float32))
    print('wrote', out, X.shape)


PREFIX = 'rubric16'


def main():
    global FIELDS, SECTIONS, ROOT, PREFIX
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True,
                    choices=['qrubrics', 'extract', 'embed'])
    ap.add_argument('--arm', default='qspec',
                    choices=['generic', 'qspec'])
    ap.add_argument('--bank', type=int, default=16, choices=[16, 32])
    args = ap.parse_args()
    FIELDS, SECTIONS, ROOT, PREFIX = bank(args.bank)
    load_dotenv()
    if args.stage == 'qrubrics':
        stage_qrubrics(os.environ['OPENROUTER_API_KEY'], args)
    elif args.stage == 'extract':
        stage_extract(os.environ['OPENROUTER_API_KEY'], args)
    else:
        stage_embed(os.environ['OPENAI_API_KEY'], args)


if __name__ == '__main__':
    main()
