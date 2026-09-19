"""Pipeline test-retest stability (the aligned definition, HH 2026-08-28):
the SAME trace, processed twice, should map to ~the same place. Stochastic
stages: rubric creation (Model 1) and extraction (Model 2). Raw-embedding
rows are deterministic (stability 1.0 by construction).

Two retest levels on q20 x 107 systems, judge deepseek-v3.1 via the
dkps.traces.qubric API:
  extraction : same (mini-written) rubrics, resampled extraction
               -> data/judge/structured-qspec-retestX/
  pipeline   : rubrics REWRITTEN by deepseek, then extraction
               -> data/judge/qspec_rubrics_ds/ + structured-qspec-dsrubric/

Usage: python scripts/stability_retest.py --stage extraction|pipeline
"""
import argparse
import json
import os
import sys

from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dkps.traces.qubric import grade_traces, write_rubrics  # noqa: E402

JUDGE = 'deepseek/deepseek-chat-v3.1'
TXT = 'data/judge/trace_texts'


def load_corpus():
    labels = json.load(open('data/leaderboard/verified_labels.json'))
    ref = 'data/judge/structured-qspec'
    q20 = sorted(f[:-5] for f in os.listdir(
        os.path.join(ref, sorted(os.listdir(ref))[0])))
    systems = sorted(s for s in os.listdir(ref)
                     if 'resolved' in labels.get(s, {}))
    return systems, q20


def run_extraction(key, systems, q20, rubrics, outdir):
    for s in tqdm(systems, desc=os.path.basename(outdir)):
        todo = {q: open(os.path.join(TXT, s, f'{q}.txt')).read()
                for q in q20
                if not os.path.exists(os.path.join(outdir, s, f'{q}.json'))}
        if not todo:
            continue
        graded = grade_traces(rubrics, todo, key, JUDGE,
                              task_ids={q: q for q in todo}, workers=10,
                              on_error='skip')
        os.makedirs(os.path.join(outdir, s), exist_ok=True)
        for q, g in graded.items():
            open(os.path.join(outdir, s, f'{q}.json'), 'w').write(json.dumps(g))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=('extraction', 'pipeline'),
                    required=True)
    args = ap.parse_args()
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
    key = os.environ['OPENROUTER_API_KEY']
    systems, q20 = load_corpus()

    if args.stage == 'extraction':
        rubrics = {q: json.loads(open(f'data/judge/qspec_rubrics/{q}.json').read())
                   for q in q20}
        run_extraction(key, systems, q20, rubrics,
                       'data/judge/structured-qspec-retestX')
    else:
        rdir = 'data/judge/qspec_rubrics_ds'
        os.makedirs(rdir, exist_ok=True)
        missing = [q for q in q20
                   if not os.path.exists(os.path.join(rdir, f'{q}.json'))]
        if missing:
            from datasets import load_dataset
            stmts = {r['instance_id']: r['problem_statement']
                     for r in load_dataset('princeton-nlp/SWE-bench_Verified',
                                           split='test')
                     if r['instance_id'] in set(missing)}
            rubs = write_rubrics({q: stmts[q] for q in missing}, key, JUDGE)
            for q, rub in rubs.items():
                open(os.path.join(rdir, f'{q}.json'), 'w').write(json.dumps(rub))
        rubrics = {q: json.loads(open(os.path.join(rdir, f'{q}.json')).read())
                   for q in q20}
        run_extraction(key, systems, q20, rubrics,
                       'data/judge/structured-qspec-dsrubric')


if __name__ == '__main__':
    main()
