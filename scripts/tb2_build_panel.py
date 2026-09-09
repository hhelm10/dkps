"""Assemble the TB2 outcome panel from downloaded result.json files.

Writes data/terminal_bench/tb2_panel.json:
  systems     list of submission dirnames
  tasks       sorted list of 89 task names
  y           mean reward per system (over all trials, official-score proxy)
  B           per-system x task mean reward across trials (for IRT we also
              keep B_first: first-job binary outcomes)
  trials      n trials per (system, task)
  meta        per system: agent display name, model names/providers, n_jobs
Also prints the confounder roster (LLM / harness tag coverage).
"""
import json
import os
from collections import defaultdict

import numpy as np
import yaml

ROOT = 'data/terminal_bench/tb2/submissions/terminal-bench/2.0'
OUT = 'data/terminal_bench/tb2_panel.json'

import re


def norm_llm(name):
    """Canonicalize a model_name string to an LLM tag."""
    n = name.lower().split('/')[-1]
    n = n.replace('.', '-')
    n = re.sub(r'-highspeed$', '', n)
    m = re.match(r'claude-(\d-\d)-(opus|sonnet)$', n)
    if m:
        n = f'claude-{m.group(2)}-{m.group(1)}'
    if n == 'gpt-3-5-codex':  # metadata typo in one submission
        n = 'gpt-5-3-codex'
    return n


HARNESS_ALIAS = {'wozcode': 'wozcode', 'terminus2': 'terminus2',
                 'terminus-2': 'terminus2', 'codebrain-1-5': 'codebrain-1',
                 'logos-ts': 'logos', 'logos-latest': 'logos'}


def norm_harness(agent):
    a = re.sub(r'[^a-z0-9.-]+', '-', agent.lower()).strip('-')
    a = a.replace('.', '-')
    return HARNESS_ALIAS.get(a, a)


def main():
    systems = sorted(d for d in os.listdir(ROOT)
                     if os.path.isdir(os.path.join(ROOT, d)))
    panel = {}
    meta = {}
    for s in systems:
        sdir = os.path.join(ROOT, s)
        mpath = os.path.join(sdir, 'metadata.yaml')
        m = yaml.safe_load(open(mpath)) if os.path.exists(mpath) else {}
        jobs = sorted(d for d in os.listdir(sdir)
                      if os.path.isdir(os.path.join(sdir, d)))
        rewards = defaultdict(list)
        for j in jobs:
            jdir = os.path.join(sdir, j)
            for t in sorted(os.listdir(jdir)):
                rp = os.path.join(jdir, t, 'result.json')
                if not os.path.exists(rp):
                    continue
                try:
                    r = json.load(open(rp))
                except json.JSONDecodeError:
                    continue
                task = r.get('task_name') or t.rsplit('__', 1)[0]
                # ~6 submissions prefix names with 'terminal-bench/'; one
                # writes the 3.11 task with dashes
                task = task.removeprefix('terminal-bench/')
                if task == 'install-windows-3-11':
                    task = 'install-windows-3.11'
                rw = ((r.get('verifier_result') or {}).get('rewards')
                      or {}).get('reward')
                if rw is not None:
                    rewards[task].append(float(rw))
        panel[s] = dict(rewards)
        raw_models = [mm.get('model_name', '?')
                      for mm in (m or {}).get('models', [])]
        agent = (m or {}).get('agent_display_name', s.split('__')[0])
        meta[s] = {
            'agent': agent,
            'models': raw_models,
            'llm_tags': sorted({norm_llm(x) for x in raw_models
                                if x not in ('?', 'multiple')}),
            'harness': norm_harness(agent),
            'n_jobs': len(jobs),
        }

    tasks = sorted({t for p in panel.values() for t in p})
    print(f'{len(systems)} systems, {len(tasks)} tasks')
    keep = [s for s in systems if len(panel[s]) >= len(tasks) * 0.9]
    drop = [s for s in systems if s not in keep]
    if drop:
        print('dropping (coverage <90%):',
              [(s, len(panel[s])) for s in drop])

    B = np.full((len(keep), len(tasks)), np.nan)
    ntr = np.zeros((len(keep), len(tasks)), int)
    for i, s in enumerate(keep):
        for j, t in enumerate(tasks):
            v = panel[s].get(t, [])
            if v:
                B[i, j] = np.mean(v)
                ntr[i, j] = len(v)
    y = np.nanmean(B, axis=1)

    out = {
        'systems': keep, 'tasks': tasks,
        'y': [round(float(v), 5) for v in y],
        'B': [[None if np.isnan(v) else round(float(v), 4) for v in row]
              for row in B],
        'n_trials': ntr.tolist(),
        'meta': {s: meta[s] for s in keep},
    }
    json.dump(out, open(OUT, 'w'))
    print(f'wrote {OUT}')

    print('\ntop/bottom by score:')
    order = np.argsort(-y)
    for i in list(order[:5]) + list(order[-3:]):
        print(f'  {y[i]:.3f}  {keep[i]}  (trials/task median '
              f'{int(np.median(ntr[i]))})')

    llms = defaultdict(list)
    for s in keep:
        for t in meta[s]['llm_tags'] or ['(multi/unknown)']:
            llms[t].append(s)
    shared = {k: v for k, v in llms.items() if len(v) > 1}
    print(f'\ncanonical LLM tags: {len(llms)}; shared by 2+ systems: '
          f'{len(shared)}')
    for k, v in sorted(shared.items(), key=lambda kv: -len(kv[1])):
        print(f'  {len(v):2d}  {k}')
    agents = defaultdict(list)
    for s in keep:
        agents[meta[s]['harness']].append(s)
    shared_a = {k: v for k, v in agents.items() if len(v) > 1}
    print(f'harness tags: {len(agents)}; shared by 2+: {len(shared_a)}')
    for k, v in sorted(shared_a.items(), key=lambda kv: -len(kv[1])):
        print(f'  {len(v):2d}  {k}')

    # protocol viability: mean allowed references per target
    M = len(keep)
    def refs(excl):
        return np.mean([sum(1 for j in range(M) if j != i and not excl(i, j))
                        for i in range(M)])
    tags = [set(meta[s]['llm_tags']) for s in keep]
    hs = [meta[s]['harness'] for s in keep]
    print(f'\nmean refs/target: LOSO {refs(lambda i, j: False):.1f}, '
          f'LLM-out {refs(lambda i, j: bool(tags[i] & tags[j])):.1f}, '
          f'harness-out {refs(lambda i, j: hs[i] == hs[j]):.1f}')


if __name__ == '__main__':
    main()
