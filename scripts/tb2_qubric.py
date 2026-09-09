"""TB2 qubric run (per HH 2026-09-09): one replicate per system.

Config mirrors the q100 era exactly:
  Model 1 = Model 2   deepseek/deepseek-v4-flash-0731 (OpenRouter)
  embedder            openai/text-embedding-3-small
  inputs              pruned, 500K-char head/tail cap
  panel               data/terminal_bench/tb2_panel.json, minus the drops
                      (2x Droid black-box, just-another-coding-agent GLM-5)

Replicate choice (deterministic): per system the job with the most
result.json files (ties -> lexicographically last, matching
tb2_fetch_traces.best_job); per task the alphabetically first trial dir.

Stages (resumable, per-file caches):
  render   -> data/terminal_bench/tb2_txt/<sys>/<task>.txt
              + tb2_chosen.json (job/trial/reward of the judged replicate)
  rubrics  -> data/terminal_bench/tb2_rubrics/<task>.json
  extract  -> data/terminal_bench/tb2-qspec-flash0731/<sys>/<task>.json
  embed    -> data/terminal_bench/tb2_emb_openai_small.npz
  qvecs    -> data/terminal_bench/tb2_query_vecs_64.npz (instruction
              embeddings, PCA-64, for the PKPS task kernel)

Usage: python scripts/tb2_qubric.py --stage render|rubrics|extract|embed|qvecs
"""
import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from dkps.traces.prune import prune_for_judging  # noqa: E402
from dkps.traces.qubric import (consensus_center, embed_graded,  # noqa: E402
                                grade_traces, write_rubrics)

JUDGE = 'deepseek/deepseek-v4-flash-0731'
ROOT = 'data/terminal_bench/tb2_full/submissions/terminal-bench/2.0'
TASKS = 'data/terminal_bench/tb2_tasks'
PANEL = 'data/terminal_bench/tb2_panel.json'
TXT = 'data/terminal_bench/tb2_txt'
CHOSEN = 'data/terminal_bench/tb2_chosen.json'
RUB = 'data/terminal_bench/tb2_rubrics'
EXT = 'data/terminal_bench/tb2-qspec-flash0731'
EMB = 'data/terminal_bench/tb2_emb_openai_small.npz'
QV = 'data/terminal_bench/tb2_query_vecs_64.npz'
DROP = {'Droid__GPT-5.3-Codex', 'Droid__Claude-Opus-4.6',
        'just-another-coding-agent__GLM-5'}
MAX_CHARS = 500_000
TEXT_EXT = ('.txt', '.json', '.sh', '.log', '.md')
SKIP_NAMES = ('return-code.txt', 'install.sh')


def load_panel():
    d = json.load(open(PANEL))
    systems = [s for s in d['systems'] if s not in DROP]
    return d, systems, d['tasks']


def norm_task(name):
    n = name.removeprefix('terminal-bench/')
    return 'install-windows-3.11' if n == 'install-windows-3-11' else n


def agent_files(tdir):
    """Includable agent-side text files of one trial (whole-trial walk:
    some harnesses put command dirs or logs outside agent/)."""
    files = []
    for dp, dns, fns in os.walk(tdir):
        dns[:] = [d for d in dns if d not in ('verifier', 'setup')]
        for fn in sorted(fns):
            if not fn.endswith(TEXT_EXT + ('.pane',)):
                continue
            if fn in SKIP_NAMES + ('result.json', 'config.json',
                                   'trial.log', 'metadata.yaml'):
                continue
            files.append(os.path.join(dp, fn))
    return files


def best_job(sdir):
    """Prefer the job whose trials actually carry agent text (some
    submitters uploaded verifier-only jobs), then completeness."""
    jobs = [j for j in os.listdir(sdir)
            if os.path.isdir(os.path.join(sdir, j))]

    def score(j):
        jdir = os.path.join(sdir, j)
        trials = [t for t in os.listdir(jdir)
                  if os.path.isdir(os.path.join(jdir, t))]
        with_text = sum(
            any(os.path.getsize(f) > 300 for f in agent_files(
                os.path.join(jdir, t)))
            for t in trials)
        n_res = sum(os.path.exists(os.path.join(jdir, t, 'result.json'))
                    for t in trials)
        return (with_text, n_res, j)

    return max(jobs, key=score) if jobs else None


def cmd_sort_key(path):
    m = re.search(r'command-(\d+)', path)
    return (0, '') if not m else (int(m.group(1)),
                                  os.path.basename(path))


def flatten_json(obj, key=''):
    """Recursively emit 'key: value' lines for string/number leaves --
    structured trajectories become prose the pruner won't destroy."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(flatten_json(v, k))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(flatten_json(v, key))
    elif isinstance(obj, str):
        if obj.strip():
            out.append(f'{key}: {obj}' if key else obj)
    return out


def render_trial(tdir):
    """Concatenate agent-side text of one trial: root files, then
    command-N in order. JSON files are flattened to text; LFS pointer
    stubs skipped; exact-duplicate contents included once."""
    files = agent_files(tdir)
    root = sorted(f for f in files if 'command-' not in f)
    cmds = sorted((f for f in files if 'command-' in f),
                  key=cmd_sort_key)
    seen, parts = set(), []
    for f in root + cmds:
        try:
            t = open(f, errors='replace').read()
        except OSError:
            continue
        if not t.strip() or t.startswith('version https://git-lfs'):
            continue
        if f.endswith('.json'):
            try:
                t = '\n'.join(flatten_json(json.loads(t)))
            except (json.JSONDecodeError, RecursionError):
                pass
        if not t.strip():
            continue
        h = hashlib.sha1(t.encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        rel = os.path.relpath(f, tdir)
        parts.append(f'### {rel}\n{t}')
    return '\n\n'.join(parts)


def cap(text):
    if len(text) <= MAX_CHARS:
        return text
    head, tail = int(MAX_CHARS * .75), int(MAX_CHARS * .25)
    return (text[:head] + '\n\n[... TRUNCATED ...]\n\n' + text[-tail:])


def stage_render(args):
    _, systems, tasks = load_panel()
    os.makedirs(TXT, exist_ok=True)
    chosen = json.load(open(CHOSEN)) if os.path.exists(CHOSEN) else {}
    for s in tqdm(systems, desc='render'):
        sdir = os.path.join(ROOT, s)
        j = best_job(sdir)
        jdir = os.path.join(sdir, j)
        os.makedirs(os.path.join(TXT, s), exist_ok=True)
        trials = sorted(os.listdir(jdir))
        for task in tasks:
            out = os.path.join(TXT, s, f'{task}.txt')
            if os.path.exists(out):
                continue
            cands = [t for t in trials
                     if norm_task(t.rsplit('__', 1)[0]) == task
                     and os.path.isdir(os.path.join(jdir, t))]
            if not cands:
                continue
            t0 = cands[0]
            txt = render_trial(os.path.join(jdir, t0))
            if not txt.strip():
                continue
            txt = cap(prune_for_judging(txt))
            open(out, 'w').write(txt)
            rw = None
            rp = os.path.join(jdir, t0, 'result.json')
            if os.path.exists(rp):
                try:
                    rw = ((json.load(open(rp)).get('verifier_result') or {})
                          .get('rewards') or {}).get('reward')
                except json.JSONDecodeError:
                    pass
            chosen.setdefault(s, {})[task] = dict(job=j, trial=t0, reward=rw)
        json.dump(chosen, open(CHOSEN, 'w'))
    n = sum(os.path.exists(os.path.join(TXT, s, f'{t}.txt'))
            for s in systems for t in tasks)
    sizes = [os.path.getsize(os.path.join(TXT, s, f'{t}.txt'))
             for s in systems for t in tasks
             if os.path.exists(os.path.join(TXT, s, f'{t}.txt'))]
    print(f'rendered {n}/{len(systems) * len(tasks)}; median '
          f'{np.median(sizes)/1e3:.0f}K chars, total {sum(sizes)/1e6:.0f}MB')


def stage_rubrics(key, args):
    _, _, tasks = load_panel()
    os.makedirs(RUB, exist_ok=True)
    missing = [t for t in tasks if not os.path.exists(f'{RUB}/{t}.json')]
    if args.limit:
        missing = missing[:args.limit]
    print(f'{len(missing)} rubrics to write')
    if not missing:
        return
    stmts = {t: open(f'{TASKS}/{t}/instruction.md').read() for t in missing}
    rubs = write_rubrics(stmts, key, JUDGE, workers=8)
    for t, rub in rubs.items():
        open(f'{RUB}/{t}.json', 'w').write(json.dumps(rub))
    print(f'wrote {len(rubs)}')


def stage_extract(key, args):
    from concurrent.futures import ThreadPoolExecutor

    import dkps.traces.qubric as qubric_mod
    qubric_mod.set_concurrency_gate(128)
    _, systems, tasks = load_panel()
    rubrics = {t: json.loads(open(f'{RUB}/{t}.json').read()) for t in tasks
               if os.path.exists(f'{RUB}/{t}.json')}

    def one_system(s):
        todo = {t: open(os.path.join(TXT, s, f'{t}.txt')).read()
                for t in tasks
                if not os.path.exists(os.path.join(EXT, s, f'{t}.json'))
                and os.path.exists(os.path.join(TXT, s, f'{t}.txt'))}
        if args.limit:
            todo = dict(list(todo.items())[:args.limit])
        if not todo:
            return 0
        graded = grade_traces(rubrics, todo, key, JUDGE,
                              task_ids={t: t for t in todo}, workers=16,
                              max_trace_chars=1_000_000, on_error='skip')
        os.makedirs(os.path.join(EXT, s), exist_ok=True)
        for t, g in graded.items():
            open(os.path.join(EXT, s, f'{t}.json'), 'w').write(json.dumps(g))
        return len(graded)

    with ThreadPoolExecutor(max_workers=12) as ex:
        list(tqdm(ex.map(one_system, systems), total=len(systems),
                  desc='extract'))
    total = sum(os.path.exists(os.path.join(EXT, s, f'{t}.json'))
                for s in systems for t in tasks)
    print(f'coverage {total}/{len(systems) * len(tasks)}')


def load_graded(systems, tasks):
    graded = []
    for s in systems:
        for t in tasks:
            try:
                d = json.loads(open(os.path.join(EXT, s, f'{t}.json')).read())
                if isinstance(d, list) and d:
                    d = d[0]
                if not isinstance(d, dict):
                    d = {}
            except (json.JSONDecodeError, FileNotFoundError):
                d = {}
            graded.append(d)
    return graded


def stage_embed(key_openai, args):
    _, systems, tasks = load_panel()
    graded = load_graded(systems, tasks)
    n_bad = sum(not g for g in graded)
    print(f'{len(graded)} graded traces ({n_bad} empty)')
    X = embed_graded(graded, key_openai, 'text-embedding-3-small')
    np.savez_compressed(EMB, X=X.astype(np.float32))
    print('wrote', EMB, X.shape)


def stage_qvecs(key_openai, args):
    import time

    import requests
    import tiktoken
    _, _, tasks = load_panel()
    enc = tiktoken.get_encoding('cl100k_base')
    texts = []
    for t in tasks:
        toks = enc.encode(open(f'{TASKS}/{t}/instruction.md').read(),
                          disallowed_special=())
        texts.append(enc.decode(toks[:8000]) or ' ')
    rows = []
    for i in tqdm(range(0, len(texts), 8), desc='qvecs'):
        for attempt in range(8):
            try:
                r = requests.post(
                    'https://api.openai.com/v1/embeddings',
                    json={'model': 'text-embedding-3-small',
                          'input': texts[i:i + 8]},
                    headers={'Authorization': f'Bearer {key_openai}'},
                    timeout=120)
            except requests.RequestException:
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code == 200:
                rows.extend(d['embedding'] for d in r.json()['data'])
                break
            time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f'embed failed at {i}')
    E = np.asarray(rows, np.float32)
    Ec = E - E.mean(0, keepdims=True)
    U, S, _ = np.linalg.svd(Ec, full_matrices=False)
    V = (U[:, :64] * S[:64]).astype(np.float32)
    np.savez_compressed(QV, ids=np.array(tasks), vecs=V)
    print('wrote', QV, V.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True,
                    choices=['render', 'rubrics', 'extract', 'embed',
                             'qvecs'])
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    load_dotenv()
    if args.stage == 'render':
        stage_render(args)
    elif args.stage in ('rubrics', 'extract'):
        globals()[f'stage_{args.stage}'](os.environ['OPENROUTER_API_KEY'],
                                         args)
    else:
        globals()[f'stage_{args.stage}'](os.environ['OPENAI_API_KEY'], args)


if __name__ == '__main__':
    main()
