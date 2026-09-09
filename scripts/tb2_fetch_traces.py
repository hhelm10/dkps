"""Fetch TB2 agent traces: for each submission, the ONE job with the most
complete trials (from the local outcome download), agent-side text only --
everything under <trial>/agent/ plus command dirs, excluding verifier
output, recordings (.cast), and archives. Same direct-GET machinery as
tb2_fetch_results.py; resumable.

Usage: python scripts/tb2_fetch_traces.py [--manifest-only]
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from huggingface_hub import HfApi

REPO = 'harborframework/terminal-bench-2-leaderboard'
ROOT = 'submissions/terminal-bench/2.0'
LOCAL = 'data/terminal_bench/tb2'
MANIFEST = 'data/terminal_bench/tb2_trace_manifest.jsonl'
WORKERS = 8
TEXT_EXT = ('.txt', '.json', '.sh', '.log', '.md', '.yaml', '.py', '.patch')
MAX_BYTES = 20_000_000  # skip pathological single files

api = HfApi()
tok = open(os.path.expanduser('~/.cache/huggingface/token')).read().strip()
sess_local = {}


def best_job(sdir):
    jobs = [d for d in os.listdir(sdir) if os.path.isdir(os.path.join(sdir, d))]
    counts = {j: sum(os.path.exists(os.path.join(sdir, j, t, 'result.json'))
                     for t in os.listdir(os.path.join(sdir, j)))
              for j in jobs}
    return max(counts, key=lambda j: (counts[j], j)) if counts else None


def keep(path, size):
    if '/agent' not in path and '/command-' not in path:
        return False
    if '/verifier/' in path or path.endswith('.cast'):
        return False
    return path.endswith(TEXT_EXT) and size <= MAX_BYTES


def build_manifest():
    base = os.path.join(LOCAL, ROOT)
    subs = sorted(d for d in os.listdir(base)
                  if os.path.isdir(os.path.join(base, d)))
    with open(MANIFEST, 'w') as f:
        for si, s in enumerate(subs):
            j = best_job(os.path.join(base, s))
            if j is None:
                print(f'!! no jobs: {s}')
                continue
            for attempt in range(6):
                try:
                    entries = [(e.path, getattr(e, 'size', 0) or 0)
                               for e in api.list_repo_tree(
                                   REPO, f'{ROOT}/{s}/{j}',
                                   repo_type='dataset', recursive=True)]
                    break
                except Exception as e:
                    print(f'list retry {s}/{j}: {e}', file=sys.stderr)
                    time.sleep(10 * (attempt + 1))
            else:
                raise RuntimeError(f'listing failed: {s}/{j}')
            want = [p for p, sz in entries if keep(p, sz)]
            tot = sum(sz for p, sz in entries if keep(p, sz))
            for p in want:
                f.write(json.dumps(p) + '\n')
            print(f'[{si+1}/{len(subs)}] {s} job={j}: {len(want)} files '
                  f'{tot/1e6:.0f} MB')
            time.sleep(0.5)
    print('manifest done')


def fetch_one(path):
    out = os.path.join(LOCAL, path)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return 'skip'
    s = sess_local.setdefault(os.getpid(), requests.Session())
    url = f'https://huggingface.co/datasets/{REPO}/resolve/main/{path}'
    for attempt in range(8):
        try:
            r = s.get(url, headers={'Authorization': f'Bearer {tok}'},
                      timeout=120, allow_redirects=True)
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 200:
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, 'wb') as f:
                f.write(r.content)
            return 'ok'
        if r.status_code == 429:
            time.sleep(min(int(r.headers.get('Retry-After', 15)), 120))
            continue
        if r.status_code == 404:
            return '404'
        time.sleep(5 * (attempt + 1))
    return 'fail'


def fetch_all():
    paths = [json.loads(x) for x in open(MANIFEST)]
    print(f'{len(paths)} files in manifest')
    stats = {'ok': 0, 'skip': 0, '404': 0, 'fail': 0}
    t0 = time.time()
    with ThreadPoolExecutor(WORKERS) as ex:
        for i, res in enumerate(ex.map(fetch_one, paths)):
            stats[res] += 1
            if (i + 1) % 500 == 0:
                rate = stats['ok'] / max(time.time() - t0, 1)
                print(f'{i+1}/{len(paths)} {stats} ({rate:.1f} dl/s)')
    print('final:', stats)


if __name__ == '__main__':
    if not os.path.exists(MANIFEST):
        build_manifest()
    if '--manifest-only' not in sys.argv:
        fetch_all()
