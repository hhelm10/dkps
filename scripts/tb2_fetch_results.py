"""Direct fetcher for TB2 leaderboard outcome files (result.json,
metadata.yaml, job config.json). snapshot_download drowns in per-file
HEAD rate limits on this repo (500K+ files); instead we list each
submission's tree once and GET the ~34K small files straight from the
resolve endpoint with our own backoff. Resumable (skips existing files).

Usage: python scripts/tb2_fetch_results.py [--manifest-only]
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
DEST = 'data/terminal_bench/tb2'
MANIFEST = 'data/terminal_bench/tb2_manifest.jsonl'
WANT = ('result.json', 'metadata.yaml', 'config.json')
WORKERS = 8

api = HfApi()
tok = open(os.path.expanduser('~/.cache/huggingface/token')).read().strip()
sess_local = {}


def build_manifest():
    from huggingface_hub.hf_api import RepoFolder
    subs = [e.path for e in api.list_repo_tree(REPO, ROOT, repo_type='dataset')
            if isinstance(e, RepoFolder)]
    print(f'{len(subs)} submissions')
    with open(MANIFEST, 'w') as f:
        for si, sub in enumerate(subs):
            for attempt in range(6):
                try:
                    entries = [e.path for e in api.list_repo_tree(
                        REPO, sub, repo_type='dataset', recursive=True)
                        if e.path.endswith(WANT)]
                    break
                except Exception as e:
                    print(f'list retry {sub}: {e}', file=sys.stderr)
                    time.sleep(10 * (attempt + 1))
            else:
                raise RuntimeError(f'listing failed: {sub}')
            for p in entries:
                f.write(json.dumps(p) + '\n')
            print(f'[{si+1}/{len(subs)}] {sub.split("/")[-1]}: '
                  f'{len(entries)} files')
            time.sleep(0.5)
    print('manifest done')


def fetch_one(path):
    out = os.path.join(DEST, path)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return 'skip'
    pid = os.getpid()
    s = sess_local.setdefault(pid, requests.Session())
    url = f'https://huggingface.co/datasets/{REPO}/resolve/main/{path}'
    for attempt in range(8):
        try:
            r = s.get(url, headers={'Authorization': f'Bearer {tok}'},
                      timeout=60, allow_redirects=True)
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 200:
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, 'wb') as f:
                f.write(r.content)
            return 'ok'
        if r.status_code == 429:
            wait = int(r.headers.get('Retry-After', 15))
            time.sleep(min(wait, 120))
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
                rate = (stats['ok']) / max(time.time() - t0, 1)
                print(f'{i+1}/{len(paths)} {stats} ({rate:.1f} dl/s)')
    print('final:', stats)


if __name__ == '__main__':
    if not os.path.exists(MANIFEST):
        build_manifest()
    if '--manifest-only' not in sys.argv:
        fetch_all()
