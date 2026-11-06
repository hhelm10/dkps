"""SWE-bench Verified leaderboard data download (shareable; no AWS account).

Sources:
  labels  github.com/swe-bench/experiments -- official per-instance resolved
          labels (evaluation/verified/<submission>/results/results.json) and
          metadata.yaml, harvested into data/leaderboard/verified_labels.json
  traces  s3://swe-bench-submissions/verified/<submission>/trajs/ -- public
          bucket read anonymously over plain HTTPS (no credentials, no boto3;
          note the key prefix has no 'evaluation/' segment)

Trajectories land at data/leaderboard/verified/<submission>/trajs/..., the
layout dkps.traces.leaderboard.load_leaderboard_corpus expects. Downloads
resume: files already present with the right size are skipped. Only trajs/
are pulled (not logs/); submissions since 2024-07 are required to include
them, older ones may list zero files.

Usage:
  python scripts/swe_download.py list                      # submissions + traj counts/sizes
  python scripts/swe_download.py labels
  python scripts/swe_download.py traces [submission ...]   # default: all with trajs
"""
import json
import os
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

BUCKET = 'https://swe-bench-submissions.s3.amazonaws.com'
PREFIX = 'verified/'
DEST = 'data/leaderboard/verified'
LABELS = 'data/leaderboard/verified_labels.json'
EXPERIMENTS = 'https://github.com/swe-bench/experiments.git'
NS = '{http://s3.amazonaws.com/doc/2006-03-01/}'
WORKERS = 16


def s3_list(prefix, delimiter=None):
    """Yield (key, size) for objects under prefix; with a delimiter, also
    (common_prefix, None) for each 'directory'."""
    token = None
    while True:
        q = {'list-type': '2', 'prefix': prefix, 'max-keys': '1000'}
        if delimiter:
            q['delimiter'] = delimiter
        if token:
            q['continuation-token'] = token
        url = BUCKET + '/?' + urllib.parse.urlencode(q)
        with urllib.request.urlopen(url, timeout=60) as r:
            root = ET.fromstring(r.read())
        for cp in root.iter(NS + 'CommonPrefixes'):
            yield cp.find(NS + 'Prefix').text, None
        for c in root.iter(NS + 'Contents'):
            yield c.find(NS + 'Key').text, int(c.find(NS + 'Size').text)
        token = root.findtext(NS + 'NextContinuationToken')
        if root.findtext(NS + 'IsTruncated') != 'true':
            break


def submissions():
    return sorted(p[len(PREFIX):].rstrip('/')
                  for p, size in s3_list(PREFIX, delimiter='/')
                  if size is None)


def traj_objects(sub):
    return [(k, s) for k, s in s3_list(f'{PREFIX}{sub}/trajs/')
            if s is not None and s > 0]


def fetch(key, size):
    dest = os.path.join(DEST, key[len(PREFIX):])
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        return 0
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = BUCKET + '/' + urllib.parse.quote(key)
    tmp = dest + '.part'
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)
    return size


def main_list():
    for sub in submissions():
        objs = traj_objects(sub)
        mb = sum(s for _, s in objs) / 1e6
        print(f'{sub}  {len(objs)} trajs  {mb:.0f} MB')


def main_traces(wanted):
    subs = wanted or submissions()
    pool = ThreadPoolExecutor(WORKERS)
    for sub in subs:
        objs = traj_objects(sub)
        if not objs:
            print(f'{sub}: no trajs, skipping')
            continue
        got = sum(pool.map(lambda o: fetch(*o), objs))
        print(f'{sub}: {len(objs)} files, {got / 1e6:.0f} MB new')


def main_labels():
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(['git', 'clone', '--depth', '1', '--filter=blob:none',
                        '--sparse', EXPERIMENTS, tmp], check=True)
        subprocess.run(['git', '-C', tmp, 'sparse-checkout', 'set',
                        'evaluation/verified'], check=True)
        root = os.path.join(tmp, 'evaluation', 'verified')
        out = {}
        for sub in sorted(os.listdir(root)):
            res = os.path.join(root, sub, 'results', 'results.json')
            meta = os.path.join(root, sub, 'metadata.yaml')
            if not os.path.exists(res):
                print(f'{sub}: no results.json, skipping')
                continue
            entry = {'resolved': json.load(open(res)).get('resolved', [])}
            if os.path.exists(meta):
                entry['metadata_yaml'] = open(meta).read()
            out[sub] = entry
    os.makedirs(os.path.dirname(LABELS), exist_ok=True)
    json.dump(out, open(LABELS, 'w'), indent=1)
    print(f'{LABELS}: {len(out)} submissions')


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'list'
    if stage == 'list':
        main_list()
    elif stage == 'labels':
        main_labels()
    elif stage == 'traces':
        main_traces(sys.argv[2:])
    else:
        sys.exit(__doc__)
