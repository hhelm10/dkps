"""Download the benchmark corpora used in the agentic-traces work.

Subcommands:

  swebench-labels   Official resolved labels + metadata for every SWE-bench
                    Verified leaderboard submission, harvested from the
                    swe-bench/experiments GitHub repo (sparse clone) into
                    data/leaderboard/verified_labels.json.

  swebench-trajs    Trajectories + final patches for submissions from the
                    public S3 bucket (anonymous access, no AWS account
                    configuration needed beyond the awscli binary):
                      s3://swe-bench-submissions/verified/<submission>/
                    into data/leaderboard/verified/<submission>/.
                    ~40GB for all submissions with trajectories.

  terminalbench     Terminal-Bench 2 leaderboard corpus (142 entries, 89
                    tasks, >=5 trials/task; ~40GB, millions of small files)
                    from Hugging Face into data/terminal_bench/tb2.

Examples:
  python scripts/data.py swebench-labels
  python scripts/data.py swebench-trajs --limit 5          # smoke test
  python scripts/data.py swebench-trajs                    # everything
  python scripts/data.py terminalbench
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

LB_ROOT = 'data/leaderboard/verified'
LABELS = 'data/leaderboard/verified_labels.json'
EXPERIMENTS_REPO = 'https://github.com/swe-bench/experiments'
S3_BUCKET = 's3://swe-bench-submissions'
TB_REPO = 'harborframework/terminal-bench-2-leaderboard'


def swebench_labels(args):
    tmp = tempfile.mkdtemp(prefix='swb_experiments_')
    try:
        subprocess.run(['git', 'clone', '--depth', '1', '--filter=blob:none',
                        '--sparse', EXPERIMENTS_REPO, tmp], check=True)
        subprocess.run(['git', '-C', tmp, 'sparse-checkout', 'set',
                        'evaluation/verified'], check=True)
        root = os.path.join(tmp, 'evaluation', 'verified')
        out = {}
        for sub in sorted(os.listdir(root)):
            d = os.path.join(root, sub)
            if not os.path.isdir(d):
                continue
            entry = {}
            res = os.path.join(d, 'results', 'results.json')
            if os.path.exists(res):
                try:
                    entry['resolved'] = json.load(open(res)).get('resolved', [])
                except json.JSONDecodeError:
                    pass
            meta = os.path.join(d, 'metadata.yaml')
            if os.path.exists(meta):
                entry['metadata_yaml'] = open(meta).read()
            if entry:
                out[sub] = entry
        os.makedirs(os.path.dirname(LABELS), exist_ok=True)
        json.dump(out, open(LABELS, 'w'))
        n_res = sum('resolved' in v for v in out.values())
        print(f'wrote {LABELS}: {len(out)} submissions, {n_res} with resolved lists')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def swebench_trajs(args):
    if shutil.which('aws') is None:
        sys.exit('needs awscli (pip install awscli); bucket is public, '
                 'no credentials required')
    ls = subprocess.run(['aws', 's3', 'ls', '--no-sign-request',
                         f'{S3_BUCKET}/verified/'],
                        capture_output=True, text=True, check=True)
    subs = [m.group(1) for m in
            re.finditer(r'PRE (\S+)/', ls.stdout)]
    print(f'{len(subs)} submissions in bucket')
    if args.limit:
        subs = subs[:args.limit]
    os.makedirs(LB_ROOT, exist_ok=True)
    for i, sub in enumerate(subs):
        dst = os.path.join(LB_ROOT, sub)
        print(f'[{i + 1}/{len(subs)}] {sub}')
        # trajectories + final patches; evaluation logs excluded (graders'
        # output, not the agents')
        subprocess.run(['aws', 's3', 'sync', '--no-sign-request', '--quiet',
                        f'{S3_BUCKET}/verified/{sub}/trajs',
                        os.path.join(dst, 'trajs')], check=True)
        subprocess.run(['aws', 's3', 'cp', '--no-sign-request', '--quiet',
                        f'{S3_BUCKET}/verified/{sub}/all_preds.jsonl',
                        os.path.join(dst, 'all_preds.jsonl')], check=False)
    print(f'done -> {LB_ROOT}')


def terminalbench(args):
    from huggingface_hub import snapshot_download
    path = snapshot_download(TB_REPO, repo_type='dataset',
                             local_dir='data/terminal_bench/tb2',
                             max_workers=args.workers)
    print('done ->', path)
    print('note: the repo holds millions of small per-trial files; expect '
          'this to take a while even on fast links')


def unpack_artifacts(args):
    import tarfile
    src = args.path
    os.makedirs('data/leaderboard', exist_ok=True)
    for name, dest in (('judge.tar', 'data'),
                       ('multiembed.tar', 'data'),
                       ('dkps_cache_lb.tar', '.')):
        p = os.path.join(src, name)
        if os.path.exists(p):
            print('extracting', name)
            with tarfile.open(p) as t:
                t.extractall(dest)
    lb = os.path.join(src, 'verified_labels.json')
    if os.path.exists(lb):
        shutil.copy(lb, LABELS)
    if os.path.exists('dkps_cache_lb') and not os.path.exists('.dkps_cache_lb'):
        os.rename('dkps_cache_lb', '.dkps_cache_lb')
    print('done')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    sp.add_parser('swebench-labels')
    t = sp.add_parser('swebench-trajs')
    t.add_argument('--limit', type=int, default=0,
                   help='only the first N submissions (smoke test)')
    tb = sp.add_parser('terminalbench')
    tb.add_argument('--workers', type=int, default=32)
    ua = sp.add_parser('unpack-artifacts')
    ua.add_argument('path', help='directory holding the shared tarballs')
    args = ap.parse_args()
    {'swebench-labels': swebench_labels,
     'swebench-trajs': swebench_trajs,
     'terminalbench': terminalbench,
     'unpack-artifacts': unpack_artifacts}[args.cmd](args)


if __name__ == '__main__':
    main()
