"""Build the full-trace pruned text cache for whole-trace judging.

Renders each q20 trajectory from the raw archive WITHOUT the historical
40K+20K truncation, applies dkps.traces.prune, and caps at --max-chars
(default 500K ~= 125K tokens, inside the 131K-token judge context) with a
4:1 head:tail split only for the extreme tail of traces.

Cache: data/judge/trace_texts_full_pruned/<sys>/<q>.txt
Prints the token budget so the judge sweep cost is known before spending.
"""
import argparse
import json
import os
import sys
from glob import glob
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dkps.traces import leaderboard as lb  # noqa: E402
from dkps.traces.prune import prune_for_judging  # noqa: E402

OUT = 'data/judge/trace_texts_full_pruned'


def _one(args):
    sub_dir, wanted, max_chars, raw = args
    ts = lb.load_leaderboard_submission(sub_dir, labels=None)
    out = {}
    for t in ts:
        if t.query_id in wanted:
            txt = (t.steps[0].assistant_text or '') if raw else \
                prune_for_judging(t.steps[0].assistant_text or '')
            if len(txt) > max_chars:
                txt = (txt[:max_chars * 4 // 5] + '\n...[omitted]...\n'
                       + txt[-max_chars // 5:])
            out[t.query_id] = txt
    return os.path.basename(sub_dir), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/leaderboard/verified')
    ap.add_argument('--labels', default='data/leaderboard/verified_labels.json')
    ap.add_argument('--ref-cache', default='data/judge/structured-qspec')
    ap.add_argument('--max-chars', type=int, default=500_000)
    ap.add_argument('--panel', default=None,
                    help='JSON panel file with {"instances": [...]}; default q20')
    ap.add_argument('--raw', action='store_true',
                    help='skip pruning (raw render; for the naive baseline); '
                         'caches to trace_texts_full_raw/')
    args = ap.parse_args()

    labels = json.load(open(args.labels))
    if args.panel:
        q20 = sorted(json.load(open(args.panel))['instances'])
    else:
        q20 = sorted(f[:-5] for f in os.listdir(
            os.path.join(args.ref_cache, sorted(os.listdir(args.ref_cache))[0])))
    systems = sorted(s for s in os.listdir(args.ref_cache)
                     if 'resolved' in labels.get(s, {}))
    global OUT
    if args.raw:
        OUT = 'data/judge/trace_texts_full_raw'
    todo = [s for s in systems
            if not all(os.path.exists(os.path.join(OUT, s, f'{q}.txt'))
                       for q in q20)]
    print(f'{len(systems)} systems; rendering {len(todo)} -> {OUT}')
    with Pool(16) as pool:
        for s, out in tqdm(pool.imap_unordered(
                _one, [(os.path.join(args.root, s), set(q20), args.max_chars,
                        args.raw) for s in todo]), total=len(todo)):
            os.makedirs(os.path.join(OUT, s), exist_ok=True)
            for q, t in out.items():
                open(os.path.join(OUT, s, f'{q}.txt'), 'w').write(t)

    sizes = np.array([os.path.getsize(p)
                      for p in glob(os.path.join(OUT, '*', '*.txt'))])
    tok = sizes.sum() / 4
    print(f'{len(sizes)} texts: median {np.median(sizes)/1e3:.0f}K chars, '
          f'p90 {np.percentile(sizes, 90)/1e3:.0f}K, capped: '
          f'{(sizes >= args.max_chars).sum()}')
    print(f'total ~{tok/1e6:.0f}M input tokens; deepseek sweep '
          f'~${tok/1e6*0.27 + 3:.0f}; gpt-oss-120b ~${tok/1e6*0.037 + 1:.0f}')


if __name__ == '__main__':
    main()
