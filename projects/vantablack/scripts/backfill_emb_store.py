"""One-off: move the batch embedding caches of an earlier real_tstep.py run into the per-id
store (real_tstep.STORE), so later windows reuse them. Rebuilds the run's miss lists exactly as
gather() did and checks each against its cache file's text hash before storing.

    python backfill_emb_store.py --end 2026-02-18 --min-steps 6
"""
import argparse
import hashlib
import os

import numpy as np
import pandas as pd

from pipeline import embed
from pipeline.ingest_moltnet import RAW
from real_tstep import _add_to_store, _lookup, _store, activity_steps, load_window


def misses(ids, sources):
    found = np.zeros(len(ids), bool)
    for _, pos in sources:
        found |= np.array([a in pos for a in ids])
    return np.flatnonzero(~found)


def cached(cache_name, texts):
    t = [(x or '')[:embed.MAX_CHARS] for x in texts]
    h = hashlib.sha1('\n'.join(embed._key(x, embed.REVISION) for x in t).encode()).hexdigest()[:16]
    p = os.path.join(embed.CACHE, f'{cache_name}_{h}.npy')
    return np.load(p) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2026-01-30')
    ap.add_argument('--end', default='2026-02-18')
    ap.add_argument('--T', type=int, default=8)
    ap.add_argument('--min-steps', type=int, default=6)
    a = ap.parse_args()
    tag = f'{a.start[5:]}_{a.end[5:]}_T{a.T}' + ('' if a.min_steps == a.T else f'_ms{a.min_steps}')
    acts, posts = load_window(a.start, a.end)
    acts['step'], _ = activity_steps(acts.t.values, a.T)
    per = acts.groupby(['agent', 'step']).size().unstack(fill_value=0)
    cohort = sorted(per.index[(per >= 1).sum(1) >= a.min_steps])
    sysm = acts.agent.isin(cohort).values
    sys_ = acts[sysm].reset_index(drop=True)
    stores = [_lookup('actions'), _lookup('ref_actions')]
    have = set().union(*[set(p) for _, p in stores])
    ref = acts[~sysm & acts.action_id.isin(have)].reset_index(drop=True)
    threads = sorted(set(sys_.thread[sys_.kind == 'comment']) | set(ref.thread[ref.kind == 'comment']))
    p = posts.drop_duplicates('id').set_index('id').reindex(threads)
    th_text = (p.title.fillna('') + '\n\n' + p.content.fillna('').str[:2000]).tolist()
    subs = sorted(set(sys_.submolt[sys_.kind == 'post'].dropna()) | set(ref.submolt[ref.kind == 'post'].dropna()))
    sm = pd.read_parquet(os.path.join(RAW, 'submolts.parquet'), columns=['name', 'display_name', 'description'])
    sm = sm.drop_duplicates('name').set_index('name').reindex(subs)
    sub_text = [f'{n if not isinstance(dn, str) or not dn else dn}\n\n{ds if isinstance(ds, str) else ""}'
                for n, dn, ds in zip(subs, sm.display_name, sm.description)]
    for kind, ids, texts, src, name in [
            ('actions', sys_.action_id.tolist(), sys_.text.tolist(), stores, f'real_tstep_sys_{tag}'),
            ('threads', threads, th_text, [_lookup('contexts'), _lookup('threads_extra')], f'real_tstep_threads_{tag}'),
            ('submolts', subs, sub_text, [_lookup('submolts')], f'real_tstep_submolts_{tag}')]:
        m = misses(ids, src)
        E = cached(name, [texts[i] for i in m])
        if E is None:
            print(f'{kind}: no cache file matches the {len(m)} misses; skipped')
            continue
        new = [i for i in m if not any(ids[i] in pos for _, pos in _store(kind))]
        keep = np.isin(m, new)
        _add_to_store(kind, [ids[i] for i in m[keep]], E[keep])
        print(f'{kind}: {keep.sum()} of {len(m)} cached embeddings added to the store')


if __name__ == '__main__':
    main()
