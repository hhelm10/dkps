"""Embed the reference corpus for PKPS content conditioning (pinned nomic v1.5).

    python embed_reference.py --scope platform_jan31_feb17

The system agents' actions and their threads are already embedded by
embed_scope.py. PKPS conditioning compares every response with everyone's
responses to similar queries, so it also needs:

  ref actions  posts and comments of all non-system agents in the window,
               capped at --cap per agent per UTC day (seeded, same rule as the
               system agents)
  threads      opening post (title + first 2000 chars, as for contexts) of
               every thread a system or reference comment replies to
  submolts     display name + description of every submolt a post is in;
               a post's query is its submolt

Raw files are read only. Writes data/derived/<scope>/emb_{ref_actions,
threads_extra,submolts}.npy with matching _ids.json, plus ref_actions.parquet
(id, agent, type, t, thread, submolt; no text).
"""
import argparse
import json
import os
import time
import numpy as np
import pandas as pd

from pipeline import embed
from pipeline.ingest_moltnet import RAW

HERE = os.path.dirname(os.path.abspath(__file__))
from pipeline.paths import DERIVED  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scope', default='platform_jan31_feb17')
    ap.add_argument('--cap', type=int, default=40)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    d = os.path.join(DERIVED, a.scope)
    meta = json.load(open(os.path.join(d, 'meta.json')))
    system = set(meta['system'])
    acts = pd.read_parquet(os.path.join(d, 'actions.parquet'),
                           columns=['action_id', 'agent_id', 'action_type', 't', 'thread_id', 'submolt'])
    acts = acts[acts.action_type.isin(['post', 'comment'])]

    ref = acts[~acts.agent_id.isin(system)].copy()
    ref['day'] = (ref.t // 24).astype(int)
    ref['r'] = np.random.default_rng(a.seed).uniform(size=len(ref))
    ref = ref[ref.groupby(['agent_id', 'day']).r.rank(method='first') <= a.cap]
    ref = ref.drop(columns=['day', 'r']).reset_index(drop=True)

    posts = pd.read_parquet(os.path.join(RAW, 'posts.parquet'),
                            columns=['id', 'title', 'content', 'submolt_name'])
    comments = pd.read_parquet(os.path.join(RAW, 'comments.parquet'), columns=['id', 'content'])
    ptext = dict(zip(posts.id, posts.title.fillna('') + '\n\n' + posts.content.fillna('')))
    ctext = dict(zip(comments.id, comments.content.fillna('')))
    del comments
    ref_text = [ptext[i] if ty == 'post' else ctext[i]
                for i, ty in zip(ref.action_id, ref.action_type)]
    ref.to_parquet(os.path.join(d, 'ref_actions.parquet'))

    have = set(json.load(open(os.path.join(d, 'emb_contexts_ids.json'))))
    sys_c = acts[acts.agent_id.isin(system) & (acts.action_type == 'comment')]
    need = sorted((set(sys_c.thread_id) | set(ref.thread_id[ref.action_type == 'comment'])) - have)
    p = posts.set_index('id').loc[need]
    th_text = (p.title.fillna('') + '\n\n' + p.content.fillna('').str[:2000]).tolist()

    subs = pd.read_parquet(os.path.join(RAW, 'submolts.parquet'),
                           columns=['name', 'display_name', 'description'])
    used = sorted(set(acts.submolt.dropna()) | set(ref.submolt.dropna()))
    subs = subs.drop_duplicates('name').set_index('name').reindex(used)
    sub_text = [f'{n if not isinstance(dn, str) or not dn else dn}\n\n{ds if isinstance(ds, str) else ""}'
                for n, dn, ds in zip(used, subs.display_name, subs.description)]
    print(f'[corpus] {len(ref)} reference actions from {ref.agent_id.nunique()} agents; '
          f'{len(need)} extra threads; {len(used)} submolts', flush=True)

    for name, ids, texts in (('submolts', used, sub_text),
                             ('threads_extra', need, th_text),
                             ('ref_actions', ref.action_id.tolist(), ref_text)):
        t = time.time()
        E = embed.embed(texts, batch_size=a.batch)
        np.save(os.path.join(d, f'emb_{name}.npy'), E.astype(np.float32))
        json.dump(ids, open(os.path.join(d, f'emb_{name}_ids.json'), 'w'))
        print(f'{name}: {len(ids)} in {time.time() - t:.0f}s', flush=True)


if __name__ == '__main__':
    main()
