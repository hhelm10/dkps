"""Embed a scope's system actions and thread contexts (pinned nomic v1.5).

    python embed_scope.py --scope platform_jan31_feb17

Writes data/derived/<scope>/emb_actions.npy (+ _ids.json) and emb_contexts.npy.
"""
import argparse
import json
import os
import time
import numpy as np
import pandas as pd

from pipeline import embed

HERE = os.path.dirname(os.path.abspath(__file__))
from pipeline.paths import DERIVED  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scope', required=True)
    ap.add_argument('--batch', type=int, default=256)
    a = ap.parse_args()
    d = os.path.join(DERIVED, a.scope)
    meta = json.load(open(os.path.join(d, 'meta.json')))
    acts = pd.read_parquet(os.path.join(d, 'actions.parquet'))
    acts = acts[acts.agent_id.isin(set(meta['system']))
                & acts.action_type.isin(['post', 'comment'])]
    ctx = pd.read_parquet(os.path.join(d, 'contexts.parquet'))
    for name, ids, texts in (('contexts', ctx.thread_id.tolist(), ctx.text.tolist()),
                             ('actions', acts.action_id.tolist(), acts.text.tolist())):
        t = time.time()
        E = embed.embed(texts, batch_size=a.batch)
        np.save(os.path.join(d, f'emb_{name}.npy'), E.astype(np.float32))
        json.dump(ids, open(os.path.join(d, f'emb_{name}_ids.json'), 'w'))
        print(f'{name}: {len(ids)} in {time.time() - t:.0f}s', flush=True)


if __name__ == '__main__':
    main()
