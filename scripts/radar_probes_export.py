"""Export our 7-embedder q100-era embeddings (q20 slice) into the
collaborator's linear-probe manifest format (experiments/linear_probes).

Representations: {openai, nomic, bge, gte, e5, mpnet, minilm} x
{raw, qubric}. Rows joined by their metadata.csv row_id
(<system>/<task>, 2129 retained rows, their exclusions inherited).
qubric = (N, 6, d) section blocks; raw = (N, 2, d) head/tail blocks
(local embedders: 32K-char slices from the pillars caches; OpenAI:
8K-token slices). Targets copied verbatim from their repo_manifest.

Writes data/linear_probes_radar/{manifest.json, metadata.csv, *.npz}.
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
from run_q100 import load_panel  # noqa: E402

OUT = 'data/linear_probes_radar'
LOCAL = {
    'nomic': 'nomic-ai_nomic-embed-text-v1.5',
    'bge': 'BAAI_bge-large-en-v1.5',
    'gte': 'thenlper_gte-large',
    'e5': 'intfloat_e5-large-v2',
    'mpnet': 'sentence-transformers_all-mpnet-base-v2',
    'minilm': 'sentence-transformers_all-MiniLM-L6-v2',
}
TARGETS = {
    'system': {'label': 'system', 'group': 'task',
               'description': 'Exact submission identity on unseen tasks'},
    'model': {'label': 'model', 'group': 'task',
              'description': 'Reported model_display on unseen tasks'},
    'vendor': {'label': 'vendor', 'group': 'task',
               'description': 'Reported model_org on unseen tasks'},
    'harness': {'label': 'harness', 'group': 'task',
                'description': 'Reported agent tag on unseen tasks'},
    'task': {'label': 'task', 'group': 'model_group',
             'description': 'Task identity on held-out model groups'},
    'outcome': {'label': 'outcome', 'group': 'model_group',
                'description': 'Resolved or not on held-out model groups'},
}


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = list(csv.DictReader(
        open('experiments/linear_probes/metadata.csv')))
    row_ids = [r['row_id'] for r in rows]
    want = set(row_ids)
    print(f'{len(row_ids)} metadata rows')

    labels, systems, q100 = load_panel()
    M, Q = len(systems), len(q100)
    all_ids = [f'{s}/{q}' for s in systems for q in q100]
    pos = {rid: i for i, rid in enumerate(all_ids)}
    missing = want - set(all_ids)
    assert not missing, f'{len(missing)} metadata rows missing from panel'
    sel = np.array([pos[rid] for rid in row_ids])

    reps = {}

    def save(name, X):
        path = f'{OUT}/{name}.npz'
        np.savez_compressed(path, row_id=np.array(row_ids), X=X[sel])
        reps[name] = f'{name}.npz'
        print(name, X[sel].shape)

    Xq = np.load('data/judge/q100_emb_openai_small.npz')['X']
    save('openai_qubric', Xq.reshape(M * Q, 6, -1).astype(np.float32))
    HT = np.load('data/judge/q100_raw_emb_openai_small.npz')['HT']
    save('openai_raw', HT.reshape(M * Q, 2, -1).astype(np.float32))

    for short, tag in LOCAL.items():
        z = np.load('data/judge/pillars_emb_q100-qspec-flash0731_'
                    f'{tag}.npz')
        d = z['H'].shape[-1]
        save(f'{short}_qubric',
             z['Xq'].reshape(M * Q, 6, -1).astype(np.float32))
        save(f'{short}_raw',
             np.stack([z['H'], z['T']], axis=1).astype(np.float32))

    man = {'metadata': 'metadata.csv', 'representations': reps,
           'targets': TARGETS}
    json.dump(man, open(f'{OUT}/manifest.json', 'w'), indent=1)
    import shutil
    shutil.copy('experiments/linear_probes/metadata.csv',
                f'{OUT}/metadata.csv')
    print('wrote', f'{OUT}/manifest.json')


if __name__ == '__main__':
    main()
