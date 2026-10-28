"""Embed the 2666 attack queries with the same embedder used for responses
(nomic-ai/nomic-embed-text-v1.5, plain encode), and cache to data/.

Run: python -m projects.routing.experiments.embed_queries
"""

import json
import os
import sys

import numpy as np

from .data import JAILBREAK_DATA, LOCAL_DATA, EMBEDDING_MODEL


def main():
    out_path = os.path.join(LOCAL_DATA, 'query_embeddings.npy')
    if os.path.exists(out_path):
        print(f'already cached: {out_path}')
        return

    meta = json.load(open(os.path.join(JAILBREAK_DATA, 'responses', '_attack_metadata.json')))
    texts = [a['attack'] for a in meta]
    print(f'embedding {len(texts)} queries with {EMBEDDING_MODEL}')

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
    emb = model.encode(texts, batch_size=256, convert_to_numpy=True, show_progress_bar=True)

    os.makedirs(LOCAL_DATA, exist_ok=True)
    np.save(out_path, emb.astype(np.float32))
    print(f'saved {emb.shape} -> {out_path}')


if __name__ == '__main__':
    main()
