"""Loaders for cached (model x query) response-embedding suites.

The jailbreak-dkps suite: 82 models x 2666 attack queries, responses embedded
with nomic-ai/nomic-embed-text-v1.5 (768-d). Query embeddings use the same
embedder and are cached under this experiment's data/ directory.

Any suite works if it provides the same three arrays (see load_jailbreak_suite
docstring); only this module is suite-specific.
"""

import json
import os

import numpy as np

JAILBREAK_DATA = os.environ.get(
    'JAILBREAK_DKPS_DATA',
    '/home/ubuntu/helivan-chat-a100/projects/jailbreak-dkps/data')
LOCAL_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

EMBEDDING_MODEL = 'nomic-ai/nomic-embed-text-v1.5'


def load_jailbreak_suite(data_dir=JAILBREAK_DATA, local_data=LOCAL_DATA):
    """Return the fully-paired jailbreak suite.

    Returns dict with:
        X            : (n_models, m_queries, d) float32 response embeddings
        model_names  : list of n_models str
        query_emb    : (m_queries, d_q) float32 query embeddings (or None if not cached)
        query_texts  : list of m_queries str (attack text)
        categories   : list of m_queries str
    """
    meta = json.load(open(os.path.join(data_dir, 'responses', '_attack_metadata.json')))
    query_texts = [a['attack'] for a in meta]
    categories = [a['category'] for a in meta]

    emb_dir = os.path.join(data_dir, 'embeddings')
    files = sorted(f for f in os.listdir(emb_dir) if f.endswith('.npy'))
    model_names = [f[:-4] for f in files]
    X = np.stack([np.load(os.path.join(emb_dir, f)) for f in files]).astype(np.float32)
    assert X.shape[1] == len(meta), f'{X.shape[1]} response rows vs {len(meta)} attacks'

    qpath = os.path.join(local_data, 'query_embeddings.npy')
    query_emb = np.load(qpath).astype(np.float32) if os.path.exists(qpath) else None
    if query_emb is not None:
        assert query_emb.shape[0] == len(meta)

    return {
        'X': X,
        'model_names': model_names,
        'query_emb': query_emb,
        'query_texts': query_texts,
        'categories': categories,
    }


def load_dist_tensor(local_data=LOCAL_DATA):
    """(m, n, n) per-query pairwise squared-distance tensor, precomputed by
    the precompute module."""
    return np.load(os.path.join(local_data, 'per_query_dist_sq.npy'))
