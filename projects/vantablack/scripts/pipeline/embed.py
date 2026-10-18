"""Action and context embeddings with a pinned nomic-embed-text-v1.5.

Same embedder family as the TDKPS work so results are comparable. Model id
and revision are pinned; embeddings are cached under data/derived/emb/ keyed
by a hash of (model, revision, prefix, text).

nomic v1.5 expects task prefixes; actions and contexts are embedded as
'clustering: ' (symmetric use), which is what the perspective space needs.
"""
import hashlib
import os
import numpy as np
from . import paths

MODEL = 'nomic-ai/nomic-embed-text-v1.5'
REVISION = 'e9b6763023c676ca8431644204f50c2b100d9aab'  # resolved 2026-10-16 via pin()
PREFIX = 'clustering: '
CACHE = os.path.join(paths.DERIVED, 'emb')
MAX_CHARS = 4000


def pin():
    """Resolve the current HF commit for MODEL once and print it for recording."""
    from huggingface_hub import HfApi
    return HfApi().model_info(MODEL).sha


def _key(text, revision):
    return hashlib.sha1(f'{MODEL}|{revision}|{PREFIX}|{text}'.encode()).hexdigest()


def embed(texts, revision=REVISION, batch_size=64, device='cuda', cache_name=None):
    """Embed texts (truncated to MAX_CHARS); returns (n, 768) float32, unit norm."""
    from sentence_transformers import SentenceTransformer
    texts = [(t or '')[:MAX_CHARS] for t in texts]
    path = None
    if cache_name:
        os.makedirs(CACHE, exist_ok=True)
        h = hashlib.sha1('\n'.join(_key(t, revision) for t in texts).encode()).hexdigest()[:16]
        path = os.path.join(CACHE, f'{cache_name}_{h}.npy')
        if os.path.exists(path):
            return np.load(path)
    model = SentenceTransformer(MODEL, revision=revision, trust_remote_code=True, device=device)
    order = np.argsort([len(t) for t in texts])  # length-sorted batches; long texts small
    out = np.zeros((len(texts), model.get_sentence_embedding_dimension()), np.float32)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        bs = batch_size if len(texts[idx[-1]]) < 1500 else max(4, batch_size // 8)
        for s2 in range(0, len(idx), bs):
            j = idx[s2:s2 + bs]
            out[j] = model.encode([PREFIX + texts[i] for i in j], normalize_embeddings=True,
                                  convert_to_numpy=True)
    if path:
        np.save(path, out)
    return out
