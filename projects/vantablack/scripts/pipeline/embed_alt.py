"""Alternative embedding functions for the sensitivity analysis (Figure 5).

The main analysis uses the pinned nomic-embed-text-v1.5 (pipeline/embed.py). This module embeds the
same texts with another model and keeps a per-model store keyed by action id, in chunks, so that an
interrupted run resumes where it stopped:

    data/derived/emb_store_alt/<model>/<kind>_<stamp>_<n>.npy / _ids.json

Models: local sentence-transformers checkpoints (from the Hugging Face cache) and OpenAI's
text-embedding-3-small (API key read from the project .env; never logged). Texts are truncated to
4,000 characters, as for nomic, and for OpenAI also to 8,000 tokens (its input limit is 8,192). Embeddings are unit-normalized.
"""
import glob
import json
import os
import time

import numpy as np

from . import paths

MODELS = {
    'minilm': 'sentence-transformers/all-MiniLM-L6-v2',
    'mpnet': 'sentence-transformers/all-mpnet-base-v2',
    'bge-small': 'BAAI/bge-small-en-v1.5',
    'openai-small': 'text-embedding-3-small',
}
STORE = os.path.join(paths.DERIVED, 'emb_store_alt')
MAX_CHARS = 4000
CHUNK = 50_000


def _store_dir(name):
    d = os.path.join(STORE, name)
    os.makedirs(d, exist_ok=True)
    return d


def _load_store(name, kind):
    out = []
    for f in sorted(glob.glob(os.path.join(_store_dir(name), f'{kind}_*_ids.json'))):
        out.append((np.load(f[:-len('_ids.json')] + '.npy', mmap_mode='r'),
                    {a: i for i, a in enumerate(json.load(open(f)))}))
    return out


def _save_chunk(name, kind, ids, E):
    stem = os.path.join(_store_dir(name), f'{kind}_{time.strftime("%Y%m%d-%H%M%S")}_{len(ids)}')
    np.save(stem + '.npy', np.asarray(E, np.float32))
    with open(stem + '_ids.json', 'w') as fh:
        json.dump(list(ids), fh)


def _openai_key():
    if os.environ.get('OPENAI_API_KEY'):
        return os.environ['OPENAI_API_KEY']
    for f in (os.path.join(paths.REPO, '.env'),):
        if os.path.exists(f):
            for line in open(f):
                if line.strip().startswith('OPENAI_API_KEY='):
                    return line.split('=', 1)[1].strip().strip('"').strip("'")
    raise RuntimeError('OPENAI_API_KEY not found')


class _Encoder:
    def __init__(self, name, device='cuda'):
        self.name = name
        if name == 'openai-small':
            from openai import OpenAI
            self.client = OpenAI(api_key=_openai_key())
        else:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(MODELS[name], device=device)

    def __call__(self, texts):
        texts = [(t or ' ')[:MAX_CHARS] or ' ' for t in texts]
        if self.name == 'openai-small':
            return self._openai(texts)
        order = np.argsort([len(t) for t in texts])
        out = None
        bs = 256
        for s in range(0, len(order), bs):
            j = order[s:s + bs]
            e = self.model.encode([texts[i] for i in j], normalize_embeddings=True, convert_to_numpy=True,
                                  batch_size=64 if len(texts[j[-1]]) < 1500 else 16)
            if out is None:
                out = np.zeros((len(texts), e.shape[1]), np.float32)
            out[j] = e
        return out

    def _openai(self, texts, batch=256, workers=8, max_tokens=8000):
        from concurrent.futures import ThreadPoolExecutor
        import tiktoken
        enc = tiktoken.get_encoding('cl100k_base')                 # the tokenizer of text-embedding-3-*
        texts = [t if len(t) < 2000 else enc.decode(enc.encode(t, disallowed_special=())[:max_tokens]) for t in texts]
        out = [None] * len(texts)

        def call(a):
            part = texts[a:a + batch]
            for attempt in range(8):
                try:
                    r = self.client.embeddings.create(model=MODELS['openai-small'], input=part)
                    return a, [d.embedding for d in r.data]
                except Exception as e:                              # rate limit / transient: back off
                    if 'invalid_request_error' in str(e):           # a bad input will not succeed on retry
                        raise
                    time.sleep(min(60, 2 ** attempt))
            raise RuntimeError('OpenAI embedding failed after retries')

        with ThreadPoolExecutor(workers) as ex:
            for a, embs in ex.map(call, range(0, len(texts), batch)):
                out[a:a + len(embs)] = embs
        E = np.asarray(out, np.float32)
        return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)


def gather(name, ids, texts, kind, log=print):
    """Embeddings of `texts` (keyed by `ids`) under model `name`, from the per-model store where
    present; the rest are embedded in chunks of CHUNK and added to the store as they finish."""
    store = _load_store(name, kind)
    out, found = None, np.zeros(len(ids), bool)
    for E, pos in store:
        idx = np.array([pos.get(a, -1) for a in ids])
        hit = (idx >= 0) & ~found
        if out is None:
            out = np.zeros((len(ids), E.shape[1]), np.float32)
        out[hit] = E[idx[hit]]
        found |= hit
    miss = np.flatnonzero(~found)
    log(f'[{name}] {kind}: {len(ids)} needed, {int(found.sum())} in store, {len(miss)} to embed')
    if len(miss):
        enc = _Encoder(name)
        for s in range(0, len(miss), CHUNK):
            j = miss[s:s + CHUNK]
            t = time.time()
            E = enc([texts[i] for i in j])
            if out is None:
                out = np.zeros((len(ids), E.shape[1]), np.float32)
            out[j] = E
            _save_chunk(name, kind, [ids[i] for i in j], E)
            log(f'[{name}] {kind}: embedded {min(s + CHUNK, len(miss))}/{len(miss)} ({time.time() - t:.0f}s per chunk)')
    return out
