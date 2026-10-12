"""Resumable, local EmbeddingGemma-2 cache and CPU/Metal numerical check.

Run with --device mps on Apple Silicon. No automatic CPU fallback or remote code.
Raw text is encoded independently: precomputing vectors does not fit preprocessing
on hidden observations. Experimental code must still apply observation masks.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

REVISION = '914f7f89142e33e77833254d9c9b90c3cef7303b'
PROMPT = 'SentenceSimilarity'


def load_model(path, device, dtype='float32', max_length=2048):
    if device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('Metal unavailable in this process; run outside the sandbox or choose --device cpu')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        raise RuntimeError('Disable CPU fallback so device/timing provenance is unambiguous')
    torch.set_num_threads(4)
    model = SentenceTransformer(
        str(path), local_files_only=True, device=device,
        config_kwargs={'vision_config': None, 'audio_config': None},
        model_kwargs={'torch_dtype': getattr(torch, dtype)},
    )
    model.max_seq_length = max_length
    return model


def encode(model, texts, batch_size):
    vectors = model.encode(texts, prompt_name=PROMPT, batch_size=batch_size,
                           normalize_embeddings=True, convert_to_numpy=True,
                           show_progress_bar=False).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or not np.allclose(norms, 1, atol=0.02):
        raise ValueError('nonfinite or non-unit embedding')
    # bfloat16's normalization can leave ~1e-3 norm error; store unit float32.
    return vectors / norms


def smoke(model, texts, batch_size, out):
    out.mkdir(parents=True, exist_ok=True)
    result = dict(platform=platform.platform(), model_revision=REVISION,
                  mps_available=torch.backends.mps.is_available(), prompt=PROMPT,
                  n_texts=len(texts), max_length=model.max_seq_length,
                  fallback_enabled=False, versions={p: importlib.metadata.version(p) for p in
                      ['torch', 'transformers', 'sentence-transformers', 'numpy']})
    gpu = None
    for device, dtype in [('mps', 'float32'), ('cpu', 'float32'), ('mps', 'bfloat16')]:
        model.to(device=device, dtype=getattr(torch, dtype))
        encode(model, texts[:2], batch_size)
        if device == 'mps':
            torch.mps.synchronize()
        start = time.perf_counter()
        vectors = encode(model, texts, batch_size)
        if device == 'mps':
            torch.mps.synchronize()
        elapsed = time.perf_counter() - start
        item = dict(seconds=elapsed, texts_per_second=len(texts) / elapsed,
                    shape=list(vectors.shape), finite=True,
                    max_unit_norm_error=float(np.max(abs(np.linalg.norm(vectors, axis=1) - 1))))
        if gpu is None:
            gpu = vectors.copy()
        else:
            item.update(min_cosine_to_mps_float32=float(np.min(np.sum(gpu * vectors, axis=1))),
                        max_absolute_error=float(np.max(abs(gpu - vectors))))
        result[f'{device}_{dtype}'] = item
        print(device, dtype, item, flush=True)
        (out / 'mps_smoke_test.json').write_text(json.dumps(result, indent=2) + '\n')


def cache(model, source, batch_size, chunk_size, dtype, model_path):
    pool = pd.read_parquet(source / 'pool.parquet')
    # Deduplicate exact text, not query IDs: some different IDs share identical text.
    texts = list(dict.fromkeys(pool['query'].tolist() + pool['response'].tolist()))
    ids = {text: i for i, text in enumerate(texts)}
    fingerprint = hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()
    config = dict(model='google/embeddinggemma-2', revision=REVISION,
                  model_path=str(model_path.resolve()), dtype=dtype, device=str(model.device),
                  prompt=PROMPT, max_length=model.max_seq_length, dimension=768,
                  text_sha256=fingerprint, n_texts=len(texts), chunk_size=chunk_size,
                  batch_size=batch_size, normalize=True, fallback_enabled=False,
                  versions={p: importlib.metadata.version(p) for p in
                            ['torch', 'transformers', 'sentence-transformers', 'numpy']})
    dest = source / 'embeddings'
    dest.mkdir(exist_ok=True)
    marker = dest / 'config.json'
    if marker.exists() and json.loads(marker.read_text()) != config:
        raise ValueError('cache configuration changed; use a separate experiment directory')
    marker.write_text(json.dumps(config, indent=2) + '\n')
    mapping = pool[['model_id', 'task_id', 'query_id', 'suite', 'score']].copy()
    mapping['q_index'] = [ids[t] for t in pool['query']]
    mapping['r_index'] = [ids[t] for t in pool['response']]
    mapping.to_parquet(dest / 'rows.parquet', index=False)
    started = time.perf_counter()
    active_batch = batch_size
    for start in range(0, len(texts), chunk_size):
        stop = min(start + chunk_size, len(texts))
        path = dest / f'{start:07d}.npy'
        if path.exists():
            old = np.load(path)
            if old.shape != (stop - start, 768) or not np.isfinite(old).all():
                raise ValueError(f'invalid cached shard {path}')
            continue
        t = time.perf_counter()
        while True:
            try:
                vectors = encode(model, texts[start:stop], active_batch)
                break
            except RuntimeError as exc:
                if model.device.type != 'mps' or 'out of memory' not in str(exc).lower() or active_batch == 1:
                    raise
                active_batch = max(1, active_batch // 2)
                print(f'Metal allocation limit reached; retrying shard {start} with batch {active_batch}', flush=True)
            # The failed exception frame must leave scope before releasing caches.
            gc.collect()
            torch.mps.empty_cache()
        with path.with_suffix('.part').open('wb') as f:
            np.save(f, vectors)
        path.with_suffix('.part').replace(path)
        with (dest / 'timings.jsonl').open('a') as f:
            f.write(json.dumps(dict(start=start, stop=stop, seconds=time.perf_counter() - t,
                                   batch_size=active_batch)) + '\n')
        print(f'{stop}/{len(texts)} texts; {time.perf_counter()-started:.1f}s this session', flush=True)
        if model.device.type == 'mps':
            # Metal driver allocations can accumulate across heterogeneous long-text
            # batches. Release unused allocations; keep the safety watermark enabled.
            torch.mps.synchronize()
            torch.mps.empty_cache()
    # A marker distinguishes a completed cache from a resumable partial run.
    (dest / 'complete.json').write_text(json.dumps(dict(n_texts=len(texts), text_sha256=fingerprint)) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=Path('data/eee-followups'))
    ap.add_argument('--model', type=Path, default=Path('data/models/embeddinggemma-2'))
    ap.add_argument('--device', choices=['cpu', 'mps'], default='mps')
    ap.add_argument('--dtype', choices=['float32', 'bfloat16'], default='float32')
    ap.add_argument('--max-length', type=int, default=2048)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--chunk-size', type=int, default=128)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--evidence', type=Path, default=Path('projects/pkps/artifacts/stress-test'))
    args = ap.parse_args()
    model = load_model(args.model, args.device, args.dtype, args.max_length)
    if args.smoke:
        if (args.source / 'pool.parquet').exists():
            texts = pd.read_parquet(args.source / 'pool.parquet').sample(n=16, random_state=9).response.tolist()
        else:
            texts = ['Two plus three equals five.', 'The sum of two and three is five.',
                     'Jupiter is a gas giant.', 'Choose the correct answer to the mathematics question.'] * 4
        smoke(model, texts, args.batch_size, args.evidence)
    else:
        cache(model, args.source, args.batch_size, args.chunk_size, args.dtype, args.model)


if __name__ == '__main__':
    main()
