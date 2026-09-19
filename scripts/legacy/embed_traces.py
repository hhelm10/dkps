"""Precompute trace text embeddings into the on-disk cache.

Run this once (slow step, GPU-friendly); the example notebook then loads from
the cache instantly. Safe to interrupt and re-run -- the cache is per-trace.

Usage:
    python scripts/embed_traces.py --traces-root data/traces [--stub]
"""
import argparse
import os
import sys
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dkps.traces import (TraceEmbedder, hash_embed_fn, load_langfuse_corpus,
                         make_sentence_transformer_embed_fn)

DEFAULT_ZIP = '/home/ubuntu/20260729_traces.zip'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traces-root', default='data/traces')
    ap.add_argument('--zip', default=DEFAULT_ZIP, help='auto-extract source if traces-root is missing')
    ap.add_argument('--cache-dir', default='.dkps_cache')
    ap.add_argument('--model', default='nomic-ai/nomic-embed-text-v1.5')
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--stub', action='store_true', help='use the deterministic hash embedder (no GPU)')
    args = ap.parse_args()

    if not os.path.isdir(args.traces_root) and os.path.exists(args.zip):
        print(f'extracting {args.zip} -> {args.traces_root}')
        os.makedirs(args.traces_root, exist_ok=True)
        with zipfile.ZipFile(args.zip) as z:
            z.extractall(args.traces_root)

    traces = load_langfuse_corpus(args.traces_root)
    print(f'loaded {len(traces)} traces')

    embed_fn = hash_embed_fn if args.stub else make_sentence_transformer_embed_fn(
        args.model, batch_size=args.batch_size)
    embedder = TraceEmbedder(embed_fn=embed_fn, cache_dir=args.cache_dir)
    embedder.transform_channels(traces)
    print(f'done. cache hits={embedder.cache_hits} misses={embedder.cache_misses} '
          f'(cache dir: {args.cache_dir}, config hash: {embedder._config_hash()})')


if __name__ == '__main__':
    main()
