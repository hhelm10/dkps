"""End-to-end pipeline check with a deterministic stub embedder (no GPU, no cache).

Usage: python scripts/smoke_test.py [--traces-root data/traces]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dkps import DataKernelPerspectiveSpace
from dkps.traces import (load_langfuse_corpus, TraceEmbedder, hash_embed_fn,
                         build_dkps_input, combine_channel_distances)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traces-root', default='data/traces')
    args = ap.parse_args()

    traces = load_langfuse_corpus(args.traces_root)
    assert traces, f'no traces under {args.traces_root}'
    n_models = len({t.model_id for t in traces})
    n_queries = len({t.query_id for t in traces})
    print(f'parsed {len(traces)} traces ({n_models} models x {n_queries} queries)')
    fallbacks = sum(t.metadata['n_parse_fallbacks'] for t in traces)
    assert fallbacks == 0, f'{fallbacks} span parse fallbacks'

    emb = TraceEmbedder(embed_fn=hash_embed_fn)
    blocks = emb.transform_channels(traces, progress=False)
    for name, b in blocks.items():
        assert b.shape[0] == len(traces) and np.isfinite(b).all(), name
        print(f'  {name:10s} {b.shape}')
    X = emb.transform(traces, progress=False)
    X2 = TraceEmbedder(embed_fn=hash_embed_fn).transform(traces, progress=False)
    assert np.array_equal(X, X2), 'pipeline is not deterministic'

    inp, report = build_dkps_input(traces, X)
    print(f'assembled {len(inp)} models, shape {next(iter(inp.values())).shape}, '
          f'imputed {len(report["imputed"])} slots')
    coords = DataKernelPerspectiveSpace().fit_transform(inp)
    assert isinstance(coords, dict) and len(coords) == n_models

    channel_inputs = {n: build_dkps_input(traces, b)[0] for n, b in blocks.items()}
    coords_b, _, _ = combine_channel_distances(channel_inputs)
    assert len(coords_b) == n_models
    print('OK: parse, channels, determinism, mode (a) concat, mode (b) distances')


if __name__ == '__main__':
    main()
