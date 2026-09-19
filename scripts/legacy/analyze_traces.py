"""DKPS analysis of agentic traces (SWE-bench Verified swe-agent runs).

Traces are decomposed into four channels (action n-grams, pooled step-text,
canonicalized final diff, scalar behavioral features) and pushed through DKPS
in three combination modes:
  (a) concatenated vector -> DataKernelPerspectiveSpace (unchanged)
  (b) weighted combination of per-channel model distance matrices -> CMDS
  (c) per-channel geometries compared via Procrustes

Validation without correctness labels: replicate consistency (same-model runs
should cluster), Mantel tests against run metadata, channel ablations.
`exit_status` is a termination reason, not a correctness score (half the models
submit 100% of the time); when SWE-bench resolved labels are available, join
them with dkps.traces.attach_labels and predict performance as in the paper.

Run `python scripts/embed_traces.py` first (populates the embedding cache).

Usage:
    python scripts/analyze_traces.py [--traces-root data/traces]
                                     [--cache-dir .dkps_cache] [--figdir figures]
"""
import argparse
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import procrustes
from graspologic.embed import ClassicalMDS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dkps import DataKernelPerspectiveSpace
from dkps.traces import (load_langfuse_corpus, TraceEmbedder, build_dkps_input,
                         model_distance_matrix, combine_channel_distances, rms_scale)

FAMILY_COLORS = {
    'gemma': 'tab:green', 'Qwen': 'tab:blue', 'NVIDIA': 'tab:orange',
    'gpt': 'tab:red', 'GLM': 'tab:purple', 'Kimi': 'tab:brown',
    'MiniMax': 'tab:pink', 'Laguna': 'tab:gray',
}


def family_color(m):
    return next((c for fam, c in FAMILY_COLORS.items() if m.startswith(fam)), 'k')


def concat(blocks, names, weights=None):
    weights = weights or {}
    return np.hstack([rms_scale(blocks[n]) * weights.get(n, 1.0) for n in names])


def build_unit_input(traces, X, queries):
    """{(model, rep) unit: (n_queries, 1, d)}, dropping units missing any query."""
    rows = defaultdict(dict)
    for tr, x in zip(traces, X):
        rows[(tr.model_id, tr.replicate)][tr.query_id] = x
    out = {}
    for (m, r), qmap in sorted(rows.items()):
        if len(qmap) == len(queries):
            out[f'{m}|r{r}'] = np.stack([qmap[q] for q in queries])[:, None, :]
    return out


def replicate_consistency(unit_input):
    """Mean within-model / mean between-model distance over (model, rep) units.
    Lower is better; 1.0 = replicates carry no model identity."""
    units, dist = model_distance_matrix(unit_input)
    unit_models = [u.rsplit('|', 1)[0] for u in units]
    same = np.array([[a == b for b in unit_models] for a in unit_models])
    triu = np.triu_indices_from(dist, k=1)
    within = dist[triu][same[triu]]
    between = dist[triu][~same[triu]]
    return within.mean() / between.mean(), units, dist


def mantel(D1, D2, n_perm=10000, seed=0):
    rng = np.random.default_rng(seed)
    triu = np.triu_indices_from(D1, k=1)
    a = D1[triu]
    r_obs = np.corrcoef(a, D2[triu])[0, 1]
    n = D1.shape[0]
    count = sum(
        np.corrcoef(a, D2[np.ix_(p, p)][triu])[0, 1] >= r_obs
        for p in (rng.permutation(n) for _ in range(n_perm))
    )
    return r_obs, (count + 1) / (n_perm + 1)


def scatter_labeled(ax, coords, labelfmt=lambda m: m, s=60, fontsize=8):
    for m, c in coords.items():
        ax.scatter(c[0], c[1], color=family_color(m.rsplit('|', 1)[0]), s=s)
        ax.annotate(labelfmt(m), (c[0], c[1]), fontsize=fontsize,
                    xytext=(4, 4), textcoords='offset points')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traces-root', default='data/traces')
    ap.add_argument('--cache-dir', default='.dkps_cache')
    ap.add_argument('--figdir', default='figures')
    args = ap.parse_args()
    os.makedirs(args.figdir, exist_ok=True)

    # ---- load ----------------------------------------------------------
    traces = load_langfuse_corpus(args.traces_root)
    models = sorted({t.model_id for t in traces})
    queries = sorted({t.query_id for t in traces})
    print(f'{len(traces)} traces | {len(models)} models x {len(queries)} queries')
    print('exit statuses:', dict(Counter(t.exit_status for t in traces).most_common()))

    # ---- embed (from cache) -------------------------------------------
    embedder = TraceEmbedder(cache_dir=args.cache_dir)
    blocks = embedder.transform_channels(traces)
    print('cache hits/misses:', embedder.cache_hits, '/', embedder.cache_misses)
    channels = list(blocks)
    structured = [c for c in channels if c != 'whole']   # 'whole' = naive baseline
    for name, b in blocks.items():
        print(f'  {name:10s} {b.shape}')
    X_all = concat(blocks, structured)

    # ---- mode (a): model perspective space ----------------------------
    X_dkps, report = build_dkps_input(traces, X_all)
    if report['imputed']:
        print('imputed replicate slots:', report['imputed'])
    coords = DataKernelPerspectiveSpace().fit_transform(X_dkps)
    fig, ax = plt.subplots(figsize=(9, 7))
    scatter_labeled(ax, coords)
    ax.set_xlabel('CMDS 1'); ax.set_ylabel('CMDS 2')
    ax.set_title('DKPS perspective space, agentic traces (structured channels)')
    fig.tight_layout()
    fig.savefig(os.path.join(args.figdir, 'perspective_space.png'), dpi=150)

    # ---- replicate consistency ----------------------------------------
    unit_input = build_unit_input(traces, X_all, queries)
    score, units, _ = replicate_consistency(unit_input)
    print(f'\n{len(units)} (model, replicate) units | '
          f'replicate-consistency ratio: {score:.3f} (lower is better, 1.0 = chance)')
    unit_coords = DataKernelPerspectiveSpace().fit_transform(unit_input)
    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.get_cmap('tab20')
    model_color = {m: cmap(i % 20) for i, m in enumerate(models)}
    for u, c in unit_coords.items():
        ax.scatter(c[0], c[1], color=model_color[u.rsplit('|', 1)[0]], s=30)
    for m in models:
        ax.scatter([], [], color=model_color[m], label=m)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title(f'(model, replicate) units — consistency {score:.3f}')
    fig.tight_layout()
    fig.savefig(os.path.join(args.figdir, 'replicate_consistency.png'), dpi=150)

    # ---- metadata Mantel tests ----------------------------------------
    per_model = {m: [t for t in traces if t.model_id == m] for m in models}
    covariates = {
        'log cost': lambda t: np.log1p(float(t.metadata.get('total_cost', 0) or 0)),
        'log tokens sent': lambda t: np.log1p(float(t.metadata.get('total_tokens_sent', 0) or 0)),
        'submit rate': lambda t: float((t.exit_status or '').startswith('submitted')),
        'n tool calls': lambda t: float(t.n_tool_calls),
    }
    _, D_dkps = model_distance_matrix(X_dkps)
    print(f'\nMantel tests, DKPS distance vs |Δ covariate|:')
    print(f'{"covariate":18s} {"r":>7s} {"p":>8s}')
    for name, fn in covariates.items():
        v = np.array([np.mean([fn(t) for t in per_model[m]]) for m in models])
        r, p = mantel(D_dkps, np.abs(v[:, None] - v[None, :]))
        print(f'{name:18s} {r:7.3f} {p:8.4f}')

    # ---- channel ablations --------------------------------------------
    configs = ({'whole trace (baseline)': ['whole']} if 'whole' in channels else {})
    configs |= ({f'{n} only': [n] for n in structured} |
                {f'without {n}': [c for c in structured if c != n] for n in structured} |
                {'all structured (concat)': structured})
    print(f'\nreplicate-consistency by channel configuration:')
    print(f'{"config":22s} {"consistency":>11s}')
    for name, chans in configs.items():
        Xc = concat(blocks, chans)
        s, _, _ = replicate_consistency(build_unit_input(traces, Xc, queries))
        print(f'{name:22s} {s:11.3f}')

    # ---- mode (b): distance combination; mode (c): per-channel geometry
    channel_inputs = {n: build_dkps_input(traces, blocks[n])[0] for n in channels}
    coords_b, D_combined, D_per_channel = combine_channel_distances(
        channel_inputs, weights={'whole': 0.0})   # baseline shown, not combined
    r, p = mantel(D_dkps, D_combined)
    print(f'\nmode (b) combined-distance vs mode (a) concat geometry: '
          f'mantel r={r:.3f}, p={p:.4f}')

    channel_coords = {n: ClassicalMDS(n_components=2).fit_transform(D_per_channel[n])
                      for n in channels}
    ncols = 3 if len(channels) > 4 else 2
    nrows = -(-len(channels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    for ax in axes.ravel()[len(channels):]:
        ax.axis('off')
    for ax, name in zip(axes.ravel(), channels):
        C = channel_coords[name]
        scatter_labeled(ax, dict(zip(models, C)),
                        labelfmt=lambda m: m.split('-')[0], s=50, fontsize=7)
        ax.set_title(f'{name} channel')
    fig.tight_layout()
    fig.savefig(os.path.join(args.figdir, 'per_channel_spaces.png'), dpi=150)

    print('\npairwise Procrustes disparity between channel geometries:')
    print(f'{"":12s}' + ''.join(f'{n:>12s}' for n in channels))
    for a in channels:
        row = ''.join(f'{procrustes(channel_coords[a], channel_coords[b])[2]:12.3f}'
                      for b in channels)
        print(f'{a:12s}' + row)

    print(f'\nfigures written to {args.figdir}/')


if __name__ == '__main__':
    main()
