"""Precompute the per-query pairwise squared-distance tensor P (m, n, n).

Run: python -m projects.routing.experiments.precompute
"""

import os

import numpy as np

from .data import LOCAL_DATA, load_jailbreak_suite
from .geometry import pairwise_query_dist_tensor


def main():
    out_path = os.path.join(LOCAL_DATA, 'per_query_dist_sq.npy')
    if os.path.exists(out_path):
        print(f'already cached: {out_path}')
        return

    suite = load_jailbreak_suite()
    X = suite['X']
    print(f'X: {X.shape} ({len(suite["model_names"])} models)')

    P = pairwise_query_dist_tensor(X)
    os.makedirs(LOCAL_DATA, exist_ok=True)
    np.save(out_path, P)
    print(f'saved {P.shape} float32 ({P.nbytes / 1e6:.0f} MB) -> {out_path}')


if __name__ == '__main__':
    main()
