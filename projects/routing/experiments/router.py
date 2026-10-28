"""Routing rules and per-query mimicry errors.

Mimicry error of routing model r for target t on query q* is the response-space
distance ||x_{r,q*} - x_{t,q*}||. The target itself is excluded from the
candidate pool (routing exists because the target is unavailable), and scores 0
by construction.
"""

import numpy as np


def route_nearest(dist_matrix, target, exclude=None):
    """Nearest candidate to `target` in a model-model distance matrix."""
    d = dist_matrix[target].copy()
    d[target] = np.inf
    if exclude is not None:
        d[list(exclude)] = np.inf
    return int(np.argmin(d))


def mimicry_errors(P_eval_query, target):
    """||x_{i,q*} - x_{t,q*}|| for all candidates i, from the (n, n) slice of
    the per-query distance tensor at the evaluation query."""
    return np.sqrt(P_eval_query[target])


def expected_random_error(errors, target):
    """Mean mimicry error over all candidates != target."""
    mask = np.ones(len(errors), dtype=bool)
    mask[target] = False
    return float(errors[mask].mean())


def oracle_error(errors, target):
    """Best achievable mimicry error by any candidate != target (hindsight)."""
    e = errors.copy()
    e[target] = np.inf
    return float(e.min())
