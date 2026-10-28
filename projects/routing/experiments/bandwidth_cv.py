"""Leak-free bandwidth selection for query-aware routing.

The anchors are cached history, so they can serve as pseudo-evaluation
queries: hold each out of its own geometry (self-weight zeroed), route every
target, and measure the true mimicry error from the precomputed tensor. This
yields a per-(pseudo-query, sigma) error table from which we select

    cv-global    : the single sigma minimizing mean pseudo-error
    cv-per-query : for each real eval query, the sigma minimizing the
                   proximity-weighted pseudo-error of nearby anchors
                   (fixed meta-bandwidth: half the median anchor distance)

Neither selection ever sees an evaluation query's responses.
"""

import numpy as np

from .geometry import rbf_weights, weighted_dist_matrix


def _route_errors(D_stack, E_true):
    """D_stack: (q, n, n) geometries; E_true: (q, n, n) true per-query errors.
    Returns (q, n) error of the nearest-candidate pick for every target."""
    q, n, _ = D_stack.shape
    out = np.empty((q, n), dtype=np.float64)
    eye = np.eye(n, dtype=bool)
    for v in range(q):
        D = D_stack[v].copy()
        D[eye] = np.inf
        picks = D.argmin(axis=1)
        out[v] = E_true[v][np.arange(n), picks]
    return out


def pseudo_error_table(P_anchor, anchor_emb, sigmas, n_pseudo=200, rng=None):
    """errors[s, v] = mean-over-targets routing error at pseudo-anchor v under
    sigma s, with v excluded from its own geometry."""
    rng = rng or np.random.default_rng(0)
    m_a = len(anchor_emb)
    V = np.sort(rng.choice(m_a, size=min(n_pseudo, m_a), replace=False))
    E_true = np.sqrt(P_anchor[V])

    table = np.empty((len(sigmas), len(V)))
    for si, s in enumerate(sigmas):
        W = rbf_weights(anchor_emb, anchor_emb[V], s)      # (V, m_a)
        W[np.arange(len(V)), V] = 0.0                      # self-exclusion
        D = weighted_dist_matrix(P_anchor, W)              # (V, n, n)
        table[si] = _route_errors(D, E_true).mean(axis=1)
    return V, table


def select_cv_global(table):
    return int(np.argmin(table.mean(axis=1)))


def select_cv_per_query(table, anchor_emb_V, eval_emb, meta_sigma):
    """sigma index per eval query: argmin of proximity-weighted pseudo-error."""
    M = rbf_weights(anchor_emb_V, eval_emb, meta_sigma)    # (q, V)
    M = M / M.sum(axis=1, keepdims=True)
    scores = M @ table.T                                   # (q, S)
    return scores.argmin(axis=1)
