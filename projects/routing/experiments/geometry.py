"""Query-weighted behavioral geometry.

For a fully-paired suite (every model answered every query), any query-weighted
DKPS distance decomposes over queries:

    D_w^2(i, k) = sum_j w_j * ||x_ij - x_kj||^2 / sum_j w_j

so we precompute the per-query pairwise squared-distance tensor
P[j] = [||x_ij - x_kj||^2]_{ik} once, and every geometry -- static DKPS
(uniform w), query-local (w_j = k_Q(q_j, q*)), any bandwidth -- is a single
tensordot over j. Uniform weights recover static DKPS; an RBF query kernel
with sigma -> inf does the same, so the routing method and the static baseline
live on one bandwidth continuum.
"""

import numpy as np
from scipy.spatial.distance import cdist


def pairwise_query_dist_tensor(X, batch=256, dtype=np.float32):
    """P[j, i, k] = ||X[i, j] - X[k, j]||^2 for a paired suite.

    X : (n_models, m_queries, d). Returns (m, n, n) float32.
    """
    n, m, d = X.shape
    P = np.empty((m, n, n), dtype=dtype)
    for start in range(0, m, batch):
        stop = min(start + batch, m)
        Xb = X[:, start:stop].astype(np.float64)          # (n, b, d)
        sq = (Xb ** 2).sum(-1)                            # (n, b)
        G = np.einsum('ibd,kbd->bik', Xb, Xb)             # (b, n, n)
        D2 = sq.T[:, :, None] + sq.T[:, None, :] - 2 * G
        P[start:stop] = np.maximum(D2, 0.0)
    return P


def weighted_dist_matrix(P, weights):
    """D_w = sqrt(sum_j w_j P[j] / sum_j w_j).

    P : (m, n, n); weights : (m,) or (q, m) for a stack of weightings.
    Returns (n, n) or (q, n, n).
    """
    w = np.asarray(weights, dtype=P.dtype)
    if w.ndim == 1:
        return np.sqrt(np.tensordot(w / w.sum(), P, axes=1))
    wn = w / w.sum(axis=1, keepdims=True)
    return np.sqrt(np.tensordot(wn, P, axes=([1], [0])))


def rbf_weights(anchor_query_emb, new_query_emb, sigma):
    """w[q, j] = exp(-||e_j - e_q*||^2 / (2 sigma^2)).

    anchor_query_emb : (m, d_q); new_query_emb : (q, d_q) or (d_q,).
    Returns (q, m) or (m,).
    """
    E = np.atleast_2d(new_query_emb)
    sq = cdist(E, anchor_query_emb, 'sqeuclidean')
    # Shift by the per-query minimum so small sigmas don't underflow to an
    # all-zero row; normalized weights are unchanged.
    sq = sq - sq.min(axis=1, keepdims=True)
    w = np.exp(-sq / (2.0 * float(sigma) ** 2))
    return w[0] if np.asarray(new_query_emb).ndim == 1 else w


def static_dist_matrix(P):
    """Static (query-agnostic) DKPS distance: uniform weights over anchors."""
    return weighted_dist_matrix(P, np.ones(P.shape[0]))
