"""APW (Anchor Point Weighted) baseline -- Vivek et al., 2024 (EACL).

The IRT baseline (Polo et al., 2024) lives in ``dkps.baselines``
(``irt_fit_difficulties`` / ``irt_estimate_ability`` / ``irt_predict``).
"""

import kmedoids as _kmedoids_lib
import numba
import numpy as np


# ==========================================================================
# APW  (Anchor Point Weighted)
# ==========================================================================

def _query_correlation_matrix(score_matrix):
    """M x M Pearson correlation matrix between queries (across models)."""
    n_ref, M = score_matrix.shape
    S = score_matrix - score_matrix.mean(axis=0, keepdims=True)
    std = score_matrix.std(axis=0)
    valid = std > 0
    S[:, ~valid] = 0.0
    std_safe = np.where(valid, std, 1.0)
    S_norm = S / std_safe
    C = (S_norm.T @ S_norm) / n_ref
    C[~valid, :] = 0.0
    C[:, ~valid] = 0.0
    np.fill_diagonal(C, 1.0)
    return C


def _kmedoids_pam_numpy(D, k, n_init=5, random_state=0):
    """PAM K-Medoids (pure numpy). Returns (medoids, labels, cost)."""
    rng = np.random.default_rng(random_state)
    M = D.shape[0]
    best_medoids, best_labels, best_cost = None, None, np.inf

    for _ in range(n_init):
        medoids = rng.choice(M, size=k, replace=False)
        for _ in range(100):
            dist_to_med = D[:, medoids]
            labels = np.argmin(dist_to_med, axis=1)
            cost = dist_to_med[np.arange(M), labels].sum()
            improved = False
            for ci in range(k):
                for c in range(M):
                    if c in medoids:
                        continue
                    nm = medoids.copy(); nm[ci] = c
                    nd = D[:, nm]; nl = np.argmin(nd, axis=1)
                    nc = nd[np.arange(M), nl].sum()
                    if nc < cost:
                        medoids, labels, cost = nm, nl, nc
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        if cost < best_cost:
            best_cost = cost
            best_medoids = medoids.copy()
            best_labels = labels.copy()

    return best_medoids, best_labels, best_cost


@numba.njit(cache=True)
def _pam_swap(D, medoids, k, M):
    """One round of PAM swap steps. Returns (medoids, labels, cost, improved)."""
    dist_to_med = np.empty((M, k))
    for i in range(M):
        for j in range(k):
            dist_to_med[i, j] = D[i, medoids[j]]

    labels = np.empty(M, dtype=np.int64)
    for i in range(M):
        best_j = 0
        best_d = dist_to_med[i, 0]
        for j in range(1, k):
            if dist_to_med[i, j] < best_d:
                best_d = dist_to_med[i, j]
                best_j = j
        labels[i] = best_j

    cost = 0.0
    for i in range(M):
        cost += dist_to_med[i, labels[i]]

    is_medoid = np.zeros(M, dtype=numba.boolean)
    for j in range(k):
        is_medoid[medoids[j]] = True

    for ci in range(k):
        for c in range(M):
            if is_medoid[c]:
                continue

            new_cost = 0.0
            for i in range(M):
                d_new = D[i, c]
                best_d = d_new
                for j in range(k):
                    if j == ci:
                        continue
                    d_j = D[i, medoids[j]]
                    if d_j < best_d:
                        best_d = d_j
                best_d = min(best_d, d_new)
                new_cost += best_d

            if new_cost < cost:
                is_medoid[medoids[ci]] = False
                medoids[ci] = c
                is_medoid[c] = True

                for i in range(M):
                    best_j = 0
                    best_d = D[i, medoids[0]]
                    for j in range(1, k):
                        d_j = D[i, medoids[j]]
                        if d_j < best_d:
                            best_d = d_j
                            best_j = j
                    labels[i] = best_j

                cost = new_cost
                return medoids, labels, cost, True

    return medoids, labels, cost, False


def _kmedoids_pam_numba(D, k, n_init=5, random_state=0):
    """PAM K-Medoids (numba). Returns (medoids, labels, cost)."""
    rng = np.random.default_rng(random_state)
    M = D.shape[0]
    best_medoids, best_labels, best_cost = None, None, np.inf

    for _ in range(n_init):
        medoids = rng.choice(M, size=k, replace=False).astype(np.int64)
        for _ in range(100):
            medoids, labels, cost, improved = _pam_swap(D, medoids, k, M)
            if not improved:
                break
        if cost < best_cost:
            best_cost = cost
            best_medoids = medoids.copy()
            best_labels = labels.copy()

    return best_medoids, best_labels, best_cost


def _kmedoids_pam(D, k, n_init=5, random_state=0, method='fasterpam'):
    """PAM K-Medoids. Returns (medoids, labels, cost).

    method: 'fasterpam' (default, Rust), 'numba', or 'numpy'
    """
    if method == 'numpy':
        return _kmedoids_pam_numpy(D, k, n_init=n_init, random_state=random_state)
    elif method == 'numba':
        return _kmedoids_pam_numba(D, k, n_init=n_init, random_state=random_state)

    result = _kmedoids_lib.fasterpam(D, int(k), n_cpu=1, random_state=random_state)
    return np.array(result.medoids), np.array(result.labels), result.loss


def apw_fit(ref_scores, m, n_init=5, random_state=0):
    """
    Select m anchor queries and compute cluster-size weights.

    Parameters
    ----------
    ref_scores : (n_ref, M) per-item scores for reference models
    m          : int        query budget

    Returns
    -------
    anchor_idx : (m,) query indices of selected anchors
    weights    : (m,) cluster-size weights (sum to 1)
    """
    M = ref_scores.shape[1]
    if m >= M:
        return np.arange(M), np.ones(M) / M

    C = _query_correlation_matrix(ref_scores)
    D = np.clip(1.0 - C, 0.0, None)
    anchor_idx, labels, _ = _kmedoids_pam(D, k=m, n_init=n_init,
                                           random_state=random_state)
    weights = np.array([np.sum(labels == i)
                        for i in range(m)], dtype=float) / M
    return anchor_idx, weights


def apw_predict(target_scores, anchor_idx, weights):
    """
    Weighted average of target model scores on anchor queries.

    Parameters
    ----------
    target_scores : (M,) per-item scores of target model
    anchor_idx    : (m,) from apw_fit()
    weights       : (m,) from apw_fit()

    Returns
    -------
    float  predicted score in [0, 1]
    """
    return float(np.clip(np.dot(weights, target_scores[anchor_idx]), 0.0, 1.0))
