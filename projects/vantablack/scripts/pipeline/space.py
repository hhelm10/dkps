"""Vantablack perspective space (one-pager Problem C, TODO section 3).

1. Residualize each action embedding on its thread's context embedding,
   so that an agent that changes threads or submolts does not look as if
   it changed.
2. Windowed positions: psi_n(t) is the mean residual of agent n's actions
   in a trailing window ending at grid time t (time- or action-count
   window); k_n(t) is the number of actions in the window.
3. Joint embedding: classical MDS on the Euclidean distances between all
   (agent, t) windowed means. For Euclidean CMDS this is PCA of the
   centred means, so we keep the basis V and project action residuals
   once (Z = (R - mu) V). Windowed positions are then a sparse linear map
   of Z, which is what makes permutation nulls cheap. d is chosen by the
   profile-likelihood elbow rule (graspologic.select_dimension).
"""
import numpy as np
import scipy.sparse as sp
from graspologic.embed import select_dimension
from sklearn.metrics import silhouette_score


def residualize(E, C_act, ridge=1.0):
    """Ridge-regress action embeddings on context embeddings; return residuals.

    E: (n, p) action embeddings, C_act: (n, q) embedding of each action's
    thread context. An intercept is included.
    """
    C1 = np.hstack([C_act, np.ones((len(C_act), 1))])
    G = C1.T @ C1 + ridge * np.eye(C1.shape[1])
    W = np.linalg.solve(G, C1.T @ E)
    return E - C1 @ W, W


def separability(X, labels, n_max=3000, seed=0):
    """Silhouette of X by label (e.g. submolt). Near 0 = no trivial clustering."""
    labels = np.asarray(labels)
    if len(set(labels)) < 2:
        return np.nan
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))[:n_max]
    return float(silhouette_score(X[idx], labels[idx]))


def window_matrix(agent_idx, t, n_agents, grid, window_h=None, window_k=None):
    """Row-normalized sparse map from actions to (agent, grid) windowed means.

    Returns M of shape (n_agents * len(grid), n_actions) and k of shape
    (n_agents, len(grid)). Exactly one of window_h (hours) or window_k
    (last-k actions) must be given.
    """
    assert (window_h is None) != (window_k is None)
    rows, cols = [], []
    G = len(grid)
    k = np.zeros((n_agents, G), dtype=int)
    for n in range(n_agents):
        idx = np.flatnonzero(agent_idx == n)
        if len(idx) == 0:
            continue
        idx = idx[np.argsort(t[idx])]
        tn = t[idx]
        hi = np.searchsorted(tn, grid, side='right')
        if window_h is not None:
            lo = np.searchsorted(tn, grid - window_h, side='right')
        else:
            lo = np.maximum(hi - window_k, 0)
        for g in range(G):
            sel = idx[lo[g]:hi[g]]
            k[n, g] = len(sel)
            rows.extend([n * G + g] * len(sel))
            cols.extend(sel.tolist())
    vals = np.ones(len(rows))
    M = sp.csr_matrix((vals, (rows, cols)), shape=(n_agents * G, len(t)))
    rs = np.asarray(M.sum(1)).ravel()
    rs[rs == 0] = 1
    M = sp.diags(1 / rs) @ M
    return M.tocsr(), k


def period_matrices(agent_idx, t, n_agents, t0):
    """Row-normalized maps from actions to each agent's burn-in / post means."""
    def build(mask):
        r = agent_idx[mask]
        c = np.flatnonzero(mask)
        M = sp.csr_matrix((np.ones(len(c)), (r, c)), shape=(n_agents, len(t)))
        cnt = np.asarray(M.sum(1)).ravel()
        return sp.diags(1 / np.maximum(cnt, 1)) @ M, cnt
    pre, n_pre = build(t <= t0)
    post, n_post = build(t > t0)
    return pre.tocsr(), post.tocsr(), n_pre, n_post


def fit_projection(R, M, k, d=None, n_elbows=2, d_max=50, n_sub=50000, seed=0):
    """Joint CMDS (= PCA) basis from the windowed means with k > 0.

    The basis comes from the eigendecomposition of the p x p covariance of
    all windowed means; the elbow rule runs on a row subsample of n_sub
    (only matters at platform scale).
    """
    X = M @ R
    X = X[k.ravel() > 0]
    mu = X.mean(0)
    Xc = X - mu
    if d is None:
        rng = np.random.default_rng(seed)
        sub = Xc[rng.permutation(len(Xc))[:n_sub]]
        elbows, _ = select_dimension(sub, n_components=min(d_max, Xc.shape[1] - 1),
                                     n_elbows=n_elbows)
        d = int(elbows[-1])
    w, V = np.linalg.eigh(Xc.T @ Xc)
    w, V = w[::-1], V[:, ::-1]
    var = np.maximum(w, 0) / np.maximum(w, 0).sum()
    return dict(mu=mu, V=V[:, :d], d=d, explained=float(var[:d].sum()))


def project(R, proj):
    return (R - proj['mu']) @ proj['V']
