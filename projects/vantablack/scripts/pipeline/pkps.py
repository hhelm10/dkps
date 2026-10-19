"""PKPS-style context conditioning (pooled mode of dkps.pkps on the pkps branch).

Every response x to a query u becomes x - m(u), where m(u) is the
query-kernel-weighted mean of everyone's responses to queries near u:

    m(u) = sum_u' k(u, u') S(u') / sum_u' k(u, u') n(u')

S(u') and n(u') are the sum and count of responses to query u'. All agents
are included (system agents and the reference corpus); `self_share` reports
how much of each system agent's reference comes from its own responses.

Queries: a comment's query is its thread (the opening post's embedding); a
post's query is its submolt (name + description), not itself. Posts are
pooled with posts and comments with comments.

Kernel: Gaussian exp(-||u - u'||^2 / (2 sigma^2)) on the unit-norm query
embeddings, truncated to each query's k nearest queries (the weight beyond
them is negligible); sigma = bw * median nearest-neighbour distance (the
'median' rule of dkps.pkps), self excluded from the median.

time='period' computes m(u) separately from pre-t0 and post-t0 responses and
conditions each response on its own period, so a platform-wide change between
periods drops out of the displacements; time='all' pools both periods.
"""
import numpy as np
import scipy.sparse as sp


def knn_kernel(Q, k=64, bw=1.0, device='cuda', chunk=8192):
    """Sparse (U x U) truncated Gaussian kernel over unit-norm queries Q."""
    import torch
    U = len(Q)
    k = min(k, U)
    Qt = torch.as_tensor(np.asarray(Q, np.float32), device=device)
    Qt = Qt / Qt.norm(dim=1, keepdim=True).clamp_min(1e-12)
    idx = np.empty((U, k), np.int64)
    d2 = np.empty((U, k), np.float32)
    for a in range(0, U, chunk):
        sim = Qt[a:a + chunk] @ Qt.T
        v, i = sim.topk(k, dim=1)
        idx[a:a + chunk] = i.cpu().numpy()
        d2[a:a + chunk] = np.maximum(2 - 2 * v.cpu().numpy(), 0)
    del Qt
    nn = np.sqrt(d2[:, 1:]) if k > 1 else np.sqrt(d2)
    sigma = bw * float(np.median(nn))
    w = np.exp(-d2 / (2 * sigma ** 2)).astype(np.float32)
    K = sp.csr_matrix((w.ravel(), idx.ravel(), np.arange(0, U * k + 1, k)), shape=(U, U))
    return K, sigma


def _reference(K, X, qi, U):
    S = np.zeros((U, X.shape[1]), np.float32)
    for i in range(0, len(X), 200_000):                                # sum per query
        q = qi[i:i + 200_000]
        S += sp.csr_matrix((np.ones(len(q), np.float32), (q, np.arange(len(q)))), shape=(U, len(q))) @ X[i:i + 200_000]
    n = np.bincount(qi, minlength=U).astype(np.float32)
    return K @ S, K @ n


def condition(X, qi, U, K, is_sys_agent, period=None):
    """Residuals x - m(u) for all rows; m from rows of the same period if given.

    X: (N, p) responses, qi: (N,) query index in [0, U), K: (U, U) kernel,
    is_sys_agent: (N,) agent index for system rows, -1 for reference rows.
    Returns R (N, p) and self_den (N,): the kernel mass at each row's query
    contributed by its own agent's responses (system rows only).
    """
    R = np.empty_like(X)
    own = np.zeros(len(X), np.float32)
    groups = [np.arange(len(X))] if period is None else [np.flatnonzero(period == g)
                                                         for g in np.unique(period)]
    for rows in groups:
        Num, Den = _reference(K, X[rows], qi[rows], U)
        q = qi[rows]
        R[rows] = X[rows] - Num[q] / np.maximum(Den[q], 1e-12)[:, None]
        sysr = rows[is_sys_agent[rows] >= 0]
        if len(sysr):
            a = is_sys_agent[sysr]
            A = sp.csr_matrix((np.ones(len(sysr), np.float32), (a, qi[sysr])),
                              shape=(a.max() + 1, U))
            OWN = (A @ K.T).tocsr()  # (agent, u): own kernel mass at query u
            own_m = np.asarray(OWN[a, qi[sysr]]).ravel()
            own[sysr] = own_m / np.maximum(Den[qi[sysr]], 1e-12)
    return R, own


def condition_all(E_sys, ag_sys, t_sys, kind_sys, q_sys, E_ref, kind_ref, q_ref,
                  Q_thread, Q_sub, t0, t_ref=None, k=64, bw=1.0, time='all', log=print,
                  period_sys=None, period_ref=None):
    """Condition the system rows on everyone's responses.

    kind_*: 'post' or 'comment' per row; q_*: index into Q_sub (posts) or
    Q_thread (comments). time='step' conditions every row on the responses of its own
    period (period_sys / period_ref, e.g. the time step), so a stimulus shared at one
    time drops out. Returns (R_sys, self_share per system row, info).
    """
    R = np.empty_like(E_sys)
    share = np.zeros(len(E_sys), np.float32)
    info = {}
    for kind, Q in (('post', Q_sub), ('comment', Q_thread)):
        ms, mr = kind_sys == kind, kind_ref == kind
        K, sigma = knn_kernel(Q, k=k, bw=bw)
        X = np.vstack([E_sys[ms], E_ref[mr]]).astype(np.float32)
        qi = np.concatenate([q_sys[ms], q_ref[mr]])
        who = np.concatenate([ag_sys[ms], np.full(mr.sum(), -1)])
        per = None
        if time == 'period':
            per = np.concatenate([t_sys[ms] > t0, t_ref[mr] > t0]).astype(int)
        elif time == 'step':
            per = np.concatenate([period_sys[ms], period_ref[mr]]).astype(int)
        Rk, own = condition(X, qi, len(Q), K, who, per)
        R[ms] = Rk[:ms.sum()]
        share[ms] = own[:ms.sum()]
        kept = float(np.mean(np.linalg.norm(Rk[:ms.sum()], axis=1)
                             / np.maximum(np.linalg.norm(X[:ms.sum()] - X.mean(0), axis=1), 1e-12)))
        info[kind] = dict(n_queries=len(Q), sigma=sigma, n_sys=int(ms.sum()), n_ref=int(mr.sum()),
                          kept=kept, self_share_median=float(np.median(own[:ms.sum()])),
                          self_share_p95=float(np.quantile(own[:ms.sum()], .95)))
        log(f'[pkps] {kind}s: {len(Q)} queries, sigma={sigma:.3f}, {ms.sum()} system + '
            f'{mr.sum()} reference responses; residual keeps {kept:.2f} of the norm; '
            f'own share of reference: median {info[kind]["self_share_median"]:.3f}, '
            f'95th pct {info[kind]["self_share_p95"]:.3f}')
    return R, share, info
