"""Simulation framework for detecting colluding subnetworks from sparse, unpaired responses.

Observation model (Section 2 of the paper):
  N agents f_i : Q -> X.  At each of two time points tau in {t0, t} agent i responds to
  m_i^(tau) queries drawn from its own query distribution (self-selected topics).  Queries
  may be observed exactly or with estimation noise (vantablack setting).

Generative model:
  latent agent position a_i in R^dl;  query embedding u in R^pq drawn near one of K topic
  centers;  response embedding x = B a_i + W u + eps.  After burn-in, agents in S shift
  a_i -> a_i + delta * v (shared direction v, effect size delta).  All agents may also share a
  common drift c.

Method (Section 3): PKPS affinity over (agent, time) units with Gaussian kernels, CMDS to
d dims, centered displacements, scan statistic over a candidate family, permutation null that
reassigns each agent's (query, response) pairs across the two time points.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional


# ----------------------------------------------------------------------------- generative model
@dataclass
class SimConfig:
    N: int = 50            # agents
    S_size: int = 5        # colluding subnetwork size
    delta: float = 1.0     # effect size (latent units; B has orthonormal columns)
    m: float = 10.0        # mean responses per agent per time point
    sigma_x: float = 0.5   # response noise sd (per coordinate)
    sigma_q: float = 1.0   # query estimation noise sd (0 = black box reference; >0 = vantablack, the default)
    alpha: float = 1.0     # Dirichlet concentration of topic preferences (small = unpaired)
    K_topics: int = 6
    d_latent: int = 3
    p_q: int = 6
    p_x: int = 12
    drift: float = 0.0     # sd of common drift c
    S2_size: int = 0       # optional second colluding subnetwork (membership experiments)
    topic_spread: float = 0.5
    query_effect: float = 0.5   # scale of the query's effect on the response
    seed: int = 0


def simulate(cfg: SimConfig, rng: np.random.Generator):
    """Return dict with per-observation arrays: unit index, agent, epoch, u_obs, x."""
    N, dl, pq, px = cfg.N, cfg.d_latent, cfg.p_q, cfg.p_x
    # fixed maps (drawn once per replicate)
    B, _ = np.linalg.qr(rng.normal(size=(px, dl)))                   # orthonormal columns
    W = cfg.query_effect * rng.normal(size=(px, pq)) / np.sqrt(pq)
    centers = rng.normal(size=(cfg.K_topics, pq)) * 1.5
    a = rng.normal(size=(N, dl))
    pref = rng.dirichlet(np.full(cfg.K_topics, cfg.alpha), size=N)
    perm = rng.permutation(N)
    S = np.sort(perm[:cfg.S_size]) if cfg.delta > 0 else np.array([], int)
    S2 = np.sort(perm[cfg.S_size:cfg.S_size + cfg.S2_size]) if (cfg.delta > 0 and cfg.S2_size > 0) else np.array([], int)
    v = rng.normal(size=dl); v /= np.linalg.norm(v)
    v2 = rng.normal(size=dl); v2 -= (v2 @ v) * v; v2 /= np.linalg.norm(v2)   # orthogonal direction
    c = rng.normal(size=dl) * cfg.drift

    agents, epochs, U, X = [], [], [], []
    for tau in (0, 1):
        a_tau = a.copy()
        if tau == 1:
            a_tau += c
            a_tau[S] += cfg.delta * v
            a_tau[S2] += cfg.delta * v2
        for i in range(N):
            mi = 2 + rng.poisson(max(cfg.m - 2.0, 0.0))   # at least two responses per unit
            topics = rng.choice(cfg.K_topics, size=mi, p=pref[i])
            u = centers[topics] + cfg.topic_spread * rng.normal(size=(mi, pq))
            x = a_tau[i] @ B.T + u @ W.T + cfg.sigma_x * rng.normal(size=(mi, px))
            U.append(u); X.append(x)
            agents.append(np.full(mi, i)); epochs.append(np.full(mi, tau))
    U = np.vstack(U); X = np.vstack(X)
    agents = np.concatenate(agents); epochs = np.concatenate(epochs)
    U_obs = U + cfg.sigma_q * rng.normal(size=U.shape) if cfg.sigma_q > 0 else U
    return dict(agents=agents, epochs=epochs, U=U_obs, X=X, S=S, S2=S2, v=v)


# ----------------------------------------------------------------------------- PKPS machinery
def sqdist(A, Bm=None):
    Bm = A if Bm is None else Bm
    aa = (A * A).sum(1)[:, None]; bb = (Bm * Bm).sum(1)[None, :]
    return np.maximum(aa + bb - 2 * A @ Bm.T, 0.0)


def gaussian_kernel(A, h):
    return np.exp(-sqdist(A) / (2 * h * h))


def median_bandwidth(A, rng, nmax=600):
    idx = rng.choice(len(A), size=min(nmax, len(A)), replace=False)
    d = sqdist(A[idx]); tri = d[np.triu_indices(len(idx), 1)]
    return np.sqrt(np.median(tri[tri > 0]) / 2.0) if np.any(tri > 0) else 1.0


def unit_sums(P, unit, n_units):
    """Return the n_units x n_units matrix of block sums of P grouped by unit (Z^T P Z)."""
    Z = np.zeros((P.shape[0], n_units)); Z[np.arange(P.shape[0]), unit] = 1.0
    return Z.T @ P @ Z


def cmds(D2, d):
    n = len(D2)
    J = np.eye(n) - 1.0 / n
    Bm = -0.5 * J @ D2 @ J
    w, V = np.linalg.eigh(Bm)
    order = np.argsort(w)[::-1][:d]
    w = np.maximum(w[order], 0.0)
    return V[:, order] * np.sqrt(w)[None, :]


def embed(KQ, PQX, unit, n_units, d):
    num = unit_sums(PQX, unit, n_units)
    den = unit_sums(KQ, unit, n_units)
    A = num / np.maximum(den, 1e-12)
    diag = np.diag(A)
    D2 = diag[:, None] + diag[None, :] - 2 * A
    np.fill_diagonal(D2, 0.0); D2 = np.maximum(D2, 0.0)
    return cmds(D2, d)


# ----------------------------------------------------------------------------- scan statistic
def trimmed_cov(Dlt, trim=0.25):
    n, d = Dlt.shape
    C = np.cov(Dlt.T) + 1e-8 * np.eye(d)
    Ci = np.linalg.inv(C)
    md = np.einsum('ij,jk,ik->i', Dlt, Ci, Dlt)
    keep = md <= np.quantile(md, 1 - trim)
    C = np.cov(Dlt[keep].T) + 1e-8 * np.eye(d)
    # rescale: trimming shrinks the covariance; consistency factor for Gaussian (approximate)
    return C / (1 - trim) ** 0.5


def kmeans(Xp, k, rng, iters=25, inits=3):
    best, best_in = None, np.inf
    n = len(Xp)
    for _ in range(inits):
        C = Xp[rng.choice(n, size=k, replace=False)].copy()
        for _ in range(iters):
            lab = np.argmin(sqdist(Xp, C), axis=1)
            newC = np.array([Xp[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(k)])
            if np.allclose(newC, C): break
            C = newC
        inertia = ((Xp - C[lab]) ** 2).sum()
        if inertia < best_in: best_in, best = inertia, lab.copy()
    return best


def candidate_family(Dlt, Ci_half, rng, Kmax=4):
    """Return list of index arrays (candidate subsets S)."""
    N = len(Dlt)
    Z = Dlt @ Ci_half                       # whitened displacements
    fam = []
    for k in range(2, Kmax + 1):
        lab = kmeans(Z, k, rng)
        for j in range(k):
            idx = np.where(lab == j)[0]
            if 2 <= len(idx) <= N - 1: fam.append(idx)
    # nested sets along the leading direction of the whitened displacements
    w = np.linalg.eigh(np.cov(Z.T))[1][:, -1]
    order = np.argsort(-np.abs(Z @ w))
    for r in range(2, N // 2 + 1):
        fam.append(order[:r])
    return fam


def family_stats(Dlt, rng, Kmax=4):
    """Centered, whitened displacements -> candidate family with raw statistics T(S)."""
    N, d = Dlt.shape
    Dlt = Dlt - np.median(Dlt, axis=0)       # remove common component
    C = trimmed_cov(Dlt)
    w, V = np.linalg.eigh(C)
    Ci_half = V @ np.diag(1 / np.sqrt(np.maximum(w, 1e-10))) @ V.T
    fam = candidate_family(Dlt, Ci_half, rng, Kmax)
    Z = Dlt @ Ci_half
    sizes = np.array([len(idx) for idx in fam])
    T = np.array([len(idx) * (Z[idx].mean(0) @ Z[idx].mean(0)) for idx in fam])
    return fam, sizes, T


def standardize(sizes, T, mu, sd):
    """Size-adjusted statistic (T - mu_size) / sd_size using per-size null moments."""
    return (T - mu[sizes]) / sd[sizes]


def null_moments(size_list, T_list, N):
    """Per-size null mean and sd of T, pooled over permutations; smoothed by isotonic-free
    fallback: sizes without enough samples borrow the nearest populated size."""
    sizes = np.concatenate(size_list); T = np.concatenate(T_list)
    mu = np.full(N + 1, np.nan); sd = np.full(N + 1, np.nan)
    for s in np.unique(sizes):
        v = T[sizes == s]
        if len(v) >= 5:
            mu[s] = v.mean(); sd[s] = v.std() + 1e-9
    filled = np.where(~np.isnan(mu))[0]
    for s in range(N + 1):
        if np.isnan(mu[s]):
            j = filled[np.argmin(np.abs(filled - s))]
            mu[s], sd[s] = mu[j], sd[j]
    return mu, sd


# ----------------------------------------------------------------------------- one replicate
def run_once(cfg: SimConfig, seed: int, B: int = 100, d: Optional[int] = None, alpha: float = 0.05,
             response_kernel: str = 'linear', q_bw_factor: float = 1.0, B_mom: int = 50):
    rng = np.random.default_rng(seed)
    data = simulate(cfg, rng)
    agents, epochs, U, X, S = data['agents'], data['epochs'], data['U'], data['X'], data['S']
    N = cfg.N; d = cfg.d_latent if d is None else d
    hq = q_bw_factor * median_bandwidth(U, rng)
    KQ = gaussian_kernel(U, hq)
    if response_kernel == 'linear':
        KX = X @ X.T
    else:
        KX = gaussian_kernel(X, median_bandwidth(X, rng))
    np.fill_diagonal(KQ, 0.0)                # U-statistic: exclude j = l pairs from the self terms
    PQX = KQ * KX
    unit = agents + N * epochs               # unit index in [0, 2N)
    psi = embed(KQ, PQX, unit, 2 * N, d)
    fam, sizes, T = family_stats(psi[N:] - psi[:N], rng)

    # permutation null: reassign each agent's pairs across epochs, preserving counts.
    # Two independent sets: B_mom permutations estimate the per-size null moments used to
    # standardize T(S); B further permutations give the null distribution of the standardized max.
    idx_by_agent = [np.where(agents == i)[0] for i in range(N)]
    def permuted_stats():
        ep = epochs.copy()
        for i in range(N):
            ii = idx_by_agent[i]
            ep[ii] = epochs[rng.permutation(ii)]
        psi_b = embed(KQ, PQX, agents + N * ep, 2 * N, d)
        _, s_b, T_b = family_stats(psi_b[N:] - psi_b[:N], rng)
        return s_b, T_b
    mom = [permuted_stats() for _ in range(B_mom)]
    mu, sd = null_moments([a for a, _ in mom], [b_ for _, b_ in mom], N)
    Lam = standardize(sizes, T, mu, sd); k = int(np.argmax(Lam)); Lam_obs = Lam[k]; Shat = fam[k]
    null = np.empty(B)
    for b in range(B):
        s_b, T_b = permuted_stats()
        null[b] = standardize(s_b, T_b, mu, sd).max()
    p = (1 + np.sum(null >= Lam_obs)) / (B + 1)
    reject = p <= alpha
    Sset, Sh = set(S.tolist()), set(Shat.tolist())
    jacc = len(Sset & Sh) / len(Sset | Sh) if (Sset | Sh) else np.nan
    prec = len(Sset & Sh) / len(Sh) if Sh else np.nan
    rec = len(Sset & Sh) / len(Sset) if Sset else np.nan
    # nomination ranking: project centered, whitened displacements onto the direction of Shat's mean
    auc = nomination_auc(psi[N:] - psi[:N], Shat, S)
    out = asdict(cfg); out.update(dict(seed=seed, p=p, reject=int(reject), Lambda=Lam_obs,
                                       jaccard=jacc, precision=prec, recall=rec, Shat_size=len(Sh),
                                       correct=int(reject and jacc >= 0.5), nom_auc=auc,
                                       S2_jaccard=(len(set(data['S2'].tolist()) & Sh) / len(set(data['S2'].tolist()) | Sh)
                                                   if len(data['S2']) else np.nan)))
    return out


def nomination_auc(Dlt, Shat, S):
    """AUC of ranking agents by projection onto the direction of Shat's mean displacement."""
    if len(S) == 0: return np.nan
    N = len(Dlt)
    Dlt = Dlt - np.median(Dlt, axis=0)
    C = trimmed_cov(Dlt); w, V = np.linalg.eigh(C)
    Z = Dlt @ (V @ np.diag(1 / np.sqrt(np.maximum(w, 1e-10))) @ V.T)
    mu = Z[Shat].mean(0); mu /= (np.linalg.norm(mu) + 1e-12)
    score = Z @ mu
    pos = np.zeros(N, bool); pos[S] = True
    sp, sn = score[pos], score[~pos]
    return float((sp[:, None] > sn[None, :]).mean() + 0.5 * (sp[:, None] == sn[None, :]).mean())


# ----------------------------------------------------------------------------- sequential (onset) experiments
# NOTE: onset estimation (P2) is deferred to future work and is not part of the paper's experiments;
# the code below is kept as a starting point.
def simulate_seq(cfg: SimConfig, rng, L: int, t_star: int, ramp: int = 1):
    """L+1 time points (0..L); agents in S shift from t_star on, with effect ramping linearly over
    `ramp` steps (ramp=1: abrupt; larger ramp: slow defection). Returns per-observation arrays."""
    N, dl, pq, px = cfg.N, cfg.d_latent, cfg.p_q, cfg.p_x
    B, _ = np.linalg.qr(rng.normal(size=(px, dl)))
    W = cfg.query_effect * rng.normal(size=(px, pq)) / np.sqrt(pq)
    centers = rng.normal(size=(cfg.K_topics, pq)) * 1.5
    a = rng.normal(size=(N, dl)); pref = rng.dirichlet(np.full(cfg.K_topics, cfg.alpha), size=N)
    S = np.sort(rng.choice(N, size=cfg.S_size, replace=False)) if cfg.delta > 0 else np.array([], int)
    v = rng.normal(size=dl); v /= np.linalg.norm(v)
    agents, epochs, U, X = [], [], [], []
    for tau in range(L + 1):
        a_tau = a + rng.normal(size=dl) * cfg.drift * np.sqrt(tau)          # random-walk common drift
        if tau >= t_star and cfg.delta > 0:
            a_tau[S] += cfg.delta * min(1.0, (tau - t_star + 1) / ramp) * v
        for i in range(N):
            mi = 2 + rng.poisson(max(cfg.m - 2.0, 0.0))
            topics = rng.choice(cfg.K_topics, size=mi, p=pref[i])
            u = centers[topics] + cfg.topic_spread * rng.normal(size=(mi, pq))
            x = a_tau[i] @ B.T + u @ W.T + cfg.sigma_x * rng.normal(size=(mi, px))
            U.append(u); X.append(x); agents.append(np.full(mi, i)); epochs.append(np.full(mi, tau))
    U = np.vstack(U); X = np.vstack(X); agents = np.concatenate(agents); epochs = np.concatenate(epochs)
    U_obs = U + cfg.sigma_q * rng.normal(size=U.shape) if cfg.sigma_q > 0 else U
    return dict(agents=agents, epochs=epochs, U=U_obs, X=X, S=S)


def run_sequential(cfg: SimConfig, seed: int, L: int = 12, t_star: int = 4, ramp: int = 1,
                   B: int = 100, B_mom: int = 50, alpha: float = 0.05, d: Optional[int] = None,
                   q_bw_factor: float = 1.0):
    """Onset estimation: for each tau >= 1 run the two-point scan test on (0, tau); first crossing of
    the Bonferroni level alpha/L is the onset estimate."""
    rng = np.random.default_rng(seed)
    data = simulate_seq(cfg, rng, L, t_star, ramp)
    agents, epochs, U, X, S = data['agents'], data['epochs'], data['U'], data['X'], data['S']
    N = cfg.N; d = cfg.d_latent if d is None else d
    hq = q_bw_factor * median_bandwidth(U, rng)
    pvals = np.ones(L + 1)
    for tau in range(1, L + 1):
        keep = (epochs == 0) | (epochs == tau)
        ag, ep, Uk, Xk = agents[keep], (epochs[keep] == tau).astype(int), U[keep], X[keep]
        KQ = gaussian_kernel(Uk, hq); np.fill_diagonal(KQ, 0.0); PQX = KQ * (Xk @ Xk.T)
        psi = embed(KQ, PQX, ag + N * ep, 2 * N, d)
        fam, sizes, T = family_stats(psi[N:] - psi[:N], rng)
        idx_by_agent = [np.where(ag == i)[0] for i in range(N)]
        def perm_stats():
            e = ep.copy()
            for i in range(N):
                ii = idx_by_agent[i]; e[ii] = ep[rng.permutation(ii)]
            ps = embed(KQ, PQX, ag + N * e, 2 * N, d)
            _, s_b, T_b = family_stats(ps[N:] - ps[:N], rng)
            return s_b, T_b
        mom = [perm_stats() for _ in range(B_mom)]
        mu, sd = null_moments([a_ for a_, _ in mom], [b_ for _, b_ in mom], N)
        Lam = standardize(sizes, T, mu, sd).max()
        null = np.array([standardize(*perm_stats(), mu, sd).max() for _ in range(B)])
        pvals[tau] = (1 + np.sum(null >= Lam)) / (B + 1)
    thr = alpha / L
    cross = np.where(pvals[1:] <= thr)[0]
    t_hat = int(cross[0] + 1) if len(cross) else -1
    out = asdict(cfg); out.update(dict(seed=seed, L=L, t_star=t_star, ramp=ramp, t_hat=t_hat,
                                       detected=int(t_hat >= 0), false_alarm=int(0 <= t_hat < t_star if cfg.delta > 0 else t_hat >= 0),
                                       delay=(t_hat - t_star if t_hat >= t_star else np.nan),
                                       pvals=';'.join(f'{p:.4f}' for p in pvals[1:])))
    return out
