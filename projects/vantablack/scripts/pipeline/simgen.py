"""Synthetic multi-agent system for the detection simulations (fully synthetic).

Design: writing/notes/sim_detection_design.md. One topic (a continuous query
space, like Moltbook's `general`), T time steps, every agent gives exactly m
responses per step, each to its own query (unpaired, R = 1).

Latent perspective of agent n at step t (t = 0..T-1), in R^dl:

    z_n(t) = eta_n + p_n(t),   p_n(0) = 0,   p_n(t) = p_n(t-1) + delta_n(t)
    delta_n(t) = sum_k B_nk sigma_f f_k(t) + sigma_e [sqrt(rho) g(t) 1{n in S}
                                                   + sqrt(1 - rho 1{n in S}) e_n(t)]

  f_k(t), g(t) ~ N(0, I_dl), new each step (platform factors; the group process)
  e_n(t)      own movement, N(0, Sigma_e), optionally AR(1)

Moltbook-matched nuisance and group features (sim_calibrate.py; each at its Moltbook median):
  drift     c(t) ~ N(0, drift^2 I_dl), the same for every agent (a platform-wide shift)
  sparse    K = round(sparse_per_dim (T-1) dl) sparse two-signed factors: agent n loads on factor
            k with probability sparse_support, loading +-|N(0, 1)| with a random sign; the
            factor's pattern f_k(t) ~ N(0, I_dl / (T-1)) over all steps, so the factors' total
            per-step variance does not depend on T: delta_n(t) += sparse_sigma sum_k b_nk f_k(t)
  event     at one step chosen at random, the colluders also make a shared jump
            sigma_e sqrt(event rho) h, h ~ N(0, I_dl) (a joint template switch). It scales with
            rho, so rho = 0 has none
  inactive  each agent-step independently has no responses with this probability
  static    a share of agents post one fixed (query, response) pair, repeated, over static_len
            consecutive steps (templated content): their positions do not move there

Effect size rho (user, 2026-10-18): the correlation between two colluders' own
(factor-free) increments. Every agent moves the same amount per step (sigma_e),
colluders included, so a colluding group differs only in that a share rho of
its members' movement is shared.

Queries: J sub-topic centers c_j ~ N(0, I_q); agent n picks sub-topic j with
probability proportional to exp(pref * <c_j, r_n>) for a random unit r_n (its
preferred region; pref = 0 means everyone draws from the same distribution),
then u = c_j + sigma_w xi. Observed query u_obs = u + sigma_q zeta.

Response to query u, in R^p:

    x = M(u) z_n(t) + W u~ + eps,   M(u) = M0 + s_mod sum_k u~_k M_k

  u~ = u / sqrt(1 + sigma_w^2) (unit scale per coordinate). M0 has orthonormal
  columns; the M_k make the query modulate how the perspective is expressed,
  smoothly in u. W u~ is what the query alone makes everyone say.
"""
from dataclasses import dataclass, asdict, field
import numpy as np


@dataclass
class SimConfig:
    N: int = 2000
    T: int = 8
    m: int = 20
    m_sd: float = 0.0       # 0: every agent gives m responses per step. >0: n(agent, step) =
                            # 1 + Poisson(mu_n), mu_n = (m - 1) exp(m_sd z_n - m_sd^2 / 2), z_n ~ N(0, 1):
                            # unequal, agent-specific activity with at least one response per step
    p: int = 100            # response embedding dimension
    dl: int = 10            # latent perspective dimension
    q: int = 50             # query space dimension
    J: int = 1000           # sub-topic centers (local structure of the query space)
    sigma_w: float = 0.9    # spread of queries around a sub-topic center
    pref: float = 0.0       # agent query preference concentration (0 = identical distributions)
    sigma_q: float = 0.5    # query estimation noise (vantablack), per coordinate
    s_mod: float = 1.0      # strength of the query's modulation of the perspective
    w_query: float = 1.0    # scale of the query-only effect W u~
    sigma_eta: float = 1.0  # stable agent effect
    sigma_e: float = 1.0    # movement per step (every agent); base m = 20 gives ~75% movement, ~25% estimation noise in an increment
    sigma_x: float = 1.0    # response noise
    # colluding group
    S_size: int = 20
    rho: float = 0.0        # effect size: correlation of colluders' increments (0 = H0)
    # nuisance
    r: int = 0              # number of platform factors
    sigma_f: float = 0.0    # factor strength
    load_mean: float = 1.0  # mean agent loading on each factor
    load_sd: float = 0.0    # agent-to-agent spread of loadings (0 = equal loadings)
    niche_size: int = 0     # agents loading on one extra niche factor (0 = none)
    niche_sigma: float = 0.0
    aniso: float = 0.0      # eigenvalue decay exponent of own movement and response noise
    hetero: float = 0.0     # log-sd of agent-specific noise scale
    ar: float = 0.0         # AR(1) coefficient of own movement
    # Moltbook-matched features (module docstring)
    drift: float = 0.0
    sparse_per_dim: float = 0.0
    sparse_support: float = 0.0
    sparse_sigma: float = 0.0
    event: float = 0.0
    inactive: float = 0.0
    static: float = 0.0     # share of agents
    static_len: int = 3
    seed: int = 0


def _aniso_scale(dim, a):
    lam = np.arange(1, dim + 1, dtype=float) ** (-a)
    return np.sqrt(lam * dim / lam.sum())


def generate(cfg: SimConfig):
    rng = np.random.default_rng(cfg.seed)
    N, T, m, p, dl, q = cfg.N, cfg.T, cfg.m, cfg.p, cfg.dl, cfg.q

    # ---- fixed maps
    M0, _ = np.linalg.qr(rng.normal(size=(p, dl)))
    Mk = rng.normal(size=(q, dl, p)) / np.sqrt(p * q)         # sum_k u~_k M_k has O(1) columns
    W = cfg.w_query * rng.normal(size=(q, p)) / np.sqrt(q)
    rot_e = np.linalg.qr(rng.normal(size=(dl, dl)))[0]        # random axes for anisotropy
    rot_x = np.linalg.qr(rng.normal(size=(p, p)))[0]
    sc_e = _aniso_scale(dl, cfg.aniso)
    sc_x = _aniso_scale(p, cfg.aniso)

    # ---- agents and the colluding group
    perm = rng.permutation(N)
    S = np.sort(perm[:cfg.S_size]) if cfg.rho > 0 else np.array([], int)
    inS = np.zeros(N, bool); inS[S] = True
    niche = np.sort(perm[cfg.S_size:cfg.S_size + cfg.niche_size]) if cfg.niche_size else np.array([], int)
    eta = cfg.sigma_eta * rng.normal(size=(N, dl))
    B = cfg.load_mean + cfg.load_sd * rng.normal(size=(N, cfg.r)) if cfg.r else np.zeros((N, 0))
    noise_scale = np.exp(cfg.hetero * rng.normal(size=N)) if cfg.hetero else np.ones(N)

    # ---- latent trajectories
    f = rng.normal(size=(T, cfg.r, dl))
    g = rng.normal(size=(T, dl))
    fn = rng.normal(size=(T, dl))
    K = int(round(cfg.sparse_per_dim * (T - 1) * dl))
    if K:
        Bs = (rng.random((N, K)) < cfg.sparse_support) * rng.choice([-1.0, 1.0], (N, K)) * np.abs(rng.normal(size=(N, K)))
        fs = rng.normal(size=(T, K, dl)) / np.sqrt(max(T - 1, 1))
    c = cfg.drift * rng.normal(size=(T, dl)) if cfg.drift else np.zeros((T, dl))
    if cfg.event:
        t_event, h = (rng.integers(1, T) if T > 1 else 0), rng.normal(size=dl)
    P = np.zeros((N, T, dl))
    e_prev = np.zeros((N, dl))
    for t in range(1, T):
        innov = (rng.normal(size=(N, dl)) * sc_e) @ rot_e.T
        e = cfg.ar * e_prev + np.sqrt(1 - cfg.ar ** 2) * innov
        e_prev = e
        own = np.where(inS, np.sqrt(1 - cfg.rho), 1.0)[:, None]
        d = cfg.sigma_e * own * e * noise_scale[:, None]
        if cfg.r:
            d += cfg.sigma_f * B @ f[t]
        if cfg.niche_size:
            d[niche] += cfg.niche_sigma * fn[t]
        d[inS] += cfg.sigma_e * np.sqrt(cfg.rho) * g[t] * noise_scale[inS, None]
        if K:
            d += cfg.sparse_sigma * Bs @ fs[t]
        d += c[t]
        if cfg.event and t == t_event:
            d[inS] += cfg.sigma_e * np.sqrt(cfg.event * cfg.rho) * h * noise_scale[inS, None]
        P[:, t] = P[:, t - 1] + d
    Zlat = eta[:, None, :] + P                                 # (N, T, dl)

    # ---- queries
    C = rng.normal(size=(cfg.J, q))
    if cfg.pref > 0:
        r = rng.normal(size=(N, q)); r /= np.linalg.norm(r, axis=1, keepdims=True)
        logit = cfg.pref * (r @ C.T)                           # <c_j, r_n> ~ N(0, 1)
        logit -= logit.max(1, keepdims=True)
        Pj = np.exp(logit); Pj /= Pj.sum(1, keepdims=True)
        cum = Pj.cumsum(1)
    if cfg.m_sd > 0:
        mu = (m - 1) * np.exp(cfg.m_sd * rng.normal(size=N) - cfg.m_sd ** 2 / 2)
        counts = 1 + rng.poisson(mu[:, None], size=(N, T))
    else:
        counts = np.full((N, T), m)
    if cfg.inactive:
        counts[rng.random((N, T)) < cfg.inactive] = 0
    static = np.zeros((N, T), bool)                            # agent-steps of templated content
    if cfg.static:
        ns = int(round(cfg.static * N))
        st_ag = rng.choice(np.setdiff1d(np.arange(N), S), ns, replace=False)
        ln = min(cfg.static_len, T)
        st0 = rng.integers(0, T - ln + 1, ns)
        for a_, t0_ in zip(st_ag, st0):
            static[a_, t0_:t0_ + ln] = True
    agent = np.repeat(np.arange(N), counts.sum(1))
    step = np.concatenate([np.repeat(np.arange(T), counts[n]) for n in range(N)])
    n_resp = len(agent)
    if cfg.pref > 0:
        uu = rng.random(n_resp)
        j = np.empty(n_resp, int)
        j = np.searchsorted(cum[0], uu)   # placeholder, overwritten per agent
        for a in range(N):
            sl = agent == a
            j[sl] = np.searchsorted(cum[a], uu[sl])
        j = np.minimum(j, cfg.J - 1)
    else:
        j = rng.integers(0, cfg.J, n_resp)
    U = C[j] + cfg.sigma_w * rng.normal(size=(n_resp, q))
    U_obs = U + cfg.sigma_q * rng.normal(size=U.shape) if cfg.sigma_q > 0 else U

    # ---- responses, in chunks
    X = np.empty((n_resp, p), np.float32)
    ut = U / np.sqrt(1 + cfg.sigma_w ** 2)
    for a in range(0, n_resp, 200_000):
        sl = slice(a, a + 200_000)
        z = Zlat[agent[sl], step[sl]]                          # (n, dl)
        mod = np.einsum('nk,nd,kdp->np', ut[sl], z, Mk, optimize=True)
        eps = (rng.normal(size=(len(z), p)) * sc_x) @ rot_x.T
        X[sl] = (z @ M0.T + cfg.s_mod * mod + ut[sl] @ W
                 + cfg.sigma_x * noise_scale[agent[sl], None] * eps)

    if static.any():
        rows = np.flatnonzero(static[agent, step])
        first = {}
        for i in rows:                                         # the agent's first templated response
            first.setdefault(agent[i], i)
        src = np.array([first[agent[i]] for i in rows])
        U_obs[rows], X[rows] = U_obs[src], X[src]

    return dict(agent=agent, step=step, U_obs=U_obs.astype(np.float32), X=X, S=S, niche=niche,
                P=P, static=static, cfg=asdict(cfg))


def query_diagnostics(U, agent, n_sample=20000, k=64, seed=0):
    """Agent share of query variance and local-overlap ratio (64th-NN / random-pair distance),
    to compare with the Moltbook `general` numbers (0.09 and 0.66)."""
    rng = np.random.default_rng(seed)
    mu = U.mean(0); tot = ((U - mu) ** 2).sum(1).mean()
    n = np.bincount(agent); Ssum = np.zeros((n.size, U.shape[1])); np.add.at(Ssum, agent, U)
    cent = Ssum / n[:, None]
    within = ((U - cent[agent]) ** 2).sum(1).mean()
    share = 1 - within / tot - n.size / len(U) * within / tot      # finite-sample corrected
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    X = torch.as_tensor(U, device=dev)
    idx = rng.choice(len(U), min(n_sample, len(U)), replace=False)
    d = torch.cdist(X[idx], X)
    d[torch.arange(len(idx)), torch.as_tensor(idx, device=dev)] = float('inf')
    knn = d.topk(k, largest=False).values.cpu().numpy()
    rand = torch.cdist(X[idx[:2000]], X[idx[2000:4000]]).cpu().numpy()
    return dict(agent_share=float(share), nn_ratio=float(np.median(knn[:, 0]) / np.median(rand)),
                knn64_ratio=float(np.median(knn[:, -1]) / np.median(rand)))
