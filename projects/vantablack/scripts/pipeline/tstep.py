"""T-step detection of colluding subnetworks: correlated movement over time.

Design: writing/notes/sim_detection_design.md. Input is per-response data
(agent, step, observed query, response embedding) with equal m per agent-step.

  1. condition  x - m(u): PKPS pooled mode, Gaussian kernel on the k nearest
                other queries (self excluded: every query is answered once)
  2. positions  mean residual per (agent, step), joint linear embedding to d
                (first profile-likelihood elbow by default)
  3. increments split-half: delta(n, t) = psi_A(n, t+1) - psi_B(n, t), with A and B
                disjoint halves of each step's responses, so consecutive increments
                share no estimation noise (overlapping increments are negatively
                autocorrelated, and independent autocorrelated series correlate
                spuriously, which the shuffle null does not reproduce)
  4. factors    remove shared directions above the Marchenko-Pastur noise edge
                whose agent loadings are diffuse (participation-ratio rule)
  5. family     clusters, kNN-graph Louvain communities, and each agent's
                nearest-neighbour balls of sizes 5/10/20/40/80 (up to N/4), and
                local-search refinements of the 50 best balls per size;
                pairs form their own size band, scanned exactly over all pairs
     statistic  per set S: mean pairwise cosine C(S) of agents' residual
                increment series, z_C = C sqrt(d_eff s(s-1)/2); Lambda_b =
                max z_C within size band b
  7. estimate   S_hat: seeded by the best set of the most significant band; members are
                the agents whose similarity to the set passes Benjamini-Hochberg at q
  6. null       per-step agent-ID shuffle of the residual increments (new
                permutation at every step), candidate family rebuilt on every
                draw; p = min_b n_bands * p_b
Comparator nulls: 'block' (shuffle per block of steps), 'rotate' (per-agent
Haar rotation of the whole trajectory), 'cshift' (per-agent circular shift).
"""
import multiprocessing as mp
import numpy as np

from . import candidates as cands

BANDS = np.array([2, 3, 11, 31, 101])   # size-band lower edges; band 0 = pairs


# ----------------------------------------------------------------------------- 1-4: features
def condition(U, X, k=64, bw=1.0, chunk=None):
    """Residuals x - kernel-weighted mean of the responses to the k nearest other queries."""
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    chunk = chunk or max(128, int(1e8 // len(U)))
    Ut = torch.as_tensor(U, device=dev)
    Ut = (Ut - Ut.mean(0)) / Ut.std(0)
    Xt = torch.as_tensor(X, device=dev)
    R = torch.empty_like(Xt)
    nn_d = []
    idx_all, d_all = [], []
    for a in range(0, len(Ut), chunk):
        d = torch.cdist(Ut[a:a + chunk], Ut)
        d[torch.arange(d.shape[0]), torch.arange(a, a + d.shape[0])] = float('inf')
        v, i = d.topk(k, largest=False)
        idx_all.append(i); d_all.append(v); nn_d.append(v[:, 0])
    sigma = bw * torch.cat(nn_d).median()
    for a, (i, v) in zip(range(0, len(Ut), chunk), zip(idx_all, d_all)):
        w = torch.exp(-v ** 2 / (2 * sigma ** 2))
        R[a:a + chunk] = Xt[a:a + chunk] - (w[:, :, None] * Xt[i]).sum(1) / w.sum(1, keepdim=True)
    out = R.cpu().numpy(), float(sigma)
    del Ut, Xt, R, idx_all, d_all, nn_d
    if dev == 'cuda':
        torch.cuda.empty_cache()
    return out


def _unit_means(R, agent, step, N, T):
    Y = np.zeros((N * T, R.shape[1])); np.add.at(Y, agent * T + step, R)
    return Y / np.maximum(np.bincount(agent * T + step, minlength=N * T), 1)[:, None]


def positions(R, agent, step, N, T, d=None):
    """Per-(agent, step) positions in a joint linear embedding (PCA of the N*T unit means).

    Also returns positions from two disjoint halves of each unit's responses (alternate
    responses), projected with the same basis, for split-half increments. d=None picks the
    dimension by the first profile-likelihood elbow."""
    Y = _unit_means(R, agent, step, N, T)
    mu = Y.mean(0)
    _, sv, Vt = np.linalg.svd(Y - mu, full_matrices=False)
    if d is None:
        from graspologic.embed import select_dimension
        d = int(select_dimension(Y - mu, n_components=min(100, Y.shape[1] - 1), n_elbows=1)[0][0])
    V = Vt[:d].T
    half = np.zeros(len(R), int)
    order = np.lexsort((np.arange(len(R)), agent * T + step))
    rank = np.empty(len(R), int); rank[order] = np.arange(len(R))
    first = np.r_[0, np.flatnonzero(np.diff((agent * T + step)[order])) + 1]
    pos_in_unit = rank - first[np.searchsorted(first, rank, side='right') - 1]
    half = pos_in_unit % 2
    proj = lambda Yh: ((Yh - mu) @ V).reshape(N, T, d)
    ZA = proj(_unit_means(R[half == 0], agent[half == 0], step[half == 0], N, T))
    ZB = proj(_unit_means(R[half == 1], agent[half == 1], step[half == 1], N, T))
    return dict(Z=proj(Y), ZA=ZA, ZB=ZB, d=d,
                explained=float((sv[:d] ** 2).sum() / (sv ** 2).sum()))


def increments(pos, mode='split'):
    """'split': psi_A(t+1) - psi_B(t), consecutive increments share no estimation noise;
    'overlap': psi(t+1) - psi(t), negatively autocorrelated by estimation noise;
    'nonoverlap': every other overlapping increment."""
    if mode == 'split':
        return pos['ZA'][:, 1:] - pos['ZB'][:, :-1]
    D = np.diff(pos['Z'], axis=1)
    return D[:, ::2] if mode == 'nonoverlap' else D


def remove_factors(D, min_support=1 / 3, kmax=None):
    """Remove shared factors from the (N, L, d) increments; returns residuals and k.

    Candidates are singular directions of the N x (L d) matrix above the Marchenko-Pastur
    noise edge (noise level re-estimated from the remaining spectrum). Of those, leading
    directions are removed while they are platform-wide: the number of agents whose loading
    clears twice its noise floor (minus the 4.6% expected from noise alone) is at least
    min_support * N. The noise floor of a loading is sqrt((edge / sv)^2 / N): a direction
    far above the edge is estimated precisely, one near it is mostly noise. A colluding
    group loads on its own members only (support |S| / N) and stops the count instead of
    being removed; a factor too weak to be told apart from a group is kept."""
    N, L, d = D.shape
    F = D.reshape(N, L * d)
    Uf, sv, Vt = np.linalg.svd(F, full_matrices=False)
    P = L * d
    edge = np.inf
    if kmax is None:
        kmax = 0
        for _ in range(20):
            sig2 = (sv[kmax:] ** 2).sum() / ((N - kmax) * (P - kmax))
            edge = np.sqrt(1.05 * sig2) * (np.sqrt(N) + np.sqrt(P))
            k_new = int((sv > edge).sum())
            if k_new == kmax:
                break
            kmax = k_new
    else:
        sig2 = (sv[kmax:] ** 2).sum() / ((N - kmax) * (P - kmax))
        edge = np.sqrt(sig2) * (np.sqrt(N) + np.sqrt(P))

    def support(j):
        floor = np.sqrt(min(1.0, (edge / sv[j]) ** 2) / N)
        return ((np.abs(Uf[:, j]) > 2 * floor).sum() - 0.0455 * N) / N

    k = 0
    while k < min(kmax, P // 2) and support(k) >= min_support:
        k += 1
    if k:
        F = F - (F @ Vt[:k].T) @ Vt[:k]
    return F.reshape(N, L, d), k


# ----------------------------------------------------------------------------- 5: statistic, family
def _units(D):
    F = D.reshape(len(D), -1)
    return F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-12)


def zc(Uu, S, d_eff):
    S = np.asarray(S); s = len(S)
    u = Uu[S].sum(0)
    C = ((u @ u) - s) / (s * (s - 1))
    return C * np.sqrt(d_eff * s * (s - 1) / 2), C


def balls(Uu, sizes=(5, 10, 20, 40, 80)):
    """Each agent with its s-1 most similar agents (cosine of increment series)."""
    G = Uu @ Uu.T
    np.fill_diagonal(G, -np.inf)
    sizes = [s for s in sizes if s <= max(3, len(Uu) // 4)]
    smax = max(sizes)
    nb = np.argpartition(-G, smax - 1, axis=1)[:, :smax - 1]
    order = np.argsort(-np.take_along_axis(G, nb, axis=1), axis=1)
    nb = np.take_along_axis(nb, order, axis=1)
    out = []
    for s in sizes:
        for i in range(len(Uu)):
            out.append((frozenset([i, *nb[i, :s - 1].tolist()]), f'ball({s})'))
    return out


def refined(Uu, d_eff, sizes=(5, 10, 20, 40, 80), top=50, rounds=3):
    """Local search from the best nearest-neighbour balls: for each size s, take the `top`
    balls with the largest z_C and repeatedly replace each by the s agents with the largest
    summed similarity to it. A ball that holds part of a colluding group pulls in the rest."""
    G = Uu @ Uu.T
    np.fill_diagonal(G, 0.0)
    sizes = [s for s in sizes if s <= max(3, len(Uu) // 4)]
    nb = np.argsort(-G, axis=1)
    out = []
    for s in sizes:
        cand = [np.r_[i, nb[i, :s - 1]] for i in range(len(Uu))]
        z = np.array([zc(Uu, S, d_eff)[0] for S in cand])
        for i in np.argsort(-z)[:top]:
            S = cand[i]
            for _ in range(rounds):
                S_new = np.argpartition(-G[:, S].sum(1), s - 1)[:s]
                if set(S_new) == set(S):
                    break
                S = S_new
            out.append((frozenset(S.tolist()), f'refined({s})'))
    return out


def family(D, seed=0, d_eff=None):
    F = D.reshape(len(D), -1)
    el = np.ones(len(D), bool)
    Uu = _units(D)
    d_eff = d_eff or F.shape[1]
    return cands.dedupe(cands.clusters(F, el, seed=seed) + cands.comove(F, el, seed=seed)
                        + balls(Uu) + refined(Uu, d_eff))


def band_max(D, d_eff, seed=0):
    """Largest z_C per size band. Band 0 (pairs) is exact: the most similar pair overall."""
    Uu = _units(D)
    out = np.full(len(BANDS), -np.inf)
    best = [None] * len(BANDS)
    G = Uu @ Uu.T
    np.fill_diagonal(G, -np.inf)
    i, j = np.unravel_index(np.argmax(G), G.shape)
    out[0], best[0] = G[i, j] * np.sqrt(d_eff), frozenset([int(i), int(j)])
    for S, _ in family(D, seed, d_eff):
        b = int(np.searchsorted(BANDS, len(S), side='right') - 1)
        if b < 1:
            continue
        z, _ = zc(Uu, sorted(S), d_eff)
        if z > out[b]:
            out[b], best[b] = z, S
    return out, best


# ----------------------------------------------------------------------------- 7: membership estimate
def estimate(D, d_eff, res, q=0.05, rounds=5):
    """Membership estimate S_hat from a scan result.

    Seed: the best set of the size band with the smallest p (ties, common at the p floor,
    broken by the observed maximum standardised by its null distribution).
    Membership: for every agent i, t_i = sqrt(d_eff) <u_i, v_-i>, with u_i its unit
    increment series and v_-i the unit mean direction of the current set without i; an
    agent independent of the set has t_i ~ N(0, 1). S_hat is the set of agents passing
    Benjamini-Hochberg at level q (one-sided), re-tested against the new set until it
    stops changing. Its size is not tied to the family's size grid; it can be empty.
    Also returns the threshold-free agent ranking score: t_i against the seed."""
    from scipy.stats import norm
    obs, nulls, pb, best = res['obs'], res['nulls'], res['p_band'], res['best']
    occ = np.flatnonzero(np.isfinite(obs))
    with np.errstate(invalid='ignore'):
        zstd = (obs[occ] - nulls[:, occ].mean(0)) / np.maximum(nulls[:, occ].std(0), 1e-12)
    b = int(occ[np.lexsort((-np.nan_to_num(zstd, nan=-np.inf), pb[occ]))[0]])
    Uu = _units(D)
    N = len(Uu)
    S = np.array(sorted(best[b]))
    score = None
    for _ in range(rounds):
        if len(S) < 2:
            break
        inS = np.zeros(N, bool); inS[S] = True
        tot = Uu[S].sum(0)
        V = tot[None, :] - inS[:, None] * Uu                       # sum over the set without i
        t = np.sqrt(d_eff) * (Uu * V).sum(1) / np.maximum(np.linalg.norm(V, axis=1), 1e-12)
        if score is None:
            score = t                                               # threshold-free ranking from the seed
        pv = norm.sf(t)
        o = np.argsort(pv)
        ok = pv[o] <= q * np.arange(1, N + 1) / N
        S_new = np.sort(o[:int(np.flatnonzero(ok).max()) + 1]) if ok.any() else np.array([], int)
        if np.array_equal(S_new, S):
            break
        S = S_new
    return S, b, score


# ----------------------------------------------------------------------------- 6: nulls
def null_draw(D, kind, rng, block=2):
    N, L, d = D.shape
    if kind == 'shuffle' or kind == 'block':
        b = 1 if kind == 'shuffle' else block
        out = np.empty_like(D)
        for a in range(0, L, b):
            out[:, a:a + b] = D[rng.permutation(N), a:a + b]
        return out
    if kind == 'rotate':
        G = rng.normal(size=(N, d, d))
        Q, Rr = np.linalg.qr(G)
        Q *= np.sign(np.diagonal(Rr, axis1=1, axis2=2))[:, None, :]
        return np.einsum('nld,nde->nle', D, Q)
    if kind == 'cshift':
        sh = rng.integers(0, L, N)
        idx = (np.arange(L)[None, :] + sh[:, None]) % L
        return D[np.arange(N)[:, None], idx]
    raise ValueError(kind)


_CTX = {}


def _worker(args):
    seed, kind = args
    D, d_eff = _CTX['D'], _CTX['d_eff']
    rng = np.random.default_rng(seed)
    return band_max(null_draw(D, kind, rng), d_eff, seed=seed)[0]


def test(D, d_eff, B=199, kind='shuffle', seed=0, workers=1, known_S=None):
    """Scan test. Returns p (global), per-band observed maxima, null maxima, best sets,
    and, if known_S is given, the known-S reference p-value from the same null kind."""
    obs, best = band_max(D, d_eff, seed=seed)
    _CTX.update(D=D, d_eff=d_eff)
    args = [(seed * 100003 + b, kind) for b in range(B)]
    if workers > 1:
        with mp.get_context('fork').Pool(workers) as pool:
            nulls = np.array(pool.map(_worker, args, chunksize=1))
    else:
        nulls = np.array([_worker(a) for a in args])
    _CTX.clear()
    occ = np.isfinite(obs)
    pb = np.where(occ, (1 + (nulls >= obs[None, :]).sum(0)) / (B + 1), 1.0)
    p = float(min(1.0, occ.sum() * pb[occ].min())) if occ.any() else 1.0
    out = dict(p=p, p_band=pb, obs=obs, best=best, nulls=nulls)
    if known_S is not None and len(known_S) > 1:
        z_obs, C_obs = zc(_units(D), known_S, d_eff)
        rng = np.random.default_rng(seed + 1)
        zn = np.array([zc(_units(null_draw(D, kind, rng)), known_S, d_eff)[0] for _ in range(B)])
        out.update(p_known=float((1 + (zn >= z_obs).sum()) / (B + 1)), C_known=float(C_obs))
    return out


def run(data, d=None, k=64, mode='split', kind='shuffle', B=199, seed=0, workers=1,
        n_factors='auto', gpu_slot=None):
    """gpu_slot: optional lock/semaphore held while conditioning (the only GPU step),
    so many replicate processes can share one GPU."""
    import contextlib
    cfg = data['cfg']
    N, T = cfg['N'], cfg['T']
    with (gpu_slot or contextlib.nullcontext()):
        R, sigma = condition(data['U_obs'], data['X'], k=k)
    pos = positions(R, data['agent'], data['step'], N, T, d=d)
    D = increments(pos, mode=mode)
    D, kf = remove_factors(D) if n_factors == 'auto' else remove_factors(D, kmax=int(n_factors))
    d_eff = D.shape[1] * D.shape[2] - kf
    res = test(D, d_eff, B=B, kind=kind, seed=seed, workers=workers,
               known_S=data['S'] if len(data['S']) else None)
    S_hat, b_hat, score = estimate(D, d_eff, res)
    res.update(n_factors=kf, d_eff=d_eff, d=pos['d'], sigma=sigma, explained=pos['explained'],
               S_hat=S_hat, band_hat=b_hat, rank_score=score)
    return res
