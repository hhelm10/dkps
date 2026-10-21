"""T-step detection of colluding subnetworks: correlated movement over time.

Design: writing/notes/sim_detection_design.md. Input is per-response data
(agent, step, observed query, response embedding) with equal m per agent-step.

  1. condition  x - m(u): PKPS pooled mode, Gaussian kernel on the k nearest
                queries, the response's own included (all positions from all the data)
  2. positions  mean residual per (agent, step), joint linear embedding to d
                (first profile-likelihood elbow by default)
  3. increments split-half: delta(n, t) = psi_A(n, t+1) - psi_B(n, t), with A and B
                disjoint halves of each step's responses, so consecutive increments
                share no estimation noise (overlapping increments are negatively
                autocorrelated, and independent autocorrelated series correlate
                spuriously, which the shuffle null does not reproduce); then centered
                per step (each step's mean increment over agents is subtracted)
  4. factors    remove shared directions above the Marchenko-Pastur noise edge
                whose agent loadings are diffuse (participation-ratio rule)
  5. family     clusters (stat='cosine': also kNN-graph Louvain communities), each agent's
                nearest-neighbour balls of sizes 5/10/20/40/80 (up to N/4), and
                local-search refinements of the 50 best balls per size;
                pairs form their own size band, scanned exactly over all pairs
     statistic  stat='rank' (default; collusion is high correlation across many steps):
                at each step, the cosine of two agents' increments is ranked among all
                pairs of agents observed at that step and turned into a normal score.
                A set's score at a step is the sum of its pairs' scores / sqrt(#pairs),
                ~ N(0, 1) under H0. Across steps, the majority statistic: the value the
                set exceeds in a majority of its observed steps (the median for an odd
                number of steps), as a z-score from the binomial law of order statistics.
                One step, however extreme, cannot make a set score high.
                stat='cosine' (earlier): mean pairwise cosine C(S) of agents' whole
                increment series, z_C = C sqrt(d_eff s(s-1)/2), which weights each step
                by the size of its increments.
                Lambda_b = max score within size band b
  7. estimate   stat='rank': from the best set of the most significant band, each agent
                is scored against the set (per-step rank of its mean pair score, summed over
                steps); a two-groups model with an empirical null gives each agent's
                posterior probability of membership; members: posterior > 1/2; repeated
                until the set stops changing. stat='cosine': seeded by that set,
                members are the agents whose similarity to it passes Benjamini-Hochberg
  6. null       per-step agent-ID shuffle of the residual increments (new
                permutation at every step, among the agents observed at that step),
                candidate family rebuilt on every draw; p: the smallest band p-value,
                calibrated by its distribution over the draws (permutation min-p). Under it a
                step's pair ranks are only relabelled, so they are computed once
  8. components (optional, split=True; not part of the method since 2026-10-20) each S_hat
                split into the connected components of its pairs whose score exceeds the
                family-wise 1 - alpha threshold over all pairs, from the same shuffle null.
                The subnetwork is S_hat itself: thresholding pairs discards the graded
                evidence of sets whose pairs co-move only moderately but consistently
Comparator nulls: 'block' (shuffle per block of steps), 'rotate' (per-agent
Haar rotation of the whole trajectory), 'cshift' (per-agent circular shift).
"""
import multiprocessing as mp
import warnings
import numpy as np

from . import candidates as cands

BANDS = np.array([2, 3, 11, 31, 101])   # size-band lower edges; band 0 = pairs


# ----------------------------------------------------------------------------- 1-4: features
def condition(U, X, k=64, bw=1.0, chunk=None):
    """Residuals x - kernel-weighted mean of all responses to the k nearest queries, the
    response's own query included (one reference from all the data, as on Moltbook,
    pipeline/pkps.py); bandwidth bw x the median distance to the nearest other query."""
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
        v, i = d.topk(k, largest=False)                                # includes the response itself
        idx_all.append(i); d_all.append(v); nn_d.append(v[:, 1])
    sigma = bw * torch.cat(nn_d).median()
    for a, (i, v) in zip(range(0, len(Ut), chunk), zip(idx_all, d_all)):
        w = torch.exp(-v ** 2 / (2 * sigma ** 2))
        R[a:a + chunk] = Xt[a:a + chunk] - (w[:, :, None] * Xt[i]).sum(1) / w.sum(1, keepdim=True)
    out = R.cpu().numpy(), float(sigma)
    del Ut, Xt, R, idx_all, d_all, nn_d
    if dev == 'cuda':
        torch.cuda.empty_cache()
    return out


def _unit_means(R, agent, step, N, T, chunk=200_000):
    from scipy import sparse
    unit = agent * T + step
    Y = np.zeros((N * T, R.shape[1]))
    for i in range(0, len(R), chunk):                                  # sum per unit, in float64
        u = unit[i:i + chunk]
        A = sparse.csr_matrix((np.ones(len(u)), (u, np.arange(len(u)))), shape=(N * T, len(u)))
        Y += A @ R[i:i + chunk].astype(np.float64)
    return Y / np.maximum(np.bincount(unit, minlength=N * T), 1)[:, None]


def elbow(sv):
    """First profile-likelihood elbow (Zhu and Ghodsi 2006) of values sorted in decreasing order:
    the split into a leading and a trailing group, each normal with its own mean and a pooled
    variance, of largest likelihood. Returns the size of the leading group (the same rule as
    graspologic's select_dimension with n_elbows=1)."""
    x = np.asarray(sv, float)
    n = len(x)
    if n < 2:
        return n
    i = np.arange(1, n + 1)                                             # size of the leading group
    c1, c2 = np.cumsum(x), np.cumsum(x ** 2)
    ss1 = c2 - c1 ** 2 / i
    n2 = n - i
    with np.errstate(invalid='ignore', divide='ignore'):
        ss2 = np.where(n2 > 0, (c2[-1] - c2) - (c1[-1] - c1) ** 2 / np.maximum(n2, 1), 0.0)
        var = (ss1 + ss2) / (n - 1 - (i < n))
        ll = -n / 2 * np.log(2 * np.pi * var) - (ss1 + ss2) / (2 * var)
    if n == 2:
        ll[0] = -np.inf
    return int(np.argmax(ll)) + 1


def positions(R, agent, step, N, T, d=None, seed=0):
    """Per-(agent, step) positions in a joint linear embedding (PCA of the N*T unit means).

    Also returns positions from two disjoint halves of each unit's responses (alternate
    responses; a unit with one response gives it to a random half), projected with the same
    basis, for split-half increments, and the half sizes nA, nB. d=None picks the dimension by
    the first profile-likelihood elbow."""
    Y = _unit_means(R, agent, step, N, T)
    occ = np.bincount(agent * T + step, minlength=N * T) > 0          # units with no responses stay out of the basis
    mu = Y[occ].mean(0)
    _, sv, Vt = np.linalg.svd(Y[occ] - mu, full_matrices=False)
    if d is None:
        d = elbow(sv[:min(100, Y.shape[1] - 1)])
    V = Vt[:d].T
    unit = agent * T + step
    order = np.lexsort((np.arange(len(R)), unit))
    rank = np.empty(len(R), int); rank[order] = np.arange(len(R))
    first = np.r_[0, np.flatnonzero(np.diff(unit[order])) + 1]
    pos_in_unit = rank - first[np.searchsorted(first, rank, side='right') - 1]
    n_unit = np.bincount(unit, minlength=N * T)
    flip = np.random.default_rng(seed).integers(0, 2, N * T)        # half of a single response
    half = np.where(n_unit[unit] == 1, flip[unit], pos_in_unit % 2)
    proj = lambda Yh: ((Yh - mu) @ V).reshape(N, T, d)
    ZA = proj(_unit_means(R[half == 0], agent[half == 0], step[half == 0], N, T))
    ZB = proj(_unit_means(R[half == 1], agent[half == 1], step[half == 1], N, T))
    nA = np.bincount(unit[half == 0], minlength=N * T).reshape(N, T)
    nB = np.bincount(unit[half == 1], minlength=N * T).reshape(N, T)
    return dict(Z=proj(Y), ZA=ZA, ZB=ZB, nA=nA, nB=nB, d=d,
                explained=float((sv[:d] ** 2).sum() / (sv ** 2).sum()))


def increments(pos, mode='split', per_step=None, center=True, weight=None, min_move=None):
    """'split': psi_A(t+1) - psi_B(t), consecutive increments share no estimation noise;
    'overlap': psi(t+1) - psi(t), negatively autocorrelated by estimation noise;
    'nonoverlap': every other overlapping increment.
    Split increments with an empty half are missing and set to 0. per_step='unit' scales
    every non-missing increment to unit length, so each (agent, step) counts equally however
    much content is behind it (needed when the amount of content varies).
    center: subtract each step's mean increment over the agents observed at that step, so a
    platform-wide drift does not make every pair of agents co-move (the per-step shuffle null
    leaves step means unchanged, so centering is exact under it).
    weight='reliability': scale each split increment by the share of its variance that is
    movement rather than estimation noise. With h = 1/nA(t+1) + 1/nB(t), E||delta||^2 = a + b h
    (a: movement, b: per-response noise), fitted by least squares over the observed increments;
    w = sqrt(a / (a + b h)). Increments resting on one or two responses then count little. The
    weight belongs to the (agent, step) cell and moves with it under the per-step shuffle.
    min_move: split increments whose raw length is below min_move times the median raw length
    are treated as missing. An agent that does not move cannot co-move, and after centering a
    zero increment would become -mu_t, identical for every static agent."""
    if mode == 'split':
        D = pos['ZA'][:, 1:] - pos['ZB'][:, :-1]
        ok = (pos['nA'][:, 1:] > 0) & (pos['nB'][:, :-1] > 0)
        if min_move is not None:
            r = np.linalg.norm(D, axis=2)
            ok &= r >= min_move * np.median(r[ok])
            pos['n_static'] = int(((pos['nA'][:, 1:] > 0) & (pos['nB'][:, :-1] > 0) & ~ok).sum())
        D = D * ok[:, :, None]
        if weight == 'reliability':
            h = 1 / np.maximum(pos['nA'][:, 1:], 1) + 1 / np.maximum(pos['nB'][:, :-1], 1)
            mu0 = D.sum(0) / np.maximum(ok.sum(0), 1)[:, None]
            y = (((D - mu0[None]) ** 2).sum(2))[ok]
            a_, b_ = np.linalg.lstsq(np.c_[np.ones(ok.sum()), h[ok]], y, rcond=None)[0]
            a_, b_ = max(a_, 1e-12), max(b_, 0.0)
            D = D * np.sqrt(a_ / (a_ + b_ * h))[:, :, None]
            pos['reliability_fit'] = (float(a_), float(b_))
    else:
        D = np.diff(pos['Z'], axis=1)
        D = D[:, ::2] if mode == 'nonoverlap' else D
        ok = np.ones(D.shape[:2], bool)
    if center:
        mu = D.sum(0) / np.maximum(ok.sum(0), 1)[:, None]
        D = (D - mu[None]) * ok[:, :, None]
    if per_step == 'unit':
        D = D / np.maximum(np.linalg.norm(D, axis=2, keepdims=True), 1e-12)
    return D


def remove_factors(D, min_support=1 / 3, kmax=None, two_signed_alpha=None):
    """Remove shared factors from the (N, L, d) increments; returns residuals and k.

    Candidates are singular directions of the N x (L d) matrix above the Marchenko-Pastur
    noise edge (noise level re-estimated from the remaining spectrum). Of those, leading
    directions are removed while they are platform-wide: the number of agents whose loading
    clears twice its noise floor (minus the 4.6% expected from noise alone) is at least
    min_support * N. The noise floor of a loading is sqrt((edge / sv)^2 / N): a direction
    far above the edge is estimated precisely, one near it is mostly noise. A colluding
    group loads on its own members only (support |S| / N) and stops the count instead of
    being removed; a factor too weak to be told apart from a group is kept.
    two_signed_alpha: also remove sparse two-signed directions (writing/notes/sim_detection_design.md
    section 1b: collusion is one-signed). For a group direction the minority side of the clear
    loadings holds noise only, ~ Binomial(N, 0.0228); a direction whose minority count exceeds that
    binomial's 1 - alpha quantile is a factor."""
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

    if two_signed_alpha is not None:
        from scipy.stats import binom
        cmax = binom.ppf(1 - two_signed_alpha, N, 0.02275)

    def two_signed(j):
        if two_signed_alpha is None:
            return False
        floor = np.sqrt(min(1.0, (edge / sv[j]) ** 2) / N)
        return min((Uf[:, j] > 2 * floor).sum(), (Uf[:, j] < -2 * floor).sum()) > cmax

    k = 0
    while k < min(kmax, P // 2) and (support(k) >= min_support or two_signed(k)):
        k += 1
    out = D
    if k:
        # missing increments (zero) stay missing: the projection would otherwise fill them with
        # the agent's factor residue, which carries no movement but counts as a step
        out = (F - (F @ Vt[:k].T) @ Vt[:k]).reshape(N, L, d) * (np.linalg.norm(D, axis=2) > 0)[:, :, None]
    return out, k


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
    smax = max(sizes)
    Gd = G.copy(); np.fill_diagonal(Gd, -np.inf)                      # self first, then neighbours
    nb = np.argpartition(-Gd, smax - 1, axis=1)[:, :smax - 1]
    nb = np.take_along_axis(nb, np.argsort(-np.take_along_axis(Gd, nb, axis=1), axis=1), axis=1)
    out = []
    for s in sizes:
        cand = np.c_[np.arange(len(Uu)), nb[:, :s - 1]]
        z = np.empty(len(Uu))
        for a in range(0, len(Uu), 256):                              # z_C of every ball at once
            u = Uu[cand[a:a + 256]].sum(1)
            z[a:a + 256] = ((u * u).sum(1) - s) / (s * (s - 1)) * np.sqrt(d_eff * s * (s - 1) / 2)
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
    el = np.linalg.norm(F, axis=1) > 0          # an agent whose increments are all missing has no series
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
    fam = [S for S, _ in family(D, seed, d_eff) if len(S) >= BANDS[1]]
    if fam:                                                          # z_C of every set at once
        from scipy import sparse
        sz = np.array([len(S) for S in fam])
        A = sparse.csr_matrix((np.ones(sz.sum()), np.concatenate([sorted(S) for S in fam]),
                               np.r_[0, np.cumsum(sz)]), shape=(len(fam), len(Uu)))
        u = A @ Uu
        z = ((u * u).sum(1) - sz) / (sz * (sz - 1)) * np.sqrt(d_eff * sz * (sz - 1) / 2)
        band = np.searchsorted(BANDS, sz, side='right') - 1
        for b in np.unique(band):
            i = np.flatnonzero(band == b)
            j = i[np.argmax(z[i])]
            if z[j] > out[b]:
                out[b], best[b] = z[j], fam[j]
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
    if kind == 'shuffle':
        # among the agents observed at the step: a missing increment stays missing, so each
        # agent keeps its own pattern of missing steps (identical to a plain shuffle when none
        # are missing). Shuffling missing increments in as well broke the pairing of the two
        # increments an inactive step removes, and inflated the pair band under H0.
        out = np.zeros_like(D)
        obs = np.linalg.norm(D, axis=2) > 0
        for a in range(L):
            o = np.flatnonzero(obs[:, a])
            out[o, a] = D[rng.permutation(o), a]
        return out
    if kind == 'block':
        out = np.empty_like(D)
        for a in range(0, L, block):
            out[:, a:a + block] = D[rng.permutation(N), a:a + block]
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
    if _CTX.get('Q') is not None:
        return band_max_rank(*shuffle_rank(D, _CTX['Q'], rng), seed=seed)[0]
    return band_max(null_draw(D, kind, rng), d_eff, seed=seed)[0]


def minp_global(obs, nulls, occ):
    """Global p-value of the scan: the smallest band p-value, calibrated by its own distribution
    under the null (permutation min-p, Westfall and Young 1993). Each shuffle draw's band
    p-values are computed against the other draws and the observed data (all B + 1 values are
    exchangeable under H0), its minimum over bands taken, and p is the share of the B + 1 minima
    at most the observed one. Exact under the shuffle null, and it accounts for the correlation
    between bands (a Bonferroni factor over bands is conservative: about 7% of null replicates
    had p <= 0.10)."""
    if not occ.any():
        return 1.0
    B = len(nulls)
    X = np.vstack([obs[None, occ], nulls[:, occ]])                     # (B + 1, bands): observed first
    X = np.where(np.isfinite(X), X, -np.inf)
    # p of each value within its band among all B + 1: share of values >= it
    ge = (X[None, :, :] >= X[:, None, :]).sum(1)                       # (B + 1, bands)
    mp = (ge / (B + 1)).min(1)
    return float((mp <= mp[0]).sum() / (B + 1))


def test(D, d_eff, B=199, kind='shuffle', seed=0, workers=1, known_S=None, Q=None):
    """Scan test. Returns p (global), per-band observed maxima, null maxima, best sets,
    and, if known_S is given, the known-S reference p-value from the same null kind.
    Q (step_scores(D)): use the rank statistic, under the per-step shuffle."""
    if Q is not None and kind != 'shuffle':
        raise ValueError('the rank statistic is tested under the per-step shuffle')
    obs, best = band_max(D, d_eff, seed=seed) if Q is None else band_max_rank(D, Q, seed=seed)
    _CTX.update(D=D, d_eff=d_eff, Q=Q)
    args = [(seed * 100003 + b, kind) for b in range(B)]
    if workers > 1:
        with mp.get_context('fork').Pool(workers) as pool:
            nulls = np.array(pool.map(_worker, args, chunksize=1))
    else:
        nulls = np.array([_worker(a) for a in args])
    _CTX.clear()
    occ = np.isfinite(obs)
    pb = np.where(occ, (1 + (nulls >= obs[None, :]).sum(0)) / (B + 1), 1.0)
    p = minp_global(obs, nulls, occ)
    out = dict(p=p, p_band=pb, obs=obs, best=best, nulls=nulls)
    if known_S is not None and Q is not None and isinstance(known_S, list):
        # several known groups: one known-S p-value each, from the same shuffle draws
        Gs = [np.asarray(G) for G in known_S if len(G) > 1]
        if Gs:
            z_obs = set_scores(Q, Gs)
            rng = np.random.default_rng(seed + 1)
            zn = np.array([set_scores(shuffle_rank(D, Q, rng)[1], Gs) for _ in range(B)])
            out.update(p_known=((1 + (zn >= z_obs[None]).sum(0)) / (B + 1)).tolist())
    elif known_S is not None and len(known_S) > 1 and Q is not None:
        z_obs = set_scores(Q, [known_S])[0]
        rng = np.random.default_rng(seed + 1)
        zn = np.array([set_scores(shuffle_rank(D, Q, rng)[1], [known_S])[0] for _ in range(B)])
        out.update(p_known=float((1 + (zn >= z_obs).sum()) / (B + 1)), C_known=float(z_obs))
    elif known_S is not None and len(known_S) > 1:
        z_obs, C_obs = zc(_units(D), known_S, d_eff)
        rng = np.random.default_rng(seed + 1)
        zn = np.array([zc(_units(null_draw(D, kind, rng)), known_S, d_eff)[0] for _ in range(B)])
        out.update(p_known=float((1 + (zn >= z_obs).sum()) / (B + 1)), C_known=float(C_obs))
    return out


def run(data, d=None, k=64, mode='split', kind='shuffle', B=199, seed=0, workers=1,
        n_factors='auto', gpu_slot=None, per_step=None, center=True, two_signed_alpha=None,
        weight=None, min_move=None, split=False, stat='rank', peel=0):
    """gpu_slot: optional lock/semaphore held while on the GPU (conditioning, and the pair
    threshold when split=True), so many replicate processes can share one GPU."""
    import contextlib
    cfg = data['cfg']
    with (gpu_slot or contextlib.nullcontext()):
        R, sigma = condition(data['U_obs'], data['X'], k=k)
    res = run_residuals(R, data['agent'], data['step'], cfg['N'], cfg['T'], d=d, mode=mode,
                        kind=kind, B=B, seed=seed, workers=workers, n_factors=n_factors,
                        per_step=per_step, center=center,
                        known_S=(list(data['groups']) if len(data.get('groups', [])) > 1
                                 else (data['S'] if len(data['S']) else None)), peel=peel,
                        two_signed_alpha=two_signed_alpha, weight=weight, min_move=min_move,
                        split=split, gpu_slot=gpu_slot, stat=stat)
    res.update(sigma=sigma)
    return res


def run_residuals(R, agent, step, N, T, d=None, mode='split', kind='shuffle', B=199, seed=0,
                  workers=1, n_factors='auto', per_step=None, center=True, known_S=None, peel=0,
                  two_signed_alpha=None, weight=None, min_move=None, split=False, gpu_slot=None,
                  stat='rank'):
    """Steps 2-7 from conditioned responses R (one row per response; agent in [0, N), step in
    [0, T)). Returns the scan result plus the factor count, dimensions, membership estimate
    and the agent ranking score; peel > 0 adds up to that many peeled groups (peel_groups).
    split=True adds step 8: S_hat's linked-pair components ('components'; with peel, also per
    peeled group and over the whole cohort). stat: 'rank' or 'cosine' (module docstring)."""
    import contextlib
    pos = positions(R, agent, step, N, T, d=d, seed=seed)
    D = increments(pos, mode=mode, per_step=per_step, center=center, weight=weight, min_move=min_move)
    D, kf = remove_factors(D, two_signed_alpha=two_signed_alpha) if n_factors == 'auto' \
        else remove_factors(D, kmax=int(n_factors))
    d_eff = D.shape[1] * D.shape[2] - kf
    Q = step_scores(D) if stat == 'rank' else None
    res = test(D, d_eff, B=B, kind=kind, seed=seed, workers=workers, known_S=known_S, Q=Q)
    S_hat, b_hat, score = estimate(D, d_eff, res) if Q is None else estimate_rank(Q, res)
    res.update(n_factors=kf, d_eff=d_eff, d=pos['d'], explained=pos['explained'],
               S_hat=S_hat, band_hat=b_hat, rank_score=score, reliability_fit=pos.get('reliability_fit'), n_static=pos.get('n_static'),
               n_missing=int((np.linalg.norm(D, axis=2) == 0).sum()))
    if peel:
        res['groups'] = peel_groups(D, d_eff, first=res, max_groups=peel, B=B, kind=kind, seed=seed,
                                    workers=workers, stat=stat)
    if split and Q is not None:
        Z = pair_scores(Q)
        res['pair_thr'] = thr = pair_threshold_rank(D, Q, seed=seed, workers=workers)
    elif split:
        with (gpu_slot or contextlib.nullcontext()):
            Z = loso_z(D, d_eff)
            res['pair_thr'] = thr = pair_threshold(D, d_eff, seed=seed)
    if split:
        res['components'] = components(Z, S_hat, thr)
        for g in res.get('groups', []):
            g['components'] = components(Z, g['members'], thr)
        if peel:
            res['cohort_components'] = components(Z, np.arange(len(D)), thr)
    return res


def loso_z(D, d_eff, rows=None):
    """Leave-one-step-out pair z: for each pair, the minimum over left-out steps s of
    sqrt(d_eff (L-1)/L) times the cosine of the two agents' increments without step s.
    Large only for pairs that co-move at more than one step. Returns an (n, n) array for
    D[rows] (all agents by default), diagonal -inf."""
    X = D if rows is None else D[rows]
    L = X.shape[1]
    try:
        import torch
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        X = torch.as_tensor(np.ascontiguousarray(X), device=dev, dtype=torch.float32)
        P = torch.einsum('itd,jtd->tij', X, X); nr = (X ** 2).sum(2)
        sq, mn, inf = torch.sqrt, torch.minimum, float('inf')
    except ImportError:
        P = np.einsum('itd,jtd->tij', X, X); nr = (X ** 2).sum(2)
        sq, mn, inf = np.sqrt, np.minimum, np.inf
    Ps, ns = P.sum(0), nr.sum(1)
    out = None
    for t in range(L):
        a = ns - nr[:, t]
        z = np.sqrt(d_eff * (L - 1) / L) * (Ps - P[t]) / sq((a[:, None] * a[None]).clip(min=1e-24))
        out = z if out is None else mn(out, z)
    out = out.cpu().numpy() if hasattr(out, 'cpu') else out
    np.fill_diagonal(out, -np.inf)
    return out


def pair_threshold(D, d_eff, B=99, alpha=0.05, seed=0):
    """Family-wise threshold for loso_z over all pairs: the 1 - alpha quantile of its maximum
    over pairs under the per-step agent-ID shuffle."""
    rng = np.random.default_rng(seed)
    mx = [loso_z(null_draw(D, 'shuffle', rng), d_eff).max() for _ in range(B)]
    return float(np.quantile(mx, 1 - alpha))


def components(Z, members, thr):
    """Connected components (size >= 2) of members under links Z > thr, largest first."""
    from scipy.sparse.csgraph import connected_components
    members = np.asarray(members)
    if len(members) < 2:
        return []
    _, lab = connected_components(Z[np.ix_(members, members)] > thr, directed=False)
    out = [members[lab == c] for c in np.unique(lab) if (lab == c).sum() > 1]
    return sorted(out, key=len, reverse=True)


def peel_groups(D, d_eff, first=None, alpha=0.05, max_groups=10, B=199, kind='shuffle', seed=0, workers=1,
                stat='cosine'):
    """Several groups by peeling: test, estimate S_hat, set its members aside, and test the
    remaining agents again; stop at the first p > alpha, an empty S_hat, or fewer than 10
    agents left. A one-agent S_hat is not a subnetwork: it is set aside and peeling continues
    (the round is kept, with subnetwork=False). Each round is a level-alpha test of 'no colluding group among the agents
    not yet set aside'. Returns one dict per round (indices into the full agent set), with
    each member's membership score against the round's S_hat ('member_score', aligned with
    'members'). With stat='rank', each round ranks pairs among the agents it tests."""
    keep = np.arange(len(D))
    out = []
    for g in range(max_groups):
        Q = step_scores(D[keep]) if stat == 'rank' else None
        if g == 0 and first is not None:
            res = first
            S, b, score = first['S_hat'], first['band_hat'], first['rank_score']
        else:
            res = test(D[keep], d_eff, B=B, kind=kind, seed=seed + g, workers=workers, Q=Q)
            S, b, score = estimate(D[keep], d_eff, res) if Q is None else estimate_rank(Q, res)
        if Q is not None and len(S) >= 2:
            score = member_z(Q, S, combine='sum')[0]                      # against the final S_hat
        out.append(dict(p=res['p'], p_band=np.asarray(res['p_band']), obs=np.asarray(res['obs']),
                        band=b, members=keep[S], n_tested=len(keep), subnetwork=len(S) >= 2,
                        member_score=np.asarray(score)[S] if score is not None and len(S) else np.array([]),
                        best=[None if x is None else keep[sorted(x)] for x in res['best']]))
        if res['p'] > alpha or len(S) == 0 or len(keep) - len(S) < 10:
            break
        keep = np.delete(keep, S)
    return out


# ----------------------------------------------------------------------------- rank statistic (stat='rank')
def step_scores(D):
    """(L, N, N) per-step pair scores, float32: at each step, the cosine of two agents'
    increments ranked among all pairs of agents observed at that step, as the normal score
    Phi^-1((rank + 1/2) / n_pairs). NaN where either agent is missing at the step, and on the
    diagonal. The per-step shuffle only relabels a step's scores, so they are computed once."""
    from scipy.special import ndtri
    N, L, _ = D.shape
    Q = np.full((L, N, N), np.nan, np.float32)
    for t in range(L):
        nr = np.linalg.norm(D[:, t], axis=1)
        o = np.flatnonzero(nr > 0)
        if len(o) < 2:
            continue
        X = D[o, t] / nr[o, None]
        i, j = np.triu_indices(len(o), 1)
        g = (X @ X.T)[i, j]
        r = np.empty(len(g)); r[np.argsort(g, kind='stable')] = np.arange(len(g))
        q = ndtri((r + 0.5) / len(g)).astype(np.float32)
        Qt = np.full((len(o), len(o)), np.nan, np.float32)
        Qt[i, j] = q; Qt[j, i] = q
        Q[t][np.ix_(o, o)] = Qt
    return Q


def majority_z(V):
    """Majority statistic over steps. V: (L, ...) per-step scores, ~ N(0, 1) and independent
    across steps under H0, NaN where missing. With k observed steps and m = floor(k/2) + 1,
    M is the m-th largest value, the level exceeded in a majority of the steps (the median
    for odd k). Returns z = Phi^-1(1 - p) and p = P(at least m of k N(0, 1) draws >= M), the
    order-statistic tail, so z is comparable across k. z = -inf, p = 1 when k < 2."""
    from scipy.special import gammaln, log_ndtr, ndtri_exp
    L = V.shape[0]
    ok = ~np.isnan(V)
    k = ok.sum(0)
    m = k // 2 + 1
    Vs = np.sort(np.where(ok, V, -np.inf), axis=0)                      # observed values last
    M = np.take_along_axis(Vs, np.clip(L - m, 0, L - 1)[None], 0)[0].astype(np.float64)
    lq, l1q = log_ndtr(-M), log_ndtr(M)                                 # log q, log(1 - q): no underflow
    terms = []
    for j in range(1, L + 1):
        use = (j >= m) & (j <= k)
        with np.errstate(invalid='ignore'):
            t = (gammaln(k + 1) - gammaln(j + 1) - gammaln(np.maximum(k - j, 0) + 1)
                 + j * lq + np.maximum(k - j, 0) * l1q)
        terms.append(np.where(use, t, -np.inf))
    T_ = np.stack(terms)
    mx = T_.max(0)
    with np.errstate(invalid='ignore', over='ignore', divide='ignore'):
        logp = np.where(np.isfinite(mx), mx + np.log(np.exp(T_ - np.where(np.isfinite(mx), mx, 0)).sum(0)), 0.0)
    logp = np.minimum(logp, 0.0)
    z = -ndtri_exp(logp)
    p = np.exp(logp)
    short = k < 2
    z[short], p[short] = -np.inf, 1.0
    return z, p


def pair_scores(Q):
    """(N, N) majority z of every pair, diagonal -inf."""
    L, N, _ = Q.shape
    i, j = np.triu_indices(N, 1)
    z = majority_z(Q[:, i, j])[0].astype(np.float32)
    Z = np.full((N, N), -np.inf, np.float32)
    Z[i, j] = z; Z[j, i] = z
    return Z


def _set_steps(sub):
    """Per-step set scores from pair-score blocks sub (L, n_sets, s, s): sum over pairs /
    sqrt(#pairs), NaN at steps where fewer than two members are observed."""
    ok = ~np.isnan(sub)
    n = ok.sum((2, 3)) / 2
    tot = np.where(ok, sub, 0).sum((2, 3)) / 2
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(n > 0, tot / np.sqrt(n), np.nan)


def set_scores(Q, sets, chunk=128):
    """Majority z of each set (index arrays)."""
    out = np.empty(len(sets))
    sizes = np.array([len(S) for S in sets])
    for s in np.unique(sizes):
        ix = np.flatnonzero(sizes == s)
        I = np.array([np.sort(np.fromiter(sets[i], int)) for i in ix])
        for a in range(0, len(ix), chunk):
            c = I[a:a + chunk]
            out[ix[a:a + chunk]] = majority_z(_set_steps(Q[:, c[:, :, None], c[:, None, :]]))[0]
    return out


def balls_rank(Q, Zp, sizes=(5, 10, 20, 40, 80), top=50, rounds=3, chunk=64):
    """Nearest-neighbour balls by pair score, all scored at once from their nested prefixes,
    and local-search refinements of the `top` best balls per size (each set replaced by the s
    agents with the largest summed pair score to it, until it stops changing; all `top` sets of
    a size at once). Returns [(frozenset, label)] and the balls' scores."""
    N, L = len(Zp), Q.shape[0]
    sizes = [s for s in sizes if s <= max(3, N // 4)]
    smax = max(sizes)
    nb = np.argpartition(-Zp, smax - 1, axis=1)[:, :smax - 1]
    nb = np.take_along_axis(nb, np.argsort(-np.take_along_axis(Zp, nb, axis=1), axis=1), axis=1)
    X = np.c_[np.arange(N), nb]                                         # self, then neighbours
    r_, c_ = np.tril_indices(smax, -1)                                  # pairs ordered by the later member,
    o_ = np.argsort(r_, kind='stable'); r_, c_ = r_[o_], c_[o_]          # so prefix s holds the first s(s-1)/2
    ks = np.array([s * (s - 1) // 2 for s in sizes])
    Qf = Q.reshape(L, N * N)
    V = np.empty((len(sizes), L, N))
    for a in range(0, N, chunk):
        c = X[a:a + chunk]
        g = np.take(Qf, (c[:, r_] * N + c[:, c_]).ravel(), axis=1).reshape(L, len(c), -1)
        ok = ~np.isnan(g)
        tot = np.where(ok, g, 0).cumsum(2, dtype=np.float64)[:, :, ks - 1]   # (L, n, sizes)
        cnt = ok.cumsum(2)[:, :, ks - 1]
        with np.errstate(invalid='ignore', divide='ignore'):
            V[:, :, a:a + chunk] = np.where(cnt > 0, tot / np.sqrt(cnt), np.nan).transpose(2, 0, 1)
    sc = majority_z(V.transpose(1, 0, 2))[0]                           # (sizes, N)
    out = [(frozenset(X[i, :s].tolist()), f'ball({s})') for bi, s in enumerate(sizes) for i in range(N)]
    G = np.where(np.isfinite(Zp), Zp, 0.0)                              # symmetric
    ref = []
    for bi, s in enumerate(sizes):
        S = X[np.argsort(-sc[bi])[:top], :s]
        for _ in range(rounds):
            Sn = np.argpartition(-G[S].sum(1), s - 1, axis=1)[:, :s]
            if (np.sort(Sn, 1) == np.sort(S, 1)).all():
                break
            S = Sn
        ref += [(frozenset(x.tolist()), f'refined({s})') for x in S]
    return out, sc.ravel(), ref


def _step_units(D):
    """Increments scaled to unit length at every observed step, flattened (each step counts
    equally); features for the cluster and community candidates."""
    nr = np.linalg.norm(D, axis=2, keepdims=True)
    return (D / np.maximum(nr, 1e-12)).reshape(len(D), -1)


def band_max_rank(D, Q, seed=0):
    """Largest majority z per size band (rank statistic). Band 0 (pairs) is exact."""
    Zp = pair_scores(Q)
    out = np.full(len(BANDS), -np.inf)
    best = [None] * len(BANDS)
    i, j = np.unravel_index(np.argmax(Zp), Zp.shape)
    if np.isfinite(Zp[i, j]):
        out[0], best[0] = Zp[i, j], frozenset([int(i), int(j)])
    F = _step_units(D)
    el = np.linalg.norm(F, axis=1) > 0
    bl, bsc, ref = balls_rank(Q, Zp)
    rest = cands.dedupe(cands.clusters(F, el, seed=seed) + ref)
    fam = [S for S, _ in bl] + [S for S, _ in rest if len(S) >= BANDS[1]]
    z = np.r_[bsc, set_scores(Q, [S for S, _ in rest if len(S) >= BANDS[1]])] if rest else bsc
    sz = np.array([len(S) for S in fam])
    band = np.searchsorted(BANDS, sz, side='right') - 1
    for b in np.unique(band):
        ii = np.flatnonzero(band == b)
        jj = ii[np.argmax(z[ii])]
        if z[jj] > out[b]:
            out[b], best[b] = z[jj], fam[jj]
    return out, best


def shuffle_rank(D, Q, rng):
    """One per-step shuffle draw, applied to the increments and their pair scores alike
    (among the agents observed at each step, as null_draw(D, 'shuffle'))."""
    N, L, _ = D.shape
    Dn = np.zeros_like(D)
    Qn = np.full_like(Q, np.nan)
    obs = np.linalg.norm(D, axis=2) > 0
    for a in range(L):
        o = np.flatnonzero(obs[:, a])
        p = rng.permutation(o)
        Dn[o, a] = D[p, a]
        Qn[a][np.ix_(o, o)] = Q[a][np.ix_(p, p)]
    return Dn, Qn


def member_z(Q, S, combine='majority'):
    """Each agent against the set S. Per step, the mean of its pair scores with the observed
    members other than itself, ranked among all agents observed at that step and turned into a
    normal score (the members of a group are correlated with each other, so the mean's spread
    for an agent unrelated to S depends on S; its rank among agents does not). Combined over
    steps by the majority statistic (z and p), or combine='sum': Stouffer's sum / sqrt(#steps)
    (z only; -inf with fewer than two steps). A step's normal score is at most
    Phi^-1(1 - 1/(2n)), so one step cannot dominate the sum."""
    from scipy.special import ndtri
    sub = Q[:, :, np.asarray(S)]
    ok = ~np.isnan(sub)
    n = ok.sum(2)
    with np.errstate(invalid='ignore', divide='ignore'):
        V = np.where(n > 0, np.where(ok, sub, 0).sum(2) / n, np.nan)
    for t in range(len(V)):
        o = np.flatnonzero(~np.isnan(V[t]))
        if len(o):
            r = np.empty(len(o)); r[np.argsort(V[t, o], kind='stable')] = np.arange(len(o))
            V[t, o] = ndtri((r + 0.5) / len(o))
    if combine == 'majority':
        return majority_z(V)
    k = np.isfinite(V).sum(0)
    z = np.nansum(V, 0) / np.sqrt(np.maximum(k, 1))
    z[k < 2] = -np.inf
    return z, None


def two_groups(z, iters=500):
    """Two-groups model (Efron 2004, empirical null): z ~ (1 - pi) N(mu0, s^2) + pi N(mu1, s^2),
    mu1 > mu0, fitted by EM. The null component is estimated, not fixed at N(0, 1): members take
    the top ranks at every step, which pushes the other agents' scores down. The two components
    share one variance, so the posterior probability of membership increases with z (with
    separate variances a wider member component also claims the far left tail). Returns each
    agent's posterior probability of membership (0 where z is not finite)."""
    from scipy.stats import norm
    z = np.asarray(z, float)
    ok = np.isfinite(z)
    x = z[ok]
    w_out = np.zeros(len(z))
    if len(x) < 10:
        return w_out
    mu0, sd = np.median(x), max(np.std(x), 1e-6)
    pi, mu1 = 0.05, max(np.quantile(x, 0.98), mu0 + sd)
    for _ in range(iters):
        f1 = pi * norm.pdf(x, mu1, sd)
        f0 = (1 - pi) * norm.pdf(x, mu0, sd)
        w = f1 / np.maximum(f0 + f1, 1e-300)
        pi_n = w.mean()
        mu1_n = (w * x).sum() / max(w.sum(), 1e-12)
        mu0 = ((1 - w) * x).sum() / max((1 - w).sum(), 1e-12)
        mu1_n = max(mu1_n, mu0 + 1e-3)
        sd = np.sqrt(max((w * (x - mu1_n) ** 2 + (1 - w) * (x - mu0) ** 2).mean(), 1e-6))
        done = abs(pi_n - pi) < 1e-8 and abs(mu1_n - mu1) < 1e-8
        pi, mu1 = pi_n, mu1_n
        if done:
            break
    w_out[ok] = w
    return w_out


def membership(Q, S):
    """One membership pass against the set S: each agent's score against S (member_z,
    combine='sum'), the two-groups model, members at posterior > 1/2. With the true group as S
    this is the oracle membership estimate (the simulations' known-S reference)."""
    return np.flatnonzero(two_groups(member_z(Q, S, combine='sum')[0]) > 0.5)


def estimate_rank(Q, res, rounds=10):
    """estimate() for the rank statistic (decisions.md, 2026-10-20). Seeded by the best set of
    the most significant band (the set the test's p-value is about). Each agent is scored
    against the set (member_z, combine='sum': every step counts), a two-groups model is fitted
    to the scores, and the members are the agents whose posterior probability of membership
    exceeds 1/2 (the Bayes rule); repeated against the new set until it stops changing. Members
    are graded by the same score."""
    obs, nulls, pb, best = res['obs'], res['nulls'], res['p_band'], res['best']
    occ = np.flatnonzero(np.isfinite(obs))
    if not len(occ):                                                    # e.g. one step: nothing can co-move repeatedly
        return np.array([], int), None, None
    with np.errstate(invalid='ignore'):
        zstd = (obs[occ] - nulls[:, occ].mean(0)) / np.maximum(nulls[:, occ].std(0), 1e-12)
    b = int(occ[np.lexsort((-np.nan_to_num(zstd, nan=-np.inf), pb[occ]))[0]])
    S = np.array(sorted(best[b]))
    if len(S) < 2:
        return S, b, None
    for _ in range(rounds):
        S_new = membership(Q, S)
        if len(S_new) < 2 or np.array_equal(S_new, S):
            break
        S = S_new
    return S, b, member_z(Q, S, combine='sum')[0]


def _pair_max_worker(seed):
    _, Qn = shuffle_rank(_CTX['D'], _CTX['Q'], np.random.default_rng(seed))
    return float(pair_scores(Qn).max())


def pair_threshold_rank(D, Q, B=99, alpha=0.05, seed=0, workers=1):
    """Family-wise threshold for the pair majority z: the 1 - alpha quantile of its maximum
    over pairs under the per-step shuffle."""
    _CTX.update(D=D, Q=Q)
    args = [seed * 7919 + b for b in range(B)]
    if workers > 1:
        with mp.get_context('fork').Pool(workers) as pool:
            mx = pool.map(_pair_max_worker, args, chunksize=1)
    else:
        mx = [_pair_max_worker(a) for a in args]
    _CTX.clear()
    return float(np.quantile(mx, 1 - alpha))
