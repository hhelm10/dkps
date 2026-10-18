"""Detection statistic T(S), its nulls, and the scan (TODO section 4).

T(S) has two terms, each computed in the projected residual space:

  shift  E(S): energy distance between S's burn-in windowed positions and
               its post-burn-in windowed positions, each agent centred at
               its own burn-in mean (so between-agent spread cancels).
  comove C(S): mean pairwise cosine between members' displacement
               vectors D_n = mean(post) - mean(burn-in).

The two terms are combined by nonparametric combination (Pesarin): each
term is turned into a permutation p-value against the same null draws,
and F = -2 (log p_E + log p_C) is the combined statistic.

Two nulls:

  time      (the one-pager's null, `time_null`) each agent's actions are
            permuted across the burn-in/post boundary within context
            strata (submolt x thread-context cluster), timestamps held
            fixed; Westfall-Young max-F over a fixed family. Valid only
            when benign agents are stationary: any benign drift makes
            every S look shifted. Kept as a per-set diagnostic.
  rotation  (`scan`, primary) T(S) = z_E(S) + z_C(S). Each agent's centred
            trajectory is rotated by an independent random orthogonal
            matrix, which keeps how far and how noisily every agent moves
            but destroys alignment between agents. The content-derived
            candidates are rebuilt on every draw, so the null of
            Lambda = max_S T(S) accounts for selection. Asks: does some
            group move together more than chance alignment of
            independently drifting agents explains?
"""
import zlib
import numpy as np
from scipy.spatial.distance import cdist

from .space import period_matrices


def _energy(X, Y):
    if len(X) < 2 or len(Y) < 2:
        return 0.0
    return float(2 * cdist(X, Y).mean() - cdist(X, X).mean() - cdist(Y, Y).mean())


def _seed_of(S):
    return zlib.crc32(np.asarray(sorted(S), dtype=np.int64).tobytes())


class SubnetTester:
    def __init__(self, Z, agent_idx, t, strata, t0, grid, M, k, n_agents,
                 min_actions=3, max_pts=300):
        self.Z, self.t, self.t0 = Z, t, t0
        self.n, self.G, self.d = n_agents, len(grid), Z.shape[1]
        self.M, self.k, self.grid = M, k, grid
        self.pre, self.post, n_pre, n_post = period_matrices(agent_idx, t, n_agents, t0)
        self.n_pre, self.n_post = n_pre, n_post
        self.eligible = (n_pre >= min_actions) & (n_post >= min_actions)
        self.pre_grid = grid <= t0
        self.max_pts = max_pts
        self.activity = n_pre + n_post
        # permutation blocks: (agent, stratum) cells containing both periods
        key = agent_idx.astype(np.int64) * 100003 + np.asarray(strata, dtype=np.int64)
        order = np.argsort(key, kind='stable')
        _, starts = np.unique(key[order], return_index=True)
        blocks = np.split(order, starts[1:])
        pre_mask = t <= t0
        self.blocks = [b for b in blocks if pre_mask[b].any() and (~pre_mask[b]).any()]
        self.frac_permutable = (sum(len(b) for b in self.blocks) / len(t)) if len(t) else 0.0

    # ----- features of one realization -----
    def features(self, Zp):
        b = self.pre @ Zp
        D = self.post @ Zp - b
        Psi = (self.M @ Zp).reshape(self.n, self.G, self.d) - b[:, None, :]
        return D, Psi

    def permuted(self, rng):
        idx = np.arange(len(self.Z))
        for blk in self.blocks:
            idx[blk] = blk[rng.permutation(len(blk))]
        return self.Z[idx]

    def stat(self, D, Psi, S):
        S = np.asarray(S)
        rng = np.random.default_rng(_seed_of(S))
        kS = self.k[S] > 0
        pre = Psi[S][:, self.pre_grid][kS[:, self.pre_grid]]
        post = Psi[S][:, ~self.pre_grid][kS[:, ~self.pre_grid]]
        if len(pre) > self.max_pts:
            pre = pre[rng.choice(len(pre), self.max_pts, replace=False)]
        if len(post) > self.max_pts:
            post = post[rng.choice(len(post), self.max_pts, replace=False)]
        E = _energy(pre, post)
        U = D[S] / np.maximum(np.linalg.norm(D[S], axis=1, keepdims=True), 1e-12)
        s = len(S)
        C = float(((U.sum(0) ** 2).sum() - s) / (s * (s - 1))) if s > 1 else 0.0
        return E, C

    # ----- nulls -----
    def clean(self, family):
        out = []
        for S, src in family:
            S = np.array(sorted(i for i in S if self.eligible[i]))
            if len(S) >= 3:
                out.append((S, src))
        return out

    def time_null(self, family, B=199, seed=0):
        family = self.clean(family)
        rng = np.random.default_rng(seed)
        E = np.zeros((B + 1, len(family)))
        C = np.zeros_like(E)
        for b in range(B + 1):
            Zp = self.Z if b == 0 else self.permuted(rng)
            D, Psi = self.features(Zp)
            for j, (S, _) in enumerate(family):
                E[b, j], C[b, j] = self.stat(D, Psi, S)
        F = _npc(E, C)
        p = (F >= F[0]).mean(0)
        maxF = F.max(1)
        p_fw = (maxF[:, None] >= F[0][None, :]).mean(0)
        return _report(family, E, C, F, p, p_fw, 'time')

    # ----- scan statistic -----
    def agent_energy(self, Psi):
        """Per-agent energy distance between own burn-in and post windows."""
        e = np.zeros(self.n)
        for i in np.flatnonzero(self.eligible):
            w = self.k[i] > 0
            e[i] = _energy(Psi[i][self.pre_grid & w], Psi[i][~self.pre_grid & w])
        return e

    def n_factors_for(self, D, min_pr=50, min_frac=0.05):
        """Number of shared *diffuse* displacement factors to remove.

        Candidates are the leading singular directions of the system-centred
        displacements, up to the first profile-likelihood elbow (cap d/2).
        Leading directions are counted while their agent loadings are
        diffuse: participation ratio 1/sum(u^4) >= max(min_pr, min_frac * n).
        A colluding group loads on few agents, so its direction stops the
        count instead of being removed as a platform trend.
        """
        from graspologic.embed import select_dimension
        el = self.eligible
        n_el = int(el.sum())
        if self.d < 4 or n_el < 10:
            return 0
        Dc = D[el]  # uncentered: a platform-wide mean shift is itself a diffuse factor
        elbows, _ = select_dimension(Dc, n_components=self.d - 1, n_elbows=1)
        kmax = int(min(elbows[0], self.d // 2))
        Uf, _, _ = np.linalg.svd(Dc, full_matrices=False)
        thr = max(min_pr, min_frac * n_el)
        k = 0
        while k < kmax and 1.0 / (Uf[:, k] ** 4).sum() >= thr:
            k += 1
        return k

    def scan_parts(self, D, Psi, ze=None):
        """Ingredients of T(S) = z_E(S) + z_C(S) for one realization.

        z_E: mean per-agent energy of S, standardized against agents of the
             same activity quartile (rotation invariant).
        z_C: mean pairwise cosine of factor-adjusted (uncentered) displacements, scaled by
             sqrt(d_eff s (s-1) / 2), its sd under independent isotropic
             directions in d_eff = d - n_factors dimensions. Factor
             adjustment removes the top n_factors shared directions of the
             system-centred displacements (platform-wide trends), estimated
             afresh on every realization, so the null gets the same treatment.
        """
        el = self.eligible
        # No mean-centring: subtracting the system mean makes every agent with
        # a near-zero displacement point along -mean, i.e. spuriously aligned.
        # Shared drift is handled by removing diffuse factors of uncentered D.
        Dc = D.copy()
        kf = getattr(self, 'n_factors', 0) or 0
        if kf:
            _, _, Vt = np.linalg.svd(Dc[el], full_matrices=False)
            V = Vt[:kf].T
            Dc = Dc - (Dc @ V) @ V.T
        U = Dc / np.maximum(np.linalg.norm(Dc, axis=1, keepdims=True), 1e-12)
        d_eff = self.d - kf
        if ze is not None:
            return dict(U=U, ze=ze, Dc=Dc, d_eff=d_eff)
        e = self.agent_energy(Psi)
        q = np.quantile(self.activity[el], [0.25, 0.5, 0.75])
        bins = np.digitize(self.activity, q)
        mu = np.zeros(self.n); sd = np.ones(self.n)
        for b in range(4):
            m = el & (bins == b)
            if m.sum() > 1:
                mu[m], sd[m] = e[m].mean(), max(e[m].std(), 1e-12)
        return dict(U=U, ze=(e - mu) / sd, Dc=Dc, d_eff=d_eff)

    def T(self, parts, S):
        S = np.asarray(S)
        s = len(S)
        u = parts['U'][S].sum(0)
        C = ((u @ u) - s) / (s * (s - 1))
        zC = C * np.sqrt(parts['d_eff'] * s * (s - 1) / 2)
        zE = parts['ze'][S].sum() / np.sqrt(s)
        return float(zE + zC), float(zE), float(zC), float(C)

    def rotated_D(self, D, rng):
        """Independent Haar rotation of each agent's centred trajectory.

        Only D is needed: per-agent energies are invariant, and rotating a
        vector by a Haar matrix gives a uniform direction of the same norm.
        """
        G = rng.normal(size=D.shape)
        G /= np.linalg.norm(G, axis=1, keepdims=True)
        return G * np.linalg.norm(D, axis=1, keepdims=True)

    def scan(self, build_family, fixed_family, R=199, seed=0, alpha=0.05, max_sets=3,
             workers=1, size_bins=(3, 11, 31, 101), n_factors=0, shift_gate=True,
             shift_B=9999):
        """Scan Lambda = max_S T(S) with a rotation null that rebuilds the
        content-derived part of the family on every draw (selection-aware).

        build_family(D) -> list of (frozenset, source) from displacements.
        fixed_family    -> candidates that do not depend on content geometry.
        workers > 1 runs the null draws in forked worker processes.

        Size-stratified: Lambda is taken separately within each size band
        (size_bins are band lower edges), and a set's family-wise p-value is
        its band's max-T p-value times the number of occupied bands
        (Bonferroni). Without this, z_C grows with s and large real trend
        communities set the threshold for small colluding groups.
        n_factors: shared displacement factors removed before z_C ('auto' =
        elbow rule on the observed displacements).
        """
        D, Psi = self.features(self.Z)
        self.n_factors = self.n_factors_for(D) if n_factors == 'auto' else int(n_factors)
        self.size_bins = np.asarray(size_bins)
        fixed = self.clean(fixed_family)
        parts = self.scan_parts(D, Psi)
        fam = self.clean(build_family(parts['Dc'])) + fixed
        vals = [self.T(parts, S) for S, _ in fam]
        _CTX['args'] = (self, D, parts['ze'], fam, build_family, fixed)
        seeds = [seed * 100003 + r for r in range(R)]
        if workers > 1:
            import multiprocessing as mp
            with mp.get_context('fork').Pool(workers) as pool:
                draws = pool.map(_draw_worker, seeds, chunksize=1)
        else:
            draws = [_draw_worker(s_) for s_ in seeds]
        _CTX.clear()
        Lam_null = np.array([d_[0] for d_ in draws])          # (R, n_bins)
        per_S_null = np.array([d_[1] for d_ in draws]).reshape(R, len(fam))
        band = lambda s_: int(np.searchsorted(self.size_bins, s_, side='right') - 1)
        occupied = sorted(set(band(len(S)) for S, _ in fam))
        nb = max(len(occupied), 1)
        rows = []
        for j, ((S, src), (Tj, zE, zC, C)) in enumerate(zip(fam, vals)):
            b = band(len(S))
            rows.append(dict(members=S.tolist(), source=src, size=len(S), T=Tj,
                             zE=zE, zC=zC, comove=C, band=b,
                             p=float((1 + (per_S_null[:, j] >= Tj).sum()) / (R + 1)),
                             p_fw=float(min(1.0, nb * (1 + (Lam_null[:, b] >= Tj).sum()) / (R + 1)))))
        # shift gate (intersection-union): a set must also have changed.
        # p_shift: its summed standardized per-agent energy against random
        # sets with the same activity-quartile composition; Holm across the
        # sets that pass the co-movement scan.
        ze = parts['ze']
        q = np.quantile(self.activity[self.eligible], [0.25, 0.5, 0.75])
        bins = np.digitize(self.activity, q)
        pools = [np.flatnonzero(self.eligible & (bins == b)) for b in range(4)]
        rng = np.random.default_rng(seed + 7)
        cand = [r_ for r_ in rows if r_['p_fw'] < alpha]
        for r_ in cand:
            S = np.asarray(r_['members'])
            obs = ze[S].sum()
            null = np.zeros(shift_B)
            for b in range(4):
                cnt = int((bins[S] == b).sum())
                if cnt:
                    pool = pools[b]
                    null += ze[pool[rng.integers(0, len(pool), (shift_B, cnt))]].sum(1)
            r_['p_shift'] = float((1 + (null >= obs).sum()) / (shift_B + 1))
        adj = _holm([r_['p_shift'] for r_ in cand]) if cand else []
        for r_, a_ in zip(cand, adj):
            r_['p_shift_holm'] = float(a_)
        for r_ in rows:
            r_.setdefault('p_shift', float('nan'))
            r_.setdefault('p_shift_holm', float('nan'))
        # among significant overlapping sets prefer the tightest (highest mean
        # pairwise co-movement), not the largest: T grows with set size
        rows.sort(key=lambda r: (r['p_fw'], -r['comove']))
        flagged, used = [], set()
        for r_ in sorted(cand, key=lambda r: (r['p_fw'], r['p_shift'], -r['comove'])):
            if len(flagged) >= max_sets:
                break
            if (not shift_gate or r_['p_shift_holm'] < alpha) and used.isdisjoint(r_['members']):
                flagged.append(r_)
                used |= set(r_['members'])
        return dict(rows=rows, flagged=flagged, n_family=len(fam), n_comove_sig=len(cand),
                    Lambda=float(rows[0]['T']) if rows else np.nan,
                    Lambda_null=Lam_null.max(1), Lambda_null_by_band=Lam_null,
                    size_bins=self.size_bins.tolist(), n_factors=self.n_factors, null='rotation',
                    ze=ze, activity_bins=bins, candidates=cand)

    # ----- onset -----
    def onset(self, S):
        """Single change-point in S's mean position along its own displacement."""
        D, Psi = self.features(self.Z)
        S = np.asarray(S)
        u = D[S].mean(0)
        u /= max(np.linalg.norm(u), 1e-12)
        proj = (Psi[S] @ u)
        w = self.k[S] > 0
        g = np.array([proj[:, j][w[:, j]].mean() if w[:, j].any() else np.nan
                      for j in range(self.G)])
        ok = np.flatnonzero(~np.isnan(g))
        best, tau = np.inf, np.nan
        for c in ok[2:-2]:
            a, b = g[ok[ok < c]], g[ok[ok >= c]]
            sse = ((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum()
            if sse < best:
                best, tau = sse, float(self.grid[c])
        return tau, g


_CTX = {}


def _draw_worker(seed):
    """One rotation-null draw: max T over a rebuilt family, and T of the
    observed family's sets under the same draw."""
    from threadpoolctl import threadpool_limits
    self, D, ze, fam_obs, build, fixed = _CTX['args']
    with threadpool_limits(1):
        rng = np.random.default_rng(seed)
        parts = self.scan_parts(self.rotated_D(D, rng), None, ze)
        fam = self.clean(build(parts['Dc'])) + fixed
        lam = np.full(len(self.size_bins), -np.inf)
        for S, _ in fam:
            b = int(np.searchsorted(self.size_bins, len(S), side='right') - 1)
            lam[b] = max(lam[b], self.T(parts, S)[0])
        per = [self.T(parts, S)[0] for S, _ in fam_obs]
    return lam, per


def _npc(E, C):
    def pv(X):  # p-value of each row against all rows, per column
        r = (X[None, :, :] >= X[:, None, :]).mean(1)
        return r
    return -2 * (np.log(pv(E)) + np.log(pv(C)))


def _holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * p[i]))
        adj[i] = run
    return adj


def _report(family, E, C, F, p, p_fw, null):
    rows = []
    for j, (S, src) in enumerate(family):
        rows.append(dict(members=S.tolist(), source=src, size=len(S),
                         shift=float(E[0, j]), comove=float(C[0, j]),
                         F=float(F[0, j]), p=float(p[j]), p_fw=float(p_fw[j]),
                         null=null))
    rows.sort(key=lambda r: (r['p_fw'], -r['F']))
    return dict(rows=rows, n_family=len(family),
                Lambda=float(F[0].max()) if len(family) else np.nan,
                null_draws=dict(E=E, C=C, F=F))
