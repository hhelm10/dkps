import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, cdist
from sklearn.utils.extmath import randomized_svd
from graspologic.embed import ClassicalMDS, select_dimension


def pca_reduce_elbow(X, n_elbows=2, max_components=128, random_state=0):
    """Center + PCA-reduce X, choosing the dimension at the n_elbows-th elbow.

    Learned only from the rows of X passed in (use observed/sampled data for a
    given run). Returns the PCA scores (n_samples, dim). dim = 2nd Zhu-Ghodsi
    elbow of the singular value spectrum, matching ClassicalMDS(n_elbows=2).
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    Xc = X - X.mean(axis=0)
    k = int(min(max_components, n - 1, d))
    if k < 1:
        return Xc
    _, S, Vt = randomized_svd(Xc, n_components=k, random_state=random_state)
    if k < 2 or len(S) < 2:                 # too few components for an elbow
        dim = k
    else:
        elbows, _ = select_dimension(S, n_elbows=n_elbows)
        dim = int(elbows[-1]) if len(elbows) else k
    dim = max(1, min(dim, k))
    return Xc @ Vt[:dim].T


def local_dist_sq(X, mid, n_models, qidx, Q=None, sigmas=(None,), max_cells=2e7, floor=1e-12):
    """Locally normalized PKPS squared distances for a linear response kernel.

    For every unique query u in the cache (an "anchor") and every model i, let

        Z_i(u) = sum_a k_Q(u, u_a)                      (model i's query mass at u)
        F_i(u) = sum_a k_Q(u, u_a) x_ia / Z_i(u)        (its kernel-smoothed response at u)

    with a running over model i's responses. The squared distance is the average
    squared difference of the smoothed responses over the anchors where both models
    have query mass,

        D^2(i, j) = sum_u Z_i(u) Z_j(u) ||F_i(u) - F_j(u)||^2 / sum_u Z_i(u) Z_j(u).

    It is a weighted average of squared norms, so it is nonnegative for any query
    kernel and any cache. With the delta kernel on a shared query set it is the paired
    DKPS distance (mean per-query squared difference), and as the bandwidth grows it
    tends to the squared distance between mean responses -- the same two limits as the
    pair normalization, whose cross term it shares up to a self-convolution of the
    query kernel.

    Parameters
    ----------
    X : (N, p) response embeddings
    mid : (N,) model index in [0, n_models) of each response
    qidx : (N,) index in [0, U) of each response's query among the unique queries
    Q : (U, q) unique query embeddings (RBF query kernel), or None (delta kernel)
    sigmas : RBF bandwidths; ignored when Q is None

    Returns {sigma: (n_models, n_models) squared distances}; NaN where two models
    share no query mass. Anchors where a model's mass is below `floor` carry no weight.
    """
    X = np.asarray(X, dtype=np.float64)
    mid, qidx = np.asarray(mid), np.asarray(qidx)
    N, p = X.shape
    n = int(n_models)
    U = int(qidx.max()) + 1 if Q is None else len(Q)
    order = np.argsort(mid, kind='stable')
    X, mid, qidx = X[order], mid[order], qidx[order]
    bounds = np.searchsorted(mid, np.arange(n + 1))
    sigmas = [None] if Q is None else list(sigmas)
    num = {s: np.zeros((n, n)) for s in sigmas}
    den = {s: np.zeros((n, n)) for s in sigmas}
    if Q is not None:
        Q = np.asarray(Q, dtype=np.float64)
        Qr, q2 = Q[qidx], (Q ** 2).sum(1)
        q2r = q2[qidx]
    chunk = max(1, int(max_cells // max(N, 1)))
    for a in range(0, U, chunk):
        cols = np.arange(a, min(a + chunk, U))
        if Q is None:
            sq = None
        else:
            sq = np.maximum(q2r[:, None] + q2[cols][None] - 2.0 * Qr @ Q[cols].T, 0.0)
        for s in sigmas:
            K = (qidx[:, None] == cols[None]).astype(np.float64) if sq is None \
                else np.exp(-sq / (2.0 * s * s))                    # (N, C)
            Z = np.add.reduceat(K, bounds[:-1], axis=0) if N else np.zeros((n, len(cols)))
            Z[np.diff(bounds) == 0] = 0.0                           # models without responses
            G = np.stack([K[bounds[i]:bounds[i + 1]].T @ X[bounds[i]:bounds[i + 1]]
                          for i in range(n)])                       # (n, C, p) = Z_i(u) F_i(u)
            live = Z > floor
            Z = np.where(live, Z, 0.0)
            G[~live] = 0.0
            q = np.where(live, (G ** 2).sum(2) / np.where(live, Z, 1.0), 0.0)   # Z_i |F_i|^2
            Gf = G.reshape(n, -1)
            num[s] += q @ Z.T + Z @ q.T - 2.0 * Gf @ Gf.T
            den[s] += Z @ Z.T
    out = {}
    for s in sigmas:
        with np.errstate(invalid='ignore', divide='ignore'):
            D2 = np.where(den[s] > 0, num[s] / den[s], np.nan)
        D2 = (D2 + D2.T) / 2.0                      # exact symmetry (roundoff only)
        np.fill_diagonal(D2, 0.0)
        out[s] = D2
    return out


def _sqrt_nonneg(d2):
    """Square root of squared distances that are nonnegative by construction: only
    floating-point roundoff below zero is tolerated (and zeroed)."""
    d2 = np.asarray(d2, dtype=float)
    fin = d2[np.isfinite(d2)]
    tol = 1e-9 * max(1.0, float(np.abs(fin).max()) if fin.size else 1.0)
    assert not (fin < -tol).any(), 'negative squared distance where none is possible'
    with np.errstate(invalid='ignore'):
        return np.sqrt(np.where(np.isfinite(d2), np.maximum(d2, 0.0), np.nan))


class ProductKernelPerspectiveSpace:
    """
    PKPS (Product Kernel Perspective Space): extends DKPS to partially-observed
    (model x query) designs.

    Uses a product kernel k_Q * k_R where k_Q weights response comparisons by
    query similarity. Reduces to standard paired DKPS when query_kernel='delta'
    and all models share the same queries.

    For each model pair (i, k):

        D^2(i,k) = A_ii + A_kk - 2 * A_ik

        S_ab = sum_{j in Q_a, l in Q_b} k_Q(q_j, q_l) * k_R(x_aj, x_bl)
        Z_ab = sum_{j in Q_a, l in Q_b} k_Q(q_j, q_l)          (query kernel mass)

        A_ab = S_ab / Z_ab                  normalization='pair'
        A_ab = S_ab / sqrt(Z_aa * Z_bb)     normalization='model'

    normalization='pair' is the original construction. It compares two models only
    through queries both have answered something similar to, but its three terms are
    averaged under different weights, so D^2 can be negative and is clipped at zero.

    normalization='local' (default for a linear response kernel) keeps that behaviour
    and removes the defect: D^2(i,k) is the average, over the queries where both
    models have query-kernel mass, of the squared difference between their
    kernel-smoothed responses (see local_dist_sq). It is nonnegative by construction.

    normalization='model' (default otherwise) makes A the Gram matrix of
    v_a = sum_j phi_Q(q_j) (x) phi_R(x_aj) / sqrt(Z_aa) for positive-semidefinite
    kernels, so the distances are Euclidean; in exchange two models are pushed apart
    when their query sets differ. All three agree when every model answers the same
    queries under the delta kernel.

    Parameters
    ----------
    query_kernel : str or callable
        'rbf' (default) — RBF kernel on query embeddings.
        'delta' — 1 if same query_id, 0 otherwise. Recovers standard paired DKPS.
        callable(Q_a, Q_b) -> (n_a, n_b) matrix.
    response_kernel : str or callable
        'linear' (default) — dot product on response embeddings.
        'rbf' — RBF kernel.
        callable(X_a, X_b) -> (n_a, n_b) matrix.
    query_bandwidth : float or 'median'
        Bandwidth for RBF query kernel.
    response_bandwidth : float or 'median'
        Bandwidth for RBF response kernel.
    task_filter : str or None
        None (default) — use all queries.
        'shared' — restrict to queries from tasks shared by both models.
        'unshared' — restrict to queries from tasks unique to each model.
        Requires task_id column in data.
    normalization : None (default), 'local', 'model' or 'pair'
        How query kernel mass is normalized (see above). None selects 'local' for a
        linear response kernel and 'model' otherwise.
    n_components_cmds : int or None
        Embedding dimension for ClassicalMDS.
    n_elbows_cmds : int
        Number of elbows for automatic dimension selection.
    """

    def __init__(
        self,
        query_kernel='rbf',
        response_kernel='linear',
        query_bandwidth='median',
        response_bandwidth='median',
        task_filter=None,
        n_components_cmds=None,
        n_elbows_cmds=2,
        normalization=None,
    ):
        if normalization is None:
            normalization = 'local' if response_kernel == 'linear' else 'model'
        if normalization not in ('local', 'model', 'pair'):
            raise ValueError(f'unknown normalization: {normalization}')
        if normalization == 'local' and (response_kernel != 'linear'
                                         or query_kernel not in ('rbf', 'delta')):
            raise ValueError("normalization='local' requires a linear response kernel and "
                             "an 'rbf' or 'delta' query kernel")
        self.normalization = normalization
        self.query_kernel = query_kernel
        self.response_kernel = response_kernel
        self.query_bandwidth = query_bandwidth
        self.response_bandwidth = response_bandwidth
        self.task_filter = task_filter
        self.n_components_cmds = n_components_cmds
        self.n_elbows_cmds = n_elbows_cmds

    def fit(self, data):
        """Compute pairwise distance matrix from partially-observed response data."""
        model_data, model_names = self._prepare_model_data(data)

        if self.query_kernel == 'rbf':
            all_q = np.vstack([md['query_emb'] for md in model_data.values()])
            self.query_bandwidth_ = self._resolve_bandwidth(all_q, self.query_bandwidth)

        if self.response_kernel == 'rbf':
            all_r = np.vstack([md['emb'] for md in model_data.values()])
            self.response_bandwidth_ = self._resolve_bandwidth(all_r, self.response_bandwidth)

        n = len(model_names)
        dist_sq = np.zeros((n, n))

        if self.normalization == 'local':
            if self.task_filter is not None:
                raise ValueError("task_filter is not supported with normalization='local'")
            sig = self.query_bandwidth_ if self.query_kernel == 'rbf' else None
            dist_sq = self._local_d2(model_data, model_names, [sig])[sig]
        elif self.task_filter is not None:
            for i in range(n):
                for k in range(i + 1, n):
                    d2 = self._pairwise_dist_sq_filtered(
                        model_data[model_names[i]], model_data[model_names[k]])
                    dist_sq[i, k] = dist_sq[k, i] = d2
        else:
            self_SZ = [self._compute_SZ(model_data[m], model_data[m]) for m in model_names]
            for i in range(n):
                for k in range(i + 1, n):
                    cross = self._compute_SZ(model_data[model_names[i]], model_data[model_names[k]])
                    dist_sq[i, k] = dist_sq[k, i] = self._d2(self_SZ[i], self_SZ[k], cross)

        self.dist_matrix_ = self._sqrt(dist_sq)
        self.model_names_ = model_names
        return self

    def fit_transform(self, data):
        """Fit and embed models into low-dimensional space via ClassicalMDS."""
        self.fit(data)
        cmds = ClassicalMDS(
            n_components=self.n_components_cmds,
            n_elbows=self.n_elbows_cmds,
            dissimilarity='precomputed',
        ).fit_transform(self.dist_matrix_)
        return {m: cmds[i] for i, m in enumerate(self.model_names_)}

    def _prepare_model_data(self, data):
        assert isinstance(data, pd.DataFrame), 'data must be a DataFrame'
        for col in ('model_id', 'query_id', 'embedding'):
            assert col in data.columns, f'missing column: {col}'
        if self.query_kernel not in ('delta',) and not callable(self.query_kernel):
            assert 'query_embedding' in data.columns, \
                'query_embedding column required when query_kernel is not delta'
        if self.task_filter is not None:
            assert 'task_id' in data.columns, \
                'task_id column required when task_filter is set'

        assert not data.duplicated(['model_id', 'query_id']).any(), \
            'duplicate (model_id, query_id) pairs'

        model_names = sorted(data['model_id'].unique())
        model_data = {}
        for name in model_names:
            sub = data[data['model_id'] == name].sort_values('query_id')
            md = {
                'emb': np.stack(sub['embedding'].values),
                'query_ids': sub['query_id'].values,
            }
            if 'query_embedding' in data.columns:
                md['query_emb'] = np.stack(sub['query_embedding'].values)
            if 'task_id' in data.columns:
                md['task_ids'] = sub['task_id'].values
            model_data[name] = md

        return model_data, model_names

    def _resolve_bandwidth(self, X, method):
        if isinstance(method, (int, float)):
            return float(method)
        if method == 'median':
            dists = pdist(X)
            return float(np.median(dists)) if len(dists) > 0 else 1.0
        if callable(method):
            return float(method(X))
        raise ValueError(f'unknown bandwidth method: {method}')

    def _filter_to_tasks(self, md, tasks):
        """Return a subset of model data restricted to the given task set."""
        mask = np.isin(md['task_ids'], list(tasks))
        if not mask.any():
            return None
        out = {'emb': md['emb'][mask], 'query_ids': md['query_ids'][mask]}
        if 'query_emb' in md:
            out['query_emb'] = md['query_emb'][mask]
        if 'task_ids' in md:
            out['task_ids'] = md['task_ids'][mask]
        return out

    def _pairwise_dist_sq_filtered(self, md_a, md_b):
        """Compute D^2 using only shared or unshared task queries."""
        tasks_a = set(md_a['task_ids'])
        tasks_b = set(md_b['task_ids'])

        if self.task_filter == 'shared':
            keep_a = tasks_a & tasks_b
            keep_b = keep_a
        elif self.task_filter == 'unshared':
            keep_a = tasks_a - tasks_b
            keep_b = tasks_b - tasks_a
        else:
            raise ValueError(f'unknown task_filter: {self.task_filter}')

        if not keep_a or not keep_b:
            return np.nan

        fa = self._filter_to_tasks(md_a, keep_a)
        fb = self._filter_to_tasks(md_b, keep_b)
        if fa is None or fb is None:
            return np.nan

        return self._d2(self._compute_SZ(fa, fa), self._compute_SZ(fb, fb),
                        self._compute_SZ(fa, fb))

    def _local_d2(self, model_data, names, sigmas):
        X = np.vstack([model_data[m]['emb'] for m in names])
        mid = np.concatenate([np.full(len(model_data[m]['emb']), i) for i, m in enumerate(names)])
        qids = np.concatenate([model_data[m]['query_ids'] for m in names])
        _, first, qidx = np.unique(qids, return_index=True, return_inverse=True)
        Q = None
        if self.query_kernel == 'rbf':
            Q = np.vstack([model_data[m]['query_emb'] for m in names])[first]
        return local_dist_sq(X, mid, len(names), qidx, Q=Q, sigmas=sigmas)

    def _d2(self, sz_a, sz_b, sz_ab):
        """Squared distance from the (S, Z) of the two self pairs and the cross pair."""
        (s_a, z_a), (s_b, z_b), (s_ab, z_ab) = sz_a, sz_b, sz_ab
        if self.normalization == 'model':
            if z_a <= 0 or z_b <= 0:
                return np.nan
            return s_a / z_a + s_b / z_b - 2 * s_ab / np.sqrt(z_a * z_b)
        if z_a <= 0 or z_b <= 0 or z_ab <= 0:
            return np.nan
        return s_a / z_a + s_b / z_b - 2 * s_ab / z_ab

    def _sqrt(self, d2):
        """Distances from squared distances. Under normalization='model' the squared
        distances are nonnegative by construction, so only floating-point roundoff is
        zeroed; normalization='pair' has no such guarantee and is clipped at zero."""
        d2 = np.asarray(d2, dtype=float)
        if self.normalization == 'local' or (self.normalization == 'model'
                                             and self._psd_kernels()):
            return _sqrt_nonneg(d2)
        with np.errstate(invalid='ignore'):
            return np.sqrt(np.where(np.isfinite(d2), np.maximum(d2, 0.0), np.nan))

    def _psd_kernels(self):
        return (self.query_kernel in ('rbf', 'delta')
                and self.response_kernel in ('linear', 'rbf'))

    def _compute_A(self, data_a, data_b):
        """Pair-normalized affinity sum(K_Q * K_R) / sum(K_Q)."""
        S, Z = self._compute_SZ(data_a, data_b)
        return S / Z if Z != 0 else np.nan

    def _compute_SZ(self, data_a, data_b):
        """Return (S_ab, Z_ab) = (sum(K_Q * K_R), sum(K_Q))."""
        if self.query_kernel == 'delta':
            ids_a, ids_b = data_a['query_ids'], data_b['query_ids']
            K_Q = (ids_a[:, None] == ids_b[None, :]).astype(np.float64)
        elif self.query_kernel == 'rbf':
            sq = cdist(data_a['query_emb'], data_b['query_emb'], 'sqeuclidean')
            K_Q = np.exp(-sq / (2 * self.query_bandwidth_ ** 2))
        elif self.query_kernel == 'linear':
            # Linear (dot-product) query kernel on L2-normalized query embeddings
            # (cosine), clipped to [0, 1] so it is a valid non-negative weight.
            qa = data_a['query_emb'] / np.clip(
                np.linalg.norm(data_a['query_emb'], axis=1, keepdims=True), 1e-12, None)
            qb = data_b['query_emb'] / np.clip(
                np.linalg.norm(data_b['query_emb'], axis=1, keepdims=True), 1e-12, None)
            K_Q = np.clip(qa @ qb.T, 0.0, None)
        elif callable(self.query_kernel):
            K_Q = np.asarray(self.query_kernel(data_a['query_emb'], data_b['query_emb']))
        else:
            raise ValueError(f'unknown query_kernel: {self.query_kernel}')

        emb_a, emb_b = data_a['emb'], data_b['emb']
        if self.response_kernel == 'linear':
            K_R = emb_a @ emb_b.T
        elif self.response_kernel == 'rbf':
            sq = cdist(emb_a, emb_b, 'sqeuclidean')
            K_R = np.exp(-sq / (2 * self.response_bandwidth_ ** 2))
        elif callable(self.response_kernel):
            K_R = np.asarray(self.response_kernel(emb_a, emb_b))
        else:
            raise ValueError(f'unknown response_kernel: {self.response_kernel}')

        return float((K_Q * K_R).sum()), float(K_Q.sum())

    # ------------------------------------------------------------------
    # Bandwidth-grid distance matrices: the response kernel and query
    # squared-distances are sigma-independent, so a single pair loop
    # produces the distance matrix for an entire RBF-query-bandwidth grid.
    # ------------------------------------------------------------------
    def _kr_qsq(self, data_a, data_b):
        ea, eb = data_a['emb'], data_b['emb']
        if self.response_kernel == 'linear':
            K_R = ea @ eb.T
        elif self.response_kernel == 'rbf':
            K_R = np.exp(-cdist(ea, eb, 'sqeuclidean') / (2 * self.response_bandwidth_ ** 2))
        else:
            K_R = np.asarray(self.response_kernel(ea, eb))
        q_sq = cdist(data_a['query_emb'], data_b['query_emb'], 'sqeuclidean')
        return K_R, q_sq

    def dist_matrices(self, data, sigmas):
        """Return (model_names, {sigma: distance_matrix}) for an RBF query kernel,
        reusing the sigma-independent response kernel + query distances."""
        model_data, names = self._prepare_model_data(data)
        if self.response_kernel == 'rbf':
            all_r = np.vstack([md['emb'] for md in model_data.values()])
            self.response_bandwidth_ = self._resolve_bandwidth(all_r, self.response_bandwidth)
        sigmas = list(sigmas)
        n = len(names)
        if self.normalization == 'local':
            self.model_names_ = names
            D2 = self._local_d2(model_data, names, sigmas)
            return names, {s: self._sqrt(D2[s]) for s in sigmas}

        def SZ_grid(KR, qsq):
            out = {}
            for s in sigmas:
                KQ = np.exp(-qsq / (2.0 * s * s))
                out[s] = (float((KQ * KR).sum()), float(KQ.sum()))
            return out

        self_SZ = [SZ_grid(*self._kr_qsq(model_data[m], model_data[m])) for m in names]
        D2 = {s: np.zeros((n, n)) for s in sigmas}
        for i in range(n):
            for k in range(i + 1, n):
                cross = SZ_grid(*self._kr_qsq(model_data[names[i]], model_data[names[k]]))
                for s in sigmas:
                    D2[s][i, k] = D2[s][k, i] = self._d2(self_SZ[i][s], self_SZ[k][s], cross[s])
        self.model_names_ = names
        return names, {s: self._sqrt(D2[s]) for s in sigmas}


# Back-compat alias. (The name PKPS now refers to the high-level estimator in
# dkps.pkps; the low-level distance machinery lives here.)
DoubleKernelDKPS = ProductKernelPerspectiveSpace
