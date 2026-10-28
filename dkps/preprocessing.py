"""Pre-processing components shared by the estimators.

- ``Whitener``: the score pre-processing used by the completion estimators (LRMC and the
  PKPS completion regressor): logit transform, per-task standardization on observed cells,
  and a two-way (model + task) additive bias. Methods predict the residual; the inverse
  map restores the score scale and clips to [0, 1].
- ``FrozenPCA``: PCA basis learned once and frozen for out-of-sample updates; dimension fixed,
  or chosen at the n-th Zhu-Ghodsi elbow of the singular value spectrum.
- ``BlockDiagonalEmbedding``: the multi-benchmark response space used for the suites: each
  benchmark's response embeddings are PCA-reduced and unit-normalized separately and placed
  in a disjoint block, so a linear response kernel is exactly zero across benchmarks.
"""

import numpy as np
from sklearn.utils.extmath import randomized_svd
from graspologic.embed import select_dimension


class Whitener:
    """logit -> per-task column standardization -> two-way additive bias.

    fit(S, observed) learns mu_/sd_ (per task, observed cells) and bias_ (model + task
    offsets minus the global mean). transform(S) returns the residual target;
    inverse_transform(R) maps residuals back to clipped scores.
    """

    def __init__(self, logit=True, standardize=True, bias=True, eps=1e-3):
        self.logit = logit
        self.standardize = standardize
        self.bias = bias
        self.eps = eps

    def _forward(self, S):
        X = np.asarray(S, dtype=float).copy()
        if self.logit:
            s = np.clip(X, self.eps, 1.0 - self.eps)
            X = np.log(s / (1.0 - s))
        return X

    def fit(self, S, observed):
        X = self._forward(S)
        mask = np.asarray(observed, dtype=bool) & np.isfinite(X)
        nM, nB = X.shape
        self.mu_ = np.zeros(nB)
        self.sd_ = np.ones(nB)
        if self.standardize:
            for b in range(nB):
                c = mask[:, b]
                if c.any():
                    self.mu_[b] = X[c, b].mean()
                    s = X[c, b].std()
                    self.sd_[b] = s if s > 1e-8 else 1.0
        Xs = (X - self.mu_) / self.sd_
        self.bias_ = np.zeros((nM, nB))
        if self.bias:
            gbar = Xs[mask].mean() if mask.any() else 0.0
            row = np.array([Xs[m, mask[m]].mean() if mask[m].any() else gbar
                            for m in range(nM)])
            col = np.array([Xs[mask[:, b], b].mean() if mask[:, b].any() else gbar
                            for b in range(nB)])
            self.bias_ = row[:, None] + col[None, :] - gbar
        return self

    def transform(self, S):
        return (self._forward(S) - self.mu_) / self.sd_ - self.bias_

    def fit_transform(self, S, observed):
        return self.fit(S, observed).transform(S)

    def inverse_transform(self, R):
        Xhat = (self.bias_ + np.asarray(R, dtype=float)) * self.sd_ + self.mu_
        P = 1.0 / (1.0 + np.exp(-Xhat)) if self.logit else Xhat
        return np.clip(P, 0.0, 1.0) if self.logit else P


class FrozenPCA:
    """PCA basis learned at fit time and frozen for out-of-sample updates.

    dim='elbow' selects the dimension at the n_elbows-th Zhu-Ghodsi elbow of the
    singular value spectrum; dim=int fixes it; dim=None disables reduction.
    """

    def __init__(self, dim='elbow', n_elbows=2, max_components=128, random_state=0):
        self.dim = dim
        self.n_elbows = n_elbows
        self.max_components = max_components
        self.random_state = random_state
        self.mean_ = None
        self.components_ = None

    def fit(self, X):
        if self.dim is None:
            return self
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        self.mean_ = X.mean(axis=0)
        k = int(min(self.max_components, n - 1, d))
        if k < 1:
            self.mean_ = None
            return self
        _, S, Vt = randomized_svd(X - self.mean_, n_components=k,
                                  random_state=self.random_state)
        if self.dim == 'elbow':
            if k < 2 or len(S) < 2:
                dim = k
            else:
                elbows, _ = select_dimension(S, n_elbows=self.n_elbows)
                dim = int(elbows[-1]) if len(elbows) else k
        else:
            dim = int(self.dim)
        dim = max(1, min(dim, k))
        self.components_ = Vt[:dim]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.components_ is None:
            return X
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class BlockDiagonalEmbedding:
    """Per-group PCA (<= reduce_dim, Zhu-Ghodsi elbow) + unit normalization, groups placed in
    disjoint coordinate blocks. fit(X, groups) learns one basis per group label; transform
    maps new rows with their group labels (unknown groups raise)."""

    def __init__(self, reduce_dim=48, n_elbows=2):
        self.reduce_dim = reduce_dim
        self.n_elbows = n_elbows

    def fit(self, X, groups):
        X = np.asarray(X, dtype=np.float64)
        groups = np.asarray(groups)
        self.pcas_, self.dims_, self.offsets_ = {}, {}, {}
        off = 0
        for g in sorted(set(groups.tolist())):
            R = X[groups == g]
            pca = FrozenPCA(dim='elbow' if R.shape[1] > self.reduce_dim else None,
                            n_elbows=self.n_elbows, max_components=self.reduce_dim).fit(R)
            d = pca.transform(R[:1]).shape[1]
            self.pcas_[g], self.dims_[g], self.offsets_[g] = pca, d, off
            off += d
        self.total_dim_ = off
        return self

    def transform(self, X, groups):
        X = np.asarray(X, dtype=np.float64)
        groups = np.asarray(groups)
        out = np.zeros((len(X), self.total_dim_))
        for g in set(groups.tolist()):
            if g not in self.pcas_:
                raise KeyError(f'unknown group {g!r}: the block layout is frozen at fit '
                               'time, so adding a new suite/benchmark requires a fresh '
                               'fit() over all records (embeddings are disk-cached, so '
                               'only the new texts are embedded)')
            idx = np.where(groups == g)[0]
            R = self.pcas_[g].transform(X[idx])
            R = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
            out[idx, self.offsets_[g]:self.offsets_[g] + self.dims_[g]] = R
        return out

    def fit_transform(self, X, groups):
        return self.fit(X, groups).transform(X, groups)
