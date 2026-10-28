"""Score-only baselines for benchmark-score prediction.

These use only the observed score matrix (no response/query embeddings), as a
contrast to the perspective-space estimators:

- ``matrix_completion_predict`` -- low-rank logit-space completion (BenchPress)
  for held-out (model, task) cells; returns a prediction matrix with values at
  the unobserved entries and NaN elsewhere.
- ``irt_fit_difficulties`` / ``irt_estimate_ability`` / ``irt_predict`` -- a 1PL
  (Rasch) item-response model for query-efficient prediction of a model's score
  from its responses to a subset of items (Polo et al., 2024).
"""
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize, minimize_scalar


def matrix_completion_predict(score_mat, observed, rank=2, lam=0.1, n_init=5,
                              max_iter=50, clip_eps=1e-3, logit=True,
                              random_state=0, fill_observed=False):
    """Logit-space rank-2 bias-decomposed ALS matrix completion (BenchPress).

    Papailiopoulos, 'You Don't Need to Run Every Eval' (arXiv:2606.24020):
      1. logit-transform scores in [0,1] (clipped away from endpoints)
      2. standardize each task column (zero mean, unit variance) on observed
      3. fit  x_hat_mb = xbar + (model offset) + (task offset) + (U V^T)_mb
         over observed entries via ALS (closed-form ridge updates, lam),
         ensemble-averaged over random inits; rank R=2
      4. invert standardization + logit; clip to [0,1]

    Returns predictions at unobserved entries (NaN at observed).
    """
    M = np.asarray(score_mat, dtype=float)
    mask = np.asarray(observed, dtype=bool)
    nM, nB = M.shape
    preds = np.full_like(M, np.nan)
    if not (~mask).any() and not fill_observed:
        return preds

    # 1. logit transform
    if logit:
        s = np.clip(M, clip_eps, 1.0 - clip_eps)
        X = np.log(s / (1.0 - s))
    else:
        X = M.copy()

    # 2. column standardization (observed entries)
    mu = np.zeros(nB)
    sd = np.ones(nB)
    for b in range(nB):
        col = mask[:, b]
        if col.any():
            mu[b] = X[col, b].mean()
            std = X[col, b].std()
            sd[b] = std if std > 1e-8 else 1.0
    Xs = (X - mu) / sd

    # 3. bias terms (global + model + task offsets) on standardized observed
    gbar = Xs[mask].mean()
    row_bar = np.array([Xs[m, mask[m]].mean() if mask[m].any() else gbar
                        for m in range(nM)])
    col_bar = np.array([Xs[mask[:, b], b].mean() if mask[:, b].any() else gbar
                        for b in range(nB)])
    bias = row_bar[:, None] + col_bar[None, :] - gbar
    R = Xs - bias

    # 4. rank-r ALS on the residual over observed entries, ensembled over inits
    r = int(max(1, min(rank, nM - 1, nB - 1)))
    rng = np.random.default_rng(random_state)
    Ir = lam * np.eye(r)
    acc = np.zeros((nM, nB))
    for _ in range(n_init):
        U = rng.normal(0, 0.1, (nM, r))
        V = rng.normal(0, 0.1, (nB, r))
        for _ in range(max_iter):
            for m in range(nM):
                cols = np.where(mask[m])[0]
                if len(cols) == 0:
                    continue
                Vc = V[cols]
                U[m] = np.linalg.solve(Vc.T @ Vc + Ir, Vc.T @ R[m, cols])
            for b in range(nB):
                rows = np.where(mask[:, b])[0]
                if len(rows) == 0:
                    continue
                Ur = U[rows]
                V[b] = np.linalg.solve(Ur.T @ Ur + Ir, Ur.T @ R[rows, b])
        acc += bias + U @ V.T
    Xhat_std = acc / n_init

    # 5. invert standardization + logit
    Xhat = Xhat_std * sd + mu
    Phat = 1.0 / (1.0 + np.exp(-Xhat)) if logit else Xhat
    Pc = np.clip(Phat, 0.0, 1.0)
    # default: predictions only at unobserved entries. fill_observed also returns the
    # low-rank reconstruction at observed cells (a denoised estimate that uses the cell's
    # own value but regularizes it toward the fitted structure).
    preds[~mask] = Pc[~mask]
    if fill_observed:
        preds[mask] = Pc[mask]
    return preds


# ==========================================================================
# IRT  (1-Parameter Logistic / Rasch)  --  Polo et al., 2024
# ==========================================================================

def irt_fit_difficulties(S_ref):
    """Fit item difficulties ``beta`` from reference model binary scores.

    Parameters
    ----------
    S_ref : (n_ref, M) binary score matrix

    Returns
    -------
    beta  : (M,) item difficulty vector
    theta : (n_ref,) ability estimates used during fitting
    """
    n_ref, M = S_ref.shape

    # closed-form ability proxy
    row_means = S_ref.mean(axis=1).clip(0.01, 0.99)
    theta = np.log(row_means / (1 - row_means))

    # initialise beta from column means
    col_means = S_ref.mean(axis=0).clip(0.01, 0.99)
    beta_init = -np.log(col_means / (1 - col_means))

    def neg_ll_and_grad(beta):
        logits = theta[:, None] - beta[None, :]   # (n_ref, M)
        p = expit(logits)
        log_p   = np.log(p + 1e-10)
        log_1mp = np.log(1 - p + 1e-10)
        loss = -(S_ref * log_p + (1 - S_ref) * log_1mp).sum()
        grad = (S_ref - p).sum(axis=0)
        return loss, grad

    result = minimize(neg_ll_and_grad, beta_init, method='L-BFGS-B',
                      jac=True, options={'maxiter': 500, 'ftol': 1e-9})
    return result.x, theta


def irt_estimate_ability(scores_m, beta_m):
    """MLE ability estimate from ``m`` query scores.

    Parameters
    ----------
    scores_m : (m,) binary scores on the m selected queries
    beta_m   : (m,) item difficulties for those queries

    Returns
    -------
    float  theta_hat, capped to [-4, 4]
    """
    def neg_ll(theta):
        logits = theta - beta_m
        p = expit(logits).clip(1e-8, 1 - 1e-8)
        return -(scores_m * np.log(p) + (1 - scores_m) * np.log(1 - p)).sum()

    result = minimize_scalar(neg_ll, bounds=(-6, 6), method='bounded')
    return float(np.clip(result.x, -4.0, 4.0))


def irt_predict(theta_hat, beta_all):
    """Mean predicted probability across all M queries."""
    return float(np.clip(expit(theta_hat - beta_all).mean(), 0.0, 1.0))


# ==========================================================================
# Estimator classes (fit / predict / update over records)
# ==========================================================================

import warnings
import pandas as pd
from .records import parse_records, merge_records, ScoreTable, parse_pairs, pairs_to_records


def max_dense_block(amask):
    """Largest fully-observed |F| x |Q| rectangle of a boolean (model x item) mask
    (greedy over item coverage). Returns (row_idx, col_idx)."""
    amask = np.asarray(amask, dtype=bool)
    order = np.argsort(-amask.sum(0))
    best_area, best = 0, (np.array([], int), np.array([], int))
    for c in range(1, len(order) + 1):
        Q = order[:c]
        F = np.where(amask[:, Q].all(1))[0]
        if len(F) == 0:
            break
        if len(F) * c > best_area:
            best_area, best = len(F) * c, (F, Q)
    return best


class _ScoreEstimator:
    """Shared fit / update / predict plumbing for score-only estimators. Subclasses
    implement _fit_matrix() -> (n_models x n_tasks) prediction array (NaN = no estimate)
    and _default_pairs()."""

    REQUIRED = ('model_id', 'query_id', 'score')

    def fit(self, records):
        """Fit from response-level records (model_id, task_id, query_id, score[, sample_score]);
        embedding columns are ignored."""
        df = parse_records(records, required=self.REQUIRED)
        if 'task_id' not in df.columns:
            df['task_id'] = '_task'
        df['score'] = pd.to_numeric(df['score'], errors='coerce')
        assert not df.duplicated(['model_id', 'task_id', 'query_id']).any(), \
            'duplicate (model_id, task_id, query_id) rows'
        self._raw = df.reset_index(drop=True)
        self._refit()
        return self

    def update(self, records):
        """Merge new rows (re-sent (model, task, query) keys replace the old row) and refit;
        these estimators are cheap enough that a full refit is the incremental update."""
        assert hasattr(self, '_raw'), 'call fit() before update()'
        df = parse_records(records, required=self.REQUIRED)
        if 'task_id' not in df.columns:
            df['task_id'] = '_task'
        df['score'] = pd.to_numeric(df['score'], errors='coerce')
        self._raw, n_rep = merge_records(self._raw, df)
        if n_rep:
            warnings.warn(f'update: {n_rep} rows replaced by their newer version', stacklevel=2)
        self._refit()
        return self

    def _refit(self):
        self.table_ = ScoreTable(self._raw)
        self.model_names_ = self.table_.models
        self.task_names_ = self.table_.tasks
        self.sample_scores_ = self.table_.sample_
        self.pred_matrix_ = np.asarray(self._fit_matrix(), dtype=float)

    def predict(self, records=None):
        """Return a list of {'model_id', 'task_id', 'score_hat'} for the requested
        (model, task) pairs (None -> this estimator's natural target cells)."""
        pairs = parse_pairs(records, self.model_names_, self.task_names_, self._default_pairs())
        mi = {m: i for i, m in enumerate(self.model_names_)}
        ti = {t: j for j, t in enumerate(self.task_names_)}
        preds = {(m, t): (self.pred_matrix_[mi[m], ti[t]] if t in ti else np.nan)
                 for m, t in pairs}
        return pairs_to_records(pairs, preds)

    def _observed_pairs(self):
        O = self.table_.observed
        return [(m, t) for i, m in enumerate(self.model_names_)
                for j, t in enumerate(self.task_names_) if O[i, j]]

    def _missing_pairs(self):
        O = self.table_.observed
        return [(m, t) for i, m in enumerate(self.model_names_)
                for j, t in enumerate(self.task_names_) if not O[i, j]]


class SampleScore(_ScoreEstimator):
    """The sample score: the mean of a cell's observed per-response scores. Predicts
    observed cells only (NaN elsewhere)."""

    def _fit_matrix(self):
        return self.table_.values

    def _default_pairs(self):
        return self._observed_pairs()


class LRMC(_ScoreEstimator):
    """Low-rank matrix completion on the sample-score matrix (BenchPress-style logit /
    standardize / two-way-bias pre-processing + rank-r ALS; see matrix_completion_predict).

    Missing cells are completed by a full fit. Observed cells are estimated by
    crossfit_k-fold cross-fitting (each cell predicted by a fit that excludes it), so
    the estimate never echoes a cell's own sample; set crossfit_k=0 to skip.
    """

    def __init__(self, rank=2, lam=0.1, n_init=2, max_iter=50, crossfit_k=3, logit=True,
                 random_state=0, crossfit_random_state=None):
        self.rank, self.lam, self.n_init, self.max_iter = rank, lam, n_init, max_iter
        self.crossfit_k, self.logit, self.random_state = crossfit_k, logit, random_state
        # fold assignment RNG (an int seed or a np.random.Generator to share); ALS inits
        # always use random_state
        self.crossfit_random_state = crossfit_random_state

    def _complete(self, M, O):
        return matrix_completion_predict(M, O, rank=self.rank, lam=self.lam, n_init=self.n_init,
                                         max_iter=self.max_iter, logit=self.logit,
                                         random_state=self.random_state)

    def _fit_matrix(self):
        M, O = self.table_.values, self.table_.observed
        pred = self._complete(M, O)
        if self.crossfit_k:
            rng = np.random.default_rng(self.random_state if self.crossfit_random_state is None
                                        else self.crossfit_random_state)
            oi = np.argwhere(O)
            if len(oi) >= self.crossfit_k:
                for fold in np.array_split(rng.permutation(len(oi)), self.crossfit_k):
                    cells = oi[fold]
                    cvo = O.copy(); cvo[cells[:, 0], cells[:, 1]] = False
                    p = self._complete(M, cvo)
                    pred[cells[:, 0], cells[:, 1]] = p[cells[:, 0], cells[:, 1]]
        return pred

    def _default_pairs(self):
        return self._missing_pairs()


class IRT(_ScoreEstimator):
    """1PL (Rasch) item-response baseline for query-efficient prediction (Polo et al.,
    2024), fit per task on binary per-response scores.

    For each task: item difficulties are fit on the largest fully-observed (model x item)
    block of the task's response table; each model's ability is the MLE on its observed
    block items, and its predicted task score is the mean predicted probability over the
    block's items. Tasks with non-binary scores get NaN (with a warning).

    binary_tasks : optional explicit collection of task ids to treat as binary. By default
        binariness is inferred from the fitted records, which can mislabel a task whose
        rare fractional scores happen to be absent from a small sample; pass the task set
        decided on the full benchmark when it is known.
    """

    def __init__(self, binary_tasks=None):
        self.binary_tasks = None if binary_tasks is None else set(binary_tasks)

    def _fit_matrix(self):
        df = self._raw.dropna(subset=['score'])
        out = np.full((len(self.model_names_), len(self.task_names_)), np.nan)
        mi = {m: i for i, m in enumerate(self.model_names_)}
        for j, t in enumerate(self.task_names_):
            sub = df[df['task_id'] == t]
            if not len(sub):
                continue
            if self.binary_tasks is not None:
                if t not in self.binary_tasks:
                    continue
            elif not np.all(np.isin(np.unique(sub['score']), [0.0, 1.0])):
                warnings.warn(f'IRT: task {t!r} has non-binary scores; skipped', stacklevel=2)
                continue
            mods = sorted(sub['model_id'].unique()); items = sorted(sub['query_id'].unique())
            mloc = {m: a for a, m in enumerate(mods)}; iloc = {q: b for b, q in enumerate(items)}
            S = np.full((len(mods), len(items)), np.nan)
            S[sub['model_id'].map(mloc).to_numpy(), sub['query_id'].map(iloc).to_numpy()] = \
                sub['score'].to_numpy(dtype=float)
            amask = np.isfinite(S)
            F, Q = max_dense_block(amask)
            if len(F) < 2 or len(Q) < 1:
                continue
            beta_Q, _ = irt_fit_difficulties(np.nan_to_num(S[np.ix_(F, Q)], nan=0.0))
            for m in mods:
                a = amask[mloc[m], Q]
                if a.any():
                    theta = irt_estimate_ability(S[mloc[m], Q[a]], beta_Q[a])
                    out[mi[m], j] = irt_predict(theta, beta_Q)
        return out

    def _default_pairs(self):
        return self._observed_pairs()
