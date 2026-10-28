"""High-level PKPS / DKPS estimators with a fit / predict / update API.

Pipeline (matching the paper):

  1. embed queries and responses (optional -- precomputed embeddings are accepted),
  2. PCA-reduce each channel (fixed dim, or Zhu-Ghodsi elbow selection),
  3. product-kernel affinities  A_ab = sum_{j,l} k_Q(u_j, u_l) k_R(x_aj, x_bl) / sum_{j,l} k_Q(u_j, u_l),
  4. squared distances          D^2_ab = A_aa + A_bb - 2 A_ab,
  5. classical MDS on D  ->  model representations Z,
  6. per-task k-NN regression from Z onto the other models' scores.

`DKPS` is the paired special case: k_Q is the delta kernel on query ids, which on a
fully paired design recovers the standard DKPS distance (mean per-query squared
distance under a linear response kernel).

Records are exchanged as JSON-friendly tables: a list of dicts, a dict of lists, a
DataFrame, or a JSON string of either. Fit/update rows carry
(model_id, task_id, query_id, query|query_embedding, response|response_embedding, score
[, reference_score]); predict rows carry (model_id[, task_id]).
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from sklearn.neighbors import KNeighborsRegressor
from graspologic.embed import ClassicalMDS

from .unpaired_dkps import local_dist_sq, _sqrt_nonneg
from .preprocessing import BlockDiagonalEmbedding, FrozenPCA, Whitener
from .records import parse_records, merge_records, parse_pairs, pairs_to_records, ScoreTable


QUERY_DEFAULTS = dict(kernel='rbf', bandwidth='median', bandwidth_ref=None, bandwidth_mult=1.0,
                      bandwidth_grid=(0.03, 0.1, 0.3, 1.0, 3.0), cv_k=8,
                      pca_dim='elbow', pca_n_elbows=2, normalization=None, pooling=False)
RESPONSE_DEFAULTS = dict(kernel='linear', bandwidth='median', bandwidth_ref=None, bandwidth_mult=1.0,
                         pca_dim='elbow', pca_n_elbows=2, suite_mode='joint')
MDS_DEFAULTS = dict(dim=None, n_elbows=2)
EMBEDDING_DEFAULTS = dict(provider='google', model=None, api_key=None)

_PROVIDER_ENV = {
    'google': 'GEMINI_API_KEY',
    'jina': 'JINA_API_KEY',
    'openrouter': 'OPENROUTER_API_KEY',
    'huggingface': 'HF_TOKEN',
}


def family(model_id):
    """Default model-family function: the developer prefix ('openai_gpt-4' -> 'openai',
    'allenai/olmo-3' -> 'allenai')."""
    m = str(model_id)
    return m.split('/')[0] if '/' in m else m.split('_')[0]


def _median_distance(X, cap=5000, random_state=0):
    X = np.asarray(X, dtype=np.float64)
    if len(X) > cap:
        rng = np.random.default_rng(random_state)
        X = X[rng.choice(len(X), cap, replace=False)]
    d = pdist(X)
    return float(np.median(d)) if len(d) else 1.0


def _kernel_matrix(kernel, bandwidth, Xa, Xb, ids_a=None, ids_b=None):
    """Kernel matrix between two row sets. 'delta' compares ids; the rest compare
    embedding vectors."""
    if kernel == 'delta':
        return (np.asarray(ids_a)[:, None] == np.asarray(ids_b)[None, :]).astype(np.float64)
    if kernel == 'rbf':
        sq = cdist(Xa, Xb, 'sqeuclidean')
        return np.exp(-sq / (2.0 * bandwidth ** 2))
    if kernel == 'linear':
        return Xa @ Xb.T
    if kernel == 'cosine':
        na = np.clip(np.linalg.norm(Xa, axis=1, keepdims=True), 1e-12, None)
        nb = np.clip(np.linalg.norm(Xb, axis=1, keepdims=True), 1e-12, None)
        return np.clip((Xa / na) @ (Xb / nb).T, 0.0, None)
    if callable(kernel):
        return np.asarray(kernel(Xa, Xb))
    raise ValueError(f'unknown kernel: {kernel}')


def _lomo_knn_error(Z, y, k):
    """Leave-one-model-out k-NN error of predicting y from Z (inverse-distance weights);
    the bandwidth-selection criterion."""
    D2 = ((Z[:, None] - Z[None]) ** 2).sum(-1)
    np.fill_diagonal(D2, np.inf)
    errs = []
    for a in range(len(y)):
        nbr = np.argsort(D2[a])[:k]
        w = 1.0 / (np.sqrt(D2[a][nbr]) + 1e-9)
        errs.append(abs(np.average(y[nbr], weights=w) - y[a]))
    return errs


class _JointNormReducer:
    """Shared-space response reducer for suite-labeled records: one frozen PCA over all
    suites, rows unit-normalized (scale comparability across suites)."""

    def __init__(self, pca):
        self.pca = pca

    def transform(self, X, groups=None):
        R = self.pca.transform(np.asarray(X, dtype=np.float64))
        return R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)


class _BlockReducer:
    """Response reducer for suite-labeled records: wraps BlockDiagonalEmbedding so the
    grouped transform threads through the frozen-reducer plumbing."""

    def __init__(self, block):
        self.block = block

    def transform(self, X, groups):
        return self.block.transform(X, groups)


class PKPS:
    """Product Kernel Perspective Space estimator.

    Parameters
    ----------
    query_kwargs : dict, optional
        Query channel configuration (defaults in QUERY_DEFAULTS):
          kernel : 'rbf' | 'delta' | 'linear' | 'cosine' | callable
          bandwidth : 'median' | 'cv' | float. 'median' uses bandwidth_mult x the
              reference scale; 'cv' picks, from bandwidth_grid x the reference scale, the
              bandwidth whose representation best predicts the observed sample scores
              under leave-one-model-out k-NN (cv_k neighbours) -- the paper's protocol.
          bandwidth_ref : reference scale for 'median'/'cv' (None = median pairwise
              distance of the (reduced) query embeddings; the paper passes the
              within-domain median).
          bandwidth_mult, bandwidth_grid, cv_k : see above
          normalization : None (default) | 'local' | 'model' | 'pair'. How query
              kernel mass is normalized when two models are compared.
              'pair' is the original A_ab = S_ab / Z_ab with S = sum k_Q k_R and
              Z = sum k_Q. It compares two models only through queries both have
              answered something similar to, but D^2 = A_aa + A_bb - 2 A_ab can be
              negative and is clipped at zero.
              'local' keeps that behaviour without the defect: D^2 is the average, over
              the queries where both models have query-kernel mass, of the squared
              difference between their kernel-smoothed responses
              (dkps.unpaired_dkps.local_dist_sq). Nonnegative by construction. Needs a
              linear response kernel and an 'rbf' or 'delta' query kernel.
              'model' uses A_ab = S_ab / sqrt(Z_aa Z_bb), a Gram matrix for
              positive-semidefinite kernels (Euclidean distances), at the price of
              pushing apart models whose query sets differ.
              None selects 'local' when it applies and 'model' otherwise.
          pooling : False (default) | True. If True, models are represented by the
              mean of their query-residualized responses instead of by pairwise
              affinities: each response has subtracted from it the query-kernel-weighted
              mean response of the *other* models to similar queries, and the distance
              between two models is the Euclidean distance between their mean residuals.
              The query kernel therefore borrows strength from every model in the
              cache, not just the pair being compared. Requires a linear response
              kernel and an 'rbf' or 'delta' query kernel; 'normalization' is unused.
          pca_dim : 'elbow' (Zhu-Ghodsi selection) | int | None (no reduction)
          pca_n_elbows : int, elbow index used when pca_dim='elbow'
    response_kwargs : dict, optional
        Response channel configuration; same keys minus the CV ones
        (defaults in RESPONSE_DEFAULTS).
    mds_kwargs : dict, optional
        Classical MDS on the model distance matrix:
          dim : int (capped at n_models - 1; the paper uses 8) or None (Zhu-Ghodsi)
          n_elbows : int, elbow index used when dim is None
        The MDS acts on the joint product-kernel distance matrix, so there is a
        single spec rather than one per channel.
    embedding_kwargs : dict, optional
        Text embedding backend, used only when fit/update rows carry raw text
        instead of precomputed embeddings:
          provider : see dkps.embed.embed_api ('google', 'jina', ...)
          model : provider-specific model name (None = provider default)
          api_key : API key (None = read from the provider's environment variable)

    Attributes (after fit)
    ----------------------
    model_names_, task_names_ : ids
    dist_matrix_ : (n, n) PKPS distance matrix at the selected bandwidth
    embedding_ : dict model_id -> MDS coordinates
    query_bandwidth_ : selected query bandwidth (None for non-RBF kernels)
    sample_scores_ : DataFrame (model x task) of observed sample scores
    reference_scores_ : DataFrame of cell-level reference scores (NaN where not given)
    """

    def __init__(self, query_kwargs=None, response_kwargs=None, mds_kwargs=None,
                 embedding_kwargs=None):
        self.query_kwargs = {**QUERY_DEFAULTS, **(query_kwargs or {})}
        self.response_kwargs = {**RESPONSE_DEFAULTS, **(response_kwargs or {})}
        self.mds_kwargs = {**MDS_DEFAULTS, **(mds_kwargs or {})}
        self.embedding_kwargs = {**EMBEDDING_DEFAULTS, **(embedding_kwargs or {})}
        local_ok = (self.response_kwargs['kernel'] == 'linear'
                    and self.query_kwargs['kernel'] in ('rbf', 'delta'))
        if self.query_kwargs['normalization'] is None:
            self.query_kwargs['normalization'] = 'local' if local_ok else 'model'
        if self.query_kwargs['normalization'] not in ('local', 'model', 'pair'):
            raise ValueError(f"unknown normalization: {self.query_kwargs['normalization']}")
        if self.query_kwargs['normalization'] == 'local' and not local_ok:
            raise ValueError("normalization='local' requires a linear response kernel and "
                             "an 'rbf' or 'delta' query kernel")
        if self.query_kwargs['pooling']:
            if self.response_kwargs['kernel'] != 'linear':
                raise ValueError('pooling requires a linear response kernel')
            if self.query_kwargs['kernel'] not in ('rbf', 'delta'):
                raise ValueError("pooling requires an 'rbf' or 'delta' query kernel")
        self._fitted = False

    # ------------------------------------------------------------------
    # fit / update / predict
    # ------------------------------------------------------------------
    def fit(self, records):
        """Fit from response-level records.

        Each record: model_id, task_id, query_id, a response (text under 'response', or
        a vector under 'response_embedding'/'embedding'), a query (text under 'query',
        or vector under 'query_embedding'; required unless the query kernel is
        'delta'), an optional per-response 'score', and optional cell-level columns:
        'reference_score' (a known benchmark score for that (model, task), e.g. from a
        fully evaluated cache; used as the regression target in place of the sample
        score when present), 'sample_score' (overrides the per-cell mean of 'score'),
        and 'suite' (a benchmark/domain label: responses are reduced into one shared
        unit-normalized space and the query kernel alone gates cross-suite
        comparisons -- the base method; pass response_kwargs=dict(suite_mode='blocked')
        for the block-diagonal ablation, where a linear response kernel is exactly
        zero across suites).
        """
        df = self._ingest(records)
        self._raw = df
        self._fit_reducers(df)
        self._model_data = {m: self._reduce_model(df[df['model_id'] == m])
                            for m in sorted(df['model_id'].unique())}
        self._A = {s: {} for s in self._sigmas}
        self._recompute_pairs(set(self._model_data))
        self._finalize()
        self._fitted = True
        return self

    def update(self, records):
        """Add records (new models, or new responses for existing models) and refresh
        the fit.

        The PCA bases and kernel bandwidth scale learned at fit time are frozen, so only
        affinities involving updated models are recomputed; the bandwidth selection
        (if 'cv'), the MDS and the score tables are refreshed. Re-run fit() to relearn
        the reducers.
        """
        assert self._fitted, 'call fit() before update()'
        df = self._ingest(records)
        touched = set(df['model_id'].unique())
        self._raw, n_rep = merge_records(self._raw, df)
        if n_rep:
            warnings.warn(f'update: {n_rep} (model, task, query) rows replaced by their '
                          'newer version', stacklevel=2)
        for m in touched:
            self._model_data[m] = self._reduce_model(self._raw[self._raw['model_id'] == m])
        self._recompute_pairs(touched)
        self._finalize()
        return self

    def predict(self, records=None, k=5, holdout='model', family_fn=None, whiten=False,
                target='auto', min_train=None, interval=None):
        """Predict scores for (model, task) pairs by per-task k-NN regression
        (inverse-distance weights) over the MDS representation.

        records : None (every cell without an observed sample score), or records with
            'model_id' and optionally 'task_id' (a model without task_id requests every task).
        holdout : 'model' -- train on every other model with a score on the task (leave-one-
            model-out); 'family' -- additionally exclude models from the target's family
            (leave-one-family-out, the paper's query-efficiency protocol).
        family_fn : model_id -> family label (default: developer prefix).
        whiten : route targets through the logit / standardize / two-way-bias Whitener
            fitted on the observed score matrix, regress the residual, and invert (the
            paper's completion protocol).
        target : 'auto' (reference score where given, else sample score), 'sample', or
            'reference'.
        min_train : minimum support per task (default 3 when holdout='family' or whiten,
            else 1); fewer -> NaN. Under holdout='model' this counts the task's observed
            models (the target included, if observed); under 'family' it counts the
            training models outside the target's family.
        interval : None, or a confidence level in (0, 1) (e.g. 0.9). When set, every
            returned dict also carries 'score_lo' and 'score_hi' from a split-conformal
            interval over the reference models: each training model on the task is
            predicted leaving itself (and, under holdout='family', its family) out,
            and the half-width is the finite-sample-corrected (1 - alpha) quantile of
            those absolute residuals. Valid marginal coverage when models are
            exchangeable draws (the paper's sampling model); the bounds are +/-inf
            when the task has too few scored reference models for the requested level
            (needs at least ceil(level / (1 - level)) calibration points).
        Returns a list of dicts with 'score_hat' (and 'score_lo'/'score_hi') added.
        """
        assert self._fitted, 'call fit() before predict()'
        if interval is not None and not (0.0 < interval < 1.0):
            raise ValueError('interval must be a confidence level in (0, 1)')
        if min_train is None:
            min_train = 3 if (holdout == 'family' or whiten) else 1
        fam = family_fn or family
        pairs = self._parse_pairs(records)
        names = self.model_names_
        idx = {m: i for i, m in enumerate(names)}
        Z = np.stack([self.embedding_[m] for m in names])
        Y = self._target_matrix(target)                        # (n_models x n_tasks)
        obs = np.isfinite(Y)
        if whiten:
            w = Whitener().fit(Y, obs)
            T = w.transform(Y)
        else:
            T = Y
        tcol = {t: j for j, t in enumerate(self.task_names_)}
        fams = np.array([fam(m) for m in names])

        def cell_pred(i, j, rows):
            """(score-scale, regression-scale) k-NN prediction for row i, column j."""
            mdl = KNeighborsRegressor(n_neighbors=min(k, len(rows)), weights='distance')
            rw = float(mdl.fit(Z[rows], T[rows, j]).predict(Z[i][None])[0])
            r = rw
            if whiten:
                R = np.full_like(Y, np.nan); R[i, j] = rw
                r = float(w.inverse_transform(R)[i, j])
            return r, rw

        cal_cache = {}

        def conformal_halfwidth(j, rows):
            """Finite-sample-corrected quantile of leave-one-reference-out |residuals|."""
            key = (j, rows.tobytes())
            if key not in cal_cache:
                res = []
                for i2 in rows:
                    tr2 = rows[rows != i2]
                    if holdout == 'family':
                        tr2 = tr2[fams[tr2] != fams[i2]]
                    if len(tr2) < 1:
                        continue
                    yhat, _ = cell_pred(i2, j, tr2)
                    if np.isfinite(yhat):
                        res.append(abs(yhat - Y[i2, j]))
                res = np.sort(res)
                rank = int(np.ceil((len(res) + 1) * interval))
                cal_cache[key] = float(res[rank - 1]) if rank <= len(res) else np.inf
            return cal_cache[key]

        preds, bounds = {}, {}
        for m, t in pairs:
            if t not in tcol:
                preds[(m, t)] = np.nan
                bounds[(m, t)] = (np.nan, np.nan)
                continue
            j = tcol[t]
            i = idx[m]
            tr = obs[:, j].copy()
            tr[i] = False
            if holdout == 'family':
                tr &= fams != fams[i]
            elif holdout != 'model':
                raise ValueError(f'unknown holdout: {holdout}')
            rows = np.where(tr)[0]
            support = len(rows) if holdout == 'family' else int(obs[:, j].sum())
            if len(rows) < 1 or support < min_train:
                preds[(m, t)] = np.nan
                bounds[(m, t)] = (np.nan, np.nan)
                continue
            r, _ = cell_pred(i, j, rows)
            preds[(m, t)] = r
            if interval is None:
                continue
            q = conformal_halfwidth(j, rows)
            lo, hi = r - q, r + q
            if whiten:  # the whitened pipeline predicts probabilities
                lo, hi = max(lo, 0.0), min(hi, 1.0)
            bounds[(m, t)] = (lo, hi)
        recs = pairs_to_records(pairs, preds)
        if interval is not None:
            for rec in recs:
                lo, hi = bounds[(rec['model_id'], rec['task_id'])]
                rec['score_lo'] = float(lo)
                rec['score_hi'] = float(hi)
        return recs

    def score_table(self, **predict_kwargs):
        """Every (model, task) cell as one tidy row: the observed sample score where the
        cell has responses, the model's prediction where it does not.

        Returns a DataFrame with columns model_id, task_id, score, source
        ('observed' | 'predicted'), and suite (when the fit records carried one).
        predict_kwargs are forwarded to predict() for the missing cells (k, holdout,
        whiten, ...). Filter by model_id for one model's row, by task_id (or suite)
        for one benchmark's column.
        """
        assert self._fitted, 'call fit() before score_table()'
        S = self.sample_scores_
        rows = [{'model_id': m, 'task_id': t, 'score': float(S.at[m, t]), 'source': 'observed'}
                for m in self.model_names_ for t in self.task_names_
                if pd.notna(S.at[m, t])]
        rows += [{'model_id': r['model_id'], 'task_id': r['task_id'],
                  'score': r['score_hat'], 'source': 'predicted',
                  **{kk: r[kk] for kk in ('score_lo', 'score_hi') if kk in r}}
                 for r in self.predict(None, **predict_kwargs)]
        df = pd.DataFrame(rows)
        if self.task_suites_:
            df['suite'] = df['task_id'].map(self.task_suites_)
        return df.sort_values(['model_id', 'task_id']).reset_index(drop=True)

    # ------------------------------------------------------------------
    # ingestion
    # ------------------------------------------------------------------
    def _needs_query_channel(self):
        return self.query_kwargs['kernel'] != 'delta'

    def _ingest(self, records):
        df = parse_records(records, required=('model_id', 'query_id'),
                           aliases=[('embedding', 'response_embedding')])
        if 'task_id' not in df.columns:
            df['task_id'] = '_task'
        for col in ('score', 'reference_score', 'sample_score'):
            if col not in df.columns:
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'response_embedding' not in df.columns:
            if 'response' not in df.columns:
                raise ValueError("records need 'response_embedding' vectors or 'response' text")
            df['response_embedding'] = list(self._embed_texts(df['response'].tolist()))
        if self._needs_query_channel() and 'query_embedding' not in df.columns:
            if 'query' not in df.columns:
                raise ValueError("records need 'query_embedding' vectors or 'query' text "
                                 "(query kernel is not 'delta')")
            df['query_embedding'] = list(self._embed_texts(df['query'].tolist()))

        keep = ['model_id', 'task_id', 'query_id', 'response_embedding', 'score',
                'reference_score', 'sample_score']
        if 'query_embedding' in df.columns:
            keep.append('query_embedding')
        if 'suite' in df.columns:
            keep.append('suite')
        df = df[keep].copy()
        assert not df.duplicated(['model_id', 'task_id', 'query_id']).any(), \
            'duplicate (model_id, task_id, query_id) rows'
        return df

    def _embed_texts(self, texts):
        """Embed texts through dkps.embed, deduplicating identical strings."""
        from .embed import embed_api
        provider = self.embedding_kwargs['provider']
        api_key = self.embedding_kwargs.get('api_key')
        if api_key and provider in _PROVIDER_ENV:
            os.environ[_PROVIDER_ENV[provider]] = api_key
        uniq = list(dict.fromkeys(texts))
        kwargs = {}
        if self.embedding_kwargs.get('model'):
            kwargs['model'] = self.embedding_kwargs['model']
        E = embed_api(provider, uniq, **kwargs)
        lookup = {t: E[i] for i, t in enumerate(uniq)}
        return [lookup[t] for t in texts]

    # ------------------------------------------------------------------
    # reduction + kernels
    # ------------------------------------------------------------------
    def _fit_reducers(self, df):
        """Learn the per-channel reducers and resolve the kernel bandwidths. Both are
        frozen afterwards so that update() can project new rows into the same space.
        With a 'suite' column the response channel is reduced per suite and placed in
        disjoint unit-normalized blocks (BlockDiagonalEmbedding); update() then only
        accepts suites seen at fit time."""
        R = np.stack([np.asarray(e, dtype=np.float64) for e in df['response_embedding']])
        if 'suite' in df.columns and self.response_kwargs.get('suite_mode') == 'blocked':
            # ablation: per-suite PCA in disjoint blocks (cross-suite k_R exactly 0)
            dim = self.response_kwargs['pca_dim']
            self._response_pca = _BlockReducer(
                BlockDiagonalEmbedding(reduce_dim=dim if isinstance(dim, int) else 48,
                                       n_elbows=self.response_kwargs['pca_n_elbows']
                                       ).fit(R, df['suite'].to_numpy()))
        elif 'suite' in df.columns:
            # base method: one shared space, unit-normalized; the query kernel alone
            # gates cross-suite comparisons (PKPS's information flow across collections).
            # Its dimension matches what the per-suite construction would retain in
            # total (second Zhu-Ghodsi elbow per suite, cap 48) -- capacity-matched to
            # the blocked ablation by definition; an int pca_dim overrides.
            dim = self.response_kwargs['pca_dim']
            if not isinstance(dim, int):
                suites = df['suite'].to_numpy()
                dim = sum(FrozenPCA('elbow', self.response_kwargs['pca_n_elbows'],
                                    max_components=48).fit(R[suites == s])
                          .components_.shape[0] for s in np.unique(suites))
            self._response_pca = _JointNormReducer(FrozenPCA(dim).fit(R))
        else:
            self._response_pca = FrozenPCA(self.response_kwargs['pca_dim'],
                                           self.response_kwargs['pca_n_elbows']).fit(R)
        self._response_bandwidth = None
        if self.response_kwargs['kernel'] == 'rbf':
            bw = self.response_kwargs['bandwidth']
            if bw == 'cv':
                raise ValueError("bandwidth='cv' is only supported on the query channel")
            if isinstance(bw, (int, float)):
                self._response_bandwidth = float(bw)
            else:
                Rr = self._response_pca.transform(R, df['suite'].to_numpy()) \
                    if isinstance(self._response_pca, (_BlockReducer, _JointNormReducer)) \
                    else self._response_pca.transform(R)
                self._response_bandwidth = self._scale(self.response_kwargs, Rr)

        self._query_pca = None
        self._sigmas = [None]
        if 'query_embedding' in df.columns:
            Q = np.stack([np.asarray(e, dtype=np.float64) for e in df['query_embedding']])
            self._query_pca = FrozenPCA(self.query_kwargs['pca_dim'],
                                        self.query_kwargs['pca_n_elbows']).fit(Q)
            if self.query_kwargs['kernel'] == 'rbf':
                bw = self.query_kwargs['bandwidth']
                if isinstance(bw, (int, float)):
                    self._sigmas = [float(bw)]
                else:
                    ref = self._scale(self.query_kwargs, self._query_pca.transform(Q))
                    if bw == 'median':
                        self._sigmas = [ref]
                    elif bw == 'cv':
                        self._sigmas = [float(ref * g) for g in self.query_kwargs['bandwidth_grid']]
                    else:
                        raise ValueError(f'unknown query bandwidth: {bw}')

    @staticmethod
    def _scale(spec, X):
        """Reference bandwidth scale: bandwidth_mult x (bandwidth_ref or median distance)."""
        ref = spec['bandwidth_ref']
        if ref is None:
            ref = _median_distance(X)
        elif callable(ref):
            ref = float(ref(X))
        return float(spec['bandwidth_mult']) * float(ref)

    def _reduce_model(self, sub):
        sub = sub.sort_values(['task_id', 'query_id'])
        X = np.stack([np.asarray(e, dtype=np.float64) for e in sub['response_embedding']])
        if isinstance(self._response_pca, (_BlockReducer, _JointNormReducer)):
            Xr = self._response_pca.transform(X, sub['suite'].to_numpy())
        else:
            Xr = self._response_pca.transform(X)
        md = {'query_ids': sub['query_id'].values, 'X': Xr,
              'qkeys': list(zip(sub['task_id'].values, sub['query_id'].values))}
        if self._query_pca is not None:
            Q = np.stack([np.asarray(e, dtype=np.float64) for e in sub['query_embedding']])
            md['Q'] = self._query_pca.transform(Q)
        return md

    def _affinities(self, a, b):
        """{sigma: (S_ab, Z_ab)} for every candidate query bandwidth, with S = sum k_Q k_R
        and Z = sum k_Q. The response kernel and the query squared distances are
        bandwidth-independent, so they are computed once."""
        K_R = _kernel_matrix(self.response_kwargs['kernel'], self._response_bandwidth,
                             a['X'], b['X'])
        qk = self.query_kwargs['kernel']
        if qk == 'rbf':
            q_sq = cdist(a['Q'], b['Q'], 'sqeuclidean')
            out = {}
            for s in self._sigmas:
                K_Q = np.exp(-q_sq / (2.0 * s * s))
                out[s] = (float((K_Q * K_R).sum()), float(K_Q.sum()))
            return out
        K_Q = _kernel_matrix(qk, None, a.get('Q'), b.get('Q'), a['query_ids'], b['query_ids'])
        return {None: (float((K_Q * K_R).sum()), float(K_Q.sum()))}

    def _recompute_pairs(self, touched):
        if self.query_kwargs['pooling'] or self.query_kwargs['normalization'] == 'local':
            return                              # these distances are rebuilt in _finalize
        self._recompute_pair_cache(touched)

    def _recompute_pair_cache(self, touched):
        """Refresh the cached affinities A[sigma][(m1, m2)] for every pair that involves a
        touched model (or is missing); untouched pairs keep their cached values, which is
        what makes update() incremental."""
        names = sorted(self._model_data)
        A0 = self._A[self._sigmas[0]]
        for i, m1 in enumerate(names):
            for m2 in names[i:]:
                if m1 in touched or m2 in touched or (m1, m2) not in A0:
                    for s, v in self._affinities(self._model_data[m1], self._model_data[m2]).items():
                        self._A[s][(m1, m2)] = v
        for s in self._sigmas:
            for p in [p for p in self._A[s] if p[0] not in self._model_data
                      or p[1] not in self._model_data]:
                del self._A[s][p]

    def _dist_matrix(self, s, names):
        """Model distance matrix at query bandwidth s.

        normalization='model': A_ab = S_ab / sqrt(Z_aa Z_bb) is a Gram matrix for
        positive-semidefinite kernels, so D_ab^2 = A_aa + A_bb - 2 A_ab >= 0 and nothing
        is clipped (floating-point roundoff aside). normalization='pair': A_ab =
        S_ab / Z_ab, clipped at zero; a pair with zero query-kernel mass has no defined
        distance and is set to the largest observed distance so the MDS still runs."""
        if self.query_kwargs['pooling']:
            return self._pooled_dist_matrix(s, names)
        if self.query_kwargs['normalization'] == 'local':
            return self._fill_undefined(_sqrt_nonneg(self._local_d2[s]))
        n = len(names)
        SZ = self._A[s]
        model_norm = self.query_kwargs['normalization'] == 'model'
        S = np.zeros((n, n)); Zm = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                S[i, j], Zm[i, j] = S[j, i], Zm[j, i] = SZ[(names[i], names[j])]
        zd = np.diag(Zm)
        with np.errstate(divide='ignore', invalid='ignore'):
            if model_norm:
                A = S / np.sqrt(np.outer(zd, zd))
            else:
                A = S / Zm
        A[~np.isfinite(A)] = np.nan
        D2 = np.diag(A)[:, None] + np.diag(A)[None, :] - 2 * A
        np.fill_diagonal(D2, 0.0)
        psd = (model_norm and self.query_kwargs['kernel'] in ('rbf', 'delta')
               and self.response_kwargs['kernel'] in ('linear', 'rbf'))
        if psd:
            fin = D2[np.isfinite(D2)]
            tol = 1e-9 * max(1.0, float(np.abs(fin).max()) if fin.size else 1.0)
            assert not (fin < -tol).any(), 'negative squared distance under model normalization'
        with np.errstate(invalid='ignore'):
            D = np.sqrt(np.where(np.isfinite(D2), np.maximum(D2, 0.0), np.nan))
        return self._fill_undefined(D)

    @staticmethod
    def _fill_undefined(D):
        if np.isnan(D).any():
            mx = np.nanmax(D) if np.isfinite(D).any() else np.nan
            if not np.isfinite(mx):
                raise ValueError('all PKPS distances undefined (zero query-kernel mass); '
                                 'increase the query bandwidth')
            warnings.warn('model pairs with zero query-kernel mass: their distance is set to '
                          'the maximum observed distance', stacklevel=4)
            D = np.nan_to_num(D, nan=mx)
        return D

    def _stack(self, names):
        """Row-stacked responses with model and unique-query indices (and the unique
        query embeddings, if any) for the representations built from all models at once."""
        md = self._model_data
        X = np.vstack([md[m]['X'] for m in names])
        mid = np.concatenate([np.full(len(md[m]['X']), i) for i, m in enumerate(names)])
        index = {}
        qidx = np.array([index.setdefault(k, len(index)) for m in names for k in md[m]['qkeys']])
        first = np.full(len(index), -1)
        first[qidx[::-1]] = np.arange(len(qidx))[::-1]
        Q = np.vstack([md[m]['Q'] for m in names])[first] if 'Q' in md[names[0]] else None
        return X, mid, qidx, Q

    def _pooled_dist_matrix(self, s, names, chunk=1500):
        """Euclidean distances between per-model means of query-residualized responses.

        For a response x of model a to query u, the residual is x - m_{-a}(u), where
        m_{-a}(u) is the query-kernel-weighted (Nadaraya-Watson) mean response of every
        other model to queries near u; where no other model has kernel mass, the overall
        mean response is used. The representation is an explicit vector per model, so
        the distances are Euclidean by construction."""
        X, mid, qidx, Qk = self._stack(names)
        U, p = int(qidx.max()) + 1, X.shape[1]
        Ssum = np.zeros((U, p)); np.add.at(Ssum, qidx, X)
        cnt = np.bincount(qidx, minlength=U).astype(float)

        if self.query_kwargs['kernel'] == 'rbf':
            q2 = (Qk ** 2).sum(1)
            inv = 1.0 / (2.0 * s * s)

            def kern(rows, cols=None):
                Qc = Qk if cols is None else Qk[cols]
                q2c = q2 if cols is None else q2[cols]
                return np.exp(-np.maximum(q2[rows, None] + q2c[None] - 2.0 * Qk[rows] @ Qc.T,
                                          0.0) * inv)
            Num = np.empty((U, p)); Den = np.empty(U)
            for a in range(0, U, chunk):
                Kc = kern(np.arange(a, min(a + chunk, U)))
                Num[a:a + chunk] = Kc @ Ssum; Den[a:a + chunk] = Kc @ cnt
        else:                                   # delta: pool over identical queries only
            kern = lambda rows, cols: np.eye(len(rows))
            Num, Den = Ssum, cnt

        gmean = X.mean(0)
        reps = np.zeros((len(names), p))
        for i in range(len(names)):
            rows = np.where(mid == i)[0]
            uq, loc = np.unique(qidx[rows], return_inverse=True)
            Si = np.zeros((len(uq), p)); np.add.at(Si, loc, X[rows])
            ci = np.bincount(loc, minlength=len(uq)).astype(float)
            Kii = kern(uq, uq)
            num, den = Num[uq] - Kii @ Si, Den[uq] - Kii @ ci
            ok = den > 1e-8
            mhat = np.where(ok[:, None], num / np.where(ok, den, 1.0)[:, None], gmean)
            reps[i] = (X[rows] - mhat[loc]).mean(0)
        self._pooled_reps = {**getattr(self, '_pooled_reps', {}), s: reps}
        return cdist(reps, reps)

    def _mds(self, D):
        """Classical MDS of the distance matrix; dim is capped at n-1 (fewer than 3 models:
        the first distance column is returned as a 1-d coordinate)."""
        n = len(D)
        if n < 3:
            return D[:, :1]
        dim = self.mds_kwargs['dim']
        return ClassicalMDS(n_components=None if dim is None else min(int(dim), n - 1),
                            n_elbows=self.mds_kwargs['n_elbows'],
                            dissimilarity='precomputed').fit_transform(D)

    def _finalize(self):
        """Distances + MDS at every candidate bandwidth, bandwidth selection, score tables."""
        names = sorted(self._model_data)
        self.model_names_ = names
        table = ScoreTable(self._raw, models=names)
        self.sample_scores_ = table.sample_
        self.task_names_ = table.tasks
        if 'suite' in self._raw.columns:
            ts = self._raw.dropna(subset=['suite']).groupby('task_id')['suite'].first()
            self.task_suites_ = ts.to_dict()
            self.suite_names_ = sorted(set(ts))
        else:
            self.task_suites_ = {}
            self.suite_names_ = []
        ref = self._raw.dropna(subset=['reference_score']).groupby(['model_id', 'task_id'])[
            'reference_score'].first().unstack()
        self.reference_scores_ = ref.reindex(index=names, columns=self.task_names_)

        if self.query_kwargs['normalization'] == 'local' and not self.query_kwargs['pooling']:
            X, mid, qidx, Q = self._stack(names)
            rbf = self.query_kwargs['kernel'] == 'rbf'
            self._local_d2 = local_dist_sq(X, mid, len(names), qidx, Q=Q if rbf else None,
                                           sigmas=self._sigmas)
        Ds = {s: self._dist_matrix(s, names) for s in self._sigmas}
        Zs = {s: self._mds(D) for s, D in Ds.items()}
        if len(self._sigmas) > 1:                                   # CV bandwidth selection
            Y = self.sample_scores_.to_numpy(dtype=float)
            k = int(self.query_kwargs['cv_k'])
            best_s, best_e = None, np.inf
            for s in self._sigmas:
                errs = []
                for j in range(Y.shape[1]):
                    rows = np.where(np.isfinite(Y[:, j]))[0]
                    if len(rows) < k + 1:
                        continue
                    errs += _lomo_knn_error(Zs[s][rows], Y[rows, j], k)
                e = float(np.mean(errs)) if errs else np.inf
                if e < best_e:
                    best_e, best_s = e, s
            if best_s is None:                                      # no task had enough models
                best_s = self._sigmas[len(self._sigmas) // 2]
        else:
            best_s = self._sigmas[0]
        self.query_bandwidth_ = best_s
        self.dist_matrix_ = Ds[best_s]
        self.embedding_ = {m: Zs[best_s][i] for i, m in enumerate(names)}
        self.embeddings_by_bandwidth_ = {s: {m: Z[i] for i, m in enumerate(names)}
                                         for s, Z in Zs.items()}

    def _target_matrix(self, target):
        """(n_models x n_tasks) regression targets: sample scores, reference scores, or
        reference-where-available ('auto')."""
        S = self.sample_scores_.to_numpy(dtype=float)
        R = self.reference_scores_.to_numpy(dtype=float)
        if target == 'sample':
            return S
        if target == 'reference':
            return R
        if target == 'auto':
            return np.where(np.isfinite(R), R, S)
        raise ValueError(f'unknown target: {target}')

    def _parse_pairs(self, records):
        S = self.sample_scores_
        default = ((m, t) for m in self.model_names_ for t in self.task_names_
                   if pd.isna(S.at[m, t]))
        return parse_pairs(records, self.model_names_, self.task_names_, default)


class DKPS(PKPS):
    """Paired special case of PKPS.

    By default the query kernel is the delta kernel on query ids, so responses are
    compared only across identical queries; on a fully paired design with a linear
    response kernel this recovers the standard DKPS distance (per-query mean squared
    distance), and queries never need embedding. Passing sigma_ratio (e.g. 0.01, the
    paper's "delta limit") instead uses an RBF query kernel at sigma_ratio x the
    reference query scale, which requires query embeddings.
    """

    def __init__(self, response_kwargs=None, mds_kwargs=None, embedding_kwargs=None,
                 sigma_ratio=None, bandwidth_ref=None, normalization=None):
        if sigma_ratio is None:
            qk = dict(kernel='delta', pca_dim=None, normalization=normalization)
        else:
            qk = dict(kernel='rbf', bandwidth='median', bandwidth_mult=float(sigma_ratio),
                      bandwidth_ref=bandwidth_ref, pca_dim=None, normalization=normalization)
        super().__init__(query_kwargs=qk, response_kwargs=response_kwargs,
                         mds_kwargs=mds_kwargs, embedding_kwargs=embedding_kwargs)
