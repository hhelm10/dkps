import warnings

import numpy as np
import pandas as pd
from graspologic.embed import ClassicalMDS, select_dimension
from scipy.spatial.distance import cdist, pdist
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity


class UnpairedDKPS:
    def __init__(
            self,
            mode='combined',
            query_kernel='constant',
            use_coverage=False,
            coverage_mode='oracle',
            paired_weight=None,
            query_bandwidth='median',
            coverage_bandwidth='scott',
            coverage_pca_components=None,
            n_components_cmds=None,
            n_elbows_cmds=2,
            dissimilarity="precomputed",
        ):

        assert mode in {'paired', 'unpaired', 'combined'}, \
            "mode must be one of {'paired', 'unpaired', 'combined'}"
        assert query_kernel in {'constant', 'rbf'}, \
            "query_kernel must be one of {'constant', 'rbf'}"
        assert coverage_mode in {'oracle', 'pca'}, \
            "coverage_mode must be one of {'oracle', 'pca'}"
        if paired_weight is not None:
            assert 0.0 <= paired_weight <= 1.0, 'paired_weight must be in [0, 1]'

        self.mode = mode
        self.query_kernel = query_kernel
        self.use_coverage = use_coverage
        self.coverage_mode = coverage_mode
        self.paired_weight = paired_weight
        self.query_bandwidth = query_bandwidth
        self.coverage_bandwidth = coverage_bandwidth
        self.coverage_pca_components = coverage_pca_components
        self.n_components_cmds = n_components_cmds
        self.n_elbows_cmds = n_elbows_cmds
        self.dissimilarity = dissimilarity

    def _validate_input_dataframe(self, data):
        assert isinstance(data, pd.DataFrame), 'data must be a DataFrame'
        for col in ('model_id', 'query_id', 'embedding', 'query_vec'):
            assert col in data.columns, f'DataFrame must have column: {col}'

        df = data.copy()
        duplicate_pairs = df.duplicated(['model_id', 'query_id'], keep=False)
        assert not duplicate_pairs.any(), 'duplicate (model_id, query_id) rows are not supported'

        embedding_shapes = {np.asarray(emb).shape for emb in df['embedding'].values}
        query_shapes = {np.asarray(q).shape for q in df['query_vec'].values}

        assert len(embedding_shapes) == 1, 'all embeddings must have the same shape'
        assert len(query_shapes) == 1, 'all query_vec values must have the same shape'

        embedding_shape = next(iter(embedding_shapes))
        query_shape = next(iter(query_shapes))
        assert len(embedding_shape) == 1, 'each embedding must be a 1D vector'
        assert len(query_shape) == 1, 'each query_vec must be a 1D vector'
        assert query_shape[0] <= embedding_shape[0], \
            'query_vec dimensionality must not exceed embedding dimensionality'

        model_names = sorted(df['model_id'].unique())
        assert len(model_names) >= 2, 'at least two models are required'

        df['query_code'] = pd.factorize(df['query_id'])[0]

        return df, model_names, embedding_shape[0], query_shape[0]

    def _resolve_bandwidth(self, X, bandwidth, context='bandwidth'):
        if callable(bandwidth):
            value = float(bandwidth(X))
        elif isinstance(bandwidth, (int, float)):
            value = float(bandwidth)
        elif bandwidth == 'scott':
            n, d = X.shape
            scale = float(np.mean(np.std(X, axis=0, ddof=0)))
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
            value = scale * (max(n, 1) ** (-1.0 / (d + 4)))
        elif bandwidth == 'median':
            if len(X) < 2:
                value = 1.0
            else:
                dists = pdist(X)
                positive = dists[dists > 0]
                value = float(np.median(positive)) if len(positive) else 1.0
        else:
            raise ValueError(f'unsupported {context}: {bandwidth}')

        if not np.isfinite(value) or value <= 0:
            value = 1.0
        return value

    def _make_query_features(self, df, embedding_dim, query_dim):
        if self.coverage_mode == 'oracle':
            # Block I synthetic experiments can use the true latent query vectors directly.
            self.query_feature_transformer_ = None
            self.coverage_pca_components_ = None
            return np.stack(df['query_vec'].values)

        X = np.stack(df['embedding'].values)
        # The plan's coverage construction uses a low-dimensional PCA space derived
        # from responses. If the caller does not force a dimension, choose one from
        # the scree/elbow rule rather than assuming it equals the latent query dim.
        max_components = min(X.shape[0], X.shape[1])
        if self.coverage_pca_components is None:
            centered = X - np.mean(X, axis=0, keepdims=True)
            elbows, _ = select_dimension(centered, n_elbows=self.n_elbows_cmds)
            n_components = int(elbows[-1]) if elbows else min(query_dim, max_components)
        else:
            n_components = int(self.coverage_pca_components)
        
        n_components = max(1, min(n_components, max_components))

        pca = PCA(n_components=n_components)
        pca.fit(X)

        padded_queries = np.zeros((len(df), embedding_dim), dtype=float)
        padded_queries[:, :query_dim] = np.stack(df['query_vec'].values)
        self.query_feature_transformer_ = pca
        self.coverage_pca_components_ = n_components
        return pca.transform(padded_queries)

    def _prepare_model_data(self, df, model_names):
        model_data = {}
        for model_name in model_names:
            sub = df[df['model_id'] == model_name].sort_values('query_code')
            model_data[model_name] = {
                'embeddings'     : np.stack(sub['embedding'].values),
                'query_features' : np.stack(sub['query_feature'].values),
                'query_codes'    : sub['query_code'].to_numpy(),
                'query_code_set' : set(sub['query_code'].to_numpy().tolist()),
            }
        return model_data

    def _infer_paired_weight(self, model_data):
        if self.mode == 'paired':
            return 1.0
        if self.mode == 'unpaired':
            return 0.0
        if self.paired_weight is not None:
            return float(self.paired_weight)

        # This is the empirical analogue of Section 3's optimal combination weight
        # alpha* for the composed kernel kappa_{alpha,w}. In the Block I synthetic
        # setup, alpha* = m_p / (m_p + m_u), so estimating it from the fraction of
        # each model's observed query budget that is globally shared recovers the
        # realized paired fraction.
        shared_query_codes = set.intersection(*(xx['query_code_set'] for xx in model_data.values()))
        avg_queries_per_model = float(np.mean([len(xx['query_codes']) for xx in model_data.values()]))
        if avg_queries_per_model == 0:
            return 0.0
        return len(shared_query_codes) / avg_queries_per_model

    def _fit_coverage_models(self, model_data):
        if not self.use_coverage:
            return None, None

        # Section 2's coverage adjustment uses one density estimate per model in
        # query space. Pairwise weights later take a harmonic mean of these KDEs,
        # so regions uncovered by either model are downweighted.
        all_query_features = np.vstack([xx['query_features'] for xx in model_data.values()])
        coverage_bandwidth = self._resolve_bandwidth(
            all_query_features,
            self.coverage_bandwidth,
            context='coverage_bandwidth',
        )

        coverage_models = {}
        for model_name, model_entry in model_data.items():
            kde = KernelDensity(kernel='gaussian', bandwidth=coverage_bandwidth)
            kde.fit(model_entry['query_features'])
            coverage_models[model_name] = kde
        return coverage_models, coverage_bandwidth

    def _evaluate_density(self, coverage_models, model_name, query_features):
        if not self.use_coverage:
            return np.ones(len(query_features), dtype=float)
        log_density = coverage_models[model_name].score_samples(query_features)
        return np.exp(log_density)

    def _resolve_query_bandwidth(self, model_i, model_k, query_features_i, query_features_k):
        if callable(self.query_bandwidth):
            value = float(self.query_bandwidth(model_i, model_k, self))
            if not np.isfinite(value) or value <= 0:
                value = 1.0
            return value

        pooled = np.vstack([query_features_i, query_features_k])
        return self._resolve_bandwidth(pooled, self.query_bandwidth, context='query_bandwidth')

    def _paired_indicator_matrix(self, query_codes_a, query_codes_b, shared_query_codes):
        """
        Build the paired-query indicator used for the 1[q=q'] component of
        kappa_{alpha,w}.

        Only query ids that are shared across the two models contribute to the
        paired term. For those shared ids, the matrix is 1 when the two entries
        correspond to the same query and 0 otherwise.

        Example
        -------
        If model i has query codes [10, 11, 20], model k has [11, 20, 30], and
        the shared query ids are {11, 20}, then the paired indicator is

            [[0, 0, 0],
             [1, 0, 0],
             [0, 1, 0]]

        because only the shared queries 11 and 20 contribute, and each shared
        query only pairs with itself.
        """
        if not shared_query_codes:
            return np.zeros((len(query_codes_a), len(query_codes_b)), dtype=float)
        shared_a = np.isin(query_codes_a, list(shared_query_codes))
        shared_b = np.isin(query_codes_b, list(shared_query_codes))
        equal = query_codes_a[:, None] == query_codes_b[None, :]
        return (equal & shared_a[:, None] & shared_b[None, :]).astype(float)

    def _unpaired_query_kernel_matrix(
            self,
            density_model_a,
            density_model_b,
            query_features_a,
            query_features_b,
            coverage_models,
        ):
        """
        Build the unpaired query-kernel block between two query collections.

        This is the w(i, k, q, q') / kappa^*(q, q') part of the estimator from
        the plan, evaluated on two finite sets of query features:

        - if `query_kernel == "constant"`, every query pair starts with weight 1
          (linear MMD over unpaired queries)
        - if `query_kernel == "rbf"`, query pairs are additionally weighted by
          similarity in query space
        - if `use_coverage` is enabled, the base kernel is multiplied by the
          harmonic-mean coverage weight so regions poorly covered by either
          model contribute less

        The same helper is used for both self terms (`ii`, `kk`) and the cross
        term (`ik`). The `density_model_*` arguments specify which pair of KDEs
        define the coverage weight; `query_features_a` and `query_features_b`
        specify where that pairwise kernel is evaluated.

        For example, when called on model i's queries against model k's queries,
        entry (j, j') is the weight assigned to comparing query j from model i
        with query j' from model k. When called for the `ii` self term of the
        `(i, k)` comparison, both query collections come from model i, but the
        coverage weight still uses model i's KDE and model k's KDE so all three
        blocks (`ii`, `kk`, `ik`) share the same pair-specific kernel.
        """
        if self.query_kernel == 'constant':
            kernel_matrix = np.ones((len(query_features_a), len(query_features_b)), dtype=float)
        elif self.query_kernel == 'rbf':
            bandwidth = self._resolve_query_bandwidth(
                density_model_a,
                density_model_b,
                query_features_a,
                query_features_b,
            )
            sq_dists = cdist(query_features_a, query_features_b, metric='sqeuclidean')
            kernel_matrix = np.exp(-sq_dists / (2.0 * bandwidth ** 2))
        else:
            raise ValueError(f'unsupported query_kernel: {self.query_kernel}')

        if not self.use_coverage:
            return kernel_matrix

        # Multiply the base query kernel by the coverage weight w(i, k, q, q').
        # This matches the identifiability correction in the plan: if either model
        # assigns low density to the relevant query region, that pair gets little
        # influence on the estimated distance.
        density_a = self._evaluate_density(coverage_models, density_model_a, query_features_a)
        density_b = self._evaluate_density(coverage_models, density_model_b, query_features_b)
        denom = density_a[:, None] + density_b[None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            coverage = np.where(
                denom > 0,
                2.0 * density_a[:, None] * density_b[None, :] / denom,
                0.0,
            )
        return kernel_matrix * coverage

    def _pair_query_kernels(self, model_i, model_k, data_i, data_k, paired_weight, coverage_models):
        # Compose the paired indicator and the unpaired/coverage-adjusted kernel
        # exactly as in kappa_{alpha,w}. The paired term captures shared-query
        # variance reduction; the unpaired term captures the remaining signal.
        unpaired_weight = 1.0 - paired_weight

        shared_query_codes = data_i['query_code_set'] & data_k['query_code_set']

        kernels = {}
        if paired_weight > 0:
            kernels['ii_paired'] = self._paired_indicator_matrix(
                data_i['query_codes'],
                data_i['query_codes'],
                shared_query_codes,
            )
            kernels['kk_paired'] = self._paired_indicator_matrix(
                data_k['query_codes'],
                data_k['query_codes'],
                shared_query_codes,
            )
            kernels['ik_paired'] = self._paired_indicator_matrix(
                data_i['query_codes'],
                data_k['query_codes'],
                shared_query_codes,
            )
        else:
            kernels['ii_paired'] = np.zeros((len(data_i['query_codes']), len(data_i['query_codes'])))
            kernels['kk_paired'] = np.zeros((len(data_k['query_codes']), len(data_k['query_codes'])))
            kernels['ik_paired'] = np.zeros((len(data_i['query_codes']), len(data_k['query_codes'])))

        if unpaired_weight > 0:
            kernels['ii_unpaired'] = self._unpaired_query_kernel_matrix(
                density_model_a   = model_i,
                density_model_b   = model_k,
                query_features_a  = data_i['query_features'],
                query_features_b  = data_i['query_features'],
                coverage_models   = coverage_models,
            )
            kernels['kk_unpaired'] = self._unpaired_query_kernel_matrix(
                density_model_a   = model_k,
                density_model_b   = model_i,
                query_features_a  = data_k['query_features'],
                query_features_b  = data_k['query_features'],
                coverage_models   = coverage_models,
            )
            kernels['ik_unpaired'] = self._unpaired_query_kernel_matrix(
                density_model_a   = model_i,
                density_model_b   = model_k,
                query_features_a  = data_i['query_features'],
                query_features_b  = data_k['query_features'],
                coverage_models   = coverage_models,
            )
        else:
            kernels['ii_unpaired'] = np.zeros((len(data_i['query_codes']), len(data_i['query_codes'])))
            kernels['kk_unpaired'] = np.zeros((len(data_k['query_codes']), len(data_k['query_codes'])))
            kernels['ik_unpaired'] = np.zeros((len(data_i['query_codes']), len(data_k['query_codes'])))

        return {
            'ii': paired_weight * kernels['ii_paired'] + unpaired_weight * kernels['ii_unpaired'],
            'kk': paired_weight * kernels['kk_paired'] + unpaired_weight * kernels['kk_unpaired'],
            'ik': paired_weight * kernels['ik_paired'] + unpaired_weight * kernels['ik_unpaired'],
        }

    def _weighted_linear_average(self, X_a, X_b, query_kernel_matrix):
        normalizer = np.sum(query_kernel_matrix)
        if normalizer <= 0:
            return np.nan
        return float(np.sum((query_kernel_matrix @ X_b) * X_a) / normalizer)

    def _pair_distance(self, model_i, model_k, data_i, data_k, paired_weight, coverage_models):
        query_kernels = self._pair_query_kernels(
            model_i,
            model_k,
            data_i,
            data_k,
            paired_weight,
            coverage_models,
        )

        term_ii = self._weighted_linear_average(
            data_i['embeddings'],
            data_i['embeddings'],
            query_kernels['ii'],
        )
        term_kk = self._weighted_linear_average(
            data_k['embeddings'],
            data_k['embeddings'],
            query_kernels['kk'],
        )
        term_ik = self._weighted_linear_average(
            data_i['embeddings'],
            data_k['embeddings'],
            query_kernels['ik'],
        )

        if not np.isfinite(term_ii) or not np.isfinite(term_kk) or not np.isfinite(term_ik):
            return np.nan

        # This is the linear-MMD form of the distance under the composed query
        # kernel:
        #   E[kappa <x_i, x_i>] + E[kappa <x_k, x_k>] - 2 E[kappa <x_i, x_k>].
        # NOTE: The plan writes the linear-MMD expression with the opposite sign.
        # We intentionally use the standard positive self + self - 2 * cross form
        # so the quantity is a squared distance and the paired linear case reduces
        # to the original DKPS distance.
        dist_sq = term_ii + term_kk - 2.0 * term_ik
        if dist_sq < 0 and abs(dist_sq) < 1e-10:
            dist_sq = 0.0
        return float(np.sqrt(max(dist_sq, 0.0)))

    def _compute_dist_matrix(self, model_names, model_data, paired_weight, coverage_models):
        # Evaluate the composed estimator for every model pair. This produces the
        # dissimilarity matrix that ClassicalMDS embeds downstream.
        
        n_models    = len(model_names)
        dist_matrix = np.zeros((n_models, n_models), dtype=float)
        
        for i, model_i in enumerate(model_names):
            for k in range(i + 1, n_models):
                model_k = model_names[k]
                
                dist_matrix[i, k] = dist_matrix[k, i] = self._pair_distance(
                    model_i         = model_i,
                    model_k         = model_k,
                    data_i          = model_data[model_i],
                    data_k          = model_data[model_k],
                    paired_weight   = paired_weight,
                    coverage_models = coverage_models,
                )
        
        return dist_matrix

    def fit(self, data):
        df, model_names, embedding_dim, query_dim = self._validate_input_dataframe(data)

        # Step 1: choose the query representation used by coverage adjustment.
        # In Block I this is either the oracle latent query space or the PCA surrogate derived from responses.
        query_features = self._make_query_features(df, embedding_dim, query_dim)
        df['query_feature'] = list(query_features)

        # Step 2: reorganize the flat DataFrame into per-model arrays so the
        # estimator can work directly with embeddings, query features, and ids.
        model_data = self._prepare_model_data(df, model_names)

        # Step 3: infer the paired fraction alpha used by kappa_{alpha,w}.
        paired_weight = self._infer_paired_weight(model_data)

        if len({len(xx['query_codes']) for xx in model_data.values()}) > 1:
            warnings.warn(
                'models have differing numbers of queries; paired_weight uses the average per-model query count',
                stacklevel=2,
            )

        # Step 4: fit per-model KDEs if coverage adjustment is enabled.
        coverage_models, coverage_bandwidth = self._fit_coverage_models(model_data)

        # Step 5: evaluate the pairwise model distances under the composed kernel.
        dist_matrix = self._compute_dist_matrix(
            model_names,
            model_data,
            paired_weight,
            coverage_models,
        )

        # Persist the fitted artifacts after the local computation is complete so
        # the data flow through fit() stays explicit rather than side-effectful.
        self.model_names_         = model_names
        self.embedding_dim_       = embedding_dim
        self.query_dim_           = query_dim
        self.query_feature_dim_   = query_features.shape[1]
        self.model_data_          = model_data
        self.shared_query_codes_  = set.intersection(*(xx['query_code_set'] for xx in model_data.values()))
        self.paired_weight_       = paired_weight
        self.coverage_models_     = coverage_models
        self.coverage_bandwidth_  = coverage_bandwidth
        self.dist_matrix_         = dist_matrix
        return self

    def fit_transform(self, data):
        self.fit(data)
        if not np.isfinite(self.dist_matrix_).all():
            raise ValueError('dist_matrix_ contains non-finite values; cannot run ClassicalMDS')

        cmds_embds = ClassicalMDS(
            n_components=self.n_components_cmds,
            n_elbows=self.n_elbows_cmds,
            dissimilarity=self.dissimilarity,
        ).fit_transform(self.dist_matrix_)

        self.embedding_ = cmds_embds
        return {model_name: cmds_embds[i] for i, model_name in enumerate(self.model_names_)}
