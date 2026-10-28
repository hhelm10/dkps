import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from graspologic.embed import ClassicalMDS

from scipy.spatial.distance import pdist, squareform


class DataKernelPerspectiveSpace:
    def __init__(
            self,
            response_distribution_fn=None,
            metric_cmds='euclidean',
            n_components_cmds=None,
            n_elbows_cmds=2,
            dissimilarity="precomputed",
        ):

        self.response_distribution_fn   = response_distribution_fn
        self.metric_cmds                = metric_cmds
        self.n_components_cmds          = n_components_cmds
        self.n_elbows_cmds              = n_elbows_cmds
        self.dissimilarity              = dissimilarity

    def _aggregate_replicates(self, emb_matrix):
        """
        Aggregate replicate embeddings for a single (model, query) group.

        Parameters
        ----------
        emb_matrix : np.ndarray, shape (n_replicates, embedding_dim)

        Returns
        -------
        np.ndarray, shape (embedding_dim,)
        """
        if self.response_distribution_fn is None:
            return emb_matrix[0]
        assert emb_matrix.shape[0] > 1, \
            'response_distribution_fn requires at least 2 replicates'
        # For a per-query (n_replicates, embedding_dim) matrix, replicates are always axis 0.
        return self.response_distribution_fn(emb_matrix, axis=0)

    def _validate_input_dataframe(self, data):
        assert isinstance(data, pd.DataFrame), 'data must be a DataFrame'
        for col in ('model_id', 'query_id', 'embedding'):
            assert col in data.columns, f'DataFrame must have column: {col}'

        df = data.copy()
        model_names = sorted(df['model_id'].unique())

        embedding_shapes = {np.asarray(emb).shape for emb in df['embedding'].values}
        assert len(embedding_shapes) == 1, 'all embeddings must have the same shape'

        # Verify all models share the same query set
        query_sets = [set(df.loc[df['model_id'] == m, 'query_id'].unique()) for m in model_names]
        assert all(qs == query_sets[0] for qs in query_sets), \
            'all models must have the same set of query_ids'

        queries = sorted(query_sets[0])
        has_replicates = 'replicate_id' in df.columns

        if has_replicates:
            duplicate_replicates = df.duplicated(['model_id', 'query_id', 'replicate_id'], keep=False)
            assert not duplicate_replicates.any(), \
                'replicate_id values must be unique within each (model_id, query_id)'

            rep_counts = df.groupby(['model_id', 'query_id']).size()
            if rep_counts.nunique() > 1:
                warnings.warn(
                    f'uneven replicate counts across (model_id, query_id) pairs: '
                    f'min={rep_counts.min()}, max={rep_counts.max()}',
                    stacklevel=2,
                )
        else:
            duplicate_pairs = df.duplicated(['model_id', 'query_id'], keep=False)
            assert not duplicate_pairs.any(), \
                'duplicate (model_id, query_id) rows without replicate_id column'

        return df, model_names, queries

    def _make_model_query_array(self, df, model_names, queries):
        grouped = df.groupby(['model_id', 'query_id'], sort=False)
        rows = []
        for model_name in model_names:
            model_vecs = []
            for query_id in queries:
                embs = np.stack(grouped.get_group((model_name, query_id))['embedding'].values)
                if embs.shape[0] == 1:
                    model_vecs.append(embs[0])
                else:
                    model_vecs.append(self._aggregate_replicates(embs))
            rows.append(np.stack(model_vecs))
        return np.stack(rows)

    def _compute_dist_matrix(self, X):
        n_models, n_queries, _ = X.shape
        X_flat = X.reshape(n_models, -1)

        if self.metric_cmds == 'euclidean':
            dist_matrix = pairwise_distances(X_flat, metric='euclidean') / np.sqrt(n_queries)
            dist_matrix = (dist_matrix + dist_matrix.T) / 2
        else:
            dist_matrix = squareform(pdist(X_flat, metric=self.metric_cmds)) / np.sqrt(n_queries)

        return dist_matrix

    def fit_transform(self, data):
        """
        Embed models into a low-dimensional space via DKPS.

        Parameters
        ----------
        data : pd.DataFrame
            Required columns: model_id, query_id, embedding
            Optional column: replicate_id
              If present, rows are replicates for a (model_id, query_id) pair
              and will be aggregated via response_distribution_fn.
              If absent, one row per (model_id, query_id) is assumed.

        Returns
        -------
        dict
        """
        df, model_names, queries = self._validate_input_dataframe(data)
        X = self._make_model_query_array(df, model_names, queries)
        dist_matrix = self._compute_dist_matrix(X)

        self.dist_matrix_ = dist_matrix
        self.model_names_ = model_names

        cmds_embds = ClassicalMDS(
            n_components=self.n_components_cmds,
            n_elbows=self.n_elbows_cmds,
            dissimilarity=self.dissimilarity,
        ).fit_transform(dist_matrix)

        return {m: cmds_embds[i] for i, m in enumerate(model_names)}
