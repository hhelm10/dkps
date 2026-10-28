import numpy as np
from graspologic.embed import ClassicalMDS

from scipy.spatial.distance import pdist, squareform

from .utils import knn_graph


class DataKernelPerspectiveSpace:
    def __init__(self,
                response_distribution_fn=np.mean,
                response_distribution_axis=1,
                metric_cmds='euclidean',
                n_components_cmds=None,
                n_elbows_cmds=2,
                ):
        self.response_distribution_fn=response_distribution_fn
        self.response_distribution_axis=response_distribution_axis
        self.metric_cmds=metric_cmds
        self.n_components_cmds=n_components_cmds
        self.n_elbows_cmds=n_elbows_cmds


    def fit_transform(self, X):
        self.model_list=None
        if isinstance(X, dict):
            self.model_list=list(X.keys())
            X = list(X.values())


        if isinstance(X, (list, np.ndarray)):
            n_queries = len(X[0])
            for i,x in enumerate(X):
                x=np.array(x)
                assert x.ndim >= 2
                assert x.shape[0] == n_queries

                X[i]=self.response_distribution_fn(x, self.response_distribution_axis)
                    
        else:
            raise ValueError('X must be a dict of array-like values or array-like with elements with at least 2 dimensions')

        X = np.array(X)
        #- this only works in cases where representation of models are matrices
        dist_matrix = squareform(pdist(X.reshape(len(X), -1), metric=self.metric_cmds)) / np.sqrt(n_queries)
        cmds_embds = ClassicalMDS(self.n_components_cmds, self.n_elbows_cmds).fit_transform(dist_matrix)
        
        if self.model_list:
            return {key: cmds_embds[i] for i, key in enumerate(self.model_list)}
        
        return cmds_embds


class DataKernelFunctionalSpace:
    def __init__(self, n_neighbors=None, n_components_joint=None, n_elbows_joint=2, n_components_cmds=None, n_elbows_cmds=2, metric='euclidean'):
        self.n_neighbors=n_neighbors
        
        self.n_components_joint=n_components_joint
        self.n_elbows_joint=n_elbows_joint
        
        self.n_components_cmds=n_components_cmds
        self.n_elbows_cmds=n_elbows_cmds
        
        self.metric=metric
        
    def fit_transform(self, X):
        self.keys=None
        if isinstance(X, dict):
            self.keys=list(X.keys())
            X = list(X.values())

        if isinstance(X, (list, np.ndarray)):
            N=len(X)
            n = len(X[0])
            for i,x in enumerate(X):
                x=np.array(x)
                assert x.ndim == 2
                X[i] = knn_graph(x, self.n_neighbors, symmetrize=True)
        else:
            raise ValueError('X must be a dict of 2-d array-like values or array-like with 2-d array-like items')
            
        dist_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(i+1, N):
                from graspologic.embed import OmnibusEmbed  # upstream graspologic only
                omni_embds = OmnibusEmbed(n_components=self.n_components_joint, n_elbows=self.n_elbows_joint).fit_transform([X[i], X[j]])
                temp_dist = np.linalg.norm(omni_embds[0] - omni_embds[1]) / np.linalg.norm( (omni_embds[0] + omni_embds[1]) / 2 )
                dist_matrix[i,j] = temp_dist
                dist_matrix[j,i] = temp_dist
                
        cmds_embds = ClassicalMDS(self.n_components_cmds, self.n_elbows_cmds).fit_transform(dist_matrix)
        if self.keys:
            return {key: cmds_embds[i] for i, key in enumerate(self.keys)}
        
        return cmds_embds