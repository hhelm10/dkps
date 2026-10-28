import numpy as np
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

def knn_graph(w, k, symmetrize=True, metric='euclidean'):
    '''
    :param w: A weighted affinity graph of shape [N, N] or 2-d array 
    :param k: The number of neighbors to use
    :kwarg symmetrize: Whether to symmetrize the resulting graph
    :kwarg metric: scipy-valid metric to use to construct pairwise distance matrix if w is 2-d array.
    :return: An undirected, binary, KNN graph of shape [N, N]
    '''
    w_shape = w.shape
    if w_shape[0] != w_shape[1]:
        w = np.array(squareform(pdist(w, metric=metric)))
            
    neighborhoods = np.argsort(w, axis=1)[:, -(k+1):-1]
    A = np.zeros_like(w)
    for i, neighbors in enumerate(neighborhoods):
        for j in neighbors:
            A[i, j] = 1
            if symmetrize:
                A[j, i] = 1
    return A


def bootstrap_null(graph, number_of_bootstraps=25, n_components=None, acorn=None):
    '''
    Constructs a bootstrap null distribution for the difference of latent positions of the nodes in the passed graph
    :param graph: [N, N] binary symmetric hollow matrix to model
    :kwarg number_of_bootstraps: the number of bootstrap replications
    :kwarg n_components: the number of components to use in initial ASE. selected automatically if None.
    :kwarg acorn: rng seed to control for randomness in umap and ase
    :return: array-like (len(graph), number_of_bootstraps), n_components.
    '''
    # upstream graspologic only (the jataware fork drops these)
    from graspologic.embed import AdjacencySpectralEmbed, OmnibusEmbed
    from graspologic.simulations import rdpg
    if acorn is not None:
        np.random.seed(acorn)

    ase_latents = AdjacencySpectralEmbed(n_components=n_components, svd_seed=acorn).fit_transform(graph)

    n, n_components = ase_latents.shape

    distances = np.zeros(((2, number_of_bootstraps, n)))
    distances = np.zeros((number_of_bootstraps, n))

    for i in tqdm(range(number_of_bootstraps)):
        graph_b = rdpg(ase_latents, directed=False)

        bootstrap_latents = OmnibusEmbed(n_components=n_components).fit_transform([graph, graph_b])
        distances[i] = np.linalg.norm(bootstrap_latents[0] - bootstrap_latents[1], axis=1)

    return distances.transpose((1, 0)), n_components


def get_cdf(pvalues, num=26):
    '''
    Get the cumulative distribution of pvalues.
    :param pvalues: array-like of pvalues
    :kwarg num: number of bins to use when constructing cdf
    :return: cdf of pvalues
    '''
    linspace = np.linspace(0, 1, num=num)

    cdf = np.zeros(num)

    for i, ii in enumerate(linspace):
        cdf[i] = np.mean(pvalues <= ii)

    return cdf