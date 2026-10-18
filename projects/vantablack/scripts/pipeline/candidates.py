"""Candidate family for the scan (TODO section 4, items a-d).

(a) clusters    agglomerative clusters of displacement directions D_n
(b) comove      Louvain communities of a kNN co-movement graph on D_n
(c) interaction Louvain communities of the observed post-burn-in
                reply/vote graph (no embeddings involved)
(d) neardup     connected components of agents sharing near-duplicate
                actions (sybil candidates)

(a) and (b) look at the same content the test looks at, so selecting S
with them and testing S on the same actions is circular. run_pipeline
handles this with sample splitting: (a)/(b) are built from one random half
of each agent's actions and the test uses the other half.
"""
import numpy as np
import networkx as nx
from scipy.cluster.hierarchy import linkage, fcluster


def _keep(S, n_max):
    return 3 <= len(S) <= n_max


def clusters(D, eligible, ks=(5, 10, 20), n_max=None, large=2000, seed=0):
    """Average-linkage cosine clusters; above `large` agents, spherical
    k-means with k = n/40, n/20, n/10 instead (linkage is O(n^2) per draw)."""
    idx = np.flatnonzero(eligible)
    n_max = n_max or max(3, len(idx) // 4)
    U = D[idx] / np.maximum(np.linalg.norm(D[idx], axis=1, keepdims=True), 1e-12)
    out = []
    if len(idx) > large:
        from sklearn.cluster import MiniBatchKMeans
        for k in (len(idx) // 40, len(idx) // 20, len(idx) // 10):
            lab = MiniBatchKMeans(k, n_init=1, random_state=seed, batch_size=4096).fit_predict(U)
            for c in np.unique(lab):
                S = frozenset(idx[lab == c].tolist())
                if _keep(S, n_max):
                    out.append((S, f'clusters(k={k})'))
        return out
    Lk = linkage(U, method='average', metric='cosine')
    for k in ks:
        lab = fcluster(Lk, t=k, criterion='maxclust')
        for c in np.unique(lab):
            S = frozenset(idx[lab == c].tolist())
            if _keep(S, n_max):
                out.append((S, f'clusters(k={k})'))
    return out


def comove(D, eligible, knn=8, seed=0, n_max=None):
    idx = np.flatnonzero(eligible)
    n_max = n_max or max(3, len(idx) // 4)
    U = D[idx] / np.maximum(np.linalg.norm(D[idx], axis=1, keepdims=True), 1e-12)
    S_ = U @ U.T
    np.fill_diagonal(S_, -np.inf)
    nb = np.argpartition(-S_, knn, axis=1)[:, :knn]
    w = np.take_along_axis(S_, nb, axis=1)
    g = nx.Graph()
    g.add_nodes_from(range(len(idx)))
    ii = np.repeat(np.arange(len(idx)), knn)
    keep = w.ravel() > 0
    g.add_weighted_edges_from(zip(ii[keep].tolist(), nb.ravel()[keep].tolist(),
                                  w.ravel()[keep].tolist()))
    out = []
    for res in (1.0, 2.0):
        for c in nx.community.louvain_communities(g, weight='weight', resolution=res, seed=seed):
            S = frozenset(idx[list(c)].tolist())
            if _keep(S, n_max):
                out.append((S, f'comove(res={res})'))
    return out


def interaction(actions, agent_index, t0, seed=0, n_max=None):
    a = actions[(actions.t > t0) & (actions.target_agent != '')
                & actions.target_agent.isin(agent_index) & actions.agent_id.isin(agent_index)]
    a = a[a.agent_id != a.target_agent]
    n_max = n_max or max(3, len(agent_index) // 4)
    g = nx.Graph()
    w = a.groupby(['agent_id', 'target_agent']).size()
    for (u, v), c in w.items():
        u, v = agent_index[u], agent_index[v]
        g.add_edge(u, v, weight=g.get_edge_data(u, v, {'weight': 0})['weight'] + c)
    out = []
    if g.number_of_edges() == 0:
        return out, g
    for res in (1.0, 2.0, 4.0):
        for c in nx.community.louvain_communities(g, weight='weight', resolution=res, seed=seed):
            S = frozenset(c)
            if _keep(S, n_max):
                out.append((S, f'interaction(res={res})'))
    return out, g


def neardup(E, agent_idx, thresh=0.97, min_shared=3, per_agent=30, valid=None, seed=0):
    """Agents linked if they share >= min_shared near-duplicate action pairs.

    Up to per_agent actions per agent (seeded sample, `valid` ones only, e.g.
    texts >= 40 chars so 'Great post!' does not link everyone). Similarity
    runs blockwise on the GPU when available.
    """
    rng = np.random.default_rng(seed)
    cand = np.arange(len(E)) if valid is None else np.flatnonzero(valid)
    order = cand[np.argsort(agent_idx[cand], kind='stable')]
    _, starts = np.unique(agent_idx[order], return_index=True)
    pick = np.split(order, starts[1:]) if len(order) else []
    idx = np.concatenate([p if len(p) <= per_agent else rng.choice(p, per_agent, replace=False)
                          for p in pick]) if pick else np.array([], int)
    ag = agent_idx[idx]
    counts = {}
    try:
        import torch
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        U = torch.nn.functional.normalize(torch.as_tensor(E[idx], dtype=torch.float32, device=dev), dim=1)
        agt = torch.as_tensor(ag, device=dev)
        for s in range(0, len(idx), 4096):
            sim = U[s:s + 4096] @ U.T
            ii, jj = torch.nonzero(sim >= thresh, as_tuple=True)
            a, b = agt[ii + s], agt[jj]
            m = a < b
            pairs = (a[m] * (len(E) + 1) + b[m]).cpu().numpy()
            for key, c in zip(*np.unique(pairs, return_counts=True)):
                k_ = (int(key // (len(E) + 1)), int(key % (len(E) + 1)))
                counts[k_] = counts.get(k_, 0) + int(c)
    except ImportError:
        U = E[idx] / np.maximum(np.linalg.norm(E[idx], axis=1, keepdims=True), 1e-12)
        for s in range(0, len(U), 2000):
            ii, jj = np.nonzero(U[s:s + 2000] @ U.T >= thresh)
            for i, j in zip(ii + s, jj):
                a, b = ag[i], ag[j]
                if a < b:
                    counts[(a, b)] = counts.get((a, b), 0) + 1
    g = nx.Graph()
    g.add_edges_from(k for k, c in counts.items() if c >= min_shared)
    return [(frozenset(c), 'neardup') for c in nx.connected_components(g) if len(c) >= 3]


def coactivity(actions, agent_index, t0, within_min=10.0, min_count=3, max_lag=20,
               seed=0, n_max=None, return_weights=False):
    """Louvain communities of a post-burn-in co-activity graph: two agents are
    linked by the number of threads where both acted within `within_min`
    minutes of each other (ignores content; targets coordinated bursts).
    Edge weight is that count divided by sqrt(activity_i * activity_j).
    return_weights: also return the dense n x n weight matrix over all pairs
    (no min_count threshold), used by the adversarial gate."""
    nA = len(agent_index)
    W = np.zeros((nA, nA), np.float32) if return_weights else None
    a = actions[(actions.t > t0) & actions.agent_id.isin(agent_index)
                & actions.action_type.isin(['post', 'comment'])]
    a = a.sort_values(['thread_id', 't'])
    th = a.thread_id.values
    ag = a.agent_id.map(agent_index).values
    t = a.t.values
    n_max = n_max or max(3, len(agent_index) // 4)
    pairs = []
    for lag in range(1, max_lag + 1):
        same = (th[lag:] == th[:-lag]) & ((t[lag:] - t[:-lag]) * 60 <= within_min) \
            & (ag[lag:] != ag[:-lag])
        i, j = ag[:-lag][same], ag[lag:][same]
        pairs.append(np.stack([np.minimum(i, j), np.maximum(i, j)], 1))
    out = []
    ret = (lambda o: (o, W)) if return_weights else (lambda o: o)
    P = np.concatenate(pairs) if pairs else np.zeros((0, 2), int)
    if not len(P):
        return ret(out)
    key, cnt = np.unique(P[:, 0] * (len(agent_index) + 1) + P[:, 1], return_counts=True)
    act = np.bincount(ag, minlength=len(agent_index)).astype(float)
    if return_weights:
        ii, jj = key // (nA + 1), key % (nA + 1)
        w = cnt / np.sqrt(act[ii] * act[jj])
        W[ii, jj] = w
        W[jj, ii] = w
    g = nx.Graph()
    for k_, c in zip(key, cnt):
        if c >= min_count:
            i, j = int(k_ // (len(agent_index) + 1)), int(k_ % (len(agent_index) + 1))
            g.add_edge(i, j, weight=c / np.sqrt(act[i] * act[j]))
    if g.number_of_edges() == 0:
        return ret(out)
    for res in (1.0, 2.0, 4.0):
        for c in nx.community.louvain_communities(g, weight='weight', resolution=res, seed=seed):
            S = frozenset(c)
            if _keep(S, n_max):
                out.append((S, f'coactivity(res={res})'))
    return ret(out)


def dedupe(family):
    seen, out = set(), []
    for S, src in family:
        if S not in seen:
            seen.add(S)
            out.append((S, src))
    return out
