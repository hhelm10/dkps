"""Explicit kernel geometry for the audited follow-up experiments.

Independent of graspologic so the mathematical checks can also use a minimal
NumPy environment. Pairwise ratio normalization is retained only as a named
comparison; the model normalization is a Gram matrix for PSD kernels.
"""
import numpy as np


def squared_distances(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.maximum((a * a).sum(1)[:, None] + (b * b).sum(1) - 2 * a @ b.T, 0)


def centered(a):
    a = np.asarray(a, dtype=float)
    return a - a.mean(0, keepdims=True) - a.mean(1, keepdims=True) + a.mean()


def spectral_embedding(affinity, dim=8):
    """Positive-eigenvalue MDS, with diagnostics *before* correction/truncation."""
    a = np.asarray(affinity, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.isfinite(a).all():
        raise ValueError('affinity must be a finite square matrix')
    if not np.allclose(a, a.T, atol=1e-10, rtol=1e-10):
        raise ValueError('affinity must be symmetric')
    if dim < 1:
        raise ValueError('dim must be positive')
    b = centered((a + a.T) / 2)
    values, vectors = np.linalg.eigh(b)
    tol = 1e-10 * max(1.0, float(np.max(np.abs(values))))
    positive = np.flatnonzero(values > tol)[::-1]
    kept = positive[:dim]
    # A zero-rank space is a legitimate constant representation.
    z = vectors[:, kept] * np.sqrt(values[kept]) if len(kept) else np.zeros((len(a), 1))
    raw_d2 = np.diag(a)[:, None] + np.diag(a)[None, :] - 2 * a
    pairs = raw_d2[np.triu_indices(len(a), 1)]
    scale = max(float(np.linalg.norm(b)), 1e-15)
    diagnostics = dict(
        min_squared_distance=float(pairs.min()) if len(pairs) else 0.0,
        negative_distance_fraction=float(np.mean(pairs < -tol)) if len(pairs) else 0.0,
        min_eigenvalue=float(values.min()),
        negative_eigenvalue_mass=float(-values[values < -tol].sum()),
        negative_eigenvalue_fraction=float(-values[values < -tol].sum()
                                          / max(np.abs(values).sum(), 1e-15)),
        positive_rank=int(len(positive)), retained_rank=int(len(kept)),
        psd_projection_relative_error=float(np.linalg.norm(np.minimum(values, 0)) / scale),
        rank_projection_relative_error=float(np.linalg.norm(b - z @ z.T) / scale),
    )
    return z, diagnostics


def affinity_components(query, response, model_ids, query_ids, task_ids,
                        kernel='rbf', bandwidth=1.0, model_order=None):
    """Return numerator and query mass matrices without normalization or clipping."""
    q, r = np.asarray(query, dtype=float), np.asarray(response, dtype=float)
    mid, qid, tid = map(np.asarray, (model_ids, query_ids, task_ids))
    if q.ndim != 2 or r.ndim != 2 or not (len(q) == len(r) == len(mid) == len(qid) == len(tid)):
        raise ValueError('inconsistent row arrays')
    if not np.isfinite(q).all() or not np.isfinite(r).all():
        raise ValueError('nonfinite embeddings')
    if kernel not in {'rbf', 'delta', 'task', 'constant'}:
        raise ValueError(f'unknown kernel: {kernel}')
    if kernel == 'rbf' and (not np.isfinite(bandwidth) or bandwidth <= 0):
        raise ValueError('RBF bandwidth must be finite and positive')
    names = sorted(set(mid)) if model_order is None else list(model_order)
    groups = [np.flatnonzero(mid == m) for m in names]
    if any(len(g) == 0 for g in groups):
        raise ValueError('each model must have at least one permitted response')
    s = np.zeros((len(names), len(names)))
    mass = np.zeros_like(s)
    for i, ai in enumerate(groups):
        for j in range(i, len(groups)):
            bj = groups[j]
            if kernel == 'rbf':
                w = np.exp(-squared_distances(q[ai], q[bj]) / (2 * bandwidth ** 2))
            elif kernel == 'delta':
                w = (qid[ai, None] == qid[bj]).astype(float)
            elif kernel == 'task':
                w = (tid[ai, None] == tid[bj]).astype(float)
            else:
                # Exact constant-kernel limit; no large response-pair matrix needed.
                mass[i, j] = mass[j, i] = len(ai) * len(bj)
                s[i, j] = s[j, i] = r[ai].sum(0) @ r[bj].sum(0)
                continue
            mass[i, j] = mass[j, i] = w.sum()
            s[i, j] = s[j, i] = np.sum(w * (r[ai] @ r[bj].T))
    return names, s, mass


def normalize_affinity(numerator, query_mass, normalization='model'):
    s, mass = np.asarray(numerator), np.asarray(query_mass)
    if normalization == 'model':
        diag = np.diag(mass)
        if np.any(diag <= 0):
            raise ValueError('model normalization requires positive self query mass')
        return s / np.sqrt(diag[:, None] * diag[None, :])
    if normalization == 'pairwise':
        if np.any(mass <= 0):
            raise ValueError('pairwise normalization undefined at zero cross query mass')
        return s / mass
    raise ValueError(f'unknown normalization: {normalization}')


def affinity_grid(query, response, model_ids, query_ids, task_ids, bandwidths,
                  shuffled_query, model_order):
    """Reuse expensive pair dot products across bandwidths and negative controls.

    Each entry is identical to a separate affinity_components call. Memory is
    bounded by one pair of model caches rather than all response pairs.
    """
    q, r, shuffled = map(lambda x: np.asarray(x, dtype=float), (query, response, shuffled_query))
    mid, qid, tid = map(np.asarray, (model_ids, query_ids, task_ids))
    groups = [np.flatnonzero(mid == name) for name in model_order]
    if any(len(g) == 0 for g in groups):
        raise ValueError('each model must have permitted responses')
    keys = ['constant', 'delta', 'task'] + [f'{kind}_{factor:g}'
        for kind in ['rbf', 'shuffled'] for factor in bandwidths]
    output = {key: (np.zeros((len(groups), len(groups))), np.zeros((len(groups), len(groups))))
              for key in keys}
    for i, ai in enumerate(groups):
        for j in range(i, len(groups)):
            bj = groups[j]
            response_dot = r[ai] @ r[bj].T
            d2 = squared_distances(q[ai], q[bj])
            shuffled_d2 = squared_distances(shuffled[ai], shuffled[bj])
            weights = {'constant': np.ones_like(d2),
                       'delta': (qid[ai, None] == qid[bj]),
                       'task': (tid[ai, None] == tid[bj])}
            for factor, bw in bandwidths.items():
                weights[f'rbf_{factor:g}'] = np.exp(-d2 / (2 * bw ** 2))
                weights[f'shuffled_{factor:g}'] = np.exp(-shuffled_d2 / (2 * bw ** 2))
            for key, w in weights.items():
                s, mass = output[key]
                s[i, j] = s[j, i] = np.sum(w * response_dot)
                mass[i, j] = mass[j, i] = w.sum()
    return output


def query_scale(query, task_ids, query_ids, cap=400, seed=0):
    """Within-task median from observed unique queries, never a hidden pool."""
    q, tid, qid = np.asarray(query), np.asarray(task_ids), np.asarray(query_ids)
    rng = np.random.default_rng(seed)
    medians = []
    for t in sorted(set(tid)):
        rows = np.flatnonzero(tid == t)
        _, first = np.unique(qid[rows], return_index=True)
        rows = rows[first]
        if len(rows) > cap:
            rows = rng.choice(rows, cap, replace=False)
        if len(rows) > 1:
            ds = np.sqrt(squared_distances(q[rows], q[rows]))[np.triu_indices(len(rows), 1)]
            medians.append(float(np.median(ds)))
    return max(float(np.median(medians)) if medians else 1.0, 1e-8)


def shuffle_queries(query, task_ids, query_ids, seed=0):
    """One permutation per task over unique IDs, shared across every model."""
    q, tid, qid = np.asarray(query), np.asarray(task_ids), np.asarray(query_ids)
    out = q.copy()
    rng = np.random.default_rng(seed)
    for t in sorted(set(tid)):
        rows = np.flatnonzero(tid == t)
        ids, first = np.unique(qid[rows], return_index=True)
        vectors = q[rows[first]]
        permutation = rng.permutation(len(ids))
        mapping = dict(zip(ids, vectors[permutation]))
        for row in rows:
            out[row] = mapping[qid[row]]
    return out
