"""Linear ridge classification and grouped evaluation; NumPy/SciPy only.

No imports from the host repository. All fitted statistics use training rows.
The dual implementation uses a *linear* Gram matrix, not a nonlinear kernel.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh


def normalize_blocks(x):
    """L2-normalize each section, concatenate, and give each section equal weight.

    Accept (rows, dimensions) or (rows, sections, dimensions). This is a
    sample-local transformation: no task labels or other examples are used.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        x = x[:, None, :]
    if x.ndim != 3 or not np.isfinite(x).all():
        raise ValueError('Expected finite (N,D) or (N,S,D) embeddings')
    if np.any(np.linalg.norm(x, axis=-1) < 1e-12):
        raise ValueError('Zero embedding block; remove or repair missing embeddings')
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    return x.reshape(len(x), -1) / np.sqrt(x.shape[1])


def balanced_accuracy(y, predicted, n_classes):
    counts = np.bincount(y, minlength=n_classes)
    hits = np.bincount(y[y == predicted], minlength=n_classes)
    if np.any(counts == 0):
        raise ValueError('Every evaluated class must have test examples')
    return float(np.mean(hits / counts))


def grouped_folds(groups, n_splits=5, seed=17):
    """Seeded group allocation, approximately balancing the number of rows."""
    groups = np.asarray(groups)
    unique, counts = np.unique(groups, return_counts=True)
    if len(unique) < n_splits:
        raise ValueError(f'Need {n_splits} groups, found {len(unique)}')
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    order = order[np.argsort(-counts[order], kind='stable')]
    allocations, sizes = [[] for _ in range(n_splits)], np.zeros(n_splits)
    for i in order:
        fold = int(np.argmin(sizes))
        allocations[fold].append(unique[i])
        sizes[fold] += counts[i]
    for selected in allocations:
        test = np.flatnonzero(np.isin(groups, selected))
        train = np.flatnonzero(~np.isin(groups, selected))
        yield train, test


@dataclass
class RidgePath:
    """Solve class-balanced squared-loss classification for multiple penalties.

    Objective: sum_i w_i ||x_i W + b - onehot(y_i)||² / n
               + alpha ||W||², w_i = n / (K * count[y_i]).
    The intercept is unpenalized; predictions are argmax of class scores.
    A symmetric eigendecomposition gives the global solution, with no
    iterative optimizer or convergence tolerance to tune.
    """
    kernel: np.ndarray
    cross_kernel: np.ndarray
    y: np.ndarray
    n_classes: int

    def __post_init__(self):
        n = len(self.y)
        counts = np.bincount(self.y, minlength=self.n_classes)
        if np.any(counts == 0):
            raise ValueError('Training fold is missing a target class')
        weights = n / (self.n_classes * counts[self.y])
        probability = weights / weights.sum()
        train_mean = probability @ self.kernel
        grand_mean = train_mean @ probability
        centered = self.kernel - train_mean[None, :] - train_mean[:, None] + grand_mean
        self.cross = (self.cross_kernel - train_mean[None, :]
                      - (self.cross_kernel @ probability)[:, None] + grand_mean)
        self.sqrt_weight = np.sqrt(weights)
        weighted = centered * self.sqrt_weight[:, None] * self.sqrt_weight[None, :]
        values, vectors = eigh(weighted, check_finite=False, driver='evd')
        if values[0] < -1e-7 * max(1., values[-1]):
            raise ValueError('Linear Gram matrix is unexpectedly indefinite')
        self.values = np.maximum(values, 0)
        labels = np.eye(self.n_classes)[self.y]
        self.label_mean = probability @ labels
        self.rhs = vectors.T @ (self.sqrt_weight[:, None] * (labels - self.label_mean))
        self.projected_cross = (self.cross * self.sqrt_weight[None, :]) @ vectors

    def scores(self, alpha):
        if alpha <= 0:
            raise ValueError('Ridge alpha must be positive')
        return (self.projected_cross @ (self.rhs / (self.values[:, None]
                                                   + len(self.y) * alpha))
                + self.label_mean)


def bootstrap_interval(y, predicted, groups, n_classes, seed=17, draws=1000):
    """Cluster-bootstrap the fixed out-of-fold predictions, not individual rows."""
    unique, gi = np.unique(groups, return_inverse=True)
    totals = np.zeros((len(unique), n_classes))
    correct = np.zeros_like(totals)
    np.add.at(totals, (gi, y), 1)
    np.add.at(correct, (gi, y), y == predicted)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(draws):
        sampled = rng.integers(len(unique), size=len(unique))
        denominator = totals[sampled].sum(0)
        if np.all(denominator > 0):
            scores.append(np.mean(correct[sampled].sum(0) / denominator))
    if len(scores) < draws * .9:
        raise ValueError('Too many bootstrap samples omit classes; choose another grouping')
    return np.quantile(scores, [.025, .975]).tolist()


def evaluate(kernel, labels, groups, *, alphas=(1e-5, 1e-4, 1e-3, 1e-2, .1, 1.),
             outer_folds=5, inner_folds=3, seed=17, bootstrap_draws=1000):
    """Nested grouped CV; select alpha by pooled inner balanced accuracy.

    Reject class-incomplete training folds instead of silently changing the
    problem. All candidates and representations must use the same row cohort.
    """
    classes, y = np.unique(labels, return_inverse=True)
    k = len(classes)
    if k < 2:
        raise ValueError('Need at least two target classes')
    n = len(y)
    predictions = np.full(n, -1, dtype=int)
    majority_predictions = np.full(n, -1, dtype=int)
    folds = []
    for number, (train, test) in enumerate(grouped_folds(groups, outer_folds, seed)):
        inner_predictions = np.full((len(alphas), len(train)), -1, dtype=int)
        inner_records = []
        for it, iv in grouped_folds(np.asarray(groups)[train], inner_folds, seed + number + 1):
            tr, va = train[it], train[iv]
            path = RidgePath(kernel[np.ix_(tr, tr)], kernel[np.ix_(va, tr)], y[tr], k)
            for a, alpha in enumerate(alphas):
                inner_predictions[a, iv] = path.scores(alpha).argmax(1)
            inner_records.append(dict(train_groups=np.unique(np.asarray(groups)[tr]).tolist(),
                                      validation_groups=np.unique(np.asarray(groups)[va]).tolist()))
        if np.any(inner_predictions < 0):
            raise AssertionError('Incomplete inner predictions')
        validation_scores = [balanced_accuracy(y[train], p, k) for p in inner_predictions]
        # Prefer stronger regularization on exact ties.
        best = max(range(len(alphas)), key=lambda a: (validation_scores[a], alphas[a]))
        path = RidgePath(kernel[np.ix_(train, train)], kernel[np.ix_(test, train)], y[train], k)
        predictions[test] = path.scores(alphas[best]).argmax(1)
        majority_predictions[test] = np.bincount(y[train], minlength=k).argmax()
        folds.append(dict(fold=number, alpha=alphas[best], inner_scores=validation_scores,
                          train_groups=np.unique(np.asarray(groups)[train]).tolist(),
                          test_groups=np.unique(np.asarray(groups)[test]).tolist(), inner=inner_records))
    if np.any(predictions < 0):
        raise AssertionError('Incomplete outer predictions')
    counts = np.bincount(y, minlength=k)
    hits = np.bincount(y[y == predictions], minlength=k)
    return dict(n_rows=n, n_classes=k, n_groups=len(np.unique(groups)),
                balanced_accuracy=balanced_accuracy(y, predictions, k),
                balanced_accuracy_ci95=bootstrap_interval(y, predictions, groups, k, seed, bootstrap_draws),
                accuracy=float(np.mean(y == predictions)), chance_balanced_accuracy=1 / k,
                majority_accuracy=float(np.mean(y == majority_predictions)),
                classes=classes.tolist(), class_counts=counts.tolist(),
                class_recall=(hits / counts).tolist(), folds=folds,
                predicted=classes[predictions].tolist(), truth=classes[y].tolist())
