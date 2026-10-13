"""Audited EEE pilot: explicit geometry and nested developer-held-out prediction.

Embeddings are frozen, normalized 768-d vectors (no fitted PCA). All geometry is
transductive over permitted response rows. This is not a Gemini/Table-1 rerun.
Full evaluation scores are joined only after prediction; full-reference training
receives a separate table with the entire outer developer removed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from graspologic.embed import ClassicalMDS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.stress_geometry import (affinity_grid, normalize_affinity,
    spectral_embedding, squared_distances, query_scale, shuffle_queries, centered)


def family(model):
    return model.split('/')[0]


def code_provenance():
    root = Path(__file__).resolve().parents[4]
    files = [Path(__file__), Path(__file__).with_name('stress_completion.py'),
             Path(__file__).resolve().parents[1]/'pipeline/stress_geometry.py',
             root/'dkps/preprocessing.py', root/'dkps/baselines.py']
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def observe(pool, m, seed, cells=None):
    """Nested budgets per seed; excludes hidden cells before accessing their rows."""
    keep = []
    for (model, task), group in pool.groupby(['model_id', 'task_id'], sort=True):
        if cells is not None and (model, task) not in cells:
            continue
        key = f'{seed}|{model}|{task}'.encode()
        rng = np.random.default_rng(int(hashlib.sha256(key).hexdigest()[:8], 16))
        indices = group.sort_values('query_id').index.to_numpy()
        if len(indices) < m:
            raise ValueError(f'budget {m} exceeds available rows for {(model, task)}')
        keep.extend(rng.permutation(indices)[:m])
    return pool.loc[keep].copy()


def matrix(rows, models, tasks, column='score'):
    return rows.groupby(['model_id', 'task_id'])[column].mean().unstack().reindex(
        index=models, columns=tasks).to_numpy(dtype=float)


def geometry_bank(rows, vectors, models, seed=0, dim=8):
    q, r = vectors[rows.q_index], vectors[rows.r_index]
    mid, qid, tid = (rows[c].to_numpy() for c in ['model_id', 'query_id', 'task_id'])
    scale = query_scale(q, tid, qid, seed=seed)
    bank, diagnostics = {}, []

    def add(name, s, mass, norm, legacy=False):
        a = normalize_affinity(s, mass, norm)
        z, info = spectral_embedding(a, dim)
        if legacy:
            d2 = np.diag(a)[:, None] + np.diag(a)[None, :] - 2 * a
            d = np.sqrt(np.maximum(d2, 0)); np.fill_diagonal(d, 0)
            z = ClassicalMDS(n_components=min(dim, len(models)-1),
                             dissimilarity='precomputed', svd_seed=0).fit_transform(d)
        bank[name] = z
        gram = centered(a)
        info['actual_embedding_relative_error'] = float(np.linalg.norm(gram-z@z.T) / max(np.linalg.norm(gram), 1e-15))
        info['actual_dimension'] = z.shape[1]
        info['embedding_kind'] = 'legacy-clipped-distance-svd' if legacy else 'positive-signed-eigenvalues'
        cross = mass[np.triu_indices(len(mass), 1)]
        diagnostics.append(dict(candidate=name, scale=scale, normalization=norm,
                                n_rows=len(rows), min_cross_mass=float(cross.min()),
                                zero_cross_mass_fraction=float(np.mean(cross == 0)),
                                min_self_mass=float(np.diag(mass).min()), **info))

    shuffled = shuffle_queries(q, tid, qid, seed)
    components = affinity_grid(q, r, mid, qid, tid,
                               {f: scale*f for f in [.03, .1, .3, 1., 3.]}, shuffled, models)
    for factor in [.03, .1, .3, 1., 3.]:
        s, mass = components[f'rbf_{factor:g}']
        add(f'model_rbf_{factor:g}', s, mass, 'model')
        # Underflow at extremely small bandwidths can make a numerical ratio undefined.
        if np.all(mass > 0):
            add(f'ratio_psd_{factor:g}', s, mass, 'pairwise')
            add(f'ratio_legacy_{factor:g}', s, mass, 'pairwise', legacy=True)
    for kernel in ['constant', 'delta', 'task']:
        s, mass = components[kernel]
        add(kernel, s, mass, 'model')
    for factor in [.03, .1, .3, 1., 3.]:
        s, mass = components[f'shuffled_{factor:g}']
        add(f'shuffled_rbf_{factor:g}', s, mass, 'model')
    return bank, diagnostics


def knn_predict(z, labels, train, targets, k=5):
    """Per-task raw-score kNN; label rows outside train are never inspected."""
    result = np.full((len(targets), labels.shape[1]), np.nan)
    for col in range(labels.shape[1]):
        refs = np.asarray(train)[np.isfinite(labels[train, col])]
        if not len(refs):
            continue
        distances = np.sqrt(squared_distances(z[targets], z[refs]))
        for j, ds in enumerate(distances):
            order = np.argsort(ds, kind='stable')[:min(k, len(refs))]
            selected = ds[order]
            if np.any(selected < 1e-10):
                weights = (selected < 1e-10).astype(float)
            else:
                weights = 1 / selected
            result[j, col] = np.average(labels[refs[order], col], weights=weights)
    return result


def nested_predict(bank, models, tasks, sample, allowed_reference, outer_family, regime):
    """The API rejects forbidden full-reference rows, even if unused by prediction."""
    fams = np.array([family(m) for m in models])
    refs = np.flatnonzero(fams != outer_family)
    targets = np.flatnonzero(fams == outer_family)
    if regime == 'full-reference':
        if allowed_reference.model_id.map(family).eq(outer_family).any():
            raise ValueError('outer-family full scores must not enter the learner')
        labels = matrix(allowed_reference, models, tasks, 'y_full')
    elif regime == 'cache-only':
        if allowed_reference is not None:
            raise ValueError('cache-only predictor cannot receive full scores')
        labels = sample.copy()
    else:
        raise ValueError(regime)
    inner = {}
    for key, z in bank.items():
        pred = np.full_like(sample, np.nan)
        for validation_family in sorted(set(fams[refs])):
            valid = np.flatnonzero(fams == validation_family)
            train = refs[fams[refs] != validation_family]
            pred[valid] = knn_predict(z, labels, train, valid)
        inner[key] = pred

    groups = {
        'model-rbf': [k for k in bank if k.startswith('model_rbf_')],
        'ratio-psd': [k for k in bank if k.startswith('ratio_psd_')],
        'ratio-legacy': [k for k in bank if k.startswith('ratio_legacy_')],
        'shuffled-rbf': [k for k in bank if k.startswith('shuffled_rbf_')],
        'constant': ['constant'], 'delta': ['delta'], 'task': ['task'],
        # The expanded selector explicitly contains both boundary kernels.
        'model-expanded': [k for k in bank if k.startswith('model_rbf_')] + ['constant', 'delta'],
    }
    predictions = {'sample': sample[targets].copy()}
    mean = np.nanmean(labels[refs], axis=0)
    predictions['task-mean'] = np.tile(mean, (len(targets), 1))
    predictions['task-median'] = np.tile(np.nanmedian(labels[refs], axis=0), (len(targets), 1))
    predictions['shrinkage-fixed'] = .5 * sample[targets] + .5 * mean
    selected = []
    if regime == 'full-reference':
        inner_mean = np.full_like(sample, np.nan)
        for validation_family in sorted(set(fams[refs])):
            train = refs[fams[refs] != validation_family]
            inner_mean[fams == validation_family] = np.nanmean(labels[train], axis=0)
        grid = np.linspace(0, 1, 21)
        errors = [np.nanmean(abs(a*sample[refs]+(1-a)*inner_mean[refs]-labels[refs])) for a in grid]
        alpha = float(grid[np.argmin(errors)])
        predictions['shrinkage-nested'] = alpha*sample[targets]+(1-alpha)*mean
        selected.append(dict(outer_family=outer_family, regime=regime, method='shrinkage-nested',
                             candidate='sample+task-mean', validation_mae=float(min(errors)), sample_weight=alpha))
    for method, candidates in groups.items():
        losses = {c: float(np.nanmean(abs(inner[c][refs] - labels[refs]))) for c in candidates}
        best = min(candidates, key=losses.get)
        predictions[method] = knn_predict(bank[best], labels, refs, targets)
        item = dict(outer_family=outer_family, regime=regime, method=method,
                    candidate=best, validation_mae=losses[best])
        if method == 'model-expanded':
            # Cache-only labels equal the sample component at m=1 (and when no
            # independent labels are purchased). Tuning alpha against them is
            # circular; use a declared fixed blend in this regime.
            if regime == 'full-reference':
                errors = [np.nanmean(abs(a * sample[refs] + (1-a) * inner[best][refs]
                                        - labels[refs])) for a in np.linspace(0, 1, 21)]
                alpha = float(np.linspace(0, 1, 21)[np.argmin(errors)])
                ensemble_name = 'ensemble-nested'
            else:
                alpha = .5
                ensemble_name = 'ensemble-fixed'
            predictions[ensemble_name] = alpha * sample[targets] + (1-alpha) * predictions[method]
            item['sample_weight'] = alpha
        selected.append(item)
    return targets, predictions, selected


def query_efficiency(pool, vectors, truth, m, seed):
    observed = observe(pool, m, seed)
    models, tasks = sorted(observed.model_id.unique()), sorted(observed.task_id.unique())
    bank, diagnostics = geometry_bank(observed, vectors, models, seed)
    sample = matrix(observed, models, tasks)
    results, selections = [], []
    for regime in ['full-reference', 'cache-only']:
        for fam in sorted(set(map(family, models))):
            allowed = truth[truth.model_id.map(family) != fam].copy() if regime == 'full-reference' else None
            indices, estimates, choices = nested_predict(bank, models, tasks, sample, allowed, fam, regime)
            selections.extend(choices)
            for method, pred in estimates.items():
                for a, i in enumerate(indices):
                    for j, task in enumerate(tasks):
                        results.append(dict(model_id=models[i], task_id=task, family=fam,
                                            regime=regime, method=method, prediction=float(pred[a, j])))
    # Evaluation is the only place target/full truth is joined to predictions.
    predictions = pd.DataFrame(results).merge(truth[['model_id', 'task_id', 'suite', 'y_full']],
                                            on=['model_id', 'task_id'], validate='many_to_one')
    predictions['absolute_error'] = abs(predictions.prediction - predictions.y_full)
    predictions['squared_error'] = (predictions.prediction - predictions.y_full) ** 2
    # Actual overlap is observed, not assumed zero.
    sets = observed.groupby('model_id').query_id.agg(set)
    overlaps = [len(sets.iloc[i] & sets.iloc[j]) / min(len(sets.iloc[i]), len(sets.iloc[j]))
                for i in range(len(sets)) for j in range(i+1, len(sets))]
    budget = dict(m=m, seed=seed, n_models=len(models), n_tasks=len(tasks), n_rows=len(observed),
                  rows_per_model=observed.groupby('model_id').size().unique().tolist(),
                  mean_pair_overlap=float(np.mean(overlaps)),
                  zero_overlap_fraction=float(np.mean(np.array(overlaps) == 0)))
    return predictions, pd.DataFrame(diagnostics), pd.DataFrame(selections), budget


def load_cache(source):
    cache = source / 'embeddings'
    if not (cache / 'complete.json').exists():
        raise ValueError('embedding cache is incomplete; rerun the resumable embedder')
    config = json.loads((cache / 'config.json').read_text())
    vectors = np.concatenate([np.load(cache / f'{start:07d}.npy')
                              for start in range(0, config['n_texts'], config['chunk_size'])])
    if vectors.shape != (config['n_texts'], config['dimension']) or not np.isfinite(vectors).all():
        raise ValueError('invalid embedding cache')
    return pd.read_parquet(cache / 'rows.parquet'), vectors


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=Path('data/eee-followups'))
    ap.add_argument('--out', type=Path, default=Path('projects/pkps/artifacts/stress-test/pilot'))
    ap.add_argument('--budgets', type=int, nargs='+', default=[1, 4, 8])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pool, vectors = load_cache(args.source)
    truth = pd.read_parquet(args.source / 'targets.parquet')
    config = dict(budgets=args.budgets, seeds=args.seeds, regressor='raw-score distance-weighted 5NN',
                  preprocessing='frozen 768d normalized text vectors; no PCA', mds_dim=8,
                  evaluation='developer-held-out; nested selection; transductive observed geometry',
                  implementation_sha256=code_provenance(),
                  dataset=json.loads((args.source/'dataset_manifest.json').read_text()),
                  embedding=json.loads((args.source/'embeddings/config.json').read_text()))
    marker = args.out / 'config.json'
    if marker.exists() and json.loads(marker.read_text()) != config:
        raise ValueError('run configuration changed; choose a new output folder')
    marker.write_text(json.dumps(config, indent=2) + '\n')
    with threadpool_limits(limits=2):
        for seed in args.seeds:
            for m in args.budgets:
                prefix = args.out / f'qe_m{m}_seed{seed}'
                if prefix.with_suffix('.complete.json').exists():
                    continue
                start = time.perf_counter()
                p, d, s, b = query_efficiency(pool, vectors, truth, m, seed)
                for frame, suffix in [(p, '.predictions.csv.gz'), (d, '.geometry.csv'), (s, '.selection.csv')]:
                    frame.assign(m=m, seed=seed).to_csv(prefix.with_suffix(suffix), index=False)
                b['seconds'] = time.perf_counter() - start
                prefix.with_suffix('.complete.json').write_text(json.dumps(b, indent=2) + '\n')
                print(b, flush=True)


if __name__ == '__main__':
    main()
