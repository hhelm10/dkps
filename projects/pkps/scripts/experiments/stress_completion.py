"""Completion pilot with validation response removal and matched score access.

Outer missing cells have no responses or labels in geometry, whitening, or tuning.
Inner validation removes the entire cell (both responses and scores), then refits
geometry. All supervised targets used for fitting/tuning are observed sample means.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.stress_followups import (load_cache, observe, matrix, geometry_bank, knn_predict, family,
                                         code_provenance)
from dkps.baselines import matrix_completion_predict
from dkps.preprocessing import Whitener


def select_cells(models, tasks, coverage, seed):
    if not 0 < coverage < 1:
        raise ValueError('completion coverage must be strictly between zero and one')
    rng = np.random.default_rng(seed)
    mask = rng.random((len(models), len(tasks))) < coverage
    # The protocol conditions on at least one permitted cell per model/task.
    for i in range(len(models)):
        if not mask[i].any():
            mask[i, rng.integers(len(tasks))] = True
    for j in range(len(tasks)):
        if not mask[:, j].any():
            mask[rng.integers(len(models)), j] = True
    return mask


def inner_split(observed, seed, fraction=.2):
    """Hold cells out without removing the final observation of any row/column."""
    train = observed.copy()
    rng = np.random.default_rng(seed)
    wanted = max(1, round(observed.sum() * fraction))
    held = 0
    for i, j in rng.permutation(np.argwhere(observed)):
        if train[i].sum() > 1 and train[:, j].sum() > 1:
            train[i, j] = False
            held += 1
            if held >= wanted:
                break
    if not held:
        raise ValueError('insufficient coverage for an inner validation split')
    return train, observed & ~train


def rows_for_mask(rows, models, tasks, mask):
    cells = {(models[i], tasks[j]) for i, j in np.argwhere(mask)}
    return rows[[key in cells for key in zip(rows.model_id, rows.task_id)]].copy()


def fit_predictions(rows, vectors, models, tasks, seed):
    bank, diagnostics = geometry_bank(rows, vectors, models, seed)
    scores = matrix(rows, models, tasks)
    observed = np.isfinite(scores)
    whitening = Whitener().fit(scores, observed)
    residual = whitening.transform(scores)
    # Critical: hidden labels are NaN before any fit or preprocessing.
    estimates = {}
    fams = np.array([family(m) for m in models])
    for name, z in bank.items():
        pred = np.full_like(scores, np.nan)
        for i in range(len(models)):
            # Match the paper's developer exclusion in perspective regression.
            refs = np.flatnonzero(fams != fams[i])
            pred[i] = knn_predict(z, residual, refs, [i])[0]
        # Absent reference task -> zero residual (additive estimate), recorded below.
        fallback = ~np.isfinite(pred)
        estimates[name] = whitening.inverse_transform(np.nan_to_num(pred, nan=0))
        for d in diagnostics:
            if d['candidate'] == name:
                d['reference_fallback_fraction'] = float(np.mean(fallback[~observed]))
    estimates['additive'] = whitening.inverse_transform(np.zeros_like(scores))
    estimates['task-mean'] = np.tile(np.nanmean(scores, axis=0), (len(models), 1))
    for rank in [1, 2, 4]:
        for lam in [.1, 1.]:
            estimates[f'lrmc_{rank}_{lam:g}'] = matrix_completion_predict(
                scores, observed, rank=rank, lam=lam, n_init=5, random_state=seed)
    return estimates, diagnostics, scores


def predict_completion(permitted_rows, vectors, models, tasks, seed):
    """Receives observed rows only; no full-depth target argument exists."""
    scores = matrix(permitted_rows, models, tasks)
    observed = np.isfinite(scores)
    train, validation = inner_split(observed, seed)
    inner_rows = rows_for_mask(permitted_rows, models, tasks, train)
    inner, inner_diag, _ = fit_predictions(inner_rows, vectors, models, tasks, seed)
    final, outer_diag, _ = fit_predictions(permitted_rows, vectors, models, tasks, seed)
    groups = {
        'model-rbf': [k for k in inner if k.startswith('model_rbf_')],
        'ratio-psd': [k for k in inner if k.startswith('ratio_psd_')],
        'ratio-legacy': [k for k in inner if k.startswith('ratio_legacy_')],
        'shuffled-rbf': [k for k in inner if k.startswith('shuffled_rbf_')],
        'model-expanded': [k for k in inner if k.startswith('model_rbf_')] + ['constant', 'delta'],
        'lrmc': [k for k in inner if k.startswith('lrmc_')],
        **{k: [k] for k in ['constant', 'delta', 'task', 'additive', 'task-mean']},
    }
    predictions, selections, choices = {}, [], {}
    for name, candidates in groups.items():
        losses = {c: float(np.mean(abs(inner[c][validation] - scores[validation]))) for c in candidates}
        best = min(candidates, key=losses.get)
        choices[name] = best
        predictions[name] = final[best]
        selections.append(dict(method=name, candidate=best, validation_mae=losses[best]))
    a, b = inner[choices['model-expanded']][validation], inner[choices['lrmc']][validation]
    grid = np.linspace(0, 1, 21)
    weight = float(grid[np.argmin([np.mean(abs(w*a+(1-w)*b - scores[validation])) for w in grid])])
    predictions['ensemble-nested'] = weight*predictions['model-expanded']+(1-weight)*predictions['lrmc']
    selections.append(dict(method='ensemble-nested', candidate='model-expanded+lrmc',
                           pkps_weight=weight))
    diagnostics = [dict(stage='inner', **d) for d in inner_diag] + [dict(stage='outer', **d) for d in outer_diag]
    counts = dict(n_observed_cells=int(observed.sum()), n_validation_cells=int(validation.sum()),
                  n_inner_rows=len(inner_rows), n_outer_rows=len(permitted_rows))
    return predictions, diagnostics, selections, counts


def run(pool, vectors, truth, n_models, coverage, m, seed):
    all_models = sorted(pool.model_id.unique())
    if not 1 <= n_models <= len(all_models):
        raise ValueError('requested cohort exceeds available models or is nonpositive')
    models = sorted(np.random.default_rng(seed).choice(all_models, n_models, replace=False))
    tasks = sorted(pool.task_id.unique())
    mask = select_cells(models, tasks, coverage, seed)
    cells = {(models[i], tasks[j]) for i, j in np.argwhere(mask)}
    permitted = observe(pool[pool.model_id.isin(models)], m, seed, cells)
    predictions, diagnostics, selections, counts = predict_completion(permitted, vectors, models, tasks, seed)
    outputs = []
    for name, matrix_ in predictions.items():
        for i, j in np.argwhere(~mask):
            outputs.append(dict(method=name, model_id=models[i], task_id=tasks[j],
                                prediction=float(matrix_[i, j])))
    # Full targets first enter only after all predictions and selections are fixed.
    df = pd.DataFrame(outputs).merge(truth[['model_id', 'task_id', 'suite', 'y_full']],
                                    on=['model_id', 'task_id'], validate='many_to_one')
    df['absolute_error'] = abs(df.prediction-df.y_full)
    df['squared_error'] = (df.prediction-df.y_full)**2
    counts.update(actual_coverage=float(mask.mean()), n_models=len(models), n_tasks=len(tasks),
                  model_list=models, mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest())
    return df, pd.DataFrame(diagnostics), pd.DataFrame(selections), counts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=Path('data/eee-followups'))
    ap.add_argument('--out', type=Path, default=Path('projects/pkps/artifacts/stress-test/completion'))
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--cohorts', type=int, nargs='+', default=[10, 45])
    ap.add_argument('--coverage', type=float, nargs='+', default=[.2, .5])
    ap.add_argument('--m', type=int, default=16, help='observed queries per cell, 16 = half the 32-row pool')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pool, vectors = load_cache(args.source)
    truth = pd.read_parquet(args.source/'targets.parquet')
    config = dict(seeds=args.seeds, cohorts=args.cohorts, coverage=args.coverage, m=args.m,
                  implementation_sha256=code_provenance(),
                  reference_regime='cache-only', inner_validation='20% cells; delete response+score',
                  preprocessing='observed-only logit/standardization/two-way bias',
                  embeddings=json.loads((args.source/'embeddings/config.json').read_text()),
                  dataset_revision=json.loads((args.source/'dataset_manifest.json').read_text())['revision'])
    marker = args.out/'config.json'
    if marker.exists() and json.loads(marker.read_text()) != config:
        raise ValueError('run configuration changed; use a new output folder')
    marker.write_text(json.dumps(config, indent=2)+'\n')
    with threadpool_limits(limits=2):
        for seed in args.seeds:
            for n in args.cohorts:
                for coverage in args.coverage:
                    prefix = args.out/f'comp_n{n}_p{coverage:g}_seed{seed}'
                    if Path(str(prefix)+'.complete.json').exists():
                        continue
                    start = time.perf_counter()
                    p, d, s, counts = run(pool, vectors, truth, n, coverage, args.m, seed)
                    for frame, suffix in [(p, '.predictions.csv.gz'), (d, '.geometry.csv'), (s, '.selection.csv')]:
                        frame.assign(seed=seed, n_models=n, coverage=coverage, m=args.m).to_csv(
                            str(prefix)+suffix, index=False)
                    counts['seconds'] = time.perf_counter()-start
                    Path(str(prefix)+'.complete.json').write_text(json.dumps(counts, indent=2)+'\n')
                    print(dict(seed=seed, coverage=coverage, **counts), flush=True)


if __name__ == '__main__':
    main()
