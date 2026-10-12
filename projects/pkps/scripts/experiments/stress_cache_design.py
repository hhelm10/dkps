"""Small E5/E6 stress probes at matched per-model response budgets.

Difficulty and poor-score-hiding masks intentionally use an oracle in the *mask
generator*. Full scores never enter the completion predictor. These are controlled
sensitivity probes, not measurements on naturally missing benchmark outcomes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.stress_followups import load_cache, observe, matrix, code_provenance
from experiments.stress_completion import predict_completion
from pipeline.stress_geometry import affinity_grid, normalize_affinity, query_scale


def nuisance_probe(pool, vectors, out, seed=0):
    models = sorted(np.random.default_rng(seed).choice(sorted(pool.model_id.unique()), 10, replace=False))
    rows = pool[pool.model_id.isin(models)]
    balanced = observe(rows, 8, seed)
    chosen_tasks = ['math-mc:math_mc_level3', 'gpqa-diamond:gpqa_diamond',
                    'judgebench:judge_judgebench_math', 'reward-bench-2:judge_rewardbench2_factuality']
    concentrated = rows[rows.task_id.isin(chosen_tasks)]
    rng = np.random.default_rng(seed)
    difficulty = rows.assign(tie=rng.random(len(rows))).sort_values(['score', 'tie']).groupby(
        ['model_id', 'task_id'], sort=True).head(8).drop(columns='tie')
    outputs = []
    for profile, changed in [('four-task-mix', concentrated), ('low-score-items', difficulty)]:
        if not (balanced.groupby('model_id').size().eq(128).all()
                and changed.groupby('model_id').size().eq(128).all()):
            raise ValueError('nuisance probe requires exactly 128 responses per model/cache')
        both = pd.concat([balanced.assign(model_id=balanced.model_id+'|base'),
                          changed.assign(model_id=changed.model_id+'|changed')], ignore_index=True)
        names = [m+'|base' for m in models]+[m+'|changed' for m in models]
        q, r = vectors[both.q_index], vectors[both.r_index]
        scale = query_scale(q, both.task_id, both.query_id, seed=seed)
        grids = affinity_grid(q, r, both.model_id, both.query_id, both.task_id, {1.: scale}, q, names)
        for kernel in ['constant', 'task', 'rbf_1']:
            a = normalize_affinity(*grids[kernel], 'model')
            d2 = np.diag(a)[:, None]+np.diag(a)[None, :]-2*a
            if d2.min() < -1e-8:
                raise ValueError('unexpected negative PSD distance')
            d = np.sqrt(np.maximum(d2, 0))
            n = len(models)
            between = float(np.median(d[:n, :n][np.triu_indices(n, 1)]))
            for i, model in enumerate(models):
                outputs.append(dict(profile=profile, kernel=kernel, model_id=model, budget=128,
                                    within_model_shift=float(d[i, i+n]), median_between_model_distance=between,
                                    shift_to_between_ratio=float(d[i, i+n]/max(between, 1e-15))))
    result = pd.DataFrame(outputs)
    result.to_csv(out/'cache_design_distances.csv', index=False)
    result.groupby(['profile', 'kernel']).agg(
        median_shift_ratio=('shift_to_between_ratio', 'median'),
        min_shift_ratio=('shift_to_between_ratio', 'min'),
        max_shift_ratio=('shift_to_between_ratio', 'max')).to_csv(out/'cache_design_summary.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, layout='constrained')
    for ax, profile in zip(axes, ['four-task-mix', 'low-score-items']):
        for i, kernel in enumerate(['constant', 'task', 'rbf_1']):
            values = result[(result.profile == profile) & (result.kernel == kernel)].shift_to_between_ratio
            ax.scatter(np.full(len(values), i), values, alpha=.75)
        ax.axhline(1, color='black', linestyle='--', linewidth=1)
        ax.set(xticks=[0, 1, 2], xticklabels=['Constant', 'Task', 'RBF'], title=profile)
    fig.supylabel('Cache shift / median between-model distance', fontsize=10)
    fig.suptitle('10 fixed models, 128 responses per cache; full PSD geometry', fontsize=12)
    fig.savefig(out/'cache_design.png', dpi=160); plt.close(fig)
    return models


def matched_mask(scores, k, mode, seed):
    """Exactly k observed tasks per model and at least one observation per task."""
    rng = np.random.default_rng(seed)
    if not 1 <= k < scores.shape[1]:
        raise ValueError('invalid task budget')
    priority = rng.random(scores.shape) if mode == 'uniform' else scores+rng.random(scores.shape)*1e-9
    mask = np.zeros(scores.shape, bool)
    for i in range(len(scores)):
        mask[i, np.argsort(priority[i])[-k:]] = True
    for task in np.flatnonzero(~mask.any(0)):
        options = [(priority[i, task]-priority[i, old], i, old)
                   for i, old in np.argwhere(mask) if mask[:, old].sum() > 1]
        if not options:
            raise ValueError('cannot preserve per-model budgets and task coverage')
        _, i, old = max(options)
        mask[i, old] = False; mask[i, task] = True
    if not np.all(mask.sum(1) == k) or not mask.any(0).all():
        raise AssertionError('mask budget invariant failed')
    return mask


def missingness_probe(pool, vectors, truth, out, seed=0):
    models, tasks = sorted(pool.model_id.unique()), sorted(pool.task_id.unique())
    # Oracle scores are confined to this design function and the final evaluator.
    y = matrix(truth, models, tasks, 'y_full')
    masks = {mode: matched_mask(y, 8, mode, seed) for mode in ['uniform', 'hide-poor']}
    common_hidden = ~masks['uniform'] & ~masks['hide-poor']
    outputs, info = [], []
    for mode, mask in masks.items():
        cells = {(models[i], tasks[j]) for i, j in np.argwhere(mask)}
        permitted = observe(pool, 16, seed, cells)
        predictions, diagnostics, selections, counts = predict_completion(permitted, vectors, models, tasks, seed)
        pd.DataFrame(diagnostics).to_csv(out/f'{mode}_geometry.csv', index=False)
        pd.DataFrame(selections).to_csv(out/f'{mode}_selection.csv', index=False)
        for method, pred in predictions.items():
            for i, j in np.argwhere(~mask):
                outputs.append(dict(mode=mode, method=method, model_id=models[i], task_id=tasks[j],
                                    prediction=float(pred[i, j]), y_full=float(y[i, j]),
                                    common_hidden=bool(common_hidden[i, j]),
                                    absolute_error=float(abs(pred[i, j]-y[i, j]))))
        info.append(dict(mode=mode, observed_cells=int(mask.sum()), rows_per_model=128,
                         selected_score_mean=float(y[mask].mean()), hidden_score_mean=float(y[~mask].mean()),
                         common_hidden_cells=int(common_hidden.sum()),
                         mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(), **counts))
        print('missingness probe', mode, flush=True)
    df = pd.DataFrame(outputs)
    df.to_csv(out/'missingness_predictions.csv.gz', index=False)
    summaries = []
    for label, subset in [('all-hidden', df), ('common-hidden', df[df.common_hidden])]:
        means = subset.groupby(['mode', 'method', 'task_id']).absolute_error.mean().groupby(['mode', 'method']).mean()
        summaries.append(means.rename('mae').reset_index().assign(evaluation=label))
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(out/'missingness_summary.csv', index=False)
    (out/'missingness_masks.json').write_text(json.dumps(info, indent=2)+'\n')
    fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    methods = ['model-expanded', 'constant', 'additive', 'lrmc', 'ensemble-nested']
    for off, mode in [(-.18, 'uniform'), (.18, 'hide-poor')]:
        sub = summary[(summary['mode'] == mode) & (summary.evaluation == 'common-hidden')].set_index('method')
        ax.bar(np.arange(len(methods))+off, sub.loc[methods, 'mae'], width=.36, label=mode)
    ax.set(xticks=np.arange(len(methods)), xticklabels=methods, ylabel='Task macro MAE on common hidden cells',
           title='One-seed oracle missingness stress test; equal response budgets')
    ax.legend(); ax.tick_params(axis='x', labelsize=8)
    fig.savefig(out/'missingness.png', dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=Path('data/eee-followups'))
    ap.add_argument('--out', type=Path, default=Path('projects/pkps/artifacts/stress-test/cache-design'))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pool, vectors = load_cache(args.source)
    truth = pd.read_parquet(args.source/'targets.parquet')
    with threadpool_limits(limits=2):
        models = nuisance_probe(pool, vectors, args.out)
        missingness_probe(pool, vectors, truth, args.out)
    (args.out/'config.json').write_text(json.dumps(dict(seed=0, nuisance_models=models,
        implementation_sha256={**code_provenance(), 'stress_cache_design.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        embedding=json.loads((args.source/'embeddings/config.json').read_text()),
        note='Controlled cache-design/oracle missingness probes; not natural missingness or deployment validation'), indent=2)+'\n')
    print('Cache-design probes complete.', flush=True)


if __name__ == '__main__':
    main()
