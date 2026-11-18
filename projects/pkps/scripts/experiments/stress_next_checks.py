"""Additional EEE checks: score-only query prediction and sparse/temporal completion.

Historical results stay unchanged. Every learner receives permitted labels only;
full hidden scores are joined by the evaluator after predictions are fixed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.stress_followups import observe, matrix, family, load_cache
from experiments.stress_completion import inner_split, rows_for_mask, predict_completion
from experiments.audit_stress_uncertainty import paired_cells, paired_seed_intervals

ROOT = Path('projects/pkps/artifacts/stress-test')
ADDITIVE_GRID = [(scale, strength) for scale in ['raw', 'logit-standardized']
                 for strength in [0., .25, .5, .75, 1.]]


def additive_predict(reference, target_sample, scale='raw', strength=1.):
    """Estimate target strength from its OTHER tasks; fit task effects on references.

    On the raw scale with strength=1 this is the balanced additive least-squares
    model. The logit variant uses reference-fitted column scaling, as completion
    does, but never fits task statistics on the target's held-out label.
    """
    ref, sample = map(lambda x: np.asarray(x, dtype=float), (reference, target_sample))
    if not len(ref) or ref.shape[1] < 2 or not np.isfinite(ref).all() or not np.isfinite(sample).all():
        raise ValueError('additive query prediction requires finite references and at least two tasks')
    if scale == 'logit-standardized':
        ref, sample = logit(np.clip(ref, .001, .999)), logit(np.clip(sample, .001, .999))
    elif scale != 'raw':
        raise ValueError(scale)
    mu = ref.mean(0)
    sd = ref.std(0) if scale == 'logit-standardized' else np.ones(ref.shape[1])
    sd = np.where(sd > 1e-8, sd, 1.)
    residual = (sample-mu)/sd
    # Algebraic leave-one-task-out: the predicted task cannot contribute to strength.
    effect = (residual.sum(1, keepdims=True)-residual)/(ref.shape[1]-1)
    pred = mu + strength*effect*sd
    return expit(pred) if scale == 'logit-standardized' else np.clip(pred, 0, 1)


def additive_fold(models, sample, allowed_reference, outer_family, regime):
    fams = np.array(list(map(family, models)))
    refs, target = np.flatnonzero(fams != outer_family), np.flatnonzero(fams == outer_family)
    if regime == 'cache-only':
        if allowed_reference is not None:
            raise ValueError('cache-only cannot receive full labels')
        labels = sample
    elif regime == 'full-reference':
        labels = np.asarray(allowed_reference, dtype=float)
        if np.isfinite(labels[target]).any():
            raise ValueError('outer family full labels must be absent')
    else:
        raise ValueError(regime)
    losses = np.zeros(len(ADDITIVE_GRID))
    count = 0
    for validation_family in sorted(set(fams[refs])):
        train = refs[fams[refs] != validation_family]
        valid = refs[fams[refs] == validation_family]
        for k, (scale, strength) in enumerate(ADDITIVE_GRID):
            pred = additive_predict(labels[train], sample[valid], scale, strength)
            losses[k] += abs(pred-labels[valid]).sum()
        count += sample[valid].size
    best = int(np.argmin(losses))
    predictions = {
        'additive-raw-fixed': additive_predict(labels[refs], sample[target]),
        'additive-logit-fixed': additive_predict(labels[refs], sample[target], 'logit-standardized'),
        'additive-nested': additive_predict(labels[refs], sample[target], *ADDITIVE_GRID[best]),
    }
    return target, predictions, dict(outer_family=outer_family, regime=regime,
        scale=ADDITIVE_GRID[best][0], strength=ADDITIVE_GRID[best][1], validation_mae=losses[best]/count)


def evaluate(predictions, truth):
    out = predictions.merge(truth[['model_id', 'task_id', 'suite', 'y_full']],
                            on=['model_id', 'task_id'], validate='many_to_one')
    if not np.isfinite(out[['prediction', 'y_full']]).all().all():
        raise ValueError('every evaluated cell needs a finite prediction and target')
    out['absolute_error'] = abs(out.prediction-out.y_full)
    out['squared_error'] = (out.prediction-out.y_full)**2
    return out


def summarize(frame, groups):
    runs = frame.groupby(groups+['seed', 'task_id']).absolute_error.mean().groupby(groups+['seed']).mean()
    out = runs.groupby(groups).agg(mae='mean', seed_sd='std', n_seeds='count').reset_index()
    out['seed_sem'] = out.seed_sd/np.sqrt(out.n_seeds)
    return out


def run_additive(source, out, seeds):
    pool = pd.read_parquet(source/'embeddings/rows.parquet')
    truth = pd.read_parquet(source/'targets.parquet')
    models, tasks = sorted(pool.model_id.unique()), sorted(pool.task_id.unique())
    full = matrix(truth, models, tasks, 'y_full')
    fams = np.array(list(map(family, models)))
    results, choices = [], []
    for seed in seeds:
        for m in [1, 4, 8]:
            sample = matrix(observe(pool, m, seed), models, tasks)
            rows = []
            for regime in ['full-reference', 'cache-only']:
                for fam in sorted(set(fams)):
                    permitted = None
                    if regime == 'full-reference':
                        permitted = full.copy()
                        permitted[fams == fam] = np.nan
                    target, estimates, selection = additive_fold(models, sample, permitted, fam, regime)
                    choices.append(dict(seed=seed, m=m, **selection))
                    for method, pred in estimates.items():
                        for a, i in enumerate(target):
                            for j, task in enumerate(tasks):
                                rows.append(dict(regime=regime, method=method, model_id=models[i],
                                    task_id=task, family=fam, prediction=pred[a, j], seed=seed, m=m))
            frame = evaluate(pd.DataFrame(rows), truth)
            # The own-sample values prove exact mask/budget agreement with the archive.
            archived = pd.read_csv(ROOT/f'query-16seed/qe_m{m}_seed{seed}.predictions.csv.gz')
            old = archived[(archived.regime == 'full-reference') & (archived.method == 'sample')]
            old_matrix = old.pivot(index='model_id', columns='task_id', values='prediction').loc[models, tasks]
            np.testing.assert_allclose(sample, old_matrix.to_numpy(), atol=1e-15)
            results.append(frame)
    frame = pd.concat(results, ignore_index=True)
    frame.to_csv(out/'additive_predictions.csv.gz', index=False)
    pd.DataFrame(choices).to_csv(out/'additive_selection.csv', index=False)
    summarize(frame, ['regime', 'm', 'method']).to_csv(out/'additive_summary.csv', index=False)
    contrasts = []
    for (regime, m), group in frame.groupby(['regime', 'm']):
        old = pd.concat([pd.read_csv(ROOT/f'query-16seed/qe_m{m}_seed{s}.predictions.csv.gz') for s in seeds])
        both = pd.concat([group, old[old.regime == regime]])
        for baseline in ['additive-raw-fixed', 'additive-logit-fixed', 'additive-nested']:
            paired = paired_cells(both, 'model-expanded', baseline)
            diffs = paired.groupby(['seed', 'task_id']).difference.mean().groupby('seed').mean()
            contrasts.append(dict(regime=regime, m=m, method='model-expanded', baseline=baseline,
                                  **paired_seed_intervals(diffs, 18, n_boot=10000)))
    pd.DataFrame(contrasts).to_csv(out/'additive_paired_intervals.csv', index=False)
    print('additive complete', flush=True)


def temporal_design(truth, metadata, fraction):
    """Retrospective evaluation-time split; not a model-release/publication split."""
    dates = pd.DataFrame([{k:r[k] for k in ['model_id','suite','evaluation_timestamp']} for r in metadata])
    dates['evaluated_at'] = pd.to_datetime(dates.evaluation_timestamp, utc=True, errors='raise')
    if dates.evaluated_at.isna().any() or dates.duplicated(['model_id', 'suite']).any():
        raise ValueError('missing or ambiguous evaluation timestamps')
    cutoff = dates.evaluated_at.sort_values().iloc[int(len(dates)*fraction)-1]
    cells = truth[['model_id','task_id','suite']].merge(dates[['model_id','suite','evaluated_at']],
                                                     on=['model_id','suite'], validate='many_to_one')
    early = cells[cells.evaluated_at <= cutoff]
    models, tasks = sorted(early.model_id.unique()), sorted(early.task_id.unique())
    future = cells[cells.evaluated_at > cutoff]
    supported = future[future.model_id.isin(models) & future.task_id.isin(tasks)]
    counts = dict(cutoff=cutoff.isoformat(), fraction=fraction, n_early_cells=len(early),
        n_future_known_cells=len(future), n_supported_future_cells=len(supported),
        n_cold_start_cells=int((~future.model_id.isin(models)).sum()),
        n_new_task_cells=int((~future.task_id.isin(tasks)).sum()))
    return early, supported, models, tasks, counts


def run_sparse(source, out, seeds):
    pool, vectors = load_cache(source)
    truth = pd.read_parquet(source/'targets.parquet')
    metadata = json.loads((source/'run_metadata.json').read_text())
    all_models, all_tasks = sorted(pool.model_id.unique()), sorted(pool.task_id.unique())
    natural = np.isfinite(matrix(truth, all_models, all_tasks, 'y_full'))
    designs = []
    for fraction in [.6, .8]:
        early, test, models, tasks, counts = temporal_design(truth, metadata, fraction)
        designs.append((f'temporal-{fraction:g}', early, test, models, tasks, counts))
    results, audits = [], []
    for seed in seeds:
        observed, test_mask = inner_split(natural, seed, fraction=.2)
        permitted_cells = {(all_models[i],all_tasks[j]) for i,j in np.argwhere(observed)}
        test = pd.DataFrame([dict(model_id=all_models[i],task_id=all_tasks[j]) for i,j in np.argwhere(test_mask)])
        specs = [('natural', permitted_cells, test, all_models, all_tasks,
                  dict(n_naturally_observed=int(natural.sum()), n_naturally_missing=int((~natural).sum()),
                       n_early_cells=int(observed.sum()), n_supported_future_cells=len(test)))]
        specs += [(label, set(zip(early.model_id,early.task_id)), test, models, tasks, counts)
                  for label,early,test,models,tasks,counts in designs]
        for label, cells, test, models, tasks, counts in specs:
            prefix = out/f'{label}_seed{seed}'
            pred_path = Path(str(prefix)+'.predictions.csv.gz')
            info_path = Path(str(prefix)+'.complete.json')
            if pred_path.exists() and info_path.exists():
                results.append(pd.read_csv(pred_path)); audits.append(json.loads(info_path.read_text()))
                continue
            permitted = observe(pool, 16, seed, cells)
            estimates, _, choices, fit_counts = predict_completion(permitted, vectors, models, tasks, seed)
            index_m, index_t = {m:i for i,m in enumerate(models)}, {t:j for j,t in enumerate(tasks)}
            rows = []
            for method, pred in estimates.items():
                for row in test.itertuples():
                    rows.append(dict(method=method, model_id=row.model_id, task_id=row.task_id,
                                     prediction=pred[index_m[row.model_id], index_t[row.task_id]]))
            frame = evaluate(pd.DataFrame(rows), truth).assign(seed=seed, design=label)
            # Mark models outside the original complete five-benchmark cohort.
            counts_per_model = truth.groupby('model_id').suite.nunique()
            frame['originally_incomplete'] = frame.model_id.map(counts_per_model) < 5
            info = dict(design=label, seed=seed, n_models=len(models), n_tasks=len(tasks),
                        observed_cell_sha256=hashlib.sha256(json.dumps(sorted(cells)).encode()).hexdigest(),
                        **counts, **fit_counts)
            frame.to_csv(pred_path,index=False)
            pd.DataFrame(choices).to_csv(str(prefix)+'.selection.csv',index=False)
            info_path.write_text(json.dumps(info,indent=2)+'\n')
            results.append(frame); audits.append(info)
            print(label,seed,'evaluated',len(test),'cells',flush=True)
    frame = pd.concat(results,ignore_index=True)
    summarize(frame,['design','method']).to_csv(out/'sparse_summary.csv',index=False)
    subset=frame[frame.originally_incomplete]
    summarize(subset,['design','method']).to_csv(out/'incomplete_models_summary.csv',index=False)
    pd.DataFrame(audits).to_csv(out/'sparse_coverage.csv',index=False)
    intervals=[]
    for design,group in frame.groupby('design'):
        for baseline in ['constant','shuffled-rbf','additive','lrmc']:
            pair=paired_cells(group,'model-expanded',baseline)
            diffs=pair.groupby(['seed','task_id']).difference.mean().groupby('seed').mean()
            intervals.append(dict(design=design,method='model-expanded',baseline=baseline,
                                  **paired_seed_intervals(diffs,12,n_boot=10000)))
    pd.DataFrame(intervals).to_csv(out/'sparse_paired_intervals.csv',index=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['additive','sparse'])
    ap.add_argument('--source',type=Path,default=Path('data/eee-followups'))
    ap.add_argument('--out',type=Path,default=ROOT/'next-checks')
    ap.add_argument('--seeds',type=int,nargs='+',default=list(range(16)))
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    config=dict(mode=args.mode,seeds=args.seeds,source=str(args.source),
                query_additive_candidates=ADDITIVE_GRID,query_target_strength='other 15 sample scores only',
                natural_holdout='20% of known cells, preserving row/column support',
                temporal_holdout='60% and 80% evaluation-time cutoffs; no future responses or scores',
                publication_times_verified=False,completion_queries_per_observed_cell=16,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.out/f'{args.mode}_config.json').write_text(json.dumps(config,indent=2)+'\n')
    with threadpool_limits(limits=2):
        (run_additive if args.mode=='additive' else run_sparse)(args.source,args.out,args.seeds)


if __name__=='__main__':
    main()
