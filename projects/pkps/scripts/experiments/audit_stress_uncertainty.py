"""Audit uncertainty in saved EEE predictions, without refitting or embedding.

Whole-run intervals target random masks/cohort selection conditional on the frozen
data and protocol. Crossed-cluster intervals are descriptive sensitivity ranges,
not calibrated confidence intervals for new developers/benchmarks. All contrasts
are paired before aggregation; a missing method/cell is a hard error.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t
from threadpoolctl import threadpool_limits


def paired_cells(frame, method, baseline):
    keys = ['seed', 'model_id', 'task_id']
    chosen = frame[frame.method.isin([method, baseline])].copy()
    if chosen.duplicated(['method']+keys).any():
        raise ValueError('duplicate prediction keys')
    if not np.isfinite(chosen[['prediction', 'y_full', 'absolute_error']]).all().all():
        raise ValueError('nonfinite prediction or loss')
    if not np.allclose(abs(chosen.prediction-chosen.y_full), chosen.absolute_error, atol=1e-12):
        raise ValueError('stored error disagrees with prediction and target')
    a = chosen[chosen.method == method].set_index(keys).sort_index()
    b = chosen[chosen.method == baseline].set_index(keys).sort_index()
    if not len(a) or not a.index.equals(b.index):
        raise ValueError('methods must predict exactly the same cells and seeds')
    if not np.array_equal(a.suite, b.suite) or not np.allclose(a.y_full, b.y_full, atol=1e-12):
        raise ValueError('mismatched evaluation targets or suites')
    out = a[['suite']].copy()
    out['difference'] = a.absolute_error-b.absolute_error
    out = out.reset_index()
    out['family'] = out.model_id.str.split('/').str[0]
    return out


def paired_seed_intervals(values, n_comparisons, n_boot=100000, rng_seed=71):
    """Approximate mean-over-random-runs CIs, with whole-run pairing intact."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('at least two finite seed differences are required')
    if n_comparisons < 1:
        raise ValueError('comparison family must be nonempty')
    n = len(values)
    mean = float(values.mean())
    sem = float(values.std(ddof=1)/np.sqrt(n))
    critical = t.ppf(.975, n-1)
    adjusted = t.ppf(1-.05/(2*n_comparisons), n-1)
    counts = np.random.default_rng(rng_seed).multinomial(n, np.ones(n)/n, size=n_boot)
    lo, hi = np.quantile(counts@values/n, [.025, .975])
    return dict(difference=mean, n_seeds=n, seed_sem=sem,
                t_lower=mean-critical*sem, t_upper=mean+critical*sem,
                bootstrap_lower=float(lo), bootstrap_upper=float(hi),
                bonferroni_lower=mean-adjusted*sem, bonferroni_upper=mean+adjusted*sem,
                comparison_family_size=n_comparisons,
                n_boot=n_boot, rng_seed=rng_seed,
                estimand='mean task-macro difference over randomized runs; frozen data/protocol')


def crossed_draws(difference, model_families, task_suites, family_counts, suite_counts):
    """Exact weighted paper statistic for complete model-by-task paired losses.

    Sampling clusters preserves their original sizes. The statistic remains
    model-weighted and task-macro, not equally weighted across families/suites.
    Counts are multiplicities, not independently sampled cells.
    """
    d = np.asarray(difference, dtype=float)
    families, fi = np.unique(model_families, return_inverse=True)
    suites, bi = np.unique(task_suites, return_inverse=True)
    if d.shape != (len(fi), len(bi)) or not np.isfinite(d).all():
        raise ValueError('complete finite model-by-task differences are required')
    fw, bw = np.asarray(family_counts), np.asarray(suite_counts)
    if fw.ndim != 2 or bw.ndim != 2 or fw.shape != (len(bw), len(families)) or bw.shape[1] != len(suites):
        raise ValueError('invalid cluster count shapes')
    if (fw < 0).any() or (bw < 0).any() or (fw.sum(1) == 0).any() or (bw.sum(1) == 0).any():
        raise ValueError('invalid cluster multiplicities')
    totals = np.zeros((len(families), len(suites)))
    for i in range(len(families)):
        for j in range(len(suites)):
            totals[i, j] = d[np.ix_(fi == i, bi == j)].sum()
    denom = (fw@np.bincount(fi))*(bw@np.bincount(bi))
    return np.einsum('rf,fg,rg->r', fw, totals, bw, optimize=True)/denom


def query_matrix(pair):
    counts = pair.groupby(['model_id', 'task_id']).seed.nunique()
    if not counts.eq(pair.seed.nunique()).all():
        raise ValueError('every query cell must have every seed')
    means = pair.groupby(['model_id', 'task_id']).difference.mean().unstack()
    if means.isna().any().any():
        raise ValueError('incomplete query design')
    if pair.groupby('task_id').suite.nunique().max() != 1:
        raise ValueError('task has inconsistent suite')
    families = np.array([m.split('/')[0] for m in means.index])
    suites = pair.groupby('task_id').suite.first().reindex(means.columns).to_numpy()
    return means.to_numpy(), families, suites


def cluster_audit(pair, n_boot=100000):
    d, families, suites = query_matrix(pair)
    fs, bs = np.unique(families), np.unique(suites)
    rows = []
    # Independent RNG runs diagnose Monte Carlo variability, not statistical validity.
    for rng_seed in [71, 1701, 9035]:
        rng = np.random.default_rng(rng_seed)
        fw = rng.multinomial(len(fs), np.ones(len(fs))/len(fs), size=n_boot)
        bw = rng.multinomial(len(bs), np.ones(len(bs))/len(bs), size=n_boot)
        for scheme, f, b in [('developer+benchmark', fw, bw),
                              ('developer-only', fw, np.ones_like(bw)),
                              ('benchmark-only', np.ones_like(fw), bw)]:
            draws = crossed_draws(d, families, suites, f, b)
            for n in [1000, n_boot]:
                lo, hi = np.quantile(draws[:n], [.025, .975])
                rows.append(dict(scheme=scheme, rng_seed=rng_seed, n_boot=n,
                                 difference=float(d.mean()), lower=float(lo), upper=float(hi),
                                 interpretation='pointwise frozen-prediction sensitivity; not calibrated population CI'))
    influence = []
    for kind, groups, axis in [('developer', families, 0), ('benchmark', suites, 1)]:
        for group in np.unique(groups):
            subset = d[groups != group] if axis == 0 else d[:, groups != group]
            influence.append(dict(cluster_type=kind, omitted_cluster=group,
                                  difference=float(subset.mean()), refitted=False))
    weights = []
    for model_rule in ['model', 'developer']:
        mw = np.ones(len(families)) if model_rule == 'model' else np.array([1/np.sum(families == f) for f in families])
        for task_rule in ['task', 'benchmark']:
            tw = np.ones(len(suites)) if task_rule == 'task' else np.array([1/np.sum(suites == b) for b in suites])
            weights.append(dict(model_weighting=model_rule, task_weighting=task_rule,
                                difference=float(mw@d@tw/mw.sum()/tw.sum())))
    return rows, influence, weights


def few_cluster_simulation(task_counts, n_datasets=2000, n_boot=4000, seed=519):
    """Counterexample to universal coverage, not an estimated coverage for EEE.

    Independent Gaussian benchmark effects, shared across all models and tasks
    inside a benchmark; true population difference zero. This intentionally simple
    setting has no model-fitting uncertainty. Scale does not affect coverage.
    """
    rng = np.random.default_rng(seed)
    sizes = np.asarray(task_counts, dtype=float)
    k = len(sizes)
    effects = rng.normal(0, .01, (n_datasets, k))
    counts = rng.multinomial(k, np.ones(k)/k, size=n_boot)
    weights = counts*sizes
    weights = weights/weights.sum(1, keepdims=True)
    covered = []
    for start in range(0, n_datasets, 100):
        draws = effects[start:start+100]@weights.T
        lo, hi = np.quantile(draws, [.025, .975], axis=1)
        covered.extend((lo <= 0) & (hi >= 0))
    coverage = float(np.mean(covered))
    return dict(n_datasets=n_datasets, n_boot=n_boot, rng_seed=seed,
                task_counts=sizes.astype(int).tolist(), nominal_coverage=.95,
                measured_coverage=coverage,
                simulation_sem=float(np.sqrt(coverage*(1-coverage)/n_datasets)),
                note='Illustrative benchmark-effect Gaussian simulation; not calibrated EEE coverage')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path('projects/pkps/artifacts/stress-test'))
    ap.add_argument('--n-boot', type=int, default=100000)
    args = ap.parse_args()
    if args.n_boot < 1000:
        raise ValueError('use at least 1000 bootstrap draws')
    out = args.root/'uncertainty-audit'
    out.mkdir(parents=True, exist_ok=True)
    q = pd.concat([pd.read_csv(p) for p in sorted((args.root/'query-16seed').glob('*.predictions.csv.gz'))], ignore_index=True)
    c = pd.concat([pd.read_csv(p) for p in sorted((args.root/'completion-16seed').glob('*.predictions.csv.gz'))], ignore_index=True)
    seeds, runs, clusters, influence, weighting, reproduction = [], [], [], [], [], []
    original = pd.read_csv(args.root/'query_paired_cluster_intervals.csv').set_index(['regime', 'm', 'baseline'])
    legacy_rng = np.random.default_rng(71)
    with threadpool_limits(limits=2):
        for (regime, m), group in q.groupby(['regime', 'm']):
            for baseline in ['sample', 'constant', 'task', 'shuffled-rbf']:
                meta = dict(experiment='query', regime=regime, m=m, method='model-expanded', baseline=baseline)
                pair = paired_cells(group, 'model-expanded', baseline)
                d, families, suites = query_matrix(pair)
                nf, nb = len(np.unique(families)), len(np.unique(suites))
                # Replay the original loop's exact random draw order independently.
                counts = [(legacy_rng.multinomial(nf, np.ones(nf)/nf),
                           legacy_rng.multinomial(nb, np.ones(nb)/nb)) for _ in range(1000)]
                fw, bw = map(np.array, zip(*counts))
                lo, hi = np.quantile(crossed_draws(d, families, suites, fw, bw), [.025, .975])
                old = original.loc[regime, m, baseline]
                error = max(abs(lo-old.cluster_lower), abs(hi-old.cluster_upper),
                            abs(d.mean()-old.paired_mae_difference))
                if error > 1e-12:
                    raise ValueError('original interval could not be reproduced')
                reproduction.append({**meta, 'original_lower': old.cluster_lower,
                                     'original_upper': old.cluster_upper, 'reproduced_lower': lo,
                                     'reproduced_upper': hi, 'max_absolute_error': error})
                run = pair.groupby(['seed', 'task_id']).difference.mean().groupby('seed').mean()
                seeds.append({**meta, **paired_seed_intervals(run.to_numpy(), 44, args.n_boot)})
                runs.extend([{**meta, 'seed': int(seed), 'difference': float(v)} for seed, v in run.items()])
                a, b, w = cluster_audit(pair, args.n_boot)
                clusters.extend([{**meta, **r} for r in a])
                influence.extend([{**meta, **r} for r in b])
                weighting.extend([{**meta, **r} for r in w])
            print('Audited query', regime, m, flush=True)
        for (n, coverage), group in c.groupby(['n_models', 'coverage']):
            for method, baseline in [('model-expanded', 'constant'), ('model-expanded', 'additive'),
                                     ('model-expanded', 'lrmc'), ('ensemble-nested', 'additive'),
                                     ('ensemble-nested', 'lrmc')]:
                meta = dict(experiment='completion', n_models=n, coverage=coverage, method=method, baseline=baseline)
                pair = paired_cells(group, method, baseline)
                run = pair.groupby(['seed', 'task_id']).difference.mean().groupby('seed').mean()
                seeds.append({**meta, **paired_seed_intervals(run.to_numpy(), 44, args.n_boot)})
                runs.extend([{**meta, 'seed': int(seed), 'difference': float(v)} for seed, v in run.items()])
        if len(seeds) != 44 or any(row['n_seeds'] != 16 for row in seeds):
            raise ValueError('audit expects the complete 44-contrast, 16-seed design')
        pd.DataFrame(seeds).to_csv(out/'paired_run_intervals.csv', index=False)
        pd.DataFrame(runs).to_csv(out/'paired_run_differences.csv', index=False)
        cluster_frame = pd.DataFrame(clusters)
        cluster_frame.to_csv(out/'cluster_sensitivity.csv', index=False)
        stability = []
        for n, group in cluster_frame[cluster_frame.scheme == 'developer+benchmark'].groupby('n_boot'):
            for key, contrast in group.groupby(['regime', 'm', 'baseline']):
                width = (contrast.upper-contrast.lower).mean()
                span = max(contrast.upper.max()-contrast.upper.min(), contrast.lower.max()-contrast.lower.min())
                stability.append(dict(regime=key[0], m=key[1], baseline=key[2], n_boot=n,
                                      n_rng_seeds=len(contrast), max_endpoint_span=span,
                                      endpoint_span_over_mean_width=span/width if width else 0.))
        pd.DataFrame(stability).to_csv(out/'bootstrap_endpoint_stability.csv', index=False)
        pd.DataFrame(influence).to_csv(out/'omit_cluster_sensitivity.csv', index=False)
        pd.DataFrame(weighting).to_csv(out/'weighting_sensitivity.csv', index=False)
        pd.DataFrame(reproduction).to_csv(out/'original_interval_reproduction.csv', index=False)
        sizes = q[['task_id', 'suite']].drop_duplicates().groupby('suite').size()
        sim = few_cluster_simulation(sizes.to_numpy())
        (out/'few_cluster_simulation.json').write_text(json.dumps(sim, indent=2)+'\n')
    config = dict(n_boot=args.n_boot, comparisons=44, query_comparisons=24, completion_comparisons=20,
                  bonferroni_family='all 44 comparisons in this audit; does not cover unlisted/post-hoc research choices',
                  clustered_population_inference='not established; fixed predictions, 13 developers, 5 benchmarks',
                  source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  input_sha256={str(p.relative_to(args.root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for folder in ['query-16seed', 'completion-16seed']
                                for p in sorted((args.root/folder).glob('*.predictions.csv.gz'))},
                  sources={'pigeonhole': 'https://arxiv.org/abs/0712.1111',
                           'few_clusters': 'https://escholarship.org/uc/item/1jq5d0pq',
                           'cv_dependence': 'https://www.jmlr.org/papers/v5/grandvalet04a.html'})
    (out/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    plot_audit(pd.DataFrame(seeds), pd.DataFrame(clusters), out)
    print('Uncertainty audit complete.', sim, flush=True)


def plot_audit(runs, clusters, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, layout='constrained')
    for ax, baseline in zip(axes, ['constant', 'shuffled-rbf']):
        a = runs[(runs.experiment == 'query') & (runs.regime == 'full-reference') & (runs.baseline == baseline)].sort_values('m')
        b = clusters[(clusters.regime == 'full-reference') & (clusters.baseline == baseline)
                     & (clusters.scheme == 'developer+benchmark') & (clusters.rng_seed == 71)
                     & (clusters.n_boot == clusters.n_boot.max())].sort_values('m')
        for offset, frame, lo, hi, label, color in [(-.12, a, 't_lower', 't_upper', 'Run-level t interval: fixed data', '#2862ba'),
                                                   (.12, b, 'lower', 'upper', 'Cluster sensitivity: frozen predictions', '#bd4e45')]:
            x = np.arange(len(frame))+offset
            ax.vlines(x, frame[lo], frame[hi], label=label, color=color)
            ax.scatter(x, frame.difference, s=20, color=color)
        ax.axhline(0, color='black', lw=.8)
        ax.set(xticks=np.arange(3), xticklabels=[1, 4, 8], xlabel='Queries per task', title=f'Expanded selector minus {baseline}')
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Paired MAE difference\nNegative favors expanded selector', fontsize=10)
    axes[1].legend(fontsize=8, loc='lower right')
    fig.suptitle('Different uncertainty questions; pointwise 95% endpoints')
    fig.savefig(out/'uncertainty_comparison.png', dpi=160)
    fig.savefig(out/'uncertainty_comparison.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
