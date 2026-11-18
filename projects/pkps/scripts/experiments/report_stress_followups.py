"""Regenerate compact evidence tables and figures from saved pilot predictions."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
from pipeline.stress_geometry import affinity_components, normalize_affinity


def mathematical_examples(out):
    q = np.array([-1., 1., -2., 2.])[:, None]
    r = np.tile(np.eye(2), (2, 1))
    _, s, mass = affinity_components(q, r, ['a', 'a', 'b', 'b'], np.arange(4), ['t']*4)
    distances = []
    for norm in ['pairwise', 'model']:
        a = normalize_affinity(s, mass, norm)
        distances.append(float(a[0, 0]+a[1, 1]-2*a[0, 1]))
    rectangle = np.array([[0, 0], [2, 0], [0, 1], [2, 1]])
    d = np.linalg.norm(rectangle[:, None]-rectangle[None, :], axis=2)
    stress = []
    for z in [np.array([-1, 1, -1, 1]), np.array([-1.1, .9, -.9, 1.1])]:
        stress.append(float(np.sum((abs(z[:, None]-z[None, :])-d)**2)))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout='constrained')
    for ax, labels, values, title in [
        (axes[0], ['Eq. 1 ratio', 'Proposed normalization'], distances, 'Two-model squared distance'),
        (axes[1], ['Classical MDS, d=1', 'Alternative coordinates'], stress, 'Rectangle: ordered-pair raw stress'),
    ]:
        bars = ax.bar(labels, values, color=['#bd4e45', '#267b76'])
        ax.axhline(0, color='black', lw=.8)
        ax.bar_label(bars, fmt='%.5f', padding=4)
        ax.set_title(title, fontsize=11)
        ax.margins(y=.25)
    fig.suptitle('Constructed counterexamples; these are not EEE estimates')
    fig.savefig(out/'mathematical_counterexamples.png', dpi=160)
    fig.savefig(out/'mathematical_counterexamples.pdf'); plt.close(fig)
    (out/'mathematical_counterexamples.json').write_text(json.dumps(
        dict(squared_distances=dict(zip(['original', 'model_normalized'], distances)),
             raw_stress=dict(zip(['classical', 'alternative'], stress))), indent=2)+'\n')


def archival(out):
    q = pd.read_csv(ROOT/'data/results-eee-rd1/rd1_suite_budget.csv.gz')
    recorded = q.groupby(['m', 'method', 'seed']).mae.mean().groupby(['m', 'method']).mean()
    paper = {1: {'sample': .224, 'irt': .165, 'dkps': .116, 'pkps': .070, 'ens': .068},
             8: {'sample': .081, 'irt': .139, 'dkps': .078, 'pkps': .060, 'ens': .050}}
    results = []
    for m, methods in paper.items():
        for method, value in methods.items():
            results.append(dict(setting=f'query m={m}', method=method, paper=value,
                                archival=float(recorded.loc[m, method])))
    n = pd.read_csv(ROOT/'data/results-eee-unified/completion_suite_n_models.csv.gz')
    c = pd.read_csv(ROOT/'data/results-eee-unified/completion_suite_coverage.csv.gz')
    for setting, data, values in [
        ('completion n=10', n[n.n_models == 10], [.109, .081, .081]),
        ('completion n=45', n[n.n_models == 45], [.096, .070, .069]),
        ('completion sparse n=10', c[(c.n_models == 10) & (c.p_task == .2)], [.144, .108, .108]),
    ]:
        for method, value in zip(['mc', 'pkps', 'ens'], values):
            results.append(dict(setting=setting, method=method, paper=value, archival=float(data[method].mean())))
    df = pd.DataFrame(results)
    df['difference'] = df.archival-df.paper
    df['rounded_csv'] = df.archival.round(3)
    df.to_csv(out/'paper_table1_eee_check.csv', index=False)
    q.groupby(['m', 'method', 'seed']).mae.count().rename('reported_task_count').reset_index().to_csv(
        out/'archival_task_coverage.csv', index=False)
    common = q.pivot(index=['m', 'seed', 'task'], columns='method', values='mae')
    common = common[['sample', 'irt', 'dkps', 'pkps', 'ens']].dropna()
    common.groupby(['m', 'seed']).mean().groupby('m').mean().to_csv(out/'archival_matched_irt_tasks.csv')


def summarize(df, grouping):
    # Match the paper's order: model/cell -> task -> seed -> mean across seeds.
    task = df.groupby(grouping+['seed', 'task_id']).agg(
        mae=('absolute_error', 'mean'), mse=('squared_error', 'mean'),
        n_cells=('prediction', 'size'), n_predicted=('prediction', 'count')).reset_index()
    seed = task.groupby(grouping+['seed'])[['mae', 'mse']].mean().reset_index()
    summary = seed.groupby(grouping).agg(mae=('mae', 'mean'), mse=('mse', 'mean'),
                                        seed_sd=('mae', 'std'), n_seeds=('seed', 'nunique')).reset_index()
    summary['seed_sem'] = summary.seed_sd / np.sqrt(summary.n_seeds)
    return summary, task


def family_benchmark_differences(df):
    """Paired crossed-cluster sensitivity intervals; five benchmarks, fixed cohort.

    Average masking repetitions first. Resample whole developers and benchmarks,
    retaining all tasks/models inside selected clusters. This is exploratory
    uncertainty, not proof of generalization to new domains or model populations.
    """
    estimates = df.groupby(['regime', 'm', 'method', 'model_id', 'task_id', 'suite', 'family']).absolute_error.mean().reset_index()
    outputs = []
    rng = np.random.default_rng(71)
    for (regime, m), group in estimates.groupby(['regime', 'm']):
        for baseline in ['sample', 'constant', 'task', 'shuffled-rbf']:
            a = group[group.method == 'model-expanded']
            b = group[group.method == baseline]
            pair = a.merge(b, on=['model_id', 'task_id', 'suite', 'family'], suffixes=('_a', '_b'))
            pair['diff'] = pair.absolute_error_a - pair.absolute_error_b
            families, benchmarks = sorted(pair.family.unique()), sorted(pair.suite.unique())
            boot = []
            for _ in range(1000):
                fw = dict(zip(families, rng.multinomial(len(families), np.ones(len(families))/len(families))))
                bw = dict(zip(benchmarks, rng.multinomial(len(benchmarks), np.ones(len(benchmarks))/len(benchmarks))))
                w = pair.family.map(fw).to_numpy()
                tmp = pair.assign(num=pair['diff'] * w, denom=w).groupby(['suite', 'task_id'])[['num', 'denom']].sum()
                task_diff = tmp.num/tmp.denom
                weights = np.array([bw[s] for s, _ in task_diff.index])
                if np.isfinite(task_diff).all():
                    boot.append(float(np.average(task_diff, weights=weights)))
            lower, upper = np.quantile(boot, [.025, .975])
            outputs.append(dict(regime=regime, m=m, method='model-expanded', baseline=baseline,
                                paired_mae_difference=float(pair.groupby('task_id')['diff'].mean().mean()),
                                cluster_lower=float(lower), cluster_upper=float(upper),
                                n_families=len(families), n_benchmarks=len(benchmarks), n_boot=len(boot)))
    return pd.DataFrame(outputs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=ROOT/'artifacts/stress-test')
    ap.add_argument('--query-dir', type=Path, help='Saved query-efficiency runs (default: OUT/pilot)')
    ap.add_argument('--completion-dir', type=Path, help='Saved completion runs (default: OUT/completion)')
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    archival(out)
    mathematical_examples(out)
    query_dir = args.query_dir or out/'pilot'
    completion_dir = args.completion_dir or out/'completion'
    paths = sorted(query_dir.glob('*.predictions.csv.gz'))
    if paths:
        df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
        if not np.isfinite(df[['prediction', 'y_full', 'absolute_error', 'squared_error']]).all().all():
            raise ValueError('Query results contain missing/nonfinite predictions; report failures explicitly')
        summary, task = summarize(df, ['regime', 'm', 'method'])
        summary.to_csv(out/'query_summary.csv', index=False)
        task.to_csv(out/'query_task_errors.csv', index=False)
        df.groupby(['regime', 'm', 'seed', 'method']).agg(
            n_cells=('prediction', 'size'), n_predicted=('prediction', 'count')).to_csv(out/'query_coverage.csv')
        benchmark, _ = summarize(df, ['regime', 'm', 'method', 'suite'])
        benchmark.to_csv(out/'query_benchmark_errors.csv', index=False)
        leave_out = []
        for suite in sorted(df.suite.unique()):
            sensitivity, _ = summarize(df[df.suite != suite], ['regime', 'm', 'method'])
            leave_out.append(sensitivity.assign(excluded_benchmark=suite))
        pd.concat(leave_out, ignore_index=True).to_csv(out/'query_leave_benchmark_out.csv', index=False)
        selection = pd.concat([pd.read_csv(p) for p in query_dir.glob('*.selection.csv')], ignore_index=True)
        selection.groupby(['regime', 'm', 'method', 'candidate']).size().rename(
            'outer_folds_selected').reset_index().to_csv(out/'query_selection_counts.csv', index=False)
        family_benchmark_differences(df).to_csv(out/'query_paired_cluster_intervals.csv', index=False)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True, layout='constrained')
        for ax, regime in zip(axes, ['full-reference', 'cache-only']):
            for method in ['sample', 'model-expanded', 'ratio-psd', 'constant', 'task', 'shuffled-rbf']:
                s = summary[(summary.regime == regime) & (summary.method == method)].sort_values('m')
                ax.plot(s.m, s.mae, marker='o', label=method)
            ax.set(title=regime, xlabel='Queries per observed model/task cell', xticks=sorted(df.m.unique()))
            ax.grid(alpha=.2)
        axes[0].set_ylabel('Mean absolute error (task macro average)')
        axes[1].legend(fontsize=8)
        fig.suptitle(f'EEE follow-up · EmbeddingGemma-2 · {df.seed.nunique()} masking seeds · nested developer holdouts')
        fig.savefig(out/'query_pilot.png', dpi=160); fig.savefig(out/'query_pilot.pdf'); plt.close(fig)
        geometry = pd.concat([pd.read_csv(p) for p in query_dir.glob('*.geometry.csv')])
        geometry.to_csv(out/'query_geometry.csv', index=False)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
        for prefix, label in [('model_rbf_', 'Model normalization'), ('ratio_psd_', 'Pairwise ratio')]:
            g = geometry[geometry.candidate.str.startswith(prefix)]
            axes[0].scatter(g.m, g.negative_distance_fraction, label=label, alpha=.6)
            axes[1].scatter(g.m, g.negative_eigenvalue_fraction, label=label, alpha=.6)
        for ax, title in zip(axes, ['Negative squared-distance fraction', 'Negative centered-Gram eigenvalue mass fraction']):
            ax.set(title=title, xlabel='Queries per cell', xticks=sorted(df.m.unique()))
            ax.grid(alpha=.2)
        axes[1].legend(fontsize=8)
        fig.suptitle('Raw diagnostics before clipping, PSD projection, or rank truncation')
        fig.savefig(out/'geometry_diagnostics.png', dpi=160); plt.close(fig)
    paths = sorted(completion_dir.glob('*.predictions.csv.gz'))
    if paths:
        df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
        if not np.isfinite(df[['prediction', 'y_full', 'absolute_error', 'squared_error']]).all().all():
            raise ValueError('Completion results contain missing/nonfinite predictions; report failures explicitly')
        summary, task = summarize(df, ['n_models', 'coverage', 'method'])
        summary.to_csv(out/'completion_summary.csv', index=False)
        task.to_csv(out/'completion_task_errors.csv', index=False)
        df.groupby(['n_models', 'coverage', 'seed', 'method']).agg(
            n_cells=('prediction', 'size'), n_predicted=('prediction', 'count')).to_csv(out/'completion_coverage.csv')
        benchmark, _ = summarize(df, ['n_models', 'coverage', 'method', 'suite'])
        benchmark.to_csv(out/'completion_benchmark_errors.csv', index=False)
        cohorts = sorted(df.n_models.unique())
        fig, axes = plt.subplots(1, len(cohorts), figsize=(5*len(cohorts), 4), squeeze=False, layout='constrained')
        for ax, n in zip(axes[0], cohorts):
            for method in ['model-expanded', 'ratio-psd', 'constant', 'additive', 'lrmc', 'ensemble-nested']:
                s = summary[(summary.n_models == n) & (summary.method == method)].sort_values('coverage')
                ax.plot(s.coverage, s.mae, marker='o', label=method)
            ax.set(title=f'{n} models', xlabel='Requested task coverage', ylabel='Task macro MAE',
                   xticks=sorted(df.coverage.unique()))
            ax.grid(alpha=.2)
        axes[0, -1].legend(fontsize=8)
        fig.suptitle(f'EEE completion · {df.seed.nunique()} masking seeds · validation responses and scores removed')
        fig.savefig(out/'completion_pilot.png', dpi=160); fig.savefig(out/'completion_pilot.pdf'); plt.close(fig)
    smoke = out/'mps_smoke_test.json'
    if smoke.exists():
        data = json.loads(smoke.read_text())
        fig, ax = plt.subplots(figsize=(6, 3), layout='constrained')
        names = ['cpu_float32', 'mps_float32', 'mps_bfloat16']
        ax.barh(names, [data[n]['texts_per_second'] for n in names], color=['#8a8a8a', '#2862ba', '#489d9c'])
        ax.set(xlabel='Texts per second (16 sampled EEE responses)', title='Local M5 smoke test; warmed model, batch size 8')
        fig.savefig(out/'mps_throughput.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    main()
