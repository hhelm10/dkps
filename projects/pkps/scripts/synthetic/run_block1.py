import argparse
from pathlib import Path

from .block1 import (
    run_exp_n_models, run_exp_n_tasks, run_exp_task_parity,
    run_exp_query_sparsity, run_exp_rho, run_exp_query_efficiency,
)
from .plots import save_figure


PRESETS = {
    'paper': {
        'n_models': dict(
            n_models=(10, 20, 50, 75, 100),
            n_seeds=50,
        ),
        'n_tasks': dict(
            n_tasks=(5, 10, 20, 30, 40, 50),
            n_seeds=50,
        ),
        'task_parity': dict(
            obs_prob=(0.1, 0.2, 0.3, 0.5, 0.7, 0.9),
            n_seeds=50,
        ),
        'query_sparsity': dict(
            query_obs_prob=(0.1, 0.2, 0.3, 0.5, 0.7, 1.0),
            n_seeds=50,
        ),
        'rho': dict(
            rhos=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            n_seeds=50,
        ),
        'query_efficiency': dict(
            budgets=(1, 2, 4, 8, 16, 32),
            rhos=(0.0, 1.0),
            n_seeds=50,
        ),
    },
    'quick': {
        'n_models': dict(
            n_models=(10, 50, 100),
            n_seeds=3,
        ),
        'n_tasks': dict(
            n_tasks=(5, 10, 20),
            n_seeds=3,
        ),
        'task_parity': dict(
            obs_prob=(0.15, 0.3, 0.7),
            n_seeds=3,
        ),
        'query_sparsity': dict(
            query_obs_prob=(0.3, 0.7, 1.0),
            n_seeds=3,
        ),
        'rho': dict(
            rhos=(0.0, 0.5, 1.0),
            n_seeds=3,
        ),
        'query_efficiency': dict(
            budgets=(1, 8, 32),
            rhos=(0.0, 1.0),
            n_seeds=3,
        ),
    },
}

RUNNERS = {
    'n_models': run_exp_n_models,
    'n_tasks': run_exp_n_tasks,
    'task_parity': run_exp_task_parity,
    'query_sparsity': run_exp_query_sparsity,
    'rho': run_exp_rho,
    'query_efficiency': run_exp_query_efficiency,
}


def _save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_block1_experiments(experiment, preset):
    config = PRESETS[preset]
    results = {}
    exps = list(RUNNERS.keys()) if experiment == 'all' else [experiment]
    for exp in exps:
        results[exp] = RUNNERS[exp](**config[exp])
    return results


def save_block1_results(results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        _save_csv(df, output_dir / f'{name}.csv')
    save_figure(output_dir, results)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Run double-kernel DKPS synthetic experiments.')
    parser.add_argument('--experiment',
                        choices=list(RUNNERS.keys()) + ['all'], default='all')
    parser.add_argument('--preset', choices=sorted(PRESETS), default='paper')
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir or f'projects/pkps/scripts/synthetic/results/{args.preset}')
    results = run_block1_experiments(args.experiment, args.preset)
    save_block1_results(results, output_dir)

    print(f'wrote outputs to {output_dir}')
    for name in sorted(results):
        print(f'  {name}: {len(results[name])} rows')


if __name__ == '__main__':
    main()
