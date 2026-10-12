"""Statistical-unit and matched-support checks for the uncertainty audit."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'projects/pkps/scripts'))
from experiments.audit_stress_uncertainty import (
    paired_cells, paired_seed_intervals, crossed_draws, query_matrix)


def paired_fixture():
    rows = []
    for seed in range(16):
        for model in ['f/a', 'f/b', 'g/c']:
            for task in ['a', 'b']:
                # Large common seed effects must cancel in a paired contrast.
                for method in ['a', 'b']:
                    pred = .05+seed/32
                    rows.append(dict(seed=seed, model_id=model, task_id=task, suite=task,
                                     method=method, prediction=pred, y_full=0., absolute_error=pred))
    return pd.DataFrame(rows)


def test_whole_run_pairing_cancels_common_seed_variation():
    frame = paired_fixture()
    pair = paired_cells(frame, 'a', 'b')
    means = pair.groupby(['seed', 'task_id']).difference.mean().groupby('seed').mean()
    result = paired_seed_intervals(means, 44, n_boot=1000)
    assert frame.groupby('seed').absolute_error.mean().std() > .1
    for key in ['difference', 'seed_sem', 't_lower', 't_upper', 'bootstrap_lower',
                'bootstrap_upper', 'bonferroni_lower', 'bonferroni_upper']:
        assert result[key] == 0


@pytest.mark.parametrize('corruption', ['missing', 'duplicate', 'target', 'nonfinite'])
def test_pairing_rejects_invalid_support(corruption):
    frame = paired_fixture()
    if corruption == 'missing':
        frame = frame.iloc[1:]
    elif corruption == 'duplicate':
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif corruption == 'target':
        frame.loc[0, 'y_full'] = .5
        frame.loc[0, 'absolute_error'] = abs(frame.loc[0, 'prediction']-.5)
    else:
        frame.loc[0, 'prediction'] = np.nan
    with pytest.raises(ValueError):
        paired_cells(frame, 'a', 'b')


def test_crossed_cluster_statistic_matches_explicit_duplicate_clusters():
    # Unequal cluster sizes expose accidental equal-family/equal-suite weighting.
    d = np.array([[0., 1., 2.], [3., 4., 5.], [20., 30., 40.]])
    families = np.array(['f', 'f', 'g'])
    suites = np.array(['a', 'b', 'b'])
    fw = np.array([[1, 1], [2, 1], [0, 3]])
    bw = np.array([[1, 1], [1, 2], [2, 0]])
    draws = crossed_draws(d, families, suites, fw, bw)
    expected = []
    for f, b in zip(fw, bw):
        rows = np.repeat(np.arange(3), f[[0, 0, 1]])
        cols = np.repeat(np.arange(3), b[[0, 1, 1]])
        expected.append(d[np.ix_(rows, cols)].mean())
    np.testing.assert_allclose(draws, expected, atol=1e-14)
    assert draws[0] == pytest.approx(d.mean())
    assert draws[2] == 20


def test_query_matrix_rejects_averaging_incomplete_seed_sets():
    pair = paired_cells(paired_fixture(), 'a', 'b')
    d, families, suites = query_matrix(pair)
    assert d.shape == (3, 2)
    with pytest.raises(ValueError, match='every seed'):
        query_matrix(pair.iloc[1:])


def test_bonferroni_contains_pointwise_interval():
    values = np.array([-.1, .2, .01, .03, -.02, .02])
    result = paired_seed_intervals(values, 44, n_boot=1000)
    assert result['bonferroni_lower'] < result['t_lower'] < result['difference']
    assert result['difference'] < result['t_upper'] < result['bonferroni_upper']
