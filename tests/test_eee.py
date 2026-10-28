"""dkps.eee: EEE datastore *_samples.jsonl -> PKPS records -> fit/predict."""
import json

import numpy as np
import pandas as pd
import pytest

import dkps.embed
from dkps import PKPS, SampleScore
from dkps.eee import load_records, record_to_row


def make_store(tmp_path, n_models=6, n_tasks=2, n_queries=10, bench='bench'):
    """A miniature EEE store: one *_samples.jsonl per model under a benchmark
    directory (store layout), real schema quirks included (string scores,
    is_correct fallback, list outputs)."""
    root = tmp_path / bench
    for i in range(n_models):
        fp = root / f'dev{i % 3}' / f'model-{i}' / 'run_samples.jsonl'
        fp.parent.mkdir(parents=True)
        with open(fp, 'w') as fh:
            for t in range(n_tasks):
                for q in range(n_queries):
                    ev = {'score': str(float((i + q) % 2))} if q % 2 == 0 else \
                         {'is_correct': 'True' if (i + q) % 2 else 'False'}
                    fh.write(json.dumps({
                        'model_id': f'dev{i % 3}/model-{i}',
                        'evaluation_id': f'{bench}/dev{i % 3}_model-{i}/1',
                        'sample_id': f'{bench}_task{t}:{q}',
                        'input': {'raw': f'question {t}-{q}?'},
                        'output': {'raw': [f'answer {i}-{t}-{q}']},
                        'evaluation': ev,
                        'metadata': {'source_task': f'{bench}_task{t}'},
                    }) + '\n')
    return root


def test_record_parsing_quirks():
    base = {'model_id': 'd/m', 'sample_id': 'task_a:0', 'input': {'raw': 'q?'},
            'output': {'raw': ['ans']}, 'evaluation': {'score': '0.5'}}
    row = record_to_row(base)
    assert row['score'] == 0.5 and row['task_id'] == 'task_a' and row['response'] == 'ans'
    assert record_to_row({**base, 'evaluation': {'is_correct': 'True'}})['score'] == 1.0
    assert record_to_row({**base, 'output': {'raw': 'plain string'}})['response'] == 'plain string'
    assert record_to_row({**base, 'output': {'raw': []}}) is None      # no response
    assert record_to_row({**base, 'metadata': None}) is not None       # falls back to sample_id prefix


def test_load_records_cap_and_dedupe(tmp_path):
    root = make_store(tmp_path)
    df = load_records(root, max_queries_per_cell=None)
    assert len(df) == 6 * 2 * 10 and set(df.columns) >= {'model_id', 'task_id', 'query_id',
                                                         'query', 'response', 'score'}
    capped = load_records(root, max_queries_per_cell=4)
    assert capped.groupby(['model_id', 'task_id']).size().max() == 4
    # a newer duplicate run for model-0 replaces the older rows
    dup = root / 'dev0' / 'model-0' / 'run2_samples.jsonl'
    line = json.dumps({'model_id': 'dev0/model-0', 'sample_id': 'bench_task0:0',
                       'input': {'raw': 'question 0-0?'}, 'output': {'raw': ['NEW']},
                       'evaluation': {'score': '1.0'},
                       'metadata': {'source_task': 'bench_task0'}})
    dup.write_text(line + '\n')
    with pytest.warns(UserWarning, match='duplicates'):
        df2 = load_records(root, max_queries_per_cell=None)
    hit = df2[(df2.model_id == 'dev0/model-0') & (df2.query_id == 'bench_task0:0')]
    assert len(hit) == 1 and hit.iloc[0]['response'] == 'NEW'


def test_model_from_path_normalizes_names(tmp_path):
    root = make_store(tmp_path)
    # in-record ids diverge from the store layout (raw repo casing); path wins by default
    fp = root / 'dev9' / 'model-9' / 'run_samples.jsonl'
    fp.parent.mkdir(parents=True)
    fp.write_text(json.dumps({
        'model_id': 'Dev9Labs/Model-9.0', 'sample_id': 'bench_task0:0',
        'input': {'raw': 'q?'}, 'output': {'raw': ['a']},
        'evaluation': {'score': '1.0'}, 'metadata': {'source_task': 'bench_task0'},
    }) + '\n')
    df = load_records(root, max_queries_per_cell=None)
    assert 'dev9/model-9' in set(df.model_id) and 'Dev9Labs/Model-9.0' not in set(df.model_id)
    df2 = load_records([fp], model_from='record', max_queries_per_cell=None)
    assert set(df2.model_id) == {'Dev9Labs/Model-9.0'}


def test_eee_records_fit_predict_via_text_embedding(tmp_path, monkeypatch):
    """The EEE-user path end to end: raw jsonl -> load_records -> PKPS.fit embeds the
    query/response TEXT -> predict. The embedding API is stubbed with a deterministic
    hash embedding so no network/key is needed."""
    def fake_embed(provider, texts, **kw):
        rng = [np.random.default_rng(abs(hash(t)) % 2**32).normal(size=16) for t in texts]
        return np.stack(rng)
    monkeypatch.setattr(dkps.embed, 'embed_api', fake_embed)

    records = load_records(make_store(tmp_path), max_queries_per_cell=None)
    est = PKPS(query_kwargs=dict(pca_dim=None), response_kwargs=dict(pca_dim=None),
               mds_kwargs=dict(dim=4)).fit(records)
    assert len(est.model_names_) == 6 and len(est.task_names_) == 2
    out = est.predict([{'model_id': 'dev0/model-0'}], holdout='family')
    assert len(out) == 2 and all(np.isfinite(r['score_hat']) for r in out)
    # score-only baselines take the same records unchanged
    ss = SampleScore().fit(records)
    assert np.isfinite(ss.predict([{'model_id': 'dev0/model-0',
                                    'task_id': 'bench_task0'}])[0]['score_hat'])


def test_suite_column_joint_default_and_blocked_option(tmp_path, monkeypatch):
    """Suite-labeled records default to one shared unit-normalized space (the base
    method: k_Q alone gates cross-suite comparisons); suite_mode='blocked' gives the
    block-diagonal ablation with exact cross-suite orthogonality."""
    def fake_embed(provider, texts, **kw):
        return np.stack([np.random.default_rng(abs(hash(t)) % 2**32).normal(size=16)
                         for t in texts])
    monkeypatch.setattr(dkps.embed, 'embed_api', fake_embed)
    make_store(tmp_path, bench='bench_a')
    make_store(tmp_path, bench='bench_b')
    records = load_records(tmp_path, max_queries_per_cell=None)
    assert set(records.suite) == {'bench_a', 'bench_b'}
    est = PKPS(query_kwargs=dict(pca_dim=None), mds_kwargs=dict(dim=4)).fit(records)
    md = est._model_data[est.model_names_[0]]
    sub = est._raw[est._raw.model_id == est.model_names_[0]].sort_values(['task_id', 'query_id'])
    X0 = md['X'][(sub.suite == 'bench_a').to_numpy()]
    X1 = md['X'][(sub.suite == 'bench_b').to_numpy()]
    np.testing.assert_allclose(np.linalg.norm(md['X'], axis=1), 1.0)   # unit-normalized
    assert np.abs(X0 @ X1.T).max() > 1e-6           # joint default: cross-suite k_R nonzero
    assert all(np.isfinite(r['score_hat']) for r in
               est.predict([{'model_id': est.model_names_[0]}], holdout='family'))
    # blocked ablation: exact cross-suite orthogonality
    estb = PKPS(query_kwargs=dict(pca_dim=None),
                response_kwargs=dict(suite_mode='blocked'),
                mds_kwargs=dict(dim=4)).fit(records)
    mdb = estb._model_data[estb.model_names_[0]]
    B0 = mdb['X'][(sub.suite == 'bench_a').to_numpy()]
    B1 = mdb['X'][(sub.suite == 'bench_b').to_numpy()]
    assert np.abs(B0 @ B1.T).max() < 1e-12


def test_multiturn_response_from_messages():
    """Arena/agent records: output is empty, transcript in messages; the model's own
    turns become the response."""
    rec = {'model_id': 'd/m', 'sample_id': 'arena_44', 'evaluation_name': 'arena',
           'input': {'raw': 'Guess the word.'}, 'output': None,
           'messages': [{'role': 'system', 'content': 'rules'},
                        {'role': 'assistant', 'content': 'CRANE'},
                        {'role': 'user', 'content': 'feedback'},
                        {'role': 'assistant', 'content': 'SLOTH'}],
           'evaluation': {'score': '0.0', 'num_turns': '4'}}
    row = record_to_row(rec)
    assert row['response'] == 'CRANE\nSLOTH'
    assert row['suite'] == 'arena' and row['task_id'] == 'arena'   # no ':' in sample_id


def test_score_table_and_listings(tmp_path, monkeypatch):
    def fake_embed(provider, texts, **kw):
        return np.stack([np.random.default_rng(abs(hash(t)) % 2**32).normal(size=16)
                         for t in texts])
    monkeypatch.setattr(dkps.embed, 'embed_api', fake_embed)
    make_store(tmp_path, bench='bench_a')
    make_store(tmp_path, n_models=4, bench='bench_b')
    records = load_records(tmp_path, max_queries_per_cell=None)
    est = PKPS(query_kwargs=dict(pca_dim=None), mds_kwargs=dict(dim=4)).fit(records)
    assert est.suite_names_ == ['bench_a', 'bench_b']
    assert est.task_suites_['bench_a_task0'] == 'bench_a'
    tbl = est.score_table(holdout='family')
    # one row per (model, task) cell; observed where responses exist, predicted otherwise
    assert len(tbl) == len(est.model_names_) * len(est.task_names_)
    assert set(tbl.source) == {'observed', 'predicted'}
    missing = tbl[tbl.source == 'predicted']
    assert (missing.suite == 'bench_b').all() or (missing.suite == 'bench_a').all()
    one = tbl[tbl.model_id == est.model_names_[0]]
    assert len(one) == len(est.task_names_)
    # joint default: a NEW suite folds in through update() (shared space transforms it)
    extra = records[records.suite == 'bench_a'].head(3).assign(
        suite='bench_c', task_id='bench_c_task0',
        query_id=[f'bench_c_task0:{i}' for i in range(3)])
    est.update(extra)
    assert 'bench_c' in est.suite_names_
    # blocked ablation: frozen block layout still refuses unseen suites, clear error
    estb2 = PKPS(query_kwargs=dict(pca_dim=None),
                 response_kwargs=dict(suite_mode='blocked'),
                 mds_kwargs=dict(dim=4)).fit(records)
    with pytest.raises(KeyError, match='fresh\\s+fit'):
        estb2.update(extra)
