#!/usr/bin/env python
"""
pipeline/loaders.py

Investigate DoubleKernelDKPS on real HELM data (MATH benchmark).

The MATH data is fully paired: 95 models x 7 subjects (tasks), every model
answered every instance of every subject. We induce sparsity and ask whether
the *combined* product-kernel estimator predicts held-out per-subject scores
better than paired/unpaired.

Methodology (per replicate, learned only from the sampled/observed data):
  1. sample observation mask over (model, task) [+ optional query subsampling]
  2. PCA-reduce the observed response embeddings (elbow dim), and SEPARATELY
     PCA-reduce the observed unique-query embeddings (elbow dim)
  3. for each estimator (paired/unpaired/combined): DoubleKernelDKPS -> model
     ClassicalMDS(n_elbows=2) -> KNN regression on held-out scores -> RMSE

No global dimensionality reduction: every reduction sees only observed data.

Data ingredients
----------------
- response embeddings : exports/math_google_embeddings.parquet (gemini, 3072-d)
- scores              : data/math.tsv
- query embeddings    : google (exports/math_query_google_embeddings.parquet),
                        mean_response (per-instance mean response), or tfidf.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge
from graspologic.embed import ClassicalMDS

from dkps.unpaired_dkps import (DoubleKernelDKPS, ProductKernelPerspectiveSpace,
                                pca_reduce_elbow)
from dkps.baselines import matrix_completion_predict


def _pca_forced(R, dim):
    """Center + PCA to exactly `dim` components (the shared-space capacity rule)."""
    from sklearn.utils.extmath import randomized_svd
    Rc = R - R.mean(axis=0)
    k = int(min(dim, Rc.shape[0] - 1, Rc.shape[1]))
    _, _, Vt = randomized_svd(Rc, n_components=k, random_state=0)
    return Rc @ Vt.T


def _matched_total(groups, per_cap):
    """Total dimension the per-dataset construction retains: second Zhu--Ghodsi elbow
    per group within per_cap, summed. The joint space uses this as its (forced)
    dimension so the two constructions are capacity-matched by definition."""
    total = 0
    for R in groups:
        total += (pca_reduce_elbow(R, max_components=per_cap).shape[1]
                  if R.shape[1] > per_cap else R.shape[1])
    return total

# Shared payloads (downloads, embedding parquets, caches) live at the repo-level data/
# directory (override with DKPS_DATA); the prep code stays in this project.
# Affinity normalization used by every experiment driver. The archived results in
# projects/pkps/data were generated with the original pair normalization (S_ab / Z_ab);
# set to 'model' (the package default, S_ab / sqrt(Z_aa Z_bb)) to regenerate them.
NORMALIZATION = os.environ.get('PKPS_NORMALIZATION', 'pair')

DATA = Path(os.environ.get('DKPS_DATA') or Path(__file__).resolve().parents[4] / 'data')


def data_path(rel):
    """'exports/x' -> DATA/exports/x; 'data/x' (HELM dumps) -> DATA/helm/x."""
    rel = str(rel)
    if rel.startswith('exports/'):
        return DATA / rel
    if rel.startswith('data/'):
        return DATA / 'helm' / rel[5:]
    return Path(rel)


class SubsampleMedianBandwidth:
    """Median pairwise distance over a random row subsample (bounds pdist cost).

    Picklable callable (works with joblib/loky); used as query/response bandwidth.
    """
    def __init__(self, max_n=5000, seed=0):
        self.max_n = max_n
        self.seed = seed

    def __call__(self, X):
        rng = np.random.default_rng(self.seed)
        if len(X) > self.max_n:
            X = X[rng.choice(len(X), self.max_n, replace=False)]
        d = pdist(X)
        return float(np.median(d)) if len(d) else 1.0


def make_estimators(query_kernel='rbf', response_kernel='rbf'):
    """Three estimators (paired/unpaired/combined) for the given kernel types."""
    bw = SubsampleMedianBandwidth(max_n=5000)
    kernels = dict(query_kernel=query_kernel, response_kernel=response_kernel,
                   query_bandwidth=bw, response_bandwidth=bw)
    return {
        'rbf_paired':   dict(**kernels, task_filter='shared'),
        'rbf_unpaired': dict(**kernels, task_filter='unshared'),
        'rbf_combined': dict(**kernels),
    }


# Default: both kernels RBF. Overridable via CLI for the both-linear supplement.
ESTIMATORS = make_estimators()


DATASETS = {
    'math': dict(parquet='exports/math_google_embeddings.parquet', tsv='data/math.tsv',
                 query_parquet='exports/math_query_google_embeddings.parquet', score_col='score'),
    'wmt_14': dict(parquet='exports/wmt_14_google_embeddings.parquet', tsv='data/wmt_14.tsv',
                   query_parquet='exports/wmt_14_query_google_embeddings.parquet', score_col='meteor'),
}

# The heterogeneous suite. MATH/WMT have rich text responses -> Google embeddings;
# med_qa/legalbench responses are single tokens (MCQ letters / class labels) -> one-hot
# answer embeddings (linear k_R on a one-hot = answer agreement, the right MCQ signal).
SUITE = {
    'math':       dict(resp='google', tsv='data/math.tsv', score_col='score',
                       query_parquet='exports/math_query_google_embeddings.parquet',
                       parquet='exports/math_google_embeddings.parquet'),
    'wmt_14':     dict(resp='google', tsv='data/wmt_14.tsv', score_col='meteor',
                       query_parquet='exports/wmt_14_query_google_embeddings.parquet',
                       parquet='exports/wmt_14_google_embeddings.parquet'),
    'med_qa':     dict(resp='onehot', tsv='data/med_qa.tsv', score_col='score',
                       query_parquet='exports/med_qa_query_google_embeddings.parquet'),
    'legalbench': dict(resp='onehot', tsv='data/legalbench.tsv', score_col='score',
                       query_parquet='exports/legalbench_query_google_embeddings.parquet'),
}


def load_pooled(keys=('math', 'wmt_14')):
    """Pool multiple datasets into one (shared-model x union-task) benchmark.
    Tasks are the union of subjects/language-pairs; restricted to shared models.
    Returns the same 11-tuple as load_helm_math."""
    longs = []
    for k in keys:
        cfg = DATASETS[k]
        emb = pd.read_parquet(data_path(cfg['parquet']))
        meta = (pd.read_csv(data_path(cfg['tsv']), sep='\t')[['dataset', 'model', 'instance_id',
                'query', cfg['score_col']]].rename(columns={cfg['score_col']: 'score'}))
        df = emb.merge(meta, on=['dataset', 'model', 'instance_id'], how='inner')
        df['task_id'] = df['dataset']
        df['model_id'] = df['model']
        df['query_id'] = df['dataset'] + '::' + df['instance_id'].astype(str)
        qdf = pd.read_parquet(data_path(cfg['query_parquet']))
        qmap = {q: np.asarray(v) for q, v in zip(qdf['query_id'], qdf['query_embedding'])}
        df['query_embedding'] = [qmap[q] for q in df['query_id']]
        longs.append(df[['model_id', 'task_id', 'query_id', 'embedding', 'query_embedding', 'score']])
    shared = set(longs[0]['model_id'])
    for l in longs[1:]:
        shared &= set(l['model_id'])
    df = pd.concat([l[l['model_id'].isin(shared)] for l in longs], ignore_index=True)
    df = df.sort_values(['model_id', 'task_id', 'query_id']).reset_index(drop=True)

    resp_X = np.stack(df['embedding'].values).astype(np.float32)
    unique_qids = sorted(df['query_id'].unique())
    qcode = {q: i for i, q in enumerate(unique_qids)}
    qid_code = df['query_id'].map(qcode).to_numpy()
    qsub = df.drop_duplicates('query_id').set_index('query_id')
    Qu = np.stack([np.asarray(qsub.loc[q, 'query_embedding']) for q in unique_qids]).astype(np.float32)
    models = sorted(df['model_id'].unique())
    tasks = sorted(df['task_id'].unique())
    m_idx = {m: i for i, m in enumerate(models)}
    t_idx = {t: i for i, t in enumerate(tasks)}
    score_mat = np.full((len(models), len(tasks)), np.nan)
    for (m, t), s in df.groupby(['model_id', 'task_id'])['score'].mean().items():
        score_mat[m_idx[m], t_idx[t]] = s
    groups = {key: g.index.to_numpy() for key, g in df.groupby(['model_id', 'task_id'])}
    row_score = df['score'].to_numpy(dtype=float)
    return (resp_X, Qu, qid_code, df['model_id'].to_numpy(), df['task_id'].to_numpy(),
            df['query_id'].to_numpy(), score_mat, models, tasks, groups, row_score)


def _suite_dataset_long(key, max_q_per_task=None, seed=0, answer_mode=None):
    """One suite dataset -> long df (model_id, task_id, query_id, resp_vec, query_embedding,
    score). resp_vec is the native response representation (Google emb or one-hot answer);
    answer_mode='string' swaps the one-hot for the Google embedding of the answer STRING
    itself (the MCQ letter / class label); answer_mode='option' resolves the answer to its
    STANDARDIZED text -- the MedQA option text the letter denotes (per-instance map,
    verified against model behavior) or the LegalBench class name the digit indexes --
    and embeds that, putting every dataset in the same semantic 3072-d text space
    (unresolvable answers fall back to the raw-string embedding)."""
    cfg = SUITE[key]
    meta = pd.read_csv(data_path(cfg['tsv']), sep='\t')
    score = meta[cfg['score_col']].to_numpy(dtype=float)
    base = pd.DataFrame({'dataset': meta['dataset'], 'model_id': meta['model'],
                         'instance_id': meta['instance_id'].astype(str),
                         'response': meta['response'].astype(str), 'score': score})
    base['query_id'] = base['dataset'] + '::' + base['instance_id']

    if cfg['resp'] == 'google':
        emb = pd.read_parquet(data_path(cfg['parquet']))[['dataset', 'model', 'instance_id', 'embedding']]
        emb['instance_id'] = emb['instance_id'].astype(str)
        emb = emb.rename(columns={'model': 'model_id'})
        df = base.merge(emb, on=['dataset', 'model_id', 'instance_id'], how='inner')
        df['resp_vec'] = list(np.stack(df['embedding'].values).astype(np.float32))
        df = df.drop(columns='embedding')
    elif answer_mode:  # embed the answer as text (shared 3072-d space across datasets)
        adf = pd.read_parquet(data_path('exports/answer_string_google_embeddings.parquet'))
        amap = {r: np.asarray(v, dtype=np.float32) for r, v in zip(adf['response'], adf['embedding'])}
        df = base.copy()
        if answer_mode == 'option':
            tmap = pd.read_parquet(data_path('exports/answer_option_text_map.parquet'))
            odf = pd.read_parquet(data_path('exports/answer_option_text_google_embeddings.parquet'))
            omap = {t: np.asarray(v, dtype=np.float32) for t, v in zip(odf['text'], odf['embedding'])}
            by_inst = {(d, i, r): t for d, i, r, t in
                       zip(tmap['dataset'], tmap['instance_id'], tmap['response'], tmap['text']) if i}
            by_ds = {(d, r): t for d, i, r, t in
                     zip(tmap['dataset'], tmap['instance_id'], tmap['response'], tmap['text']) if not i}
            def vec(ds_, iid, resp):
                rr = str(resp).rstrip('.')
                t = by_inst.get((ds_, iid, rr)) or by_ds.get((ds_, rr))
                return omap[t] if t is not None else amap[str(resp)]
            df['resp_vec'] = [vec(d, i, r) for d, i, r in
                              zip(df['dataset'], df['instance_id'], df['response'])]
        else:
            df['resp_vec'] = [amap[r] for r in df['response']]
    else:  # one-hot answer agreement
        vocab = sorted(base['response'].unique())
        vi = {v: i for i, v in enumerate(vocab)}
        eye = np.eye(len(vocab), dtype=np.float32)
        df = base.copy()
        df['resp_vec'] = [eye[vi[r]] for r in df['response']]

    qdf = pd.read_parquet(data_path(cfg['query_parquet']))
    qmap = {q: np.asarray(v, dtype=np.float32) for q, v in zip(qdf['query_id'], qdf['query_embedding'])}
    df = df[df['query_id'].isin(qmap)]
    df['query_embedding'] = [qmap[q] for q in df['query_id']]
    df['task_id'] = df['dataset']

    if max_q_per_task:  # cap queries per task so datasets contribute comparably
        rng = np.random.default_rng(seed)
        keep_q = []
        for t, g in df.drop_duplicates('query_id').groupby('task_id'):
            q = g['query_id'].to_numpy()
            keep_q.append(q if len(q) <= max_q_per_task
                          else rng.choice(q, max_q_per_task, replace=False))
        df = df[df['query_id'].isin(np.concatenate(keep_q))]
    return df[['model_id', 'task_id', 'query_id', 'resp_vec', 'query_embedding', 'score']]


def load_suite(keys=('math', 'wmt_14', 'med_qa', 'legalbench'), reduce_dim=48,
               max_q_per_task=120, seed=0, resp_mode='option-text', response_space='joint',
               joint_force_dim=None):
    """Heterogeneous joint benchmark. BASE METHOD (option-text, joint): every response
    is embedded text -- free text for MATH/WMT, and for the MCQ/label datasets the
    STANDARDIZED answer text (the MedQA option text the letter denotes, the LegalBench
    class name the digit indexes; see data/resolve_answer_option_text.py) -- reduced by
    ONE shared PCA whose dimension matches what the per-dataset construction retains in
    total (capacity-matched by definition), then unit-normalized. No hard cross-dataset
    zeroing: the query kernel alone decides which response comparisons carry weight.
    The published one-hot protocol is resp_mode='native' (one-hot answers, forces
    blocked); response_space='blocked' gives the per-dataset disjoint-block ablation
    (linear cross-dataset k_R = 0 exactly). Query embeddings share one Google space;
    the within-domain median query distance (returned as query_med) keeps the RBF query
    kernel ~0 across domains. Restricted to shared models. Returns the load_helm_math
    11-tuple plus query_med."""
    answer_mode = {'native': None, 'text': 'string', 'option-text': 'option'}[resp_mode]
    if response_space == 'joint' and answer_mode is None:
        response_space = 'blocked'  # one-hot answers cannot share a space: published protocol
    longs = {k: _suite_dataset_long(k, max_q_per_task, seed, answer_mode=answer_mode)
             for k in keys}
    shared = set.intersection(*[set(l['model_id']) for l in longs.values()])

    if response_space == 'joint':
        # one shared PCA over all datasets' 3072-d responses (capacity-matched to the
        # blocked construction: reduce_dim components per dataset), then unit-norm --
        # cross-dataset response similarity is real and the query kernel gates it
        parts = []
        for k in keys:
            l = longs[k][longs[k]['model_id'].isin(shared)].reset_index(drop=True)
            parts.append(l)
        df = pd.concat(parts, ignore_index=True)
        R = np.stack(df['resp_vec'].values).astype(np.float64)
        # shared-space dimension = what the per-dataset construction retains in total
        # (capacity-matched by definition); joint_force_dim overrides for controls
        dim = joint_force_dim or _matched_total(
            (np.stack(longs[k]['resp_vec'].values).astype(np.float64) for k in keys),
            reduce_dim)
        R = _pca_forced(R, dim)
        R /= (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
        df['embedding'] = list(R.astype(np.float32))
    else:
        blocks, dims = {}, {}
        for k, l in longs.items():
            R = np.stack(l['resp_vec'].values).astype(np.float64)
            if R.shape[1] > reduce_dim:
                R = pca_reduce_elbow(R, max_components=reduce_dim)
            R /= (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
            blocks[k] = R.astype(np.float32)
            dims[k] = R.shape[1]
        total = sum(dims.values())
        offset, off = {}, 0
        for k in keys:
            offset[k] = off
            off += dims[k]

        parts = []
        for k in keys:
            l = longs[k][longs[k]['model_id'].isin(shared)].reset_index(drop=True)
            idx = longs[k].index[longs[k]['model_id'].isin(shared)]
            B = np.zeros((len(l), total), np.float32)
            B[:, offset[k]:offset[k] + dims[k]] = blocks[k][[longs[k].index.get_loc(i) for i in idx]]
            l = l.copy()
            l['embedding'] = list(B)
            parts.append(l)
        df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(['model_id', 'task_id', 'query_id']).reset_index(drop=True)

    resp_X = np.stack(df['embedding'].values).astype(np.float32)
    unique_qids = sorted(df['query_id'].unique())
    qcode = {q: i for i, q in enumerate(unique_qids)}
    qid_code = df['query_id'].map(qcode).to_numpy()
    qsub = df.drop_duplicates('query_id').set_index('query_id')
    Qu = np.stack([np.asarray(qsub.loc[q, 'query_embedding']) for q in unique_qids]).astype(np.float32)
    # Reduce the 3072-dim Google query embeddings once (PCA) so the per-model-pair query
    # cdist is cheap; domain separation (what blocks k_Q across datasets) is preserved.
    Qu = pca_reduce_elbow(Qu.astype(np.float64), max_components=64).astype(np.float32)
    q_task = qsub['task_id'].to_dict()

    models = sorted(df['model_id'].unique())
    tasks = sorted(df['task_id'].unique())
    m_idx = {m: i for i, m in enumerate(models)}
    t_idx = {t: i for i, t in enumerate(tasks)}
    score_mat = np.full((len(models), len(tasks)), np.nan)
    for (m, t), s in df.groupby(['model_id', 'task_id'])['score'].mean().items():
        score_mat[m_idx[m], t_idx[t]] = s
    groups = {key: g.index.to_numpy() for key, g in df.groupby(['model_id', 'task_id'])}
    row_score = df['score'].to_numpy(dtype=float)

    # within-domain median query distance (per task, pooled) for the k_Q bandwidth
    meds = []
    rng = np.random.default_rng(seed)
    for t in tasks:
        qi = [i for i, q in enumerate(unique_qids) if q_task[q] == t]
        if len(qi) > 1:
            sub = Qu[qi] if len(qi) <= 400 else Qu[rng.choice(qi, 400, replace=False)]
            meds.append(np.median(pdist(sub)))
    query_med = float(np.median(meds)) if meds else 1.0

    return (resp_X, Qu, qid_code, df['model_id'].to_numpy(), df['task_id'].to_numpy(),
            df['query_id'].to_numpy(), score_mat, models, tasks, groups, row_score, query_med)


def load_eee(reduce_dim=None, seed=0, emb_tag=None, response_space='joint'):
    """The Every Eval Ever suite (data/eee.py + data/embed_eee.py): 5 benchmarks
    (math-mc, gsm-mc, gpqa-diamond, judgebench, reward-bench-2) -> 16 tasks over the
    models shared by all five. Same block-diagonal construction as load_suite -- each
    benchmark's responses are PCA-reduced, unit-normalized, and placed in a disjoint
    block (linear cross-benchmark k_R = 0) -- but every domain is Google-embedded text.
    score_mat holds the FULL-depth per-cell means from eee_cell_targets (42-1288
    queries/cell), while the loaded rows are the capped per-cell pools; the pools are
    sampled with model-dependent seeds, so the suite is unpaired by construction.
    Returns the load_suite 12-tuple."""
    sfx = f'__{emb_tag}' if emb_tag else ''    # alternative embedding models (sensitivity study)
    emb = pd.read_parquet(data_path(f'exports/eee_response_embeddings{sfx}.parquet'))
    qdf = pd.read_parquet(data_path(f'exports/eee_query_embeddings{sfx}.parquet'))
    tgt = pd.read_parquet(data_path('exports/eee_cell_targets.parquet'))

    shared = set.intersection(*[set(g['model']) for _, g in emb.groupby('bench')])
    df = emb[emb['model'].isin(shared)].reset_index(drop=True)
    df = df.rename(columns={'model': 'model_id', 'task': 'task_id'})

    n_bench = df['bench'].nunique()
    if response_space == 'joint':
        # BASE METHOD: one shared response space -- a single PCA over all responses,
        # unit-normalized, capacity matched to the blocked ablation (48 per benchmark).
        # No hard cross-benchmark zeroing: the query kernel alone decides which response
        # comparisons carry weight (soft gating via its bandwidth). PKPS exists to let
        # information flow across unpaired, sparse collections; blocking is the ablation.
        R = np.stack(df['emb'].values).astype(np.float64)
        if reduce_dim is not None:      # explicit cap: elbow within it (legacy protocol)
            if R.shape[1] > reduce_dim:
                R = pca_reduce_elbow(R, max_components=reduce_dim)
        else:                           # capacity-matched to the blocked construction
            dim = _matched_total((np.stack(g['emb'].values).astype(np.float64)
                                  for _, g in df.groupby('bench')), 48)
            R = _pca_forced(R, dim)
        R /= (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
        resp_X = R.astype(np.float32)
    else:
        # ABLATION: block-diagonal response space, one block per benchmark
        per_block = (reduce_dim if reduce_dim is not None else 48 * n_bench) // n_bench
        blocks, dims = {}, {}
        for b, g in df.groupby('bench'):
            R = np.stack(g['emb'].values).astype(np.float64)
            if R.shape[1] > per_block:
                R = pca_reduce_elbow(R, max_components=per_block)
            R /= (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
            blocks[b] = (g.index.to_numpy(), R.astype(np.float32))
            dims[b] = R.shape[1]
        total, off = sum(dims.values()), 0
        resp_X = np.zeros((len(df), total), np.float32)
        for b in sorted(dims):
            idx, R = blocks[b]
            resp_X[idx, off:off + dims[b]] = R
            off += dims[b]

    df = df.assign(_row=np.arange(len(df))).sort_values(
        ['model_id', 'task_id', 'query_id']).reset_index(drop=True)
    resp_X = resp_X[df['_row'].to_numpy()]

    unique_qids = sorted(df['query_id'].unique())
    qcode = {q: i for i, q in enumerate(unique_qids)}
    qid_code = df['query_id'].map(qcode).to_numpy()
    qmap = {q: np.asarray(v, dtype=np.float32) for q, v in zip(qdf['query_id'], qdf['emb'])}
    Qu = np.stack([qmap[q] for q in unique_qids]).astype(np.float64)
    Qu = pca_reduce_elbow(Qu, max_components=64).astype(np.float32)
    q_task = df.drop_duplicates('query_id').set_index('query_id')['task_id'].to_dict()

    models = sorted(df['model_id'].unique())
    tasks = sorted(df['task_id'].unique())
    m_idx = {m: i for i, m in enumerate(models)}
    t_idx = {t: i for i, t in enumerate(tasks)}
    # targets = FULL-depth cell means (never limited to the embedded pool)
    score_mat = np.full((len(models), len(tasks)), np.nan)
    for r in tgt.itertuples():
        if r.model in m_idx and r.task in t_idx:
            score_mat[m_idx[r.model], t_idx[r.task]] = r.y_full
    groups = {key: g.index.to_numpy() for key, g in df.groupby(['model_id', 'task_id'])}
    row_score = df['score'].to_numpy(dtype=float)

    meds, rng = [], np.random.default_rng(seed)
    for t in tasks:
        qi = [i for i, q in enumerate(unique_qids) if q_task[q] == t]
        if len(qi) > 1:
            sub = Qu[qi] if len(qi) <= 400 else Qu[rng.choice(qi, 400, replace=False)]
            meds.append(np.median(pdist(sub)))
    query_med = float(np.median(meds)) if meds else 1.0

    return (resp_X, Qu, qid_code, df['model_id'].to_numpy(), df['task_id'].to_numpy(),
            df['query_id'].to_numpy(), score_mat, models, tasks, groups, row_score, query_med)


def load_helm_math(parquet_path, tsv_path, query_source='google', query_parquet=None,
                   score_col='score'):
    """Return raw (unreduced) response/query embeddings + metadata.

    Reduction is deferred to each replicate (observed data only).

    Returns
    -------
    resp_X : (N, Dr) float32   raw response embeddings (one row per (model,instance))
    Qu     : (n_uniq, Dq) float32   raw query embedding per unique instance
    qid_code : (N,) int        index into Qu for each row
    model_id, task_id, query_id : (N,) object arrays
    score_mat : (n_models, n_tasks)   ground-truth mean score per (model, task)
    models, tasks : sorted label lists
    groups : {(model_id, task_id): row indices}
    """
    parquet_path, tsv_path = data_path(parquet_path), data_path(tsv_path)
    emb = pd.read_parquet(parquet_path)
    meta = (pd.read_csv(tsv_path, sep='\t')[['dataset', 'model', 'instance_id', 'query', score_col]]
            .rename(columns={score_col: 'score'}))
    df = emb.merge(meta, on=['dataset', 'model', 'instance_id'], how='inner')
    df['task_id'] = df['dataset']
    df['model_id'] = df['model']
    df['query_id'] = df['dataset'] + '::' + df['instance_id'].astype(str)
    df = df.sort_values(['model_id', 'task_id', 'query_id']).reset_index(drop=True)

    resp_X = np.stack(df['embedding'].values).astype(np.float32)

    unique_qids = sorted(df['query_id'].unique())
    qcode = {q: i for i, q in enumerate(unique_qids)}
    qid_code = df['query_id'].map(qcode).to_numpy()

    # Raw per-instance query embeddings Qu (no reduction here).
    if query_source == 'google':
        qpath = data_path(query_parquet) if query_parquet else \
            Path(parquet_path).parent / 'math_query_google_embeddings.parquet'
        assert qpath.exists(), f'{qpath} missing — run embed_math_queries.py first'
        qdf = pd.read_parquet(qpath)
        qmap = {q: v for q, v in zip(qdf['query_id'].values, qdf['query_embedding'].values)}
        Qu = np.stack([np.asarray(qmap[q]) for q in unique_qids]).astype(np.float32)
    elif query_source == 'mean_response':
        tmp = pd.DataFrame(resp_X)
        tmp['query_id'] = df['query_id'].values
        means = tmp.groupby('query_id').mean()
        Qu = means.loc[unique_qids].to_numpy().astype(np.float32)
    elif query_source == 'tfidf':
        qtext = (df[['query_id', 'query']].drop_duplicates('query_id')
                 .set_index('query_id').loc[unique_qids]['query'].fillna('').astype(str))
        tfidf = TfidfVectorizer(max_features=4096, ngram_range=(1, 2), stop_words='english')
        Qu = tfidf.fit_transform(qtext).toarray().astype(np.float32)
    else:
        raise ValueError(f'unknown query_source: {query_source}')

    models = sorted(df['model_id'].unique())
    tasks = sorted(df['task_id'].unique())
    m_idx = {m: i for i, m in enumerate(models)}
    t_idx = {t: i for i, t in enumerate(tasks)}
    score_mat = np.full((len(models), len(tasks)), np.nan)
    for (m, t), s in df.groupby(['model_id', 'task_id'])['score'].mean().items():
        score_mat[m_idx[m], t_idx[t]] = s

    groups = {key: g.index.to_numpy() for key, g in df.groupby(['model_id', 'task_id'])}
    row_score = df['score'].to_numpy(dtype=float)  # per-(model,instance) score

    return (resp_X, Qu, qid_code,
            df['model_id'].to_numpy(), df['task_id'].to_numpy(), df['query_id'].to_numpy(),
            score_mat, models, tasks, groups, row_score)


def sample_observed(n_models, n_tasks, obs_prob, rng):
    """Boolean (model, task) mask; guarantee >=1 obs per row and per column."""
    observed = rng.random((n_models, n_tasks)) < obs_prob
    for i in range(n_models):
        if not observed[i].any():
            observed[i, rng.integers(n_tasks)] = True
    for k in range(n_tasks):
        if not observed[:, k].any():
            observed[rng.integers(n_models), k] = True
    return observed


def _logit(M, eps=1e-3):
    s = np.clip(M, eps, 1.0 - eps)
    return np.log(s / (1.0 - s))


def two_way_bias(score_mat, observed, logit=True):
    """Transform (optional logit) + two-way additive bias from observed entries.

    Returns (X transformed scores, bias) where
      bias_mb = global + (model offset) + (task offset),
    matching the bias decomposition of the BenchPress matrix-completion baseline.
    The model offset uses the model's *other* observed tasks; task offset uses
    observed models on that task — all available info.
    """
    M = np.asarray(score_mat, dtype=float)
    mask = np.asarray(observed, dtype=bool)
    X = _logit(M) if logit else M.copy()
    gbar = X[mask].mean()
    row = np.array([X[m, mask[m]].mean() if mask[m].any() else gbar
                    for m in range(M.shape[0])])
    col = np.array([X[mask[:, b], b].mean() if mask[:, b].any() else gbar
                    for b in range(M.shape[1])])
    bias = row[:, None] + col[None, :] - gbar
    return X, bias


def _make_predictor(name, k, n_train):
    if name == 'knn':
        return KNeighborsRegressor(n_neighbors=min(k, n_train), weights='distance')
    if name == 'ols':
        return LinearRegression()
    if name == 'ridge':
        return Ridge(alpha=1.0)
    raise ValueError(f'unknown predictor: {name}')


def _whiten(M, mask, logit=True, standardize=True):
    """logit -> per-task column standardization (observed) -> two-way bias. Returns
    (target = standardized residual, bias, mu, sd) and lets callers invert. Matches the
    preprocessing of the BenchPress matrix-completion baseline so the heads are
    comparable (without standardization, the model offset is dominated by the
    highest-logit-variance tasks, which hurts on heterogeneous, multi-scale suites)."""
    X = _logit(M) if logit else np.asarray(M, dtype=float).copy()
    nB = X.shape[1]
    mu = np.zeros(nB); sd = np.ones(nB)
    if standardize:
        for b in range(nB):
            c = mask[:, b]
            if c.any():
                mu[b] = X[c, b].mean()
                s = X[c, b].std()
                sd[b] = s if s > 1e-8 else 1.0
    Xs = (X - mu) / sd
    gbar = Xs[mask].mean() if mask.any() else 0.0
    row = np.array([Xs[m, mask[m]].mean() if mask[m].any() else gbar for m in range(X.shape[0])])
    col = np.array([Xs[mask[:, b], b].mean() if mask[:, b].any() else gbar for b in range(nB)])
    bias = row[:, None] + col[None, :] - gbar
    return Xs - bias, bias, mu, sd


def predict_from_embedding(Z, score_mat, observed, predictor='ols',
                           bias_decomp=True, logit=True, k=5, standardize=True):
    """Per-task regression from the model embedding Z to (logit, column-standardized,
    bias-decomposed) scores. Predicts held-out entries, inverts, clips."""
    M = np.asarray(score_mat, dtype=float)
    mask = np.asarray(observed, dtype=bool)
    if bias_decomp:
        target, bias, mu, sd = _whiten(M, mask, logit=logit, standardize=standardize)
    else:
        X = _logit(M) if logit else M.copy()
        bias = np.zeros_like(M); mu = np.zeros(M.shape[1]); sd = np.ones(M.shape[1])
        target = X

    resid = np.full_like(M, np.nan)
    for t in range(M.shape[1]):
        train = mask[:, t]
        test = ~mask[:, t]
        if train.sum() < 2 or test.sum() < 1:
            continue
        mdl = _make_predictor(predictor, k, int(train.sum()))
        mdl.fit(Z[train], target[train, t])
        resid[test, t] = mdl.predict(Z[test])

    Xhat = (bias + resid) * sd + mu
    Phat = 1.0 / (1.0 + np.exp(-Xhat)) if logit else Xhat
    preds = np.full_like(M, np.nan)
    held = ~mask & np.isfinite(resid)
    preds[held] = np.clip(Phat, 0.0, 1.0)[held]
    return preds


def predict_from_embedding_all(Z, score_mat, observed, predictor='knn',
                               bias_decomp=True, logit=True, k=5, standardize=True):
    """Like predict_from_embedding but returns a cross-model estimate for EVERY cell:
    held-out cells by the standard per-task regression, and observed cells by
    leave-one-out (the cell's own score is excluded), so the estimate never sees the
    target it denoises. This is the unified RD1/RD2 perspective predictor."""
    M = np.asarray(score_mat, dtype=float)
    mask = np.asarray(observed, dtype=bool)
    if bias_decomp:
        target, bias, mu, sd = _whiten(M, mask, logit=logit, standardize=standardize)
    else:
        X = _logit(M) if logit else M.copy()
        bias = np.zeros_like(M); mu = np.zeros(M.shape[1]); sd = np.ones(M.shape[1])
        target = X

    resid = np.full_like(M, np.nan)
    for t in range(M.shape[1]):
        train = np.where(mask[:, t])[0]
        if len(train) < 3:
            continue
        ytr = target[train, t]
        test = np.where(~mask[:, t])[0]
        if len(test):
            mdl = _make_predictor(predictor, k, len(train))
            resid[test, t] = mdl.fit(Z[train], ytr).predict(Z[test])
        for pos, j in enumerate(train):  # leave-one-out for observed cells
            keep = np.delete(np.arange(len(train)), pos)
            mdl = _make_predictor(predictor, k, len(keep))
            resid[j, t] = mdl.fit(Z[train[keep]], ytr[keep]).predict(Z[j:j + 1])[0]

    Xhat = (bias + resid) * sd + mu
    Phat = 1.0 / (1.0 + np.exp(-Xhat)) if logit else Xhat
    preds = np.full_like(M, np.nan)
    fin = np.isfinite(resid)
    preds[fin] = np.clip(Phat, 0.0, 1.0)[fin]
    return preds


def mae_heldout(preds, score_mat, observed):
    held = ~observed & np.isfinite(preds) & np.isfinite(score_mat)
    if held.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(score_mat[held] - preds[held])))


def fit_predict(df_obs, train_score, eval_score, observed, models, predictor='ols',
                mds_dim=0, bias_decomp=True, logit=True, k=5, **dkps_kwargs):
    """Fit DoubleKernelDKPS, embed models via ClassicalMDS (fixed mds_dim or
    n_elbows=2 if mds_dim<=0), then predict held-out scores from the embedding.

    train_score = observed-pair scores used to fit (sample means under query
    subsampling); eval_score = true full scores used only for held-out MAE.
    """
    est = DoubleKernelDKPS(**{'normalization': NORMALIZATION, **dkps_kwargs})
    est.fit(df_obs)
    dist = est.dist_matrix_.copy()

    name2row = {m: i for i, m in enumerate(est.model_names_)}
    present = [m for m in models if m in name2row]
    if np.any(np.isnan(dist)):
        mx = np.nanmax(dist)
        if not np.isfinite(mx):
            return np.nan
        dist = np.nan_to_num(dist, nan=mx)
    if len(present) < 3:
        return np.nan

    if mds_dim and mds_dim > 0:
        Z = ClassicalMDS(n_components=min(mds_dim, len(present) - 1),
                         dissimilarity='precomputed').fit_transform(dist)
    else:
        Z = ClassicalMDS(n_components=None, n_elbows=2,
                         dissimilarity='precomputed').fit_transform(dist)

    full = np.zeros((len(models), Z.shape[1]))
    for m in present:
        full[models.index(m)] = Z[name2row[m]]

    preds = predict_from_embedding(full, train_score, observed, predictor=predictor,
                                   bias_decomp=bias_decomp, logit=logit, k=k)
    return mae_heldout(preds, eval_score, observed)


def _mds_full(D, names, models, mds_dim=8):
    name2row = {m: i for i, m in enumerate(names)}
    present = [m for m in models if m in name2row]
    if np.any(np.isnan(D)):
        mx = np.nanmax(D)
        if not np.isfinite(mx) or len(present) < 4:
            return None
        D = np.nan_to_num(D, nan=mx)
    if len(present) < 4:
        return None
    Z = ClassicalMDS(n_components=min(mds_dim, len(present) - 1),
                     dissimilarity='precomputed').fit_transform(D)
    full = np.zeros((len(models), Z.shape[1]))
    for m in present:
        full[models.index(m)] = Z[name2row[m]]
    return full


def cv_predict_all(df_obs, score_mat, observed, models, k=5, seed=0, mds_dim=8,
                   train_score=None, n_splits=5, response_kernel='rbf', query_med=None):
    """Combined-PKPS + matrix-completion + their ensemble for RD2 (missing tasks).

    The query bandwidth is fixed to the median query distance and the predictor to
    KNN: empirically the combined held-out MAE is flat in sigma for sigma >= 0.5*med
    (0.090-0.091 on MATH) and KNN robustly beats OLS (0.091 vs 0.106), so a per-run
    grid-CV over (sigma, predictor) only injects selection noise on the sparse
    observed split. The one hyper-parameter that genuinely varies across regimes is
    the ensemble weight alpha, which we pick by repeated held-out CV (averaged over
    n_splits validation masks). dist_matrices still builds the single median-bandwidth
    distance matrix from one response-kernel pair-loop."""
    train_score = score_mat if train_score is None else train_score
    if query_med is not None:
        med = query_med
    else:
        qsub = df_obs.drop_duplicates('query_id')
        qemb = np.stack(qsub['query_embedding'].values)
        med = float(np.median(pdist(qemb))) if len(qemb) > 1 else 1.0
    bw = SubsampleMedianBandwidth(max_n=5000)
    est = ProductKernelPerspectiveSpace(query_kernel='rbf', response_kernel=response_kernel,
                                        query_bandwidth=bw, response_bandwidth=bw,
                                        normalization=NORMALIZATION)
    names, Ds = est.dist_matrices(df_obs, [med])
    Z = _mds_full(Ds[med], names, models, mds_dim)

    mc = matrix_completion_predict(train_score, observed)
    if Z is None:
        return dict(combined=np.full_like(score_mat, np.nan), matcomplete=mc,
                    ensemble=mc, sigma=None, predictor='knn', alpha=None)
    P_comb = predict_from_embedding(Z, train_score, observed, predictor='knn',
                                    bias_decomp=True, logit=True, k=k)

    # CV the ensemble weight over several held-out validation masks. With few observed
    # cells a fine grid overfits the handful of validation cells -- and blending in a
    # collapsed base (e.g. MC at tiny cohorts) is actively harmful -- so fall back to
    # {0, 1}: pick whichever base (MC or combined) is stronger here, never blend.
    obs_idx = np.argwhere(observed)
    grid = np.linspace(0, 1, 21) if len(obs_idx) >= 40 else np.array([0.0, 1.0])
    curves = []
    for sp in range(n_splits):
        rng = np.random.default_rng(seed * 100 + sp)
        val = obs_idx[rng.choice(len(obs_idx), max(1, int(0.25 * len(obs_idx))), replace=False)]
        vr, vc = val[:, 0], val[:, 1]
        # validate against the NOISY observed score (train_score) -- the only thing the
        # method actually sees. Using the full score_mat here would leak the truth into
        # the ensemble-weight CV and is unfair to MC under evaluation noise.
        vt = train_score[vr, vc]
        cv_obs = observed.copy()
        cv_obs[vr, vc] = False
        # embedding is unchanged across splits; only the regressed scores use cv_obs
        Pc = predict_from_embedding(Z, train_score, cv_obs, predictor='knn',
                                    bias_decomp=True, logit=True, k=k)
        mcv = matrix_completion_predict(train_score, cv_obs)[vr, vc]
        pcv = Pc[vr, vc]
        h = np.isfinite(pcv) & np.isfinite(mcv) & np.isfinite(vt)
        if h.any():
            curves.append([np.mean(np.abs(np.clip(a * pcv[h] + (1 - a) * mcv[h], 0, 1) - vt[h]))
                           for a in grid])
    alpha = float(grid[int(np.argmin(np.mean(curves, axis=0)))]) if curves else 0.5
    ens = np.clip(alpha * P_comb + (1 - alpha) * mc, 0, 1)
    return dict(combined=P_comb, matcomplete=mc, ensemble=ens,
                sigma=med, predictor='knn', alpha=alpha)


def run_one(resp_X, Qu, qid_code, model_id, task_id, query_id, groups, row_score,
            score_mat, models, tasks, obs_prob, query_obs_prob, seed, k,
            n_models_use=None, pca_elbows=2, pca_max_components=128,
            predictor='ols', mds_dim=0, bias_decomp=True, logit=True):
    """One replicate: sample, PCA-reduce (observed only, separately q/r), estimate."""
    rng = np.random.default_rng(seed)
    if n_models_use is not None and n_models_use < len(models):
        sel = np.sort(rng.choice(len(models), n_models_use, replace=False))
        models = [models[i] for i in sel]
        score_mat = score_mat[sel]
    observed = sample_observed(len(models), len(tasks), obs_prob, rng)
    # Only (model, task) cells that actually exist can be observed (datasets like
    # WMT are not fully paired — some models miss some tasks).
    observed &= np.isfinite(score_mat)

    # sample_mat: observed-pair score = mean over the *answered* queries
    # (= full score when query_obs_prob==1). Used to fit; full score_mat is the
    # held-out evaluation target.
    sample_mat = np.full_like(score_mat, np.nan)
    keep = []
    for i in range(len(models)):
        for t in range(len(tasks)):
            if not observed[i, t]:
                continue
            idx = groups.get((models[i], tasks[t]))
            if idx is None or len(idx) == 0:
                continue
            if query_obs_prob < 1.0:
                m = rng.random(len(idx)) < query_obs_prob
                if not m.any():
                    m[rng.integers(len(idx))] = True
                idx = idx[m]
            sample_mat[i, t] = row_score[idx].mean()
            keep.append(idx)
    if not keep:
        return []
    keep = np.concatenate(keep)

    # PCA on the observed responses (per run, observed data only).
    resp_red = pca_reduce_elbow(resp_X[keep], n_elbows=pca_elbows,
                                max_components=pca_max_components)

    # PCA on the observed *unique* queries, separately.
    obs_codes = qid_code[keep]
    uniq = np.unique(obs_codes)
    q_red = pca_reduce_elbow(Qu[uniq], n_elbows=pca_elbows,
                             max_components=pca_max_components)
    code2vec = {c: q_red[j] for j, c in enumerate(uniq)}

    df_obs = pd.DataFrame({
        'model_id': model_id[keep],
        'task_id': task_id[keep],
        'query_id': query_id[keep],
        'embedding': list(resp_red),
        'query_embedding': [code2vec[c] for c in obs_codes],
    })

    def _row(name, mae, pred=predictor, md=mds_dim):
        return dict(estimator=name, obs_prob=obs_prob, query_obs_prob=query_obs_prob,
                    n_models=len(models), seed=seed, predictor=pred,
                    mds_dim=md, bias_decomp=bias_decomp, logit=logit, mae=mae)

    rows = []
    for name, kw in ESTIMATORS.items():
        mae = fit_predict(df_obs, sample_mat, score_mat, observed, models,
                          predictor=predictor, mds_dim=mds_dim,
                          bias_decomp=bias_decomp, logit=logit, k=k, **kw)
        rows.append(_row(name, mae))

    # Score-only baseline: logit-space rank-2 matrix completion (BenchPress),
    # fit on the observed sample scores, evaluated against full scores.
    rows.append(_row('matcomplete', mae_heldout(
        matrix_completion_predict(sample_mat, observed), score_mat, observed),
        pred='na', md=0))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='math',
                    help="dataset key from DATASETS (e.g. 'math', 'wmt_14'); overrides --parquet/--tsv")
    ap.add_argument('--parquet', default=None)
    ap.add_argument('--tsv', default=None)
    ap.add_argument('--query_parquet', default=None,
                    help='google query-embedding parquet (default: alongside --parquet)')
    ap.add_argument('--score_col', default=None, help='tsv score column (default per --dataset)')
    ap.add_argument('--sweep',
                    choices=['task_parity', 'query_sparsity', 'n_models',
                             'mds_predictor', 'obs_x_query'],
                    default='task_parity')
    ap.add_argument('--obs_probs', type=float, nargs='+',
                    default=[0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
    ap.add_argument('--query_obs_probs', type=float, nargs='+',
                    default=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    ap.add_argument('--n_models_values', type=int, nargs='+',
                    default=[10, 20, 40, 60, 80, 95])
    ap.add_argument('--fixed_obs_prob', type=float, default=0.3)
    ap.add_argument('--n_seeds', type=int, default=25)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--pca_elbows', type=int, default=2)
    ap.add_argument('--pca_max_components', type=int, default=128)
    ap.add_argument('--query_source', choices=['google', 'mean_response', 'tfidf'],
                    default='google')
    ap.add_argument('--query_kernel', choices=['rbf', 'linear', 'delta'], default='rbf')
    ap.add_argument('--response_kernel', choices=['rbf', 'linear'], default='rbf')
    # Prediction head: bias-decomposition + logit (matching the MC baseline),
    # pluggable predictor, and MDS embedding dimension.
    ap.add_argument('--predictor', choices=['ols', 'knn', 'ridge'], default='ols')
    ap.add_argument('--mds_dim', type=int, default=0, help='0 = ClassicalMDS n_elbows=2')
    ap.add_argument('--bias_decomp', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--logit', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--predictors', nargs='+', default=['ols', 'knn', 'ridge'],
                    help='predictors swept by --sweep mds_predictor')
    ap.add_argument('--mds_dims', type=int, nargs='+', default=[2, 4, 8, 16, 32],
                    help='MDS dims swept by --sweep mds_predictor')
    ap.add_argument('--n_jobs', type=int, default=-1)
    ap.add_argument('--outdir', default='results-doublekernel')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    global ESTIMATORS
    ESTIMATORS = make_estimators(args.query_kernel, args.response_kernel)
    print(f'kernels: query={args.query_kernel}, response={args.response_kernel}; '
          f'reduction: PCA(n_elbows={args.pca_elbows}) per run, observed only')

    print(f'loading HELM data: dataset={args.dataset} ...')
    if args.dataset == 'pooled':
        (resp_X, Qu, qid_code, model_id, task_id, query_id,
         score_mat, models, tasks, groups, row_score) = load_pooled(('math', 'wmt_14'))
    else:
        cfg = DATASETS.get(args.dataset, {})
        parquet = args.parquet or cfg.get('parquet')
        tsv = args.tsv or cfg.get('tsv')
        query_parquet = args.query_parquet or cfg.get('query_parquet')
        score_col = args.score_col or cfg.get('score_col', 'score')
        (resp_X, Qu, qid_code, model_id, task_id, query_id,
         score_mat, models, tasks, groups, row_score) = load_helm_math(
            parquet, tsv, query_source=args.query_source,
            query_parquet=query_parquet, score_col=score_col)
    print(f'  query source: {args.query_source}; resp {resp_X.shape}, Qu {Qu.shape}')
    print(f'  {len(models)} models, {len(tasks)} tasks, {len(model_id)} response rows')

    pred_cfg = dict(predictor=args.predictor, mds_dim=args.mds_dim,
                    bias_decomp=args.bias_decomp, logit=args.logit)
    if args.sweep == 'task_parity':
        x_col = 'obs_prob'
        specs = [dict(obs_prob=p, query_obs_prob=1.0, n_models_use=None, **pred_cfg)
                 for p in args.obs_probs]
    elif args.sweep == 'query_sparsity':
        x_col = 'query_obs_prob'
        specs = [dict(obs_prob=args.fixed_obs_prob, query_obs_prob=q, n_models_use=None, **pred_cfg)
                 for q in args.query_obs_probs]
    elif args.sweep == 'n_models':
        x_col = 'n_models'
        specs = [dict(obs_prob=args.fixed_obs_prob, query_obs_prob=1.0, n_models_use=n, **pred_cfg)
                 for n in args.n_models_values]
    elif args.sweep == 'obs_x_query':  # 2D grid: obs_prob x query_obs_prob
        x_col = 'obs_prob'
        specs = [dict(obs_prob=p, query_obs_prob=q, n_models_use=None, **pred_cfg)
                 for p in args.obs_probs for q in args.query_obs_probs]
    else:  # mds_predictor: vary MDS dim x predictor at fixed obs_prob
        x_col = 'mds_dim'
        specs = [dict(obs_prob=args.fixed_obs_prob, query_obs_prob=1.0, n_models_use=None,
                      predictor=pr, mds_dim=md, bias_decomp=args.bias_decomp, logit=args.logit)
                 for pr in args.predictors for md in args.mds_dims]
    print(f'  sweep: {args.sweep}; predictor={args.predictor} mds_dim={args.mds_dim} '
          f'bias_decomp={args.bias_decomp} logit={args.logit}')

    jobs = [delayed(run_one)(resp_X, Qu, qid_code, model_id, task_id, query_id,
                             groups, row_score, score_mat, models, tasks,
                             seed=s, k=args.k, pca_elbows=args.pca_elbows,
                             pca_max_components=args.pca_max_components, **spec)
            for spec in specs for s in range(args.n_seeds)]
    nested = Parallel(n_jobs=args.n_jobs, verbose=10)(jobs)
    res = pd.DataFrame([r for sub in nested for r in sub])
    res.to_csv(outdir / f'{args.sweep}.csv', index=False)

    if args.sweep == 'mds_predictor':
        group_cols = ['mds_dim', 'predictor', 'estimator']
    elif args.sweep == 'obs_x_query':
        group_cols = ['obs_prob', 'query_obs_prob', 'estimator']
    else:
        group_cols = [x_col, 'estimator']
    summary = (res.groupby(group_cols)['mae']
               .agg(['mean', 'std', 'count']).reset_index())
    print(summary.to_string(index=False))
    summary.to_csv(outdir / f'{args.sweep}_summary.csv', index=False)
    print(f'wrote {outdir}/{args.sweep}.csv')


if __name__ == '__main__':
    main()
