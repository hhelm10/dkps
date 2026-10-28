#!/usr/bin/env python
"""Standardize the HELM suite's MCQ/label answers to semantic option text.

Motivation: MedQA and LegalBench responses are stored as bare answer tokens
(MCQ letters, class-index digits) which carry no question-specific content.
This script resolves each token to the TEXT it denotes so responses can be
embedded in the same semantic space as the free-text datasets:

  * med_qa -- the stored `query` column's option ordering does NOT match the
    ordering the models actually saw (a data-prep reconstruction artifact:
    among models with score==1, 997/998 instances show a single letter, so all
    models shared one true ordering, but resolving letters against the stored
    query text contradicts `score` for every model). The true ordering is the
    ORIGINAL MedQA dataset ordering: matching instances to the MedQA test /
    validation / train splits by option-text sets and comparing the dataset
    answer key against the letter that score==1 models chose verifies
    999/1000 instances (the one inconsistent instance is dropped and falls
    back to raw-string behavior). The stored `target` column is likewise
    unreliable and is never used.
  * legalbench -- digits index the alphabetically sorted class names of each
    subset, 1-based, with 0 the unparseable-answer bucket (verified exactly by
    the score==1 response/target crosstab). 0 maps to 'invalid response'.

Writes (under DKPS data root):
  exports/answer_option_text_map.parquet      (dataset, instance_id, response, text);
                                              instance_id == '' rows apply dataset-wide
  exports/answer_option_text_google_embeddings.parquet  (text, embedding)

Requires GEMINI_API_KEY (repo-root .env). MedQA splits are fetched from
GBaker/MedQA-USMLE-4-options (test/train jsonl) and bigbio/med_qa
(validation parquet) on Hugging Face.
"""
import io
import json
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv(Path(__file__).resolve().parents[4] / '.env')
from pipeline.loaders import data_path      # noqa: E402
from dkps.embed import embed_api            # noqa: E402

GBAKER = 'https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options/resolve/main'
BIGBIO = 'https://huggingface.co/datasets/bigbio/med_qa/resolve/main/med_qa_en_4options_source'


def fetch(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def norm(t):
    """Aggressive normalization for option matching (unicode dashes etc. differ)."""
    return re.sub(r'[^a-z0-9]', '', str(t).casefold())


def medqa_pool():
    pool = []
    for split in ['phrases_no_exclude_test.jsonl', 'phrases_no_exclude_train.jsonl']:
        for line in fetch(f'{GBAKER}/{split}').decode().splitlines():
            r = json.loads(line)
            pool.append(dict(options=r['options'], answer_idx=r['answer_idx']))
    v = pd.read_parquet(io.BytesIO(fetch(f'{BIGBIO}/validation-00000-of-00001.parquet')))
    for _, r in v.iterrows():
        pool.append(dict(options={d['key']: d['value'] for d in r['options']},
                         answer_idx=r['answer_idx']))
    return pool


def resolve_medqa():
    by_opts = {}
    for r in medqa_pool():
        by_opts.setdefault(frozenset(norm(v) for v in r['options'].values()), []).append(r)

    m = pd.read_csv(data_path('data/med_qa.tsv'), sep='\t')
    m['iid'] = m['instance_id'].astype(str)
    m['bare'] = m['query'].astype(str).map(lambda q: q.split('\n\n')[-1].strip())
    opt_re = re.compile(r'^([A-E])\.\s+(.*\S)\s*$')

    def opts(q):
        return {mo.group(1): mo.group(2) for line in q.split('\n')
                if (mo := opt_re.match(line.strip()))}

    # consensus gold letter per instance (models that scored 1 share the true letter)
    m['resp'] = m['response'].astype(str).str.rstrip('.')
    letters = m[m['resp'].isin(list('ABCDE'))]
    gold = letters[letters['score'] == 1].groupby('iid')['resp'].agg(
        lambda x: x.mode().iloc[0])

    rows, checked, verified = [], 0, 0
    for iid, g in m.drop_duplicates('iid').set_index('iid').iterrows():
        cand = by_opts.get(frozenset(norm(v) for v in opts(g['bare']).values()), [])
        if len(cand) != 1:
            continue                        # unmatched instance -> raw-string fallback
        r = cand[0]
        if iid in gold.index:
            checked += 1
            if r['answer_idx'] != gold.loc[iid]:
                continue                    # behavioral contradiction -> drop
            verified += 1
        for letter, text in r['options'].items():
            rows.append(dict(dataset='med_qa', instance_id=iid, response=letter,
                             text=' '.join(str(text).split())))
    df = pd.DataFrame(rows)
    print(f'med_qa: {df.instance_id.nunique()}/1000 instances mapped, '
          f'behaviorally verified {verified}/{checked}')
    return df


def resolve_legalbench():
    l = pd.read_csv(data_path('data/legalbench.tsv'), sep='\t')
    rows = []
    for ds, g in l.groupby('dataset'):
        classes = sorted(g['target'].astype(str).unique())
        for d, name in enumerate(classes, start=1):
            rows.append(dict(dataset=ds, instance_id='', response=str(d), text=name))
        rows.append(dict(dataset=ds, instance_id='', response='0',
                         text='invalid response'))
    return pd.DataFrame(rows)


def main():
    tab = pd.concat([resolve_medqa(), resolve_legalbench()], ignore_index=True)
    tab.to_parquet(data_path('exports/answer_option_text_map.parquet'))
    uniq = sorted(tab['text'].unique())
    print(f'{len(tab)} map rows, embedding {len(uniq)} unique texts')
    E = np.asarray(embed_api('google', uniq, model='gemini-embedding-001'),
                   dtype=np.float32)
    pd.DataFrame({'text': uniq, 'embedding': list(E)}).to_parquet(
        data_path('exports/answer_option_text_google_embeddings.parquet'))
    print('wrote exports/answer_option_text_map.parquet and '
          'exports/answer_option_text_google_embeddings.parquet')


if __name__ == '__main__':
    main()
