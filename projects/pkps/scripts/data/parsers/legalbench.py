"""
    parsers/legalbench.py
"""

import pandas as pd
from tqdm import tqdm

def _clean_answer(x):
    x = x.lower().strip().rstrip('.')
    
    prefixes = [
        "function: ",
        "answer: ",
        "label: ",
    ]
    for p in prefixes:
        if x.startswith(p):
            x = x[len(p):]
            break
    
    return x

class LegalBenchParser:
    @staticmethod
    def parse(run, dps, _params):
        id2score = {x['instance_id']: x['stats']['quasi_exact_match'] for x in dps}
        
        choices     = sorted(set([x['instance']['references'][0]['output']['text'] for x in run['request_states']]))
        choices2idx = {x.lower(): i + 1 for i, x in enumerate(choices)}
        
        df_tmp = pd.DataFrame([{
            **_params,
            "instance_id" : _params['dataset'] + '--' + x['instance']['id'],
            "query"       : x['request']['prompt'],
            "response"    : choices2idx.get(_clean_answer(x['result']['completions'][0]['text'].strip()), 0),
            "target"      : x['instance']['references'][0]['output']['text'],
            "score"       : id2score[x['instance']['id']],
        } for x in run['request_states']])
        return df_tmp
    
    @staticmethod
    def postprocess(df):
        vcs  = df[['model', 'stop']].drop_duplicates().model.value_counts()
        bad  = list(vcs.index[vcs == 2])
        drop = df.model.isin(bad) & (~df.stop.isna())
        df   = df[~drop]
        del df['stop']
        
        bad_models = [
            'mistralai_mistral-large-2402', # abercrombie
            'mistralai_mistral-small-2402'
        ]
        df = df[~df.model.isin(bad_models)]
        df = df.reset_index(drop=True)
        
        return df