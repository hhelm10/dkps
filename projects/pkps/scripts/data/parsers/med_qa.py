"""
    parsers/wmt_14.py
"""

import pandas as pd
from tqdm import tqdm

class MEDQAParser:
    @staticmethod
    def parse(run, dps, _params):
        id2score = {x['instance_id']: x['stats']['quasi_exact_match'] for x in dps}
        
        df_tmp = pd.DataFrame([{
            "dataset"     : _params['dataset'],
            "model"       : _params['model'],
            "instance_id" : _params['dataset'] + '--' + x['instance']['id'],
            "query"       : x['request']['prompt'],
            "response"    : x['result']['completions'][0]['text'].strip(),
            "target"      : x['instance']['references'][0]['output']['text'],
            "score"       : id2score[x['instance']['id']],
        } for x in run['request_states']])
        return df_tmp
    
    @staticmethod
    def postprocess(df):
        return df