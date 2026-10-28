"""
    parsers/wmt_14.py
"""

import pandas as pd
from tqdm import tqdm

import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
nltk.download('wordnet')

from sacrebleu.metrics import CHRF
from joblib import Parallel, delayed

def bleu_4(gold: str, pred: str) -> float:
    return sentence_bleu([word_tokenize(gold)], word_tokenize(pred), weights=(0, 0, 0, 1))

def bleu_flat(gold: str, pred: str) -> float:
    return sentence_bleu([word_tokenize(gold)], word_tokenize(pred), weights=(0.25, 0.25, 0.25, 0.25))

def chrf(gold: str, pred: str) -> float:
    metric = CHRF(word_order=2) # chrF++; set word_order=0 for plain chrF
    return metric.sentence_score(pred, [gold]).score

def meteor(gold: str, pred: str) -> float:
    return meteor_score([word_tokenize(gold)], word_tokenize(pred))   # returns 0‑1; multiply by 100 if you prefer %

class WMT14Parser:
    @staticmethod
    def parse(run, dps, _params):
        id2score = {x['instance_id']: x['stats']['bleu_4'] for x in dps}
        
        df_tmp = pd.DataFrame([{
            "dataset"     : _params['dataset'],
            "model"       : _params['model'],
            "instance_id" : _params['dataset'] + '--' + x['instance']['id'],
            "query"       : x['request']['prompt'],
            "response"    : x['result']['completions'][0]['text'],
            "target"      : x['instance']['references'][0]['output']['text'],
            "score"       : id2score[x['instance']['id']],
        } for x in run['request_states']])
        return df_tmp
    
    @staticmethod
    def postprocess(df):
        lens = df.groupby('instance_id').response.apply(lambda x: min([len(xx) for xx in x]))
        bad  = lens[lens < 25]
        df   = df[~df.instance_id.isin(bad.index)]

        df = df.reset_index(drop=True)

        def _compute_scores(target, response):
            return {
                "bleu_4"    : bleu_4(target, response),
                "bleu_flat" : bleu_flat(target, response),
                "chrf"      : chrf(target, response),
                "meteor"    : meteor(target, response),
            }

        jobs    = [delayed(_compute_scores)(q, r) for q, r in tqdm(df[['target', 'response']].values, desc='computing scores')]
        _scores = Parallel(n_jobs=-1, verbose=10)(jobs)

        df[['bleu_4', 'bleu_flat', 'chrf', 'meteor']] = pd.DataFrame(_scores)
        return df