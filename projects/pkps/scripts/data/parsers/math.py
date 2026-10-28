import pandas as pd


class MATHParser:
    @staticmethod
    def parse(run, dps, _params):
        id2score = {x['instance_id']: x['stats']['math_equiv_chain_of_thought'] for x in dps}
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
        # [TODO] drop problems that have empty responses?
        # or deal w/ empty responses somehow?
        
        # drop models w/ zero score - what's the deal here?
        y_acts = df.groupby('model').score.mean().to_dict()
        for model, score in y_acts.items():
            if score == 0:
                df = df[df.model != model]

        df = df.reset_index(drop=True)
        return df