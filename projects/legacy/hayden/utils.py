import numpy as np
from tqdm import tqdm
from dkps.dkps import DataKernelPerspectiveSpace

def make_embedding_dict(df):
    model_names  = df.model.unique()
    instance_ids = df.instance_id.unique()
    
    embedding_dict = {}
    for model_name in model_names:
        sub = df[df.model == model_name]
        assert (sub.instance_id.values == instance_ids).all(), f'instance_ids are not the same for model {model_name}'
        embedding_dict[model_name] = np.vstack(sub.embedding.values)
    
    embedding_dict = {k:v[:,None] for k,v in embedding_dict.items()}
    
    return embedding_dict


def dkps_df(df, **kwargs):
    embedding_dict = make_embedding_dict(df)
    return DataKernelPerspectiveSpace(**kwargs).fit_transform(embedding_dict, return_dict=True)


# --

def onehot_embedding(df, dataset):
    if dataset == 'med_qa':
        lookup = {'A' : 0, 'B' : 1, 'C' : 2, 'D' : 3}
        
        embeddings = np.zeros((len(df), 4))
        for i, xx in enumerate(df.response.values):
            if xx in lookup:
                embeddings[i, lookup[xx]] = 1
        
        df['embedding'] = embeddings.tolist()
    
    elif 'legalbench' in dataset:
        # slightly different - bad values get mapped to 0
        n_levels   = len(df.response.unique())
        embeddings = np.zeros((len(df), n_levels))
        for i, xx in enumerate(df.response.values):
            embeddings[i, xx] = 1

        df['embedding'] = embeddings.tolist()
    else:
        raise ValueError(f'{dataset} is not supported for onehot embeddings')
    
    return df