"""Reuse the pinned EG2 cache and embed only new natural-cache or styled texts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def style_text(text, variant):
    if variant == 'original':
        return text
    if variant == 'json':
        return json.dumps({'response': text}, ensure_ascii=False)
    if variant == 'blockquote':
        return '\n'.join('> '+line for line in text.split('\n'))
    raise ValueError(variant)


def restore_text(text, variant):
    if variant == 'original':
        return text
    if variant == 'json':
        return json.loads(text)['response']
    if variant == 'blockquote':
        if not all(line.startswith('> ') for line in text.split('\n')):
            raise ValueError('invalid blockquote')
        return '\n'.join(line[2:] for line in text.split('\n'))
    raise ValueError(variant)


def prepare_style(base, out, model_path):
    from transformers import AutoTokenizer
    from experiments.stress_followups import observe
    pool = pd.read_parquet(base/'pool.parquet')
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    prompt = json.loads((model_path/'config_sentence_transformers.json').read_text())['prompts']['SentenceSimilarity']
    variants = ['original', 'json', 'blockquote']
    # Exclude very long responses before tokenization, then check every rendering.
    texts = list(dict.fromkeys(pool.loc[pool.response.str.len() <= 16000, 'response']))
    eligible = set()
    for start in range(0, len(texts), 128):
        chunk = texts[start:start+128]
        renderings = [prompt+style_text(t,v) for t in chunk for v in variants]
        lengths = np.array(tokenizer(renderings, truncation=False, padding=False,
                                    return_length=True, return_attention_mask=False)['length']).reshape(-1,3)
        eligible.update(t for t, length in zip(chunk, lengths.max(1)) if length <= 2048)
    allowed = pool[pool.response.isin(eligible)]
    counts = allowed.groupby('model_id').task_id.nunique()
    complete_models = counts[counts == pool.task_id.nunique()].index
    excluded_models = sorted(set(pool.model_id)-set(complete_models))
    eligible_rows_before_cohort = len(allowed)
    allowed = allowed[allowed.model_id.isin(complete_models)]
    if len(complete_models) < 10:
        raise ValueError('fewer than ten models have eligible responses on every task')
    selected = pd.concat([observe(allowed,1,seed).assign(seed=seed) for seed in [0,1,2]])
    selected[['seed','model_id','task_id','query_id']].to_parquet(out/'observations.parquet',index=False)
    unique = selected.drop_duplicates(['model_id','task_id','query_id']).drop(columns='seed')
    frames=[]
    for variant in variants:
        changed = unique.copy()
        changed['response'] = changed.response.map(lambda text: style_text(text,variant))
        assert all(restore_text(t,variant)==s for t,s in zip(changed.response,unique.response))
        frames.append(changed.assign(variant=variant))
    styled = pd.concat(frames,ignore_index=True)
    styled.to_parquet(out/'pool.parquet',index=False)
    pd.read_parquet(base/'targets.parquet').to_parquet(out/'targets.parquet',index=False)
    info=dict(variants=variants,seeds=[0,1,2],m=1,n_models=len(complete_models),
              n_tasks=pool.task_id.nunique(),pool_rows=len(pool),eligible_rows=len(allowed),
              eligible_rows_before_cohort=eligible_rows_before_cohort,excluded_models=excluded_models,
              excluded_rows=len(pool)-len(allowed),unique_selected_rows=len(unique),
              max_tokens_including_prompt=2048,response_truncation=False,
              intervention='reversible JSON and blockquote serialization; original content exactly recoverable',
              evaluation='change one outer developer cache at a time; other developers remain original',
              note='controlled formatting interventions; not a test of generated prose styles')
    (out/'style_design.json').write_text(json.dumps(info,indent=2)+'\n')
    print('style design',info,flush=True)


def extend_cache(base, source, model_path, device):
    from experiments.stress_followups import load_cache
    from data.embed_eee_followups import load_model, encode
    import torch
    _, base_vectors = load_cache(base)
    base_pool = pd.read_parquet(base/'pool.parquet')
    base_texts = list(dict.fromkeys(base_pool['query'].tolist()+base_pool.response.tolist()))
    base_config = json.loads((base/'embeddings/config.json').read_text())
    if hashlib.sha256(json.dumps(base_texts,ensure_ascii=False).encode()).hexdigest()!=base_config['text_sha256']:
        raise ValueError('base text order differs from cached embeddings')
    lookup = {t:i for i,t in enumerate(base_texts)}
    pool = pd.read_parquet(source/'pool.parquet')
    texts = list(dict.fromkeys(pool['query'].tolist()+pool.response.tolist()))
    ids = {t:i for i,t in enumerate(texts)}
    fingerprint = hashlib.sha256(json.dumps(texts,ensure_ascii=False).encode()).hexdigest()
    cache = source/'embeddings'; cache.mkdir(exist_ok=True)
    config = dict(base_config, text_sha256=fingerprint, n_texts=len(texts),
                  reused_texts=sum(t in lookup for t in texts),base_text_sha256=base_config['text_sha256'],
                  device='mps:0' if device=='mps' else 'cpu')
    marker=cache/'config.json'
    if marker.exists() and json.loads(marker.read_text())!=config:
        raise ValueError('extension cache configuration changed')
    marker.write_text(json.dumps(config,indent=2)+'\n')
    mapping=pool.drop(columns=['query','response']).copy()
    mapping['q_index']=[ids[t] for t in pool['query']]
    mapping['r_index']=[ids[t] for t in pool.response]
    mapping.to_parquet(cache/'rows.parquet',index=False)
    model=None; chunk=config['chunk_size']; started=time.perf_counter()
    print('cache',source,'texts',len(texts),'new',len(texts)-config['reused_texts'],flush=True)
    for start in range(0,len(texts),chunk):
        path=cache/f'{start:07d}.npy'; batch=texts[start:start+chunk]
        if path.exists():
            arr=np.load(path)
            if arr.shape!=(len(batch),768) or not np.isfinite(arr).all(): raise ValueError('invalid shard')
            continue
        arr=np.empty((len(batch),768),dtype=np.float32)
        new=[i for i,t in enumerate(batch) if t not in lookup]
        for i,t in enumerate(batch):
            if t in lookup: arr[i]=base_vectors[lookup[t]]
        if new:
            if model is None: model=load_model(model_path,device,'float32',2048)
            arr[new]=encode(model,[batch[i] for i in new],8)
        with path.with_suffix('.part').open('wb') as f: np.save(f,arr)
        path.with_suffix('.part').replace(path)
        if new:
            print(source.name,min(start+chunk,len(texts)),'/',len(texts),'elapsed',round(time.perf_counter()-started),flush=True)
        if model is not None and device=='mps':
            torch.mps.synchronize(); torch.mps.empty_cache()
    (cache/'complete.json').write_text(json.dumps(dict(n_texts=len(texts),text_sha256=fingerprint))+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['natural','style'])
    ap.add_argument('--base',type=Path,default=Path('data/eee-followups'))
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--model',type=Path,default=Path('data/models/embeddinggemma-2'))
    ap.add_argument('--device',choices=['mps','cpu'],default='mps')
    args=ap.parse_args(); args.source.mkdir(parents=True,exist_ok=True)
    if args.mode=='style': prepare_style(args.base,args.source,args.model)
    extend_cache(args.base,args.source,args.model,args.device)


if __name__=='__main__':
    main()
