"""Paired, reversible response-format interventions on one target developer at a time."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.stress_followups import load_cache, matrix, family, geometry_bank, nested_predict
from experiments.stress_next_checks import evaluate, summarize
from pipeline.stress_geometry import affinity_grid, normalize_affinity, query_scale


def geometry_shift(base, changed, vectors, models, seed, variant):
    both=pd.concat([base.assign(model_id=base.model_id+'|base'),
                    changed.assign(model_id=changed.model_id+'|changed')],ignore_index=True)
    names=[m+'|base' for m in models]+[m+'|changed' for m in models]
    q,r=vectors[both.q_index],vectors[both.r_index]
    scale=query_scale(vectors[base.q_index],base.task_id,base.query_id,seed=seed)
    grid=affinity_grid(q,r,both.model_id,both.query_id,both.task_id,{1.:scale},q,names)
    result=[]; n=len(models)
    for kernel in ['constant','rbf_1']:
        a=normalize_affinity(*grid[kernel],'model')
        d=np.sqrt(np.maximum(np.diag(a)[:,None]+np.diag(a)[None,:]-2*a,0))
        between=float(np.median(d[:n,:n][np.triu_indices(n,1)]))
        for i,m in enumerate(models):
            result.append(dict(seed=seed,variant=variant,kernel=kernel,model_id=m,
                               within_shift=d[i,i+n],median_between=between,ratio=d[i,i+n]/between))
    return result


def run(source,out):
    pool,vectors=load_cache(source)
    truth=pd.read_parquet(source/'targets.parquet')
    observations=pd.read_parquet(source/'observations.parquet')
    keys=['model_id','task_id','query_id']
    results=[]; shifts=[]; selections=[]
    for seed,observed in observations.groupby('seed'):
        rows=pool.merge(observed[keys],on=keys,validate='many_to_one')
        base=rows[rows.variant=='original'].copy().reset_index(drop=True)
        models,tasks=sorted(base.model_id.unique()),sorted(base.task_id.unique())
        sample=matrix(base,models,tasks)
        if len(base)!=len(models)*len(tasks): raise ValueError('style test must use exactly m=1 per cell')
        original_bank,_=geometry_bank(base,vectors,models,int(seed))
        for variant in ['original','json','blockquote']:
            changed=rows[rows.variant==variant].copy()
            lookup=changed.set_index(keys).r_index
            if variant!='original': shifts+=geometry_shift(base,changed,vectors,models,int(seed),variant)
            output=[]
            for fam in sorted(set(map(family,models))):
                target_rows=base.model_id.map(family)==fam
                styled=base.copy()
                styled.loc[target_rows,'r_index']=[lookup.loc[tuple(row)] for row in base.loc[target_rows,keys].to_numpy()]
                bank=original_bank if variant=='original' else geometry_bank(styled,vectors,models,int(seed))[0]
                for regime in ['full-reference','cache-only']:
                    permitted=truth[truth.model_id.map(family)!=fam] if regime=='full-reference' else None
                    indices,estimates,choices=nested_predict(bank,models,tasks,sample,permitted,fam,regime)
                    selections += [dict(seed=int(seed),variant=variant,**choice) for choice in choices]
                    for method,pred in estimates.items():
                        for a,i in enumerate(indices):
                            for j,task in enumerate(tasks):
                                output.append(dict(model_id=models[i],task_id=task,method=method,
                                                   regime=regime,prediction=pred[a,j]))
            frame=evaluate(pd.DataFrame(output),truth).assign(seed=int(seed),variant=variant)
            frame.to_csv(out/f'style_{variant}_seed{seed}.predictions.csv.gz',index=False)
            results.append(frame)
            print('style',seed,variant,'complete',flush=True)
    frame=pd.concat(results,ignore_index=True)
    summarize(frame,['variant','regime','method']).to_csv(out/'style_summary.csv',index=False)
    pd.DataFrame(selections).to_csv(out/'style_selection.csv',index=False)
    distances=pd.DataFrame(shifts)
    distances.to_csv(out/'style_distances.csv',index=False)
    distances.groupby(['variant','kernel']).ratio.agg(['median','min','max']).to_csv(out/'style_geometry_summary.csv')
    keys=['seed','regime','method','model_id','task_id']
    original=frame[frame.variant=='original'].set_index(keys)
    comparisons=[]
    for variant in ['json','blockquote']:
        changed=frame[frame.variant==variant].set_index(keys)
        if not changed.index.equals(original.index): raise ValueError('style support mismatch')
        delta=pd.DataFrame({'prediction_change':abs(changed.prediction-original.prediction),
                            'error_change':changed.absolute_error-original.absolute_error}).reset_index()
        for method in ['sample','task-mean','task-median','shrinkage-fixed','shrinkage-nested']:
            assert np.allclose(delta.loc[delta.method==method,'prediction_change'],0)
        per_seed=delta.groupby(['regime','method','seed','task_id'])[['prediction_change','error_change']].mean().groupby(['regime','method','seed']).mean()
        summary=per_seed.groupby(['regime','method']).agg(
            mean_absolute_prediction_change=('prediction_change','mean'),
            mean_mae_change=('error_change','mean'),min_seed_mae_change=('error_change','min'),
            max_seed_mae_change=('error_change','max')).reset_index().assign(variant=variant,n_seeds=3)
        comparisons.append(summary)
    pd.concat(comparisons).to_csv(out/'style_paired_changes.csv',index=False)
    config=json.loads((source/'style_design.json').read_text())
    config.update(embedding=json.loads((source/'embeddings/config.json').read_text()),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'style_config.json').write_text(json.dumps(config,indent=2)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=Path('data/eee-style'))
    ap.add_argument('--out',type=Path,default=Path('projects/pkps/artifacts/stress-test/next-checks'))
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=2): run(args.source,args.out)
