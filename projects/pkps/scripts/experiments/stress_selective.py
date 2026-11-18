"""Repeat the existing oracle selective-reporting probe over 16 masking seeds.

This is distinct from the evaluation on the naturally incomplete datastore graph.
Methods and masks use the same per-model budgets; compare only common hidden cells.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.stress_followups import load_cache, code_provenance
from experiments.stress_cache_design import missingness_probe
from experiments.stress_next_checks import summarize
from experiments.audit_stress_uncertainty import paired_cells, paired_seed_intervals


def main():
    out=Path('projects/pkps/artifacts/stress-test/next-checks/selective-16seed')
    out.mkdir(parents=True,exist_ok=True)
    pool,vectors=load_cache(Path('data/eee-followups'))
    truth=pd.read_parquet('data/eee-followups/targets.parquet')
    results=[]
    for seed in range(16):
        dest=out/f'seed{seed}'; dest.mkdir(exist_ok=True)
        path=dest/'missingness_predictions.csv.gz'
        if not path.exists(): missingness_probe(pool,vectors,truth,dest,seed=seed)
        results.append(pd.read_csv(path).assign(seed=seed))
    frame=pd.concat(results,ignore_index=True)
    common=frame[frame.common_hidden].copy()
    summarize(common,['mode','method']).to_csv(out/'summary.csv',index=False)
    contrasts=[]
    for mode,group in common.groupby('mode'):
        # Recover benchmark labels from task IDs without consulting hidden scores.
        group=group.assign(suite=group.task_id.str.split(':').str[0])
        for baseline in ['constant','additive','lrmc']:
            pair=paired_cells(group,'model-expanded',baseline)
            diff=pair.groupby(['seed','task_id']).difference.mean().groupby('seed').mean()
            contrasts.append(dict(mode=mode,method='model-expanded',baseline=baseline,
                                  **paired_seed_intervals(diff,6,n_boot=10000)))
    pd.DataFrame(contrasts).to_csv(out/'paired_intervals.csv',index=False)
    common.groupby('seed').agg(n_cells=('model_id',lambda x:len(x)//(2*common.method.nunique())),
                               n_tasks=('task_id','nunique')).to_csv(out/'coverage.csv')
    config=dict(seeds=list(range(16)),observed_tasks_per_model=8,responses_per_observed_task=16,
                evaluation='same hidden cells under both masks within each seed',
                limitation='oracle score-dependent mask; not natural selection or deployment validation',
                implementation_sha256={**code_provenance(),
                    str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    print('selective reporting 16 seeds complete',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2): main()
