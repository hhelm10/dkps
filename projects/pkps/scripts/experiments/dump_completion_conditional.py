import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from joblib import Parallel, delayed
import pandas as pd
from pipeline.crossval import load
from experiments.completion import trial
data, qmed, suite = load('suite')
# levers vary one at a time; others held at MEDIAN (n=40, p_task=0.5, p_query=0.5)
specs = ([(n,0.5,0.5,'cohort') for n in [10,40,93]]
       + [(40,p,0.5,'coverage') for p in [0.2,0.5,0.9]]
       + [(40,0.5,q,'querydepth') for q in [0.25,0.5,1.0]])
def run(n,pt,pq,sw,s):
    rows = trial(data, qmed, n, None, pt, s, p_query=pq, dump_cells=True)
    for r in rows: r['sweep']=sw
    return rows
jobs=[delayed(run)(n,pt,pq,sw,s) for (n,pt,pq,sw) in specs for s in range(8)]
rows=[r for sub in Parallel(n_jobs=-1)(jobs) if sub for r in sub]
pd.DataFrame(rows).to_csv('/home/paperspace/projects/dkps/projects/pkps/scripts/results-pkps-unified/comp_cond_cells.csv',index=False)
print('comp_cond_cells.csv',len(rows))
