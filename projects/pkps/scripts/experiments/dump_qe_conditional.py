import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from joblib import Parallel, delayed
import pandas as pd
from pipeline import loaders as H
from experiments.query_efficiency import run_seed
data = H.load_suite()
# each sweep varies one lever; the others held at the MEDIAN (m=4, n=40, p_task=0.5)
specs = ([(m,40,0.5,'budget') for m in [1,2,4,8,16]]
       + [(4,n,0.5,'cohort') for n in [10,40,93]]
       + [(4,40,p,'coverage') for p in [0.2,0.5,0.9]])
def run(m,n,p,sw,s):
    rows = run_seed(data, m, s, n_models=n, p_task=p, n_paired=None, dump_cells=True)
    for r in rows: r['sweep']=sw
    return rows
jobs=[delayed(run)(m,n,p,sw,s) for (m,n,p,sw) in specs for s in range(8)]
rows=[r for sub in Parallel(n_jobs=-1)(jobs) if sub for r in sub]
pd.DataFrame(rows).to_csv('/home/paperspace/projects/dkps/projects/pkps/scripts/results-pkps-rd1/qe_cond_cells.csv',index=False)
print('qe_cond_cells.csv',len(rows))
