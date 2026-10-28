import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from joblib import Parallel, delayed
import pandas as pd
from pipeline.crossval import load
from experiments.completion import trial
data, qmed, suite = load('eee')
# the two cohort operating points used for the per-task Wilcoxon tests (Section 4.2)
specs = [(10, 0.5, 0.5, 'cohort'), (45, 0.5, 0.5, 'cohort')]
def run(n, pt, pq, sw, s):
    rows = trial(data, qmed, n, None, pt, s, p_query=pq, dump_cells=True)
    for r in rows: r['sweep'] = sw
    return rows
jobs = [delayed(run)(n, pt, pq, sw, s) for (n, pt, pq, sw) in specs for s in range(16)]
rows = [r for sub in Parallel(n_jobs=-1)(jobs) if sub for r in sub]
pd.DataFrame(rows).to_csv('results-eee-unified/comp_cond_cells.csv', index=False)
print('eee comp cells:', len(rows))
