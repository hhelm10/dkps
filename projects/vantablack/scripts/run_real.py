"""Real-data run on an ingested scope: embed -> detect -> estimate -> figure.

    python -m pipeline.ingest_moltnet --start ... --end ... --submolts general \\
        --top-n 300 --donor-submolts philosophy ponderings --name <scope>
    python run_real.py --scope <scope> --t0 2026-02-12T00:00 [--B 199]

Writes artifacts/figure2_<scope>.{png,pdf} and data/results/results_<scope>.json.
Gated on the section-0 decisions (writing/notes/decisions.md) and on the
pre-registered ground truth (writing/notes/ground_truth_preregistration.md).
"""
import argparse
import json
import os
import pickle
import numpy as np
import pandas as pd

from pipeline import embed
from pipeline.ingest_moltnet import ORIGIN
from pipeline.run import run_pipeline, PipeConfig
from pipeline.schema import validate, coverage
from pipeline.figure2 import draw

HERE = os.path.dirname(os.path.abspath(__file__))
from pipeline.paths import DERIVED, RESULTS, ART  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scope', required=True)
    ap.add_argument('--t0', required=True, help='burn-in end, UTC')
    ap.add_argument('--B', type=int, default=199)
    ap.add_argument('--window-h', type=float, default=24.0)
    ap.add_argument('--grid-step', type=float, default=4.0)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    d = os.path.join(DERIVED, a.scope)
    acts = pd.read_parquet(os.path.join(d, 'actions.parquet'))
    ctx = pd.read_parquet(os.path.join(d, 'contexts.parquet'))
    ev = pd.read_parquet(os.path.join(d, 'score_events.parquet'))
    meta = json.load(open(os.path.join(d, 'meta.json')))
    validate(acts, ctx, ev)
    cov = coverage(acts)
    print('[coverage]', json.dumps(cov, default=str))

    lo, hi = meta['window_hours']
    t0 = (pd.Timestamp(a.t0, tz='UTC') - ORIGIN).total_seconds() / 3600
    # shift the clock so the scope starts at 0
    acts = acts.assign(t=acts.t - lo)
    ctx = ctx.assign(t=ctx.t - lo)
    ev = ev.assign(t=ev.t - lo)
    content = acts[acts.agent_id.isin(set(meta['system']))]
    E = embed.embed(content.text.tolist(), cache_name=f'{a.scope}_actions')
    C = embed.embed(ctx.text.tolist(), cache_name=f'{a.scope}_contexts')
    data = dict(actions=acts, contexts=ctx, score_events=ev,
                emb=dict(action_ids=content.action_id.tolist(), E=E,
                         thread_ids=ctx.thread_id.tolist(), C=C),
                meta=dict(t0=t0 - lo, T=hi - lo, source=meta['source']),
                system=meta['system'], donors=meta.get('donors') or [])
    res = run_pipeline(data, PipeConfig(B=a.B, seed=a.seed, window_h=a.window_h,
                                        grid_step=a.grid_step))
    out = dict(scope=a.scope, meta=meta, coverage=cov, t0=a.t0, d=res['proj']['d'],
               separability=res['separability'], family_size=res['scan']['n_family'],
               Lambda=res['scan']['Lambda'],
               Lambda_null_q95=float(np.quantile(res['scan']['Lambda_null'], .95)),
               delta_hat=res['delta_T'], delta_ci=res['delta_ci'],
               flagged=[{k: f[k] for k in ('agents', 'source', 'size', 'T', 'zE', 'zC', 'p_fw',
                                           'onset', 'own_gain', 'own_gain_ci', 'adversarial',
                                           'time_null_p')} for f in res['flagged']])
    with open(os.path.join(RESULTS, f'results_{a.scope}.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    with open(os.path.join(d, 'res.pkl'), 'wb') as fh:
        pickle.dump(res, fh)
    draw(res, os.path.join(ART, f'figure2_{a.scope}.png'))
    print(f'wrote artifacts/figure2_{a.scope}.png and data/results/results_{a.scope}.json')


if __name__ == '__main__':
    main()
