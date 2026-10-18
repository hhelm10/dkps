"""Synthetic end-to-end demo: simulate, detect, estimate, score against truth.

    python demo_synthetic.py [--seed 0] [--B 199]

Writes artifacts/figure2_synthetic.{png,pdf} and data/results/results_synthetic.json.
The true Delta comes from re-running the simulator with the ring switched
off (same seed), which is only possible in simulation.
"""
import argparse
import json
import os
import pickle
import numpy as np

from pipeline.simulate import simulate, SimConfig
from pipeline.run import run_pipeline, PipeConfig
from pipeline import scores
from pipeline.figure2 import draw

HERE = os.path.dirname(os.path.abspath(__file__))
from pipeline.paths import DERIVED, RESULTS, ART, NOTES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--B', type=int, default=199)
    a = ap.parse_args()

    cfg = SimConfig(seed=a.seed)
    data = simulate(cfg)
    data['system'], data['donors'] = data['truth']['system'], data['truth']['donors']
    res = run_pipeline(data, PipeConfig(B=a.B, seed=a.seed))

    # ground truth Delta: same seed, ring off, same complement
    cf = simulate(cfg, collude=False)
    Ycf = scores.trajectories(cf['score_events'], res['system'], res['grid'])
    comp = res['comp']
    true_delta = float(Ycf[comp].mean(0)[-1] - res['Y'][comp].mean(0)[-1])

    ring, coh = set(data['truth']['ring']), set(data['truth']['cohesive'])
    out = dict(sim=data['meta']['cfg'], pipe=res['cfg'], d=res['proj']['d'],
               separability=res['separability'], family_size=res['scan']['n_family'],
               Lambda=res['scan']['Lambda'],
               Lambda_null_q95=float(np.quantile(res['scan']['Lambda_null'], .95)),
               delta_hat=res['delta_T'], delta_ci=res['delta_ci'], delta_true=true_delta,
               flagged=[])
    for f in res['flagged']:
        m = set(f['agents'])
        out['flagged'].append(dict(
            source=f['source'], size=f['size'], T=f['T'], zE=f['zE'], zC=f['zC'],
            p_fw=f['p_fw'], onset=f['onset'], own_gain=f['own_gain'],
            own_gain_ci=f['own_gain_ci'], adversarial=f['adversarial'],
            time_null_p=f['time_null_p'],
            ring_precision=len(m & ring) / len(m), ring_recall=len(m & ring) / len(ring),
            cohesive_overlap=len(m & coh)))
    print(f'[truth] Delta(T) on unflagged agents = {true_delta:+.2f} '
          f'(estimates: {", ".join(f"{k}={v:+.2f}" for k, v in res["delta_T"].items())})')
    for f in out['flagged']:
        print(f'[truth] {f["source"]}: ring precision={f["ring_precision"]:.2f} '
              f'recall={f["ring_recall"]:.2f}, cohesive members={f["cohesive_overlap"]}, '
              f'onset={f["onset"]} (true ring onset {cfg.t_star})')

    os.makedirs(os.path.join(NOTES), exist_ok=True)
    with open(os.path.join(RESULTS, 'results_synthetic.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    os.makedirs(os.path.join(DERIVED), exist_ok=True)
    with open(os.path.join(DERIVED, 'res_synthetic.pkl'), 'wb') as fh:
        pickle.dump(res, fh)
    draw(res, os.path.join(ART, 'figure2_synthetic.png'))
    print('wrote artifacts/figure2_synthetic.png and data/results/results_synthetic.json')


if __name__ == '__main__':
    main()
