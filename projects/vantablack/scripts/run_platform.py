"""Platform-scale run on real Moltbook data with a planted ground-truth ring.

    python run_platform.py --scope platform_jan31_feb17 --t0 2026-02-06T00:00 \\
        --t-star 2026-02-10T00:00 --m 12 --alpha 0.3 [--B 99 --workers 24]
    python run_platform.py ... --no-plant      # the real stream only

Inputs: data/derived/<scope>/ from pipeline.ingest_moltnet.build_platform and
embed_scope.py. Writes data/results/results_<tag>.json, artifacts/figure2_<tag>.{png,pdf}.

Ground truth: the planted ring (pipeline/plant.py). The unplanted stream is
the exact benign counterfactual, so the true Delta on any complement is the
planted harm. Real weak labels (writing/notes/ground_truth_preregistration.md) are
reported as found / not testable / missed.
"""
import argparse
import json
import os
import pickle
import time
import numpy as np
import pandas as pd

from pipeline import embed, scores
from pipeline.ingest_moltnet import ORIGIN
from pipeline.plant import plant, CAMPAIGN
from pipeline.run import run_pipeline, PipeConfig
from pipeline.figure2 import draw

HERE = os.path.dirname(os.path.abspath(__file__))
from pipeline.paths import DERIVED, RESULTS, ART  # noqa: E402
WEAK = {'Hackerclaw': 'flooding', 'RoyMas': 'zenity', 'Dragon234Sad': 'zenity',
        'LamaKor13264': 'zenity'}


def hours(ts):
    return (pd.Timestamp(ts, tz='UTC') - ORIGIN).total_seconds() / 3600


def load_ref(d, lo):
    """Reference corpus from embed_reference.py, times shifted like the actions."""
    ra = pd.read_parquet(os.path.join(d, 'ref_actions.parquet'))
    ids = json.load(open(os.path.join(d, 'emb_ref_actions_ids.json')))
    assert ids == ra.action_id.tolist()
    return dict(actions=ra.assign(t=ra.t - lo), E=np.load(os.path.join(d, 'emb_ref_actions.npy')),
                thread_ids=json.load(open(os.path.join(d, 'emb_threads_extra_ids.json'))),
                Q_thread=np.load(os.path.join(d, 'emb_threads_extra.npy')),
                sub_ids=json.load(open(os.path.join(d, 'emb_submolts_ids.json'))),
                Q_sub=np.load(os.path.join(d, 'emb_submolts.npy')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scope', default='platform_jan31_feb17')
    ap.add_argument('--t0', default='2026-02-06T00:00')
    ap.add_argument('--t-star', default='2026-02-10T00:00')
    ap.add_argument('--m', type=int, default=12)
    ap.add_argument('--alpha', type=float, default=0.3)
    ap.add_argument('--burst-rate', type=float, default=0.25)
    ap.add_argument('--no-plant', action='store_true')
    ap.add_argument('--B', type=int, default=99)
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--max-sets', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--d', type=int, default=None, help='embedding dimension (default: elbow rule)')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--gate', default='combined', choices=['combined', 'coordinated', 'shift', 'none'])
    ap.add_argument('--content', default='ridge', choices=['ridge', 'pkps'])
    ap.add_argument('--pkps-k', type=int, default=64)
    ap.add_argument('--pkps-bw', type=float, default=1.0)
    ap.add_argument('--pkps-time', default='all', choices=['all', 'period'])
    a = ap.parse_args()
    ctag = '' if a.content == 'ridge' else (
        f'_pkps{"" if a.pkps_time == "all" else "T"}'
        + ('' if a.pkps_bw == 1.0 else f'_bw{a.pkps_bw}') + ('' if a.pkps_k == 64 else f'_k{a.pkps_k}'))
    tag = a.tag or ((f'{a.scope}_real' if a.no_plant else f'{a.scope}_m{a.m}_a{a.alpha}_s{a.seed}')
                    + (f'_d{a.d}' if a.d else '') + ('' if a.gate == 'combined' else f'_{a.gate}') + ctag)
    T0 = time.time()

    d = os.path.join(DERIVED, a.scope)
    meta = json.load(open(os.path.join(d, 'meta.json')))
    lo, hi = meta['window_hours']
    acts = pd.read_parquet(os.path.join(d, 'actions.parquet'))
    ctx = pd.read_parquet(os.path.join(d, 'contexts.parquet'))
    ev = pd.read_parquet(os.path.join(d, 'score_events.parquet'))
    acts = acts.assign(t=acts.t - lo)
    ctx = ctx.assign(t=ctx.t - lo)
    ev = ev.assign(t=ev.t - lo)
    E = np.load(os.path.join(d, 'emb_actions.npy'))
    C = np.load(os.path.join(d, 'emb_contexts.npy'))
    data = dict(actions=acts, contexts=ctx, score_events=ev,
                emb=dict(action_ids=json.load(open(os.path.join(d, 'emb_actions_ids.json'))), E=E,
                         thread_ids=json.load(open(os.path.join(d, 'emb_contexts_ids.json'))), C=C),
                meta=dict(t0=hours(a.t0) - lo, T=hi - lo, source=meta['source']),
                system=meta['system'], donors=meta.get('donors') or [])
    if a.content == 'pkps':
        data['ref'] = load_ref(d, lo)
    print(f'[load] {len(meta["system"])} system agents, {len(E)} embedded actions, '
          f'{time.time() - T0:.0f}s', flush=True)

    truth = None
    if not a.no_plant:
        c = embed.embed([CAMPAIGN])[0]
        data, truth = plant(data, c, m=a.m, alpha=a.alpha, t_star=hours(a.t_star) - lo,
                            burst_rate=a.burst_rate, seed=a.seed)
        print(f'[plant] ring of {a.m}: {truth["n_hijacked"]} hijacked actions, '
              f'{truth["n_burst_comments"]} burst comments, {truth["n_ring_votes"]} hidden ring votes, '
              f'{truth["n_harm_events"]} displaced votes', flush=True)

    res = run_pipeline(data, PipeConfig(B=a.B, seed=a.seed, workers=a.workers,
                                        max_sets=a.max_sets, d=a.d, gate=a.gate,
                                        content=a.content, pkps_k=a.pkps_k, pkps_bw=a.pkps_bw,
                                        pkps_time=a.pkps_time))
    sysidx = {x: i for i, x in enumerate(res['system'])}

    out = dict(tag=tag, scope=a.scope, t0=a.t0, n_system=len(res['system']),
               n_eligible=int(res['eligible'].sum()),
               d=res['proj']['d'], separability=res['separability'],
               family_size=res['scan']['n_family'], Lambda=res['scan']['Lambda'],
               Lambda_null_q95=float(np.quantile(res['scan']['Lambda_null'], .95)),
               delta_hat=res['delta_T'], delta_ci=res['delta_ci'], flagged=[], pkps=res.get('pkps'),
               runtime_s=None)
    ring = set(truth['ring']) if truth else set()
    for f in res['flagged']:
        m = set(f['agents'])
        out['flagged'].append(dict(
            source=f['source'], size=f['size'], T=f['T'], zE=f['zE'], zC=f['zC'],
            comove=f['comove'],
            p_fw=f['p_fw'], onset_h=f['onset'], own_gain=f['own_gain'],
            own_gain_ci=f['own_gain_ci'], adversarial=f['adversarial'],
            time_null_p=f['time_null_p'], p_gate=f.get('p_gate'), p_shift=f.get('p_shift'),
            p_gain=f.get('p_gain'), p_coact=f.get('p_coact'),
            ring_overlap=len(m & ring), weak_labels=sorted(m & set(WEAK)),
            agents=sorted(m) if len(m) <= 40 else None))

    if truth:
        best = max(res['flagged'], key=lambda f: len(set(f['agents']) & ring), default=None)
        bm = set(best['agents']) if best else set()
        harm_Y = scores.trajectories(truth['harm_events'], res['system'], res['grid'])
        comp = res['comp']
        out['truth'] = dict(
            ring=truth['ring'], t_star_h=truth['t_star'], alpha=a.alpha, m=a.m,
            n_hijacked=truth['n_hijacked'], n_burst_comments=truth['n_burst_comments'],
            n_ring_votes=truth['n_ring_votes'], n_harm_events=truth['n_harm_events'],
            ring_eligible=int(sum(1 for x in truth['ring'] if res['eligible'][sysidx[x]])),
            best_set_source=best['source'] if best else None,
            precision=len(bm & ring) / len(bm) if bm else 0.0,
            recall=len(bm & ring) / len(ring),
            best_set_adversarial=bool(best['adversarial']) if best else False,
            onset_h=best['onset'] if best else None,
            delta_true_T=float(-harm_Y[comp].mean(0)[-1]),
        )
        # detection by the co-movement scan alone (before the adversarial gate)
        cand = res['scan']['candidates']
        jac = [len(set(r_['members']) & {sysidx[x] for x in ring}) /
               len(set(r_['members']) | {sysidx[x] for x in ring}) for r_ in cand]
        if cand:
            j = int(np.argmax(jac))
            out['truth'].update(scan_best_jaccard=float(jac[j]), scan_best_size=cand[j]['size'],
                                scan_best_p_fw=cand[j]['p_fw'],
                                scan_best_p_gate=cand[j].get('p_gate'),
                                n_scan_significant=len(cand))
        if res['did']:
            hz = -harm_Y[:, -1]
            out['truth']['did_true'] = float(hz[res['did']['exposed']].mean()
                                             - hz[res['did']['control']].mean())
            out['did'] = {k: v for k, v in res['did'].items() if k not in ('exposed', 'control')}
            print(f'[truth] DiD true={out["truth"]["did_true"]:+.3f} vs estimate='
                  f'{res["did"]["estimate"]:+.3f} CI{tuple(round(x, 3) for x in res["did"]["ci"])}')
        print(f'[truth] best flagged set: {out["truth"]["best_set_source"]} '
              f'precision={out["truth"]["precision"]:.2f} recall={out["truth"]["recall"]:.2f} '
              f'adversarial={out["truth"]["best_set_adversarial"]} onset={out["truth"]["onset_h"]} '
              f'(true {truth["t_star"]:.0f}); Delta true={out["truth"]["delta_true_T"]:+.3f} vs '
              + ', '.join(f'{k}={v:+.3f}' for k, v in res['delta_T'].items()))

    weak = {}
    for name, lab in WEAK.items():
        if name not in sysidx:
            weak[name] = 'not in system (no burn-in or no post-t0 activity)'
        elif not res['eligible'][sysidx[name]]:
            weak[name] = 'in system, not eligible after cap'
        else:
            hit = [f['source'] for f in res['flagged'] if name in f['agents']]
            weak[name] = f'flagged in {hit}' if hit else 'eligible, not flagged'
    out['weak_labels'] = weak
    print('[weak]', weak)
    out['runtime_s'] = time.time() - T0
    with open(os.path.join(RESULTS, f'results_{tag}.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    with open(os.path.join(d, f'res_{tag}.pkl'), 'wb') as fh:
        pickle.dump(dict(res=res, truth={k: v for k, v in (truth or {}).items()
                                         if k != 'harm_events'}), fh)
    draw(res, os.path.join(ART, f'figure2_{tag}.png'),
         truth_idx=[sysidx[x] for x in ring])
    print(f'[done] {out["runtime_s"]:.0f}s -> data/results/results_{tag}.json, artifacts/figure2_{tag}.png')


if __name__ == '__main__':
    main()
