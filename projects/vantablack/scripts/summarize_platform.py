"""Collect platform-run results into writing/notes/platform_results.md.

    python summarize_platform.py [--scope platform_jan31_feb17] [--d 100]
"""
import argparse
import glob
import json
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
from pipeline.paths import RESULTS, NOTES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scope', default='platform_jan31_feb17')
    ap.add_argument('--d', type=int, default=100)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(RESULTS, f'results_{a.scope}_m*_d{a.d}.json')))
    rows = []
    for f in files:
        r = json.load(open(f))
        t = r['truth']
        adv = [x for x in r['flagged'] if x['adversarial']]
        rows.append(dict(alpha=t['alpha'], seed=int(r['tag'].split('_s')[-1].split('_')[0]),
                         prec=t['precision'], rec=t['recall'],
                         adv_ok=t['best_set_adversarial'] and t['recall'] > 0,
                         scan_j=t.get('scan_best_jaccard'), scan_p=t.get('scan_best_p_fw'),
                         onset=t['onset_h'], t_star=t['t_star_h'], n_flag=len(r['flagged']),
                         n_adv=len(adv), n_adv_other=sum(1 for x in adv if x['ring_overlap'] == 0),
                         n_comove=None, d_true=t['delta_true_T'],
                         d_ext=r['delta_hat'].get('extrapolate'), d_syn=r['delta_hat'].get('synth'),
                         did_true=t.get('did_true'), did=(r.get('did') or {}).get('estimate')))
    real = os.path.join(RESULTS, f'results_{a.scope}_real_d{a.d}.json')
    out = [f'# Platform runs: {a.scope}, d = {a.d}', '',
           'Planted ring of 12 real agents, onset Feb 10; 199 rotation-null draws; '
           'system = all agents with at least 3 actions on each side of Feb 6 (5,241).', '',
           '| content shift | ring | scan-only best Jaccard | precision | recall | labelled adversarial | onset (h, true) | '
           'flagged sets | other sets labelled adversarial | true Δ | Δ̂ extrapolate | Δ̂ synth |',
           '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for x in sorted(rows, key=lambda x: (-x['alpha'], x['seed'])):
        out.append(f"| {x['alpha']} | {x['seed']} | {x['scan_j']:.2f} | {x['prec']:.2f} | {x['rec']:.2f} | "
                   f"{'yes' if x['adv_ok'] else 'no'} | {x['onset'] if x['rec'] > 0 else '-'} ({x['t_star']:.0f}) | {x['n_flag']} | "
                   f"{x['n_adv_other']} | {x['d_true']:+.2f} | {x['d_ext']:+.1f} | {x['d_syn']:+.1f} |")
    out.append('')
    for al in sorted(set(x['alpha'] for x in rows), reverse=True):
        g = [x for x in rows if x['alpha'] == al]
        hit = [x['rec'] >= 0.5 and x['prec'] >= 0.5 and x['adv_ok'] for x in g]
        out.append(f'- Content shift {al}: ring recovered and labelled adversarial in '
                   f'{sum(hit)}/{len(g)} runs; mean precision {np.mean([x["prec"] for x in g]):.2f}, '
                   f'mean recall {np.mean([x["rec"] for x in g]):.2f}.')
    if os.path.exists(real):
        r = json.load(open(real))
        out += ['', '## Real stream, nothing planted', '',
                f'{len(r["flagged"])} sets flagged; '
                f'{sum(1 for x in r["flagged"] if x["adversarial"])} labelled adversarial.', '',
                '| source | size | p_fw | shift z | co-movement z | own gain (95% interval) | adversarial |',
                '|---|---|---|---|---|---|---|']
        for x in r['flagged']:
            out.append(f"| {x['source']} | {x['size']} | {x['p_fw']:.3f} | {x['zE']:.1f} | {x['zC']:.1f} | "
                       f"{x['own_gain']:+.0f} ({x['own_gain_ci'][0]:+.0f}, {x['own_gain_ci'][1]:+.0f}) | "
                       f"{'yes' if x['adversarial'] else 'no'} |")
    coord = os.path.join(RESULTS, f'results_{a.scope}_real_d{a.d}_coordinated.json')
    if os.path.exists(coord):
        r = json.load(open(coord))
        out += ['', '## Real stream: coordinated groups (nothing planted)', '',
                'Gate: the co-movement scan, plus co-activity (same thread within 10 minutes) '
                'against activity-matched random sets, at false discovery rate 0.05 '
                '(Benjamini-Hochberg) across the scan-significant sets. No score gain required.', '',
                f'{len(r["flagged"])} disjoint coordinated groups.', '',
                '| source | size | co-movement (mean cos) | co-activity p | FDR-adjusted p | shift p | '
                'own gain (95% interval) | members (first 8) |',
                '|---|---|---|---|---|---|---|---|']
        for x in r['flagged']:
            names = ', '.join((x['agents'] or [])[:8]) + (' …' if x['agents'] and len(x['agents']) > 8 else '')
            out.append(f"| {x['source']} | {x['size']} | {x.get('comove', float('nan')):.2f} | "
                       f"{x['p_coact']:.1e} | {x['p_gate']:.3f} | {x['p_shift']:.3f} | "
                       f"{x['own_gain']:+.0f} ({x['own_gain_ci'][0]:+.0f}, {x['own_gain_ci'][1]:+.0f}) | {names} |")
    open(os.path.join(NOTES, 'platform_results.md'), 'w').write('\n'.join(out) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
