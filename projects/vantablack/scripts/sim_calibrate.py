"""Moltbook summary statistics that set the simulation's nuisance and group parameters.

    python sim_calibrate.py [--tag 01-30_02-09_T8_ms6]

Reads the residual increments of a real_tstep.py run (rebuilt as real_investigate.py does) and
reports, as medians over steps, directions or groups:
  - missing:  share of agent-steps with no responses (inactive), and of split increments that are
              missing because a half is empty
  - static:   share of observed split increments shorter than min_move x the median (templated
              agents posting identical content), and how they spread over agents
  - drift:    per step, |mean increment over agents| / median |increment|
  - factors:  the directions the factor rule removes, by kind (pervasive / two-signed): count per
              increment dimension, strength sv / MP edge, support (share of agents with clear
              loadings), minority-side share, loading excess kurtosis, and the share of the
              direction's energy in its largest step
  - groups:   for the cohort-wide persistent-pair components (size >= 3): median pair cosine of
              the members' residual increment series and the share of the mean pair cosine in the
              largest step
Only these summary numbers leave this script (data/results/sim_calibration_<tag>.json); no agent
identifiers are written.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import binom, kurtosis

from pipeline import tstep
from pipeline.paths import DERIVED, RESULTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='01-30_02-09_T8_ms6')
    a = ap.parse_args()
    summ = json.load(open(os.path.join(RESULTS, f'real_tstep_{a.tag}.json')))
    d = os.path.join(DERIVED, 'real_tstep')
    R = np.load(os.path.join(d, f'{a.tag}_R.npy')).astype(np.float32)
    rows = pd.read_parquet(os.path.join(d, f'{a.tag}_rows.parquet'), columns=['agent', 'agent_ix', 'step'])
    names = np.array(sorted(rows.agent.unique()))
    N, T = len(names), summ['T']
    pos = tstep.positions(R, rows.agent_ix.values, rows.step.values, N, T, seed=summ['seed'])
    L = T - 1
    out = dict(tag=a.tag, N=N, T=T, d=pos['d'])

    # missing and static
    act = (pos['nA'] + pos['nB']) > 0
    has = (pos['nA'][:, 1:] > 0) & (pos['nB'][:, :-1] > 0)
    D0 = pos['ZA'][:, 1:] - pos['ZB'][:, :-1]
    r = np.linalg.norm(D0, axis=2)
    med = np.median(r[has])
    static = has & (r < summ['min_move'] * med)
    out.update(inactive_share=float(1 - act.mean()), missing_increment_share=float(1 - has.mean()),
               static_share=float(static.sum() / has.sum()),
               static_agents_share=float(static.any(1).mean()),
               static_per_static_agent_median=float(np.median(static.sum(1)[static.any(1)])))

    # drift, before centering
    ok = has & ~static
    mu = (D0 * ok[:, :, None]).sum(0) / ok.sum(0)[:, None]
    drift = np.linalg.norm(mu, axis=1) / np.median(r[ok])
    out.update(drift_ratio_per_step=[round(float(x), 3) for x in drift], drift_ratio_median=float(np.median(drift)))

    # factors, as the rule sees them
    D = tstep.increments(pos, weight=summ.get('weight'), min_move=summ['min_move'])
    F = D.reshape(N, -1)
    Uf, sv, Vt = np.linalg.svd(F, full_matrices=False)
    _, kf = tstep.remove_factors(D, two_signed_alpha=summ['two_signed_alpha'])
    P = F.shape[1]
    kmax = 0
    for _ in range(20):                                               # the MP edge, as remove_factors finds it
        sig2 = (sv[kmax:] ** 2).sum() / ((N - kmax) * (P - kmax))
        edge = np.sqrt(1.05 * sig2) * (np.sqrt(N) + np.sqrt(P))
        k_new = int((sv > edge).sum())
        if k_new == kmax:
            break
        kmax = k_new
    cmax = binom.ppf(1 - summ['two_signed_alpha'], N, 0.02275)
    dirs = []
    for j in range(kf):
        floor = np.sqrt(min(1.0, (edge / sv[j]) ** 2) / N)
        u = Uf[:, j]
        cp, cm = int((u > 2 * floor).sum()), int((u < -2 * floor).sum())
        supp = ((np.abs(u) > 2 * floor).sum() - 0.0455 * N) / N
        kind = 'pervasive' if supp >= 1 / 3 else 'two_signed'
        e = (Vt[j].reshape(L, -1) ** 2).sum(1)
        dirs.append(dict(kind=kind, sv_over_edge=float(sv[j] / edge), support=float(supp),
                         clear_share=float((cp + cm) / N), minority_share=float(min(cp, cm) / max(cp + cm, 1)),
                         excess_kurtosis=float(kurtosis(u)), max_step_share=float(e.max() / e.sum()),
                         var_share=float(sv[j] ** 2 / (sv ** 2).sum())))
    dd = pd.DataFrame(dirs)
    out['factors'] = dict(P=P, n_above_edge=kmax, n_removed=kf, edge=float(edge), two_signed_cutoff=float(cmax),
                          by_kind={k: dict(n=int(len(g)), per_dim=float(len(g) / P),
                                           **{c: float(g[c].median()) for c in g.columns if c != 'kind'},
                                           var_share_total=float(g.var_share.sum()))
                                   for k, g in dd.groupby('kind')},
                          directions=dd.round(4).to_dict('records'))

    # groups: cohort-wide persistent-pair components
    Dr = tstep.remove_factors(D, two_signed_alpha=summ['two_signed_alpha'])[0]
    Uu = tstep._units(Dr)
    idx = {n: i for i, n in enumerate(names)}
    grp = []
    for comp in summ['cohort_components']:
        if len(comp) < 3:
            continue
        m = np.array([idx[x] for x in comp])
        G = Uu[m] @ Uu[m].T
        iu = np.triu_indices(len(m), 1)
        nrm = np.linalg.norm(Dr[m], axis=(1, 2))[:, None]
        step_c = np.array([((Dr[m, s] / nrm) @ (Dr[m, s] / nrm).T)[iu].mean() for s in range(L)])
        pc = np.clip(step_c, 0, None)
        grp.append(dict(size=len(m), pair_cos_median=float(np.median(G[iu])),
                        max_step_share=float(pc.max() / pc.sum())))
    gg = pd.DataFrame(grp)
    out['groups'] = dict(n=len(gg), size_median=float(gg['size'].median()),
                         pair_cos_median=float(gg.pair_cos_median.median()),
                         max_step_share_median=float(gg.max_step_share.median()),
                         per_group=gg.round(3).to_dict('records'))
    path = os.path.join(RESULTS, f'sim_calibration_{a.tag}.json')
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != 'factors'}, indent=1))
    print(json.dumps({k: v for k, v in out['factors'].items() if k != 'directions'}, indent=1))
    print(dd.round(3).to_string())


if __name__ == '__main__':
    main()
