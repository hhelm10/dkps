"""IRT + DKPS blend on identical informative probes (collaborator round 1).

Per target: 2PL fit on leave-one-LLM-out references (outcome_baselines.py,
bkj), probes = that model's informative order; qubric-geometry kNN on the
same probes; honest per-target alpha chosen on references. The blend beats
IRT alone at every m -- trace content carries information beyond graded
outcomes -- and sets the current best-known QUENCH numbers (m=1 .0905).

Usage: python scripts/irt_dkps_blend.py   -> figures/irt_dkps_blend.json
"""
import json
import sys

import numpy as np
from scipy.spatial.distance import cdist

sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
from outcome_baselines import ItemModel, load_panel  # noqa: E402
from dkps.traces.qubric import consensus_center  # noqa: E402


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', default=None)
    ap.add_argument('--emb', default='data/judge/quench_emb_structured-qspec_'
                                     'nomic-ai_nomic-embed-text-v1.5.npz')
    ap.add_argument('--out', default='figures/irt_dkps_blend.json')
    ap.add_argument('--global-alpha', action='store_true',
                    help='pool the alpha selection across targets (shrinkage)')
    args = ap.parse_args()
    systems, q20, y, B, allowed = load_panel(panel=args.panel)
    M, Q = B.shape
    models = [ItemModel(B[allowed[i]], y[allowed[i]], 10.0) for i in range(M)]
    info_order = [m.informative_order() for m in models]
    X = np.load(args.emb)['X']
    Xc = consensus_center(X, np.tile(np.arange(Q), M)).reshape(M, Q, -1)
    alphas = np.linspace(0, 1, 11)

    res = {}
    for m in (1, 2, 3, 5, 10, 20):
        e_irt, e_geo, e_bl = [], [], []
        for i in range(M):
            cols = info_order[i][:m]
            Z = Xc[:, cols, :].reshape(M, -1)
            D = cdist(Z, Z)
            np.fill_diagonal(D, np.inf)
            refs = np.where(allowed[i])[0]
            Dref = D[:, refs]
            nn = np.argsort(Dref, axis=1)[:, :3]
            w = 1 / (np.take_along_axis(Dref, nn, 1) + 1e-12)
            geo_all = (w * y[refs][nn]).sum(1) / w.sum(1)
            irt_all = np.array([models[i].predict(cols, B[j, cols])
                                for j in range(M)])
            E = np.abs(alphas[None] * irt_all[refs, None]
                       + (1 - alphas[None]) * geo_all[refs, None]
                       - y[refs, None]).mean(0)
            a = alphas[int(E.argmin())]
            e_irt.append(abs(irt_all[i] - y[i]))
            e_geo.append(abs(geo_all[i] - y[i]))
            e_bl.append(abs(a * irt_all[i] + (1 - a) * geo_all[i] - y[i]))
        res[m] = dict(irt=float(np.mean(e_irt)), geom=float(np.mean(e_geo)),
                      blend=float(np.mean(e_bl)))
        print(f'{m:2d}  irt {res[m]["irt"]:.4f}  geom {res[m]["geom"]:.4f}  '
              f'blend {res[m]["blend"]:.4f}')
    json.dump(res, open(args.out, 'w'), indent=2)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
