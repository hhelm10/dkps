"""16-field sensitivity, part A (+ frozen part-B manifest).

Ordering discipline (per proposal): step 0 writes the part-B manifest --
seeded 80/20 model-group split, five inner folds, the candidate mask
library, selection and evaluation probe banks -- BEFORE any part-A MAE
is computed. Part B consumes the manifest later; revisiting it after
seeing part-A results is not allowed.

Part A protocol (fixed pipeline, no tuning):
  reference pool per target = all systems minus the target's
  model_display group; per-(query, field) median centering fit on the
  POOL only, L2; D^2 averaged over selected fields S and probes P;
  inverse-distance 3-NN. r in {1,2,4,6,8,12,16}; <=1000 uniform subsets
  per r (enumerated when fewer); shared masks and probe draws across
  arms; m in {1,2,3,5,10,20} (singletons at m=1, full panel at m=20,
  else 100 fixed draws).

Writes data/judge/rubric16/manifest.json,
       figures/rubric16_sensitivity.json (+ per-subset table npz),
       figures/fig_sensitivity.png.
"""
import itertools
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
from rubric16 import FIELDS, SECTIONS, panel  # noqa: E402

RS = (1, 2, 4, 6, 8, 12, 16)
MSA = (1, 2, 3, 5, 10, 20)
N_SUB = 1000
N_DRAW = 100
MANIFEST = 'data/judge/rubric16/manifest.json'
OUT_JSON = 'figures/rubric16_sensitivity.json'
OUT_NPZ = 'figures/rubric16_subset_table.npz'
OUT_PNG = 'figures/fig_sensitivity.png'
F = len(SECTIONS)
ORIG6 = tuple(range(6))


def llm_groups(labels, systems):
    tags = []
    for s in systems:
        m = re.search(r'^\s+model_display:\s*(.*)$',
                      labels[s].get('metadata_yaml', ''), re.M)
        tags.append(m.group(1).strip() if m else f'__solo__{s}')
    return tags


def build_manifest(labels, systems, q20):
    rng = np.random.default_rng(160)
    tags = llm_groups(labels, systems)
    groups = sorted(set(tags))
    # part B: 80/20 group split + five inner folds (frozen now, used later)
    perm = list(rng.permutation(groups))
    n_test = max(1, int(round(len(groups) * .2)))
    test_groups = sorted(perm[:n_test])
    dev_groups = sorted(perm[n_test:])
    folds = [[] for _ in range(5)]
    for i, g in enumerate(rng.permutation(dev_groups)):
        folds[i % 5].append(str(g))
    # candidate mask library (shared by A and B) and probe banks
    masks = {}
    for r in RS:
        allsub = list(itertools.combinations(range(F), r))
        if len(allsub) <= N_SUB:
            sel = allsub
        else:
            idx = rng.choice(len(allsub), N_SUB, replace=False)
            sel = [allsub[i] for i in sorted(idx)]
        masks[str(r)] = [list(s) for s in sel]
    draws = {}
    for m in MSA:
        if m == 1:
            draws[str(m)] = [[q] for q in range(20)]
        elif m == 20:
            draws[str(m)] = [list(range(20))]
        else:
            draws[str(m)] = [sorted(rng.choice(20, m, replace=False)
                                    .tolist()) for _ in range(N_DRAW)]
    eval_draws = {}
    rng2 = np.random.default_rng(161)
    for m in MSA:
        if m == 1:
            eval_draws[str(m)] = [[q] for q in range(20)]
        elif m == 20:
            eval_draws[str(m)] = [list(range(20))]
        else:
            eval_draws[str(m)] = [sorted(rng2.choice(20, m, replace=False)
                                         .tolist()) for _ in range(N_DRAW)]
    man = dict(seed=160, fields=list(SECTIONS),
               field_defs=FIELDS, systems=systems, q20=q20,
               groups={g: [s for s, t in zip(systems, tags) if t == g]
                       for g in groups},
               test_groups=[str(g) for g in test_groups],
               dev_groups=[str(g) for g in dev_groups],
               dev_folds=folds, masks=masks,
               selection_draws=draws, eval_draws=eval_draws,
               primary=dict(m=5, r=6))
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    json.dump(man, open(MANIFEST, 'w'))
    print(f'manifest frozen: {len(groups)} groups '
          f'({len(test_groups)} held out), masks per r: '
          f'{ {r: len(v) for r, v in masks.items()} }')
    return man


def d2_blocks(arm, systems, q20, tags):
    """(Q, F, M, M) float32: row i centered under target i's pool
    (everyone outside i's model_display group)."""
    M, Q = len(systems), len(q20)
    X = np.load(f'data/judge/rubric16_emb_{arm}.npz')['X'] \
        .reshape(M, Q, F, -1)
    tag_arr = np.array(tags)
    D2 = np.zeros((Q, F, M, M), np.float32)
    for g in sorted(set(tags)):
        tgt = np.where(tag_arr == g)[0]
        pool = np.where(tag_arr != g)[0]
        med = np.median(X[pool], axis=0, keepdims=True)      # (1,Q,F,d)
        Z = X - med
        Z = Z / np.maximum(np.linalg.norm(Z, axis=-1, keepdims=True), 1e-9)
        for q in range(Q):
            for f in range(F):
                A = Z[tgt, q, f]                             # (|g|, d)
                Bm = Z[:, q, f]                              # (M, d)
                D2[q, f, tgt] = ((A ** 2).sum(1)[:, None]
                                 + (Bm ** 2).sum(1)[None]
                                 - 2 * A @ Bm.T)
    return D2


def part_a(man, arm):
    systems, q20 = man['systems'], man['q20']
    labels, _, _ = panel()
    tags = llm_groups(labels, systems)
    tag_arr = np.array(tags)
    M = len(systems)
    y = np.array([len(labels[s]['resolved']) / 500 for s in systems])
    D2 = d2_blocks(arm, systems, q20, tags)
    allowed = tag_arr[:, None] != tag_arr[None, :]

    def knn_mae(D):
        Dm = np.where(allowed, D, np.inf)
        np.fill_diagonal(Dm, np.inf)
        nn = np.argpartition(Dm, 3, axis=1)[:, :3]
        dd = np.take_along_axis(Dm, nn, 1)
        w = 1 / (np.sqrt(np.maximum(dd, 0)) + 1e-9)
        pred = (w * y[nn]).sum(1) / w.sum(1)
        return float(np.abs(pred - y).mean())

    results = {}
    table = {}
    controls = {'orig6': list(ORIG6), 'all16': list(range(F))}
    for m in MSA:
        drs = man['selection_draws'][str(m)]
        Pmats = [np.asarray(P) for P in drs]
        for r in RS:
            subs = [tuple(s) for s in man['masks'][str(r)]]
            maes = np.zeros(len(subs))
            for si, S in enumerate(subs):
                Dq = D2[:, S].mean(1)                       # (Q, M, M)
                vals = [knn_mae(Dq[P].mean(0)) for P in Pmats]
                maes[si] = np.mean(vals)
            table[(m, r)] = maes
            results.setdefault(str(m), {})[str(r)] = dict(
                mean=float(maes.mean()), median=float(np.median(maes)),
                std=float(maes.std(ddof=1)) if len(maes) > 1 else 0.0,
                p10=float(np.percentile(maes, 10)),
                p90=float(np.percentile(maes, 90)),
                min=float(maes.min()), max=float(maes.max()),
                n=len(subs))
        for cname, S in controls.items():
            Dq = D2[:, S].mean(1)
            vals = [knn_mae(Dq[np.asarray(P)].mean(0)) for P in Pmats]
            results[str(m)][cname] = float(np.mean(vals))
        print(arm, f'm={m}', {r: round(results[str(m)][str(r)]['mean'], 4)
                              for r in RS},
              'orig6', round(results[str(m)]['orig6'], 4),
              'all16', round(results[str(m)]['all16'], 4))
    return results, table


def render(all_res):
    import matplotlib
    matplotlib.use('Agg')
    import hv_style
    hv_style.apply()
    import matplotlib.pyplot as plt

    SZ = hv_style.SIZES
    m0 = 5
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9), sharey=True)
    for ax, arm, nice in zip(axes, ('generic', 'qspec'),
                             ('Generic rubric', 'qubric')):
        res = all_res[arm][str(m0)]
        rs = np.array(RS)
        mean = [res[str(r)]['mean'] for r in RS]
        p10 = [res[str(r)]['p10'] for r in RS]
        p90 = [res[str(r)]['p90'] for r in RS]
        mn = [res[str(r)]['min'] for r in RS]
        mx = [res[str(r)]['max'] for r in RS]
        st = hv_style.ROLES['focus' if arm == 'qspec' else 'comparator']
        ax.fill_between(rs, mn, mx, color=st['color'], alpha=.10, lw=0,
                        label='min–max over subsets')
        ax.fill_between(rs, p10, p90, color=st['color'], alpha=.22, lw=0,
                        label='10th–90th pct')
        ax.plot(rs, mean, color=st['color'], lw=2.6, marker='o', ms=4,
                label='mean subset')
        ax.scatter([6], [res['orig6']], marker='D', s=70,
                   color=hv_style.ROLES['anchor']['color'], zorder=5,
                   label='original six fields')
        ax.scatter([16], [res['all16']], marker='s', s=70,
                   color=hv_style.INK_MUTE, zorder=5, label='all 16 fields')
        ax.set_title(nice, fontsize=SZ['label'], color=hv_style.INK_TITLE)
        ax.set_xlabel('number of rubric fields $r$', fontsize=SZ['subtitle'])
        ax.set_xticks(RS)
        ax.tick_params(labelsize=SZ['tick'])
    axes[0].set_ylabel(f'MAE$(\\hat{{y}}, y)$ at $m={m0}$',
                       fontsize=SZ['label'])
    axes[0].legend(fontsize=SZ['legend'] - 2, handlelength=2.4,
                   loc='upper right')
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200, bbox_inches='tight', pad_inches=0.03)
    print('wrote', OUT_PNG)


def main():
    labels, systems, q20 = panel()
    if os.path.exists(MANIFEST):
        man = json.load(open(MANIFEST))
    else:
        man = build_manifest(labels, systems, q20)
    all_res = {}
    tables = {}
    for arm in ('generic', 'qspec'):
        all_res[arm], tables[arm] = part_a(man, arm)
    json.dump(all_res, open(OUT_JSON, 'w'), indent=1)
    np.savez_compressed(
        OUT_NPZ, **{f'{arm}_m{m}_r{r}': tables[arm][(m, r)]
                    for arm in tables for (m, r) in tables[arm]})
    print('wrote', OUT_JSON, OUT_NPZ)
    render(all_res)


if __name__ == '__main__':
    main()
