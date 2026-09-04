"""Outcome-only baselines for the QUENCH figure: predictors of a system's true
full-500 resolve rate that see graded probe outcomes on the q20 panel but no
traces.

  count_lookup     : average true rate of reference systems that got the same
                     number of the m probes right ("average score by n correct").
  irt_2pl_random   : two-parameter IRT (per-item difficulty and slope) fit on
                     the references' panel outcomes; posterior over the target's
                     ability from its m probe outcomes; prediction = posterior
                     mean of a fitted ability -> full-500 score map. Probes random.
  irt_2pl_informative : same model, but the m probes are the items with the
                     highest Fisher information averaged over the reference
                     population (one fixed panel for everyone).
  irt_2pl_adaptive : same model, probes chosen one at a time per target to
                     maximise expected information under the current posterior.

All predictors use leave-one-LLM-out references (same rule as quench.py) and
are scored on all systems. Random-probe numbers average over draws (all 20
single tasks at m=1, all draws of the full panel at m=20).

Usage:  python scripts/outcome_baselines.py [--ms 1,2,3,5,10,20] [--draws 100]
                                             [--ridge-a 10] [--out figures/outcome_baselines.json]
"""
import argparse
import json
import os
import re

import numpy as np
from scipy.optimize import least_squares, minimize

GRID = np.linspace(-8, 8, 801)          # ability grid for the posterior


# ---------------------------------------------------------------- data ----
def load_panel(labels_path='data/leaderboard/verified_labels.json',
               judge_dir='data/judge/structured-qspec', panel=None):
    """Systems, panel tasks, true full-500 rates y, panel outcomes B (M x Q),
    and the leave-one-LLM-out reference mask `allowed` (M x M).
    `panel`: optional JSON file with {"instances": [...]} overriding the
    instance list derived from judge_dir."""
    labels = json.load(open(labels_path))
    systems = sorted(s for s in os.listdir(judge_dir) if 'resolved' in labels.get(s, {}))
    if panel:
        q20 = sorted(json.load(open(panel))['instances'])
    else:
        q20 = sorted(f[:-5] for f in os.listdir(os.path.join(judge_dir, systems[0])))
    y = np.array([len(labels[s]['resolved']) / 500 for s in systems])
    B = np.array([[q in set(labels[s]['resolved']) for q in q20] for s in systems], float)

    def llm_tag(s):
        m = re.search(r'^\s+model_display:\s*(.*)$', labels[s].get('metadata_yaml', ''), re.M)
        return m.group(1).strip().strip('"\'') if m else None
    tags = [llm_tag(s) for s in systems]
    allowed = np.array([[j != i and not (tags[i] and tags[j] == tags[i])
                         for j in range(len(systems))] for i in range(len(systems))])
    return systems, q20, y, B, allowed


# ------------------------------------------------- metadata baselines ----
SCAFFOLDS = ('sweagent', 'swe-agent', 'openhands', 'agentless', 'autocoderover',
             'moatless', 'composio', 'marscode', 'lingma', 'gru', 'masai',
             'codeact', 'tools', 'epam', 'solver', 'aime', 'blackbox',
             'devlo', 'emergent', 'nemotron', 'trae', 'refact', 'zencoder')
UNINFORMATIVE_MODEL_TAGS = {None, '', 'Multiple', 'Undisclosed', 'None'}


def scaffold_of(system):
    """Scaffold keyword from the submission name (same heuristic as pillars.py)."""
    low = system.lower()
    return next((h for h in SCAFFOLDS if h in low), None)


def group_mean_baseline(y, groups, exclude):
    """Predict each system as the mean true rate of the other systems in its
    group, restricted to references not excluded for it; fall back to the
    mean of its allowed references when the group is empty or untagged.
    Returns (MAE over all systems, MAE over covered systems only, MAE of the
    fallback on those same covered systems, number covered)."""
    preds, is_cov = [], []
    for i in range(len(y)):
        ok = exclude[i] & np.array([g is not None and g == groups[i] for g in groups])
        covered = groups[i] is not None and ok.any()
        preds.append(y[ok].mean() if covered else y[exclude[i]].mean()); is_cov.append(covered)
    preds, is_cov = np.array(preds), np.array(is_cov)
    fallback = np.array([y[exclude[i]].mean() for i in range(len(y))])
    return (float(np.abs(preds - y).mean()), float(np.abs(preds - y)[is_cov].mean()),
            float(np.abs(fallback - y)[is_cov].mean()), int(is_cov.sum()))


# ------------------------------------------------------- count lookup ----
def count_lookup(B, y, allowed, i, cols):
    """Mean true rate of allowed references with the same number correct on
    `cols` as system i; widen the tolerance until some reference matches."""
    k = B[i, cols].sum()
    ks = B[:, cols].sum(1)
    for tol in range(len(cols) + 1):
        ok = allowed[i] & (np.abs(ks - k) <= tol)
        if ok.any():
            return y[ok].mean()


# ----------------------------------------------------------- 2PL IRT ----
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def fit_2pl(B_ref, ridge=1.0, ridge_a=10.0):
    """MAP fit of P(i solves q) = sigmoid(a_q * (theta_i - b_q)) on the
    reference outcome matrix. Priors: N(0,1) on theta and b, N(0, 1/ridge_a) on
    log a (so slope 1, the Rasch value, is the prior centre).
    Returns theta (n,), b (Q,), a (Q,)."""
    n, Q = B_ref.shape

    def unpack(p):
        return p[:n], p[n:n + Q], np.exp(p[n + Q:])

    def neg_log_post(p):
        th, b, a = unpack(p)
        z = a[None] * (th[:, None] - b[None])
        ll = (B_ref * z - np.logaddexp(0, z)).sum()
        return -ll + 0.5 * ridge * (th @ th + b @ b) + 0.5 * ridge_a * (np.log(a) @ np.log(a))

    def grad(p):
        th, b, a = unpack(p)
        z = a[None] * (th[:, None] - b[None])
        R = sigmoid(z) - B_ref                                  # d(-ll)/dz
        return np.concatenate([(R * a[None]).sum(1) + ridge * th,
                               -(R * a[None]).sum(0) + ridge * b,
                               (R * z).sum(0) + ridge_a * np.log(a)])

    res = minimize(neg_log_post, np.zeros(n + 2 * Q), jac=grad, method='L-BFGS-B')
    return unpack(res.x)


class ItemModel:
    """A fitted 2PL model for one target's reference pool, plus the map from
    ability to full-500 score and the population prior on ability."""

    def __init__(self, B_ref, y_ref, ridge_a):
        self.theta, self.b, self.a = fit_2pl(B_ref, ridge_a=ridge_a)
        self.mu, self.sd = self.theta.mean(), self.theta.std()
        # score map: y ~ sigmoid(c1 * theta + c0), least squares on references
        c = least_squares(lambda p: sigmoid(p[0] * self.theta + p[1]) - y_ref, [1.0, 0.0]).x
        self.score_of_theta = sigmoid(c[0] * GRID + c[1])
        # item response curves on the grid, P[grid, item]
        self.P_grid = sigmoid(self.a[None] * (GRID[:, None] - self.b[None]))

    def posterior(self, cols, outcomes, mu=None, sd=None):
        """Posterior weights over GRID after observing `outcomes` on `cols`.
        The prior is the population normal unless (mu, sd) is given."""
        mu = self.mu if mu is None else mu
        sd = self.sd if sd is None else sd
        logp = -0.5 * ((GRID - mu) / sd) ** 2
        for q, o in zip(cols, outcomes):
            p = self.P_grid[:, q]
            logp += np.log(p if o else 1 - p)
        w = np.exp(logp - logp.max())
        return w / w.sum()

    def predict(self, cols, outcomes, mu=None, sd=None):
        return float(self.posterior(cols, outcomes, mu, sd) @ self.score_of_theta)

    def item_information(self, weights):
        """Expected Fisher information per item under ability weights on GRID."""
        return weights @ (self.a[None] ** 2 * self.P_grid * (1 - self.P_grid))

    def informative_order(self):
        """Items ranked by information averaged over the reference abilities."""
        P = sigmoid(self.a[None] * (self.theta[:, None] - self.b[None]))
        return np.argsort(-(self.a[None] ** 2 * P * (1 - P)).mean(0))

    def adaptive_path(self, outcomes_all):
        """Greedy adaptive testing: at each step pick the unused item with the
        highest expected information under the current posterior, observe it
        (from the target's full outcome row), and continue. Returns the item
        order and the prediction after each step."""
        used, preds = [], []
        for _ in range(len(self.b)):
            info = self.item_information(self.posterior(used, outcomes_all[used]))
            info[used] = -np.inf
            used.append(int(np.argmax(info)))
            preds.append(self.predict(used, outcomes_all[used]))
        return used, preds


# --------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ms', default='1,2,3,5,10,20')
    ap.add_argument('--draws', type=int, default=100, help='random probe draws per m (1 < m < 20)')
    ap.add_argument('--ridge-a', type=float, default=10.0, help='2PL prior precision on log-slope')
    ap.add_argument('--out', default='figures/outcome_baselines.json')
    ap.add_argument('--panel', default=None)
    args = ap.parse_args()

    systems, q20, y, B, allowed = load_panel(panel=args.panel)
    M, Q = B.shape
    ms = [int(x) for x in args.ms.split(',')]
    print(f'{M} systems x {Q} panel tasks; fitting {M} leave-one-LLM-out 2PL models')
    models = [ItemModel(B[allowed[i]], y[allowed[i]], args.ridge_a) for i in range(M)]

    rng = np.random.default_rng(0)
    def draws_for(m):
        if m == Q:
            return [np.arange(Q)]
        if m == 1:
            return [np.array([q]) for q in range(Q)]
        return [rng.choice(Q, m, replace=False) for _ in range(args.draws)]

    # adaptive paths and informative orders are per target, independent of m
    adaptive = [models[i].adaptive_path(B[i]) for i in range(M)]
    info_order = [models[i].informative_order() for i in range(M)]

    labels = json.load(open('data/leaderboard/verified_labels.json'))
    model_tags = [None if (t := re.search(r'^\s+model_display:\s*(.*)$', labels[s].get('metadata_yaml', ''), re.M))
                  is None or t.group(1).strip().strip('"\'') in UNINFORMATIVE_MODEL_TAGS
                  else t.group(1).strip().strip('"\'') for s in systems]
    scaffolds = [scaffold_of(s) for s in systems]
    loo = ~np.eye(M, dtype=bool)                                     # exclude only the target itself
    scaf = group_mean_baseline(y, scaffolds, allowed)    # figure's rule: LLM siblings excluded
    modl = group_mean_baseline(y, model_tags, loo)       # needs LLM siblings: leave-one-out only
    for name, (mae, mae_cov, mae_fb, n) in (('same-scaffold mean (LLM-out)', scaf), ('same-model mean (leave-one-out)', modl)):
        print(f'{name}: MAE {mae:.4f} over all {M} (fallback = mean of references); '
              f'on the {n} covered systems {mae_cov:.4f} vs {mae_fb:.4f} for the fallback')

    out = {'m': ms, 'ridge_a': args.ridge_a, 'draws': args.draws,
           'constant_mean_mae': float(np.abs(y.mean() - y).mean()),
           'same_scaffold_mean': dict(zip(('mae', 'mae_covered', 'fallback_mae_covered', 'n_covered'), scaf)),
           'same_model_mean': dict(zip(('mae', 'mae_covered', 'fallback_mae_covered', 'n_covered'), modl)),
           'count_lookup': [], 'irt_2pl_random': [], 'irt_2pl_informative': [], 'irt_2pl_adaptive': []}
    print(f"{'m':>3} {'count lookup':>13} {'2PL random':>11} {'2PL informative':>16} {'2PL adaptive':>13}")
    for m in ms:
        e_count, e_rand = [], []
        for cols in draws_for(m):
            e_count.append(np.mean([abs(count_lookup(B, y, allowed, i, cols) - y[i]) for i in range(M)]))
            e_rand.append(np.mean([abs(models[i].predict(cols, B[i, cols]) - y[i]) for i in range(M)]))
        e_info = np.mean([abs(models[i].predict(info_order[i][:m], B[i, info_order[i][:m]]) - y[i]) for i in range(M)])
        e_adapt = np.mean([abs(adaptive[i][1][m - 1] - y[i]) for i in range(M)])
        for key, v in (('count_lookup', np.mean(e_count)), ('irt_2pl_random', np.mean(e_rand)),
                       ('irt_2pl_informative', e_info), ('irt_2pl_adaptive', e_adapt)):
            out[key].append(float(v))
        print(f'{m:3d} {np.mean(e_count):13.4f} {np.mean(e_rand):11.4f} {e_info:16.4f} {e_adapt:13.4f}')

    out['informative_panel_example'] = [q20[q] for q in info_order[0][:10]]
    json.dump(out, open(args.out, 'w'), indent=1)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
