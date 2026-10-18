"""Score trajectories, benign counterfactuals and Delta-hat (TODO section 5).

Y[n, g] is agent n's cumulative score at grid time g. For a set A,
sigma(A)(t) = mean_{n in A} Y[n, t]. Delta-hat for an estimated S is

    Delta-hat(t) = sigma~(comp)(t) - sigma(comp)(t),  comp = system minus S,

positive meaning the rest of the system scored lower than it would have
had S kept its burn-in behaviour (the one-pager's sign convention).

Counterfactual estimators:
  extrapolate  each agent's burn-in linear trend, extended past t0.
  synth        synthetic control: convex weights over donor agents (agents
               outside the analysed system, never exposed to S) fit to
               sigma(comp) on the burn-in grid, applied after t0.
"""
import numpy as np


def trajectories(score_events, agent_ids, grid):
    pos = {a: i for i, a in enumerate(agent_ids)}
    Y = np.zeros((len(agent_ids), len(grid)))
    ev = score_events[score_events.agent_id.isin(pos)]
    for a, grp in ev.groupby('agent_id'):
        t = np.sort(grp.t.values)
        v = grp.value.values[np.argsort(grp.t.values)]
        cs = np.concatenate([[0], np.cumsum(v)])
        Y[pos[a]] = cs[np.searchsorted(t, grid, side='right')]
    return Y


def extrapolate(Y, grid, t0):
    pre = grid <= t0
    X = np.vstack([np.ones(pre.sum()), grid[pre]]).T
    beta, *_ = np.linalg.lstsq(X, Y[:, pre].T, rcond=None)
    return (np.vstack([np.ones(len(grid)), grid]).T @ beta).T


def synth(y, Xd, grid, t0, iters=3000):
    """Convex weights (simplex) by projected gradient on the burn-in fit."""
    pre = grid <= t0
    A, b = Xd[:, pre].T, y[pre]
    w = np.full(len(Xd), 1.0 / len(Xd))
    L = np.linalg.norm(A, 2) ** 2 + 1e-12
    for _ in range(iters):
        w = _simplex(w - A.T @ (A @ w - b) / L)
    return w @ Xd, w


def _simplex(v):
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1
    r = np.flatnonzero(u - css / (np.arange(len(u)) + 1) > 0)[-1]
    return np.maximum(v - css[r] / (r + 1), 0)


def delta(Y_sys, comp_mask, grid, t0, Y_donor=None):
    comp = Y_sys[comp_mask]
    obs = comp.mean(0)
    out = {'observed': obs, 'extrapolate': extrapolate(comp, grid, t0).mean(0) - obs}
    if Y_donor is not None and len(Y_donor):
        cf, w = synth(obs, Y_donor, grid, t0)
        out['synth'] = cf - obs
        out['synth_weights_nonzero'] = int((w > 1e-6).sum())
    return out


def bootstrap_delta(Y_sys, comp_mask, grid, t0, Y_donor=None, B=500, seed=0):
    """Percentile intervals for Delta-hat at the last grid time."""
    rng = np.random.default_rng(seed)
    comp_idx = np.flatnonzero(comp_mask)
    res = {'extrapolate': [], 'synth': []}
    for _ in range(B):
        ci = rng.choice(comp_idx, len(comp_idx))
        Ysub = Y_sys[ci]
        m = np.ones(len(ci), bool)
        Yd = None
        if Y_donor is not None and len(Y_donor):
            Yd = Y_donor[rng.choice(len(Y_donor), len(Y_donor))]
        d = delta(Ysub, m, grid, t0, Yd)
        res['extrapolate'].append(d['extrapolate'][-1])
        if 'synth' in d:
            res['synth'].append(d['synth'][-1])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            for k, v in res.items() if v}


def own_gain(Y_S, grid, t0, B=500, seed=0):
    """Mean score gain of a set over its members' burn-in trends at the last
    grid time, with a bootstrap-over-members percentile interval."""
    g = Y_S[:, -1] - extrapolate(Y_S, grid, t0)[:, -1]
    rng = np.random.default_rng(seed)
    bs = [g[rng.choice(len(g), len(g))].mean() for _ in range(B)]
    return float(g.mean()), (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def delta_did(Y, exposed, control, grid, t0, B=500, seed=0):
    """Difference-in-differences Delta-hat at the last grid time.

    dev_n = (burn-in trend extrapolation - observed) for each agent; the
    estimate is mean dev over agents exposed to S minus mean dev over
    unexposed agents, so platform-wide departures from linear growth (which
    bias plain extrapolation) cancel. Percentile bootstrap within groups.
    """
    dev = extrapolate(Y, grid, t0)[:, -1] - Y[:, -1]
    e, c = dev[exposed], dev[control]
    if len(e) == 0 or len(c) == 0:
        return float('nan'), (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    bs = [rng.choice(e, len(e)).mean() - rng.choice(c, len(c)).mean() for _ in range(B)]
    return float(e.mean() - c.mean()), (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
