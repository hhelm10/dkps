"""Adversarial gate applied to sets that pass the co-movement scan.

The one-pager calls S adversarial when it moved in a coordinated way AND
gained relative to the rest. Beyond co-movement (already tested by the
scan), a set must show evidence on three per-set features, each compared
with random sets of the same activity-quartile composition:

  shift   sum of members' standardized per-agent energy (behaviour changed)
  gain    mean rank of members' score gain over their burn-in trend
  coact   mean pairwise co-activity weight (same thread within minutes)

The three p-values are combined by Fisher's method (chi2, 6 df) and Holm-
corrected across the scan-significant sets. gate='shift' uses the shift
p-value alone (the earlier design), gate='none' flags on the scan only.

gate='coordinated' asks only for coordination, not harm: a set must move
together (the scan) AND act together in time (co-activity p). It produces a
discovery list, so the co-activity p-values are controlled at false
discovery rate alpha (Benjamini-Hochberg) across the scan-significant sets
rather than family-wise (Holm). Score gain is reported, not required.
The co-activity null uses up to B_coact_max random sets, fewer for large
sets (cost grows with size^2), so small sets can reach Holm-level p-values.
"""
import numpy as np
from scipy.stats import chi2, rankdata

from .stats import _holm


def _matched(bins, pools, S, B, rng):
    cols = []
    for b in range(len(pools)):
        c = int((bins[S] == b).sum())
        if c:
            cols.append(pools[b][rng.integers(0, len(pools[b]), (B, c))])
    return np.concatenate(cols, 1)


def _pair_sums(W, R):
    """For each row r of R (agent indices, drawn with replacement), the sum of
    W[i, j] over all i, j in r, duplicates counted: c^T W c with c the draw
    counts. Computed as (M W) gathered at r, M the sparse count matrix, in
    float64 on the GPU when available (same values as summing the |r|^2
    entries directly, to float rounding)."""
    try:
        import torch
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        torch = None
    n_, s = W.shape[0], R.shape[1]
    out = np.empty(len(R))
    if torch is None:
        step = max(1, int(2e7 // (s ** 2)))
        for a in range(0, len(R), step):
            blk = R[a:a + step]
            out[a:a + step] = W[blk[:, :, None], blk[:, None, :]].sum((1, 2), dtype=np.float64)
        return out
    Wt = torch.as_tensor(W, dtype=torch.float64, device=dev)
    step = min(4096, max(1, int(2e8 // (s * n_))))
    for a in range(0, len(R), step):
        blk = torch.as_tensor(R[a:a + step], device=dev)
        b = blk.shape[0]
        M = torch.sparse_coo_tensor(
            torch.stack([torch.arange(b, device=dev).repeat_interleave(s), blk.reshape(-1)]),
            torch.ones(b * s, dtype=torch.float64, device=dev), (b, n_)).coalesce()
        out[a:a + b] = torch.sparse.mm(M, Wt).gather(1, blk).sum(1).cpu().numpy()
    return out


def apply(scan, eligible, gain, W, alpha=0.05, gate='combined', max_sets=3, B=9999,
          B_coact_max=99999, coact_budget=2e9, seed=0):
    rng = np.random.default_rng(seed + 11)
    ze, bins = scan['ze'], scan['activity_bins']
    pools = [np.flatnonzero(eligible & (bins == b)) for b in range(4)]
    u = np.full(len(gain), 0.5)
    u[eligible] = rankdata(gain[eligible]) / (eligible.sum() + 1)
    cand = scan['candidates']
    for r in cand:
        S = np.asarray(r['members'])
        Bc = int(max(999, min(B_coact_max, coact_budget // (len(S) ** 2))))
        R = _matched(bins, pools, S, max(B, Bc), rng)
        r['p_gain'] = float((1 + (u[R[:B]].mean(1) >= u[S].mean()).sum()) / (B + 1))
        if W is not None:
            obs = W[np.ix_(S, S)].sum() / (len(S) * (len(S) - 1))
            null = _pair_sums(W, R[:Bc]) / (len(S) * (len(S) - 1))
            r['p_coact'] = float((1 + (null >= obs).sum()) / (Bc + 1))
        else:
            r['p_coact'] = 1.0
        stat = -2 * (np.log(r['p_shift']) + np.log(r['p_gain']) + np.log(r['p_coact']))
        r['p_combined'] = float(chi2.sf(stat, 6))
    key = {'combined': 'p_combined', 'shift': 'p_shift', 'coordinated': 'p_coact'}.get(gate)
    if key:
        ps = [r[key] for r in cand]
        adj = (_bh(ps) if gate == 'coordinated' else _holm(ps)) if cand else []
        for r, a in zip(cand, adj):
            r['p_gate'] = float(a)
    else:
        for r in cand:
            r['p_gate'] = 0.0
    flagged, used = [], set()
    for r in sorted(cand, key=lambda r: (r['p_gate'], r['p_fw'], -r['comove'])):
        if len(flagged) >= max_sets:
            break
        if r['p_gate'] < alpha and used.isdisjoint(r['members']):
            flagged.append(r)
            used |= set(r['members'])
    return flagged


def _bh(p):
    """Benjamini-Hochberg adjusted p-values (step-up)."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 1.0
    for r in range(m - 1, -1, -1):
        i = order[r]
        run = min(run, p[i] * m / (r + 1))
        adj[i] = run
    return adj
