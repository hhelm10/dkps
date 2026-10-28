"""Two-member ensembles of score estimators.

Blend  y_hat = alpha * member[0] + (1 - alpha) * member[1], clipped to `clip`.

Paper settings
--------------
Query efficiency:  Ensemble([SampleScore(), PKPS(...)], mode='cv', holdout='family',
                   predict_kwargs=[{}, {'holdout': 'family'}])
    alpha is a per-family scalar chosen on a grid to minimize the error, over the OTHER
    families' observed cells, of the blend against their reference (full) scores --
    records must carry 'reference_score' for those cells.
Completion:        Ensemble([PKPS(...), LRMC()], mode='cv', holdout=None,
                   predict_kwargs=[{'whiten': True}, {}])
    one alpha chosen on the observed cells against their sample scores (both members
    are cross-fit / leave-one-out there, so neither echoes a cell's own sample).
mode='precision' is the per-cell shrinkage of the own sample (member[0]) toward the
prior (member[1]): weight on the sample = e_i / (e_i + p(1-p)/m), with e_i the model's
depth-weighted prior error; mode='fixed' uses the given alpha.
"""

import numpy as np
import pandas as pd

from .pkps import family
from .records import parse_records, ScoreTable, parse_pairs, pairs_to_records


class Ensemble:
    """Blend of two fitted (or to-be-fitted) score estimators.

    Parameters
    ----------
    members : [a, b]
        Two estimators with the fit / predict / update API. y_hat = alpha*a + (1-alpha)*b.
        By convention member[1] is the prior / model-based estimate (it wins the fallback
        when only one member has a value).
    mode : 'cv' | 'precision' | 'fixed'
        'cv'        alpha on a grid of `grid` points in [0, 1], minimizing the blend's MAE on the
                    observed cells against `target` ('auto' = reference score where given, else
                    sample score). With holdout='family' a separate alpha is learned for each
                    family from the OTHER families' cells (leave-one-family-out).
        'precision' per-cell shrinkage weight on member[0] (the own sample):
                    e_i / (e_i + p(1-p)/m), e_i = the model's depth-weighted prior error.
        'fixed'     the given `alpha`.
    holdout : 'family' or None (per-family vs. one global alpha; 'cv' mode only)
    family_fn : model_id -> family label (default: developer prefix)
    predict_kwargs : [kwargs_a, kwargs_b] forwarded to each member's predict(), both when
        learning alpha and when predicting (e.g. {'holdout': 'family'} or {'whiten': True}).
    clip : output range, or None
    fallback : if one member has no estimate for a cell, use the other's (else NaN)

    Attributes (after fit): alpha_global_, alpha_ (per family, when holdout='family'),
    prior_err_ (precision mode), model_names_, task_names_.
    """

    def __init__(self, members, mode='cv', alpha=None, holdout='family', family_fn=None,
                 predict_kwargs=None, grid=21, target='auto', clip=(0.0, 1.0), fallback=True):
        assert len(members) == 2, 'Ensemble blends exactly two members'
        self.members = list(members)
        self.mode = mode
        self.alpha = alpha
        self.holdout = holdout
        self.family_fn = family_fn or family
        self.predict_kwargs = list(predict_kwargs or [{}, {}])
        self.grid = np.linspace(0, 1, grid)
        self.target = target
        self.clip = clip
        self.fallback = fallback   # one member missing: use the other (prefer member[1]); else NaN

    # ------------------------------------------------------------------
    def fit(self, records):
        """Fit any unfitted member on `records`, then learn the blend weight(s)."""
        df = parse_records(records, required=('model_id', 'query_id'))
        if 'task_id' not in df.columns:
            df['task_id'] = '_task'
        for m in self.members:
            if not getattr(m, '_fitted', hasattr(m, 'pred_matrix_')):
                m.fit(records)
        self._records = records
        self._learn(df)
        return self

    def update(self, records):
        """Forward the new records to both members and re-learn the blend weight(s)."""
        for m in self.members:
            m.update(records)
        df = parse_records(records, required=('model_id', 'query_id'))
        self._learn(df, incremental=True)
        return self

    def predict(self, records=None):
        """Blend the members' predictions for the requested (model, task) pairs (None = the
        union of the members' natural targets). Returns a list of dicts with 'score_hat'."""
        pairs = parse_pairs(records, self.model_names_, self.task_names_, self._default_pairs())
        P = self._member_preds(pairs)
        alpha = self._alpha_for(pairs, P)
        preds = {}
        for n, (m, t) in enumerate(pairs):
            a, b = P[0][n], P[1][n]
            if np.isfinite(a) and np.isfinite(b):
                v = alpha[n] * a + (1 - alpha[n]) * b
            elif self.fallback:
                v = b if np.isfinite(b) else a
            else:
                v = np.nan
            preds[(m, t)] = self._clip(v) if np.isfinite(v) else np.nan
        return pairs_to_records(pairs, preds)

    # ------------------------------------------------------------------
    def _clip(self, v):
        return float(np.clip(v, *self.clip)) if self.clip is not None else float(v)

    def _member_preds(self, pairs):
        out = []
        for m, kw in zip(self.members, self.predict_kwargs):
            recs = [{'model_id': a, 'task_id': b} for a, b in pairs]
            r = m.predict(recs, **kw)
            out.append(np.array([x['score_hat'] for x in r], dtype=float))
        return out

    def _default_pairs(self):
        # union of the members' natural targets (observed cells for SampleScore/IRT,
        # missing cells for PKPS/LRMC), in member order
        seen, out = set(), []
        for m, kw in zip(self.members, self.predict_kwargs):
            for r in m.predict(None, **kw):
                key = (r['model_id'], r['task_id'])
                if key not in seen:
                    seen.add(key); out.append(key)
        return out

    def _learn(self, df, incremental=False):
        """Learn alpha from the members' predictions on the observed cells. Both members are
        asked for every observed cell with their predict_kwargs, so a member that is
        leave-one-out / cross-fit there (PKPS, LRMC) contributes clean estimates."""
        # score tables (sample + counts, and reference scores) from the members' own records
        raw = getattr(self.members[1], '_raw', None)
        raw = raw if raw is not None else df
        table = ScoreTable(raw)
        self.model_names_, self.task_names_ = table.models, table.tasks
        S, N = table.values, table.counts_.to_numpy(dtype=float)
        R = np.full_like(S, np.nan)
        if 'reference_score' in raw.columns:
            ref = raw.dropna(subset=['reference_score']).groupby(['model_id', 'task_id'])[
                'reference_score'].first().unstack().reindex(index=table.models, columns=table.tasks)
            R = ref.to_numpy(dtype=float)
        Y = {'sample': S, 'reference': R, 'auto': np.where(np.isfinite(R), R, S)}[self.target]
        obs = np.isfinite(S)
        pairs = [(m, t) for i, m in enumerate(table.models) for j, t in enumerate(table.tasks) if obs[i, j]]
        P = self._member_preds(pairs)
        ii = np.array([table.index(m) for m, _ in pairs]); jj = np.array([table.tasks.index(t) for _, t in pairs])
        y = Y[ii, jj]; a, b = P
        ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(y)
        fams = np.array([self.family_fn(m) for m, _ in pairs])

        self.alpha_ = {}
        if self.mode == 'fixed':
            assert self.alpha is not None, "mode='fixed' needs alpha"
            self.alpha_global_ = float(self.alpha)
        elif self.mode == 'cv':
            def best(mask):
                if mask.sum() < 5:
                    return 0.5
                errs = [np.nanmean(np.abs(np.clip(g * a[mask] + (1 - g) * b[mask], *self.clip) - y[mask]))
                        if self.clip is not None else
                        np.nanmean(np.abs(g * a[mask] + (1 - g) * b[mask] - y[mask])) for g in self.grid]
                return float(self.grid[int(np.argmin(errs))])
            self.alpha_global_ = best(ok)
            if self.holdout == 'family':
                for f in sorted(set(fams)):
                    self.alpha_[f] = best(ok & (fams != f))
        elif self.mode == 'precision':
            # per-model prior error from the observed cells, depth-weighted; member[1] = prior
            n = np.maximum(N[ii, jj], 1.0)
            pcl = np.clip(np.where(np.isfinite(b), b, 0.5), 1e-3, 1 - 1e-3)
            sig = pcl * (1 - pcl) / n
            self.prior_err_ = {}
            for m in table.models:
                sel = ok & np.array([mm == m for mm, _ in pairs])
                if sel.any():
                    err = (b[sel] - a[sel]) ** 2 - sig[sel]
                    self.prior_err_[m] = max(float(np.sum(n[sel] * err) / np.sum(n[sel])), 1e-6)
                else:
                    self.prior_err_[m] = 1e-6
            self._counts = table.counts_
            self.alpha_global_ = None
        else:
            raise ValueError(f'unknown mode: {self.mode}')

    def _alpha_for(self, pairs, P):
        """Per-pair blend weight on member[0]."""
        if self.mode == 'precision':
            out = np.empty(len(pairs))
            for n, (m, t) in enumerate(pairs):
                cnt = float(self._counts.at[m, t]) if (m in self._counts.index and t in self._counts.columns) else 0.0
                b = P[1][n]
                pcl = np.clip(b if np.isfinite(b) else 0.5, 1e-3, 1 - 1e-3)
                sig = pcl * (1 - pcl) / max(cnt, 1.0)
                e = self.prior_err_.get(m, 1e-6)
                out[n] = e / (e + sig)
            return out
        if self.mode == 'cv' and self.holdout == 'family':
            return np.array([self.alpha_.get(self.family_fn(m), self.alpha_global_) for m, _ in pairs])
        return np.full(len(pairs), self.alpha_global_)
