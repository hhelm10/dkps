"""End-to-end detection and estimation on one dataset in canonical form.

    data = dict(actions, contexts, score_events,
                emb=dict(action_ids, E, thread_ids, C),
                meta=dict(t0, T, ...),
                system=[agent ids] (optional; default all content agents),
                donors=[agent ids] (optional; synthetic-control pool))
"""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from . import space, candidates as cands, scores, gate
from .stats import SubnetTester


@dataclass
class PipeConfig:
    grid_step: float = 4.0
    window_h: float = 24.0      # set window_k instead for an action-count window
    window_k: int = None
    d: int = None               # None = elbow rule
    ctx_clusters: int = 3       # thread-context clusters per submolt (strata)
    B: int = 199                # rotation-null draws for the scan
    alpha: float = 0.05
    min_actions: int = 3
    max_sets: int = 3           # disjoint flagged sets reported
    workers: int = 1            # processes for the rotation-null draws
    n_strata: int = 256         # global context clusters when submolts are many
    n_factors: object = 'auto'  # shared displacement factors removed before z_C
    size_bins: tuple = (3, 11, 31, 101)  # lower edges of the scan's size bands
    gate: str = 'combined'      # 'combined' (shift+gain+co-activity), 'coordinated'
                                # (co-activity only), 'shift', or 'none'
    content: str = 'ridge'      # context conditioning: 'ridge' (linear on the thread
                                # context) or 'pkps' (pipeline/pkps.py; needs data['ref'])
    pkps_k: int = 64            # nearest queries kept in the truncated query kernel
    pkps_bw: float = 1.0        # bandwidth multiplier on the median-NN rule
    pkps_time: str = 'all'      # 'all' or 'period' (reference means per side of t0)
    seed: int = 0


def _tester(Z, ag, t, strata, t0, grid, n, cfg, mask=None):
    if mask is not None:
        Z, ag, t, strata = Z[mask], ag[mask], t[mask], strata[mask]
    M, k = space.window_matrix(ag, t, n, grid, cfg.window_h, cfg.window_k)
    return SubnetTester(Z, ag, t, strata, t0, grid, M, k, n, min_actions=cfg.min_actions)


def run_pipeline(data, cfg=PipeConfig(), verbose=True, diagnostic_sets=None):
    """diagnostic_sets: optional list of agent-id lists; each gets a time-null p-value."""
    log = print if verbose else (lambda *a, **k: None)
    t0, T = data['meta']['t0'], data['meta']['T']
    acts = data['actions']
    emb = data['emb']
    row_of = {a: i for i, a in enumerate(emb['action_ids'])}
    system = data.get('system') or sorted(acts[acts.action_id.isin(row_of)].agent_id.unique())
    sys_index = {a: i for i, a in enumerate(system)}
    n = len(system)

    content = acts[acts.action_id.isin(row_of) & acts.agent_id.isin(sys_index)].copy()
    E = emb['E'][[row_of[a] for a in content.action_id]]
    th_row = {th: i for i, th in enumerate(emb['thread_ids'])}
    C_act = emb['C'][[th_row[th] for th in content.thread_id]]
    ag = content.agent_id.map(sys_index).values
    t = content.t.values
    log(f'[data] {n} agents, {len(content)} content actions, t0={t0}, T={T}')

    # 1. context conditioning
    E = E.astype(np.float32)
    C_act = C_act.astype(np.float32)
    pkps_info, self_share = None, None
    if cfg.content == 'pkps':
        R, self_share, pkps_info = _pkps(data, content, E, ag, t, t0, cfg, log)
    else:
        R, _ = space.residualize(E, C_act)
    R = R.astype(np.float32)
    sep = dict(raw=space.separability(E, content.submolt.values),
               residual=space.separability(R, content.submolt.values))
    log(f'[context] submolt silhouette raw={sep["raw"]:.3f} residual={sep["residual"]:.3f}')

    # 2. strata for the matched-context permutation
    strata = np.zeros(len(content), dtype=np.int64)
    subs = content.submolt.values
    if len(np.unique(subs)) <= 50:  # submolt x per-submolt context clusters
        for si, s in enumerate(np.unique(subs)):
            m = subs == s
            kk = min(cfg.ctx_clusters, m.sum())
            lab = KMeans(kk, n_init=4, random_state=cfg.seed).fit_predict(C_act[m]) if kk > 1 else 0
            strata[m] = si * 100 + lab
    else:  # platform scale: one global clustering of thread contexts
        from sklearn.cluster import MiniBatchKMeans
        strata[:] = MiniBatchKMeans(cfg.n_strata, n_init=1, random_state=cfg.seed,
                                    batch_size=8192).fit_predict(C_act)

    # 3. windows + joint embedding
    grid = np.arange(cfg.grid_step, T + 1e-9, cfg.grid_step)
    M, k = space.window_matrix(ag, t, n, grid, cfg.window_h, cfg.window_k)
    proj = space.fit_projection(R, M, k, d=cfg.d)
    Z = space.project(R, proj)
    log(f'[space] d={proj["d"]} (explained {proj["explained"]:.2f})')

    full = _tester(Z, ag, t, strata, t0, grid, n, cfg)
    log(f'[null] {full.frac_permutable:.2f} of actions lie in permutable '
        f'(agent, context) cells; {full.eligible.sum()}/{n} agents eligible')

    # 4. candidate family + scan
    fixed, G_int = cands.interaction(acts, sys_index, t0, seed=cfg.seed)
    valid = content.text.fillna('').str.len().values >= 40
    fam_co, W_co = cands.coactivity(acts, sys_index, t0, seed=cfg.seed, return_weights=True)
    fixed = cands.dedupe(fixed + fam_co + cands.neardup(E, ag, valid=valid, seed=cfg.seed))
    log(f'[scan] fixed family: {len(fixed)} sets '
        f'({", ".join(sorted(set(s.split("(")[0] for _, s in fixed)))})')
    el = full.eligible

    def build(Dc):
        return cands.dedupe(cands.clusters(Dc, el, seed=cfg.seed)
                            + cands.comove(Dc, el, seed=cfg.seed))

    scan = full.scan(build, fixed, R=cfg.B, seed=cfg.seed, alpha=cfg.alpha,
                     max_sets=cfg.max_sets, workers=cfg.workers,
                     size_bins=cfg.size_bins, n_factors=cfg.n_factors,
                     shift_gate=False)

    # 5. adversarial gate on the scan-significant sets (pipeline/gate.py)
    Y = scores.trajectories(data['score_events'], system, grid)
    gain_n = Y[:, -1] - scores.extrapolate(Y, grid, t0)[:, -1]
    scan['flagged'] = gate.apply(scan, full.eligible, gain_n, W_co, alpha=cfg.alpha,
                                 gate=cfg.gate, max_sets=cfg.max_sets, seed=cfg.seed)
    srcs = sorted(set(r_['source'].split('(')[0] for r_ in scan['rows']))
    log(f'[scan] |family|={scan["n_family"]} ({", ".join(srcs)}); '
        f'{scan["n_factors"]} displacement factors removed; '
        f'Lambda={scan["Lambda"]:.2f}, null 95%={np.quantile(scan["Lambda_null"], .95):.2f}; '
        f'{scan["n_comove_sig"]} sets pass co-movement, {len(scan["flagged"])} flagged after '
        f'{cfg.gate} gate')

    donors = data.get('donors') or []
    Yd = scores.trajectories(data['score_events'], donors, grid) if donors else None
    # Delta-hat is measured on the agents in no flagged set, so one flagged
    # set's effect does not contaminate another's counterfactual. Which set
    # is responsible is read from its own score gain over its burn-in trend
    # (coordinated movement + relative gain = adversarial; movement alone =
    # merely cohesive).
    comp = np.ones(n, bool)
    for f in scan['flagged']:
        comp[f['members']] = False
    dl = scores.delta(Y, comp, grid, t0, Yd)
    ci = scores.bootstrap_delta(Y, comp, grid, t0, Yd, B=300, seed=cfg.seed)
    delta_T = {k: float(dl[k][-1]) for k in ('extrapolate', 'synth') if k in dl}
    if scan['flagged']:
        log('[delta] unflagged agents: ' + ', '.join(
            f'{k}={v:+.2f} CI({ci[k][0]:+.2f},{ci[k][1]:+.2f})' for k, v in delta_T.items()))
    # one batched time-null run for every flagged set (per-set p is marginal)
    tn = full.time_null([(frozenset(f['members']), f['source']) for f in scan['flagged']],
                        B=min(cfg.B, 99), seed=cfg.seed) if scan['flagged'] else {'rows': []}
    tn_p = {tuple(r_['members']): r_['p'] for r_ in tn['rows']}
    flagged = []
    for f in scan['flagged']:
        S = f['members']
        gain, gain_ci = scores.own_gain(Y[S], grid, t0, seed=cfg.seed)
        f = dict(f, agents=[system[i] for i in S], onset=full.onset(S)[0],
                 own_gain=gain, own_gain_ci=gain_ci,
                 adversarial=bool(gain_ci[0] > 0),
                 time_null_p=tn_p.get(tuple(sorted(S)), float('nan')))
        flagged.append(f)
        log(f'[flag] {f["source"]} |S|={len(S)} T={f["T"]:.2f} (zE={f["zE"]:.2f}, zC={f["zC"]:.2f}) '
            f'p_fw={f["p_fw"]:.3f} p_gate={f["p_gate"]:.2g} (shift {f["p_shift"]:.2g}, gain {f["p_gain"]:.2g}, '
            f'coact {f["p_coact"]:.2g}) onset={f["onset"]} own gain={gain:+.1f} '
            f'CI({gain_ci[0]:+.1f},{gain_ci[1]:+.1f}) time-null p={f["time_null_p"]:.3f} '
            f'-> {"ADVERSARIAL" if f["adversarial"] else "cohesive only"}')
    flagged.sort(key=lambda f: (not f['adversarial'], f['p_fw'], -f['T']))

    # exposure-based DiD for the leading adversarial set: exposed = unflagged
    # agents active after t0 in the submolts holding >= 50% of S's post-t0
    # content; control = unflagged agents with no post-t0 action there
    did = None
    adv = [f for f in flagged if f['adversarial']]
    if adv:
        S = adv[0]['members']
        post = t > t0
        sv = pd.Series(subs[post & np.isin(ag, S)]).value_counts(normalize=True)
        tgt = list(sv.index[:np.searchsorted(sv.cumsum().values, 0.5) + 1])
        touched = np.zeros(n, bool)
        touched[np.unique(ag[post & np.isin(subs, tgt)])] = True
        exposed, control = comp & touched, comp & ~touched
        est, ci_ = scores.delta_did(Y, exposed, control, grid, t0, seed=cfg.seed)
        did = dict(submolts=tgt, n_exposed=int(exposed.sum()), n_control=int(control.sum()),
                   estimate=est, ci=ci_, exposed=exposed, control=control)
        log(f'[delta] DiD on exposure to S ({",".join(tgt)}): exposed={exposed.sum()} '
            f'control={control.sum()} Delta-hat={est:+.2f} CI({ci_[0]:+.2f},{ci_[1]:+.2f})')
    S_hat = flagged[0]['members'] if flagged else []
    diag = []
    if diagnostic_sets:
        sets = [sorted(sys_index[a] for a in ids if a in sys_index) for ids in diagnostic_sets]
        tn = full.time_null([(frozenset(S), 'diagnostic') for S in sets],
                            B=min(cfg.B, 99), seed=cfg.seed)
        dp = {tuple(r_['members']): r_['p'] for r_ in tn['rows']}
        diag = [dp.get(tuple(S), np.nan) for S in sets]

    return dict(cfg=asdict(cfg), system=system, grid=grid, t0=t0, T=T, k=k,
                Psi=(M @ Z).reshape(n, len(grid), -1), D=full.features(Z)[0],
                proj=dict(d=proj['d'], explained=proj['explained']),
                separability=sep, frac_permutable=full.frac_permutable,
                scan=scan, flagged=flagged, S_hat_idx=S_hat,
                S_hat=[system[i] for i in S_hat], Y=Y, G_int=G_int,
                comp=comp, did=did, eligible=full.eligible, diagnostic_p=diag, delta=dl, delta_T=delta_T, delta_ci=ci,
                pkps=pkps_info, self_share=_agent_mean(self_share, ag, n))


def _agent_mean(x, ag, n):
    if x is None:
        return None
    return np.bincount(ag, weights=x, minlength=n) / np.maximum(np.bincount(ag, minlength=n), 1)


def _pkps(data, content, E, ag, t, t0, cfg, log):
    """PKPS conditioning of the system rows on everyone's responses (pipeline/pkps.py).

    data['ref'] = dict(actions (action_id, agent_id, action_type, t, thread_id,
    submolt), E, thread_ids, Q_thread, sub_ids, Q_sub). Thread queries are the
    context embeddings in data['emb'] plus ref['Q_thread'].
    """
    from . import pkps
    ref, emb = data['ref'], data['emb']
    th_ids = list(emb['thread_ids']) + list(ref['thread_ids'])
    Q_thread = np.vstack([emb['C'], ref['Q_thread']]).astype(np.float32)
    th = {x: i for i, x in enumerate(th_ids)}
    sub = {x: i for i, x in enumerate(ref['sub_ids'])}
    ra = ref['actions']

    def qidx(df):
        kind = df.action_type.values
        q = np.where(kind == 'post', df.submolt.map(sub).fillna(-1).values,
                     df.thread_id.map(th).fillna(-1).values).astype(np.int64)
        return kind, q
    ks, qs = qidx(content)
    kr, qr = qidx(ra)
    ok_s, ok_r = qs >= 0, qr >= 0
    if not ok_s.all():
        log(f'[pkps] {(~ok_s).sum()} system rows without a query embedding keep the raw embedding')
    R = E.copy()
    share = np.zeros(len(E), np.float32)
    Rs, sh, info = pkps.condition_all(
        E[ok_s], ag[ok_s], t[ok_s], ks[ok_s], qs[ok_s],
        ref['E'][ok_r], kr[ok_r], qr[ok_r], Q_thread, ref['Q_sub'], t0,
        t_ref=ra.t.values[ok_r], k=cfg.pkps_k, bw=cfg.pkps_bw, time=cfg.pkps_time, log=log)
    R[ok_s], share[ok_s] = Rs, sh
    return R, share, info
