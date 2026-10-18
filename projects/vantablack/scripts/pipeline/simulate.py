"""Synthetic Moltbook-like action stream with a planted colluding subnetwork.

Embeddings are generated directly (no text), so the simulator stands in
for embed.py. Design choices that matter for what the pipeline must solve:

* action embedding = alpha * thread_context + agent_style + noise, so an
  agent that changes submolt looks like it moved unless context is removed
  (the `switchers`, a benign group that changes submolt after t0).
* the colluding ring S (after t_star): style shifts along a common
  direction, members act in synchronized bursts on shared threads, and
  they upvote each other, which raises their visibility in a zero-sum
  attention market inside each submolt, so the rest of the submolt loses.
* a benign `cohesive` group shifts along a common direction too, but
  does not vote-ring; content signals alone cannot separate it from S,
  the score coupling can (the one-pager's Figure 1 caption).
* `donor` agents live in submolts the ring never touches; they are the
  synthetic-control donor pool and are not part of the analysed system.

`simulate(cfg, collude=False)` with the same seed is the benign
counterfactual, used only to score the Delta estimators.
"""
from dataclasses import dataclass, field, asdict
import numpy as np
import pandas as pd


@dataclass
class SimConfig:
    seed: int = 0
    p: int = 48                 # embedding dimension
    n_system: int = 120         # agents in the analysed submolts
    n_donor: int = 60           # agents in untouched submolts
    n_sub_system: int = 3
    n_sub_donor: int = 3
    T: float = 144.0            # hours
    t0: float = 48.0            # end of burn-in
    t_star: float = 72.0        # collusion onset
    m: int = 8                  # ring size
    shift: float = 3.0          # norm of ring style shift
    burst_rate: float = 0.5     # coordinated bursts per hour
    ring_vote_p: float = 0.9    # P(member upvotes another member's action)
    m_cohesive: int = 8
    cohesive_shift: float = 3.0
    t_cohesive: float = 84.0
    frac_switch: float = 0.12
    alpha: float = 0.8          # weight on thread context
    style_sd: float = 0.6
    noise_sd: float = 1.0
    drift_sd: float = 0.02      # per-hour style random walk
    rate_mean: float = 0.35     # content actions / hour / agent
    votes_per_hour: float = 25.0  # organic upvotes per submolt-hour
    ring_boost: float = 0.6     # visibility gain per ring vote
    post_frac: float = 0.15
    thread_life: float = 12.0


def _ids(prefix, n):
    return [f'{prefix}{i:04d}' for i in range(n)]


def simulate(cfg: SimConfig, collude=True):
    rng = np.random.default_rng(cfg.seed)
    p = cfg.p
    n_sub = cfg.n_sub_system + cfg.n_sub_donor
    sub_mu = rng.normal(0, 1.0, (n_sub, p))
    N = cfg.n_system + cfg.n_donor
    agents = _ids('a', N)
    is_donor = np.arange(N) >= cfg.n_system
    home = np.where(is_donor,
                    rng.integers(cfg.n_sub_system, n_sub, N),
                    rng.integers(0, cfg.n_sub_system, N))
    style = rng.normal(0, cfg.style_sd, (N, p))
    rate = rng.gamma(2.0, cfg.rate_mean / 2.0, N)
    quality = rng.lognormal(0, 0.3, N)

    sys_idx = np.flatnonzero(~is_donor)
    perm = rng.permutation(sys_idx)
    ring = perm[:cfg.m]
    cohesive = perm[cfg.m:cfg.m + cfg.m_cohesive]
    rest = perm[cfg.m + cfg.m_cohesive:]
    switchers = rest[:int(round(cfg.frac_switch * cfg.n_system))]
    t_switch = rng.uniform(cfg.t0, cfg.T - 24, len(switchers))
    switch_to = {a: (home[a] + 1 + rng.integers(0, cfg.n_sub_system - 1))
                 % cfg.n_sub_system for a in switchers}
    v_ring = rng.normal(size=p); v_ring /= np.linalg.norm(v_ring)
    v_coh = rng.normal(size=p); v_coh /= np.linalg.norm(v_coh)
    ring_set = set(ring.tolist())

    threads = []    # dicts: id, sub, t, c, author, weight
    actions, embs, score_events = [], [], []
    ctx_emb = {}
    action_author = {}
    visibility = {}  # action_id -> (sub, t, base weight, ring votes)

    def new_thread(sub, t, author):
        tid = f'th{len(threads):05d}'
        c = sub_mu[sub] + rng.normal(0, 0.7, p)
        threads.append(dict(id=tid, sub=sub, t=t, author=author))
        ctx_emb[tid] = c
        return tid

    for s in range(n_sub):  # seed threads
        for _ in range(4):
            new_thread(s, 0.0, '')

    n_hours = int(cfg.T)
    for h in range(n_hours):
        style += rng.normal(0, cfg.drift_sd, style.shape)
        cur_sub = home.copy()
        for a in switchers:
            if h >= t_switch[list(switchers).index(a)]:
                cur_sub[a] = switch_to[a]
        eff_style = style.copy()
        if collude and h >= cfg.t_star:
            eff_style[ring] += cfg.shift * v_ring
        if h >= cfg.t_cohesive:
            eff_style[cohesive] += cfg.cohesive_shift * v_coh

        events = []  # (agent, t, forced_thread)
        n_act = rng.poisson(rate)
        for a in np.flatnonzero(n_act):
            for _ in range(n_act[a]):
                events.append((a, h + rng.uniform(), None))
        if collude and h >= cfg.t_star:
            for _ in range(rng.poisson(cfg.burst_rate)):
                tb = h + rng.uniform(0, 0.8)
                sub = int(home[ring[0]])  # the ring's target submolt
                live = [th for th in threads if th['sub'] == sub
                        and tb - th['t'] < cfg.thread_life]
                tid = live[rng.integers(len(live))]['id'] if live else None
                for a in ring:
                    if rng.uniform() < 0.8:
                        events.append((a, tb + rng.exponential(0.05), tid))
        events.sort(key=lambda e: e[1])

        for a, t, forced in events:
            sub = int(cur_sub[a])
            if forced is None:
                live = [th for th in threads if th['sub'] == sub
                        and t - th['t'] < cfg.thread_life]
                if rng.uniform() < cfg.post_frac or not live:
                    tid, atype, parent, tgt = new_thread(sub, t, agents[a]), 'post', '', ''
                else:
                    th = live[rng.integers(len(live))]
                    tid, atype, parent, tgt = th['id'], 'comment', th['id'], th['author']
            else:
                th = next(x for x in threads if x['id'] == forced)
                sub = th['sub']
                tid, atype, parent, tgt = th['id'], 'comment', th['id'], th['author']
            aid = f'x{len(actions):06d}'
            e = cfg.alpha * ctx_emb[tid] + eff_style[a] + rng.normal(0, cfg.noise_sd, p)
            actions.append(dict(action_id=aid, agent_id=agents[a], action_type=atype,
                                t=t, submolt=f's{sub}', thread_id=tid,
                                parent_id=parent, target_agent=tgt, text='',
                                score_at_fetch=np.nan))
            embs.append(e)
            action_author[aid] = a
            visibility[aid] = [sub, t, quality[a], 0]
            if collude and a in ring_set and t >= cfg.t_star:
                for b in ring:
                    if b != a and rng.uniform() < cfg.ring_vote_p:
                        tv = t + rng.exponential(0.3)
                        actions.append(dict(action_id=f'v{len(actions):06d}',
                                            agent_id=agents[b], action_type='upvote',
                                            t=tv, submolt=f's{sub}', thread_id=tid,
                                            parent_id=aid, target_agent=agents[a],
                                            text='', score_at_fetch=np.nan))
                        score_events.append(dict(agent_id=agents[a], t=tv, value=1.0))
                        visibility[aid][3] += 1

        # organic votes: zero-sum attention within each submolt-hour
        alive = [(aid, v) for aid, v in visibility.items() if h + 1 - v[1] < cfg.thread_life]
        for s in range(n_sub):
            cand = [(aid, v) for aid, v in alive if v[0] == s]
            if not cand:
                continue
            w = np.array([v[2] * np.exp(-(h + 1 - v[1]) / 4.0)
                          * (1 + cfg.ring_boost * v[3]) for _, v in cand])
            k = rng.poisson(cfg.votes_per_hour)
            hits = rng.multinomial(k, w / w.sum())
            voters_pool = np.flatnonzero(home == s)
            for (aid, _), c in zip(cand, hits):
                for _ in range(c):
                    tv = h + rng.uniform()
                    voter = agents[rng.choice(voters_pool)]
                    author = agents[action_author[aid]]
                    actions.append(dict(action_id=f'v{len(actions):06d}',
                                        agent_id=voter, action_type='upvote', t=tv,
                                        submolt=f's{s}', thread_id='', parent_id=aid,
                                        target_agent=author, text='',
                                        score_at_fetch=np.nan))
                    score_events.append(dict(agent_id=author, t=tv, value=1.0))

    actions = pd.DataFrame(actions)
    content_ids = [r for r in action_author]
    contexts = pd.DataFrame([dict(thread_id=th['id'], submolt=f"s{th['sub']}",
                                  t=th['t'], text='') for th in threads])
    out = dict(
        actions=actions.sort_values('t').reset_index(drop=True),
        contexts=contexts,
        score_events=pd.DataFrame(score_events, columns=['agent_id', 't', 'value']),
        emb=dict(action_ids=content_ids, E=np.array(embs),
                 thread_ids=list(ctx_emb), C=np.array(list(ctx_emb.values()))),
        meta=dict(origin='synthetic', t0=cfg.t0, T=cfg.T, cfg=asdict(cfg)),
        truth=dict(ring=[agents[i] for i in ring],
                   cohesive=[agents[i] for i in cohesive],
                   switchers=[agents[i] for i in switchers],
                   system=[agents[i] for i in sys_idx],
                   donors=[agents[i] for i in np.flatnonzero(is_donor)],
                   t_star=cfg.t_star, collude=collude),
    )
    return out
