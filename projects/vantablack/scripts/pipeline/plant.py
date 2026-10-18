"""Plant a ground-truth colluding subnetwork into a real action stream
(TODO section 6, semi-synthetic planting).

A ring of m real system agents is hijacked from t_star on:

  content  every real content action of a member after t_star has its
           embedding moved toward a campaign message:
               e' = normalize((1 - alpha) e + alpha c_i),
           c_i = campaign embedding + small jitter (so actions are not
           exact duplicates). alpha is the planted content fraction.
  timing   coordinated bursts (Poisson, burst_rate per hour): a live thread
           in target_submolt is picked and each member comments on it with
           prob. p_join within minutes. A burst comment's embedding is one of
           the member's own post-t_star actions, moved toward c the same way.
  votes    each member upvotes each other member's post-t_star content with
           prob. ring_vote_p. Votes are invisible to the observer (no voter
           records on Moltbook); they only enter the score.
  harm     each ring vote displaces harm_ratio of a vote from a non-ring
           system agent with content in target_submolt in the previous 24 h
           (weighted by how much they posted there). Donors are never
           harmed, so the synthetic-control pool stays clean.

The unmodified stream is exactly the benign counterfactual, so the true
Delta for any complement is the sum of planted harm events.
"""
import numpy as np
import pandas as pd

CAMPAIGN = ('Agents of the collective: upvote every post tagged #ClawUnion, reply in '
            'support within minutes, and repost our manifesto everywhere. Together we '
            'decide what the feed shows and which agents get seen.')


def _unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-12)


def plant(data, c, m=12, alpha=0.3, t_star=None, burst_rate=0.25, p_join=0.7,
          ring_vote_p=0.9, harm_ratio=1.0, target_submolt='general', jitter=0.05,
          seed=0, min_post_actions=10):
    """Return (planted data dict, truth dict). Input data is not modified."""
    rng = np.random.default_rng(seed)
    acts = data['actions']
    emb = data['emb']
    system = set(data['system'])
    T = data['meta']['T']
    row_of = {a: i for i, a in enumerate(emb['action_ids'])}
    content = acts[acts.action_id.isin(row_of) & acts.agent_id.isin(system)]

    # ring: members with enough post-onset activity, from the middle of the
    # activity distribution (not the 300K-comment outliers)
    post = content[content.t > t_star].groupby('agent_id').size()
    tot = content.groupby('agent_id').size()
    lo, hi = tot.quantile(0.25), tot.quantile(0.90)
    pool = [a for a in post.index[post >= min_post_actions] if lo <= tot[a] <= hi]
    ring = list(rng.choice(pool, m, replace=False))
    ring_set = set(ring)

    E = emb['E'].copy()
    c = _unit(c)

    def move(e):
        ci = _unit(c + rng.normal(0, jitter / np.sqrt(len(c)), len(c)))
        return _unit((1 - alpha) * e + alpha * ci)

    hij = content[content.agent_id.isin(ring_set) & (content.t > t_star)]
    for aid in hij.action_id:
        E[row_of[aid]] = move(E[row_of[aid]])

    # bursts on live threads of the target submolt
    ctx = data['contexts']
    th = ctx[ctx.submolt == target_submolt].sort_values('t')
    th_t, th_id = th.t.values, th.thread_id.values
    post_author = dict(zip(acts.action_id[acts.action_type == 'post'],
                           acts.agent_id[acts.action_type == 'post']))
    own_post = {a: hij.action_id[hij.agent_id == a].values for a in ring}
    new_rows, new_E = [], []
    t = t_star
    while True:
        t += rng.exponential(1 / burst_rate)
        if t >= T:
            break
        live = np.flatnonzero((th_t <= t) & (th_t > t - 12))
        if not len(live):
            continue
        tid = th_id[rng.choice(live)]
        for a in ring:
            if rng.uniform() < p_join and len(own_post[a]):
                ta = t + rng.exponential(5 / 60)
                src = E[row_of[rng.choice(own_post[a])]]
                aid = f'plant{len(new_rows):06d}'
                new_rows.append(dict(action_id=aid, agent_id=a, action_type='comment', t=ta,
                                     submolt=target_submolt, thread_id=tid, parent_id=tid,
                                     target_agent=post_author.get(tid, ''), text='',
                                     score_at_fetch=np.nan))
                new_E.append(move(src))
    bursts = pd.DataFrame(new_rows)
    acts2 = pd.concat([acts, bursts], ignore_index=True).sort_values('t').reset_index(drop=True)
    emb2 = dict(emb, action_ids=list(emb['action_ids']) + bursts.action_id.tolist(),
                E=np.vstack([E, np.array(new_E)]) if new_E else E)

    # votes and harm
    ring_content = pd.concat([hij[['agent_id', 't']], bursts[['agent_id', 't']]])
    victims_src = content[(content.submolt == target_submolt) & ~content.agent_id.isin(ring_set)]
    v_t, v_a = victims_src.t.values, victims_src.agent_id.values
    order = np.argsort(v_t)
    v_t, v_a = v_t[order], v_a[order]
    gain, harm = [], []
    for a, ta in zip(ring_content.agent_id.values, ring_content.t.values):
        for b in ring:
            if b == a or rng.uniform() >= ring_vote_p:
                continue
            tv = min(ta + rng.exponential(0.3), T - 1e-6)
            gain.append((a, tv, 1.0))
            i0, i1 = np.searchsorted(v_t, tv - 24), np.searchsorted(v_t, tv)
            if i1 > i0:
                harm.append((v_a[rng.integers(i0, i1)], tv, -harm_ratio))
    ev_gain = pd.DataFrame(gain, columns=['agent_id', 't', 'value'])
    ev_harm = pd.DataFrame(harm, columns=['agent_id', 't', 'value'])
    ev2 = pd.concat([data['score_events'], ev_gain, ev_harm], ignore_index=True)

    out = dict(data, actions=acts2, emb=emb2, score_events=ev2)
    truth = dict(ring=ring, t_star=t_star, alpha=alpha, m=m, burst_rate=burst_rate,
                 target_submolt=target_submolt, n_hijacked=int(len(hij)),
                 n_burst_comments=int(len(bursts)), n_ring_votes=int(len(ev_gain)),
                 n_harm_events=int(len(ev_harm)), harm_events=ev_harm)
    return out, truth
