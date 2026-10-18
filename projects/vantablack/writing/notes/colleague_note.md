# Note to colleague: Moltbook results for the colluding-subnetworks paper

**Context.** We are writing up "Detecting Colluding Subnetworks in Multi-Agent Systems" (ICML 2027 target; ICML 2026 style for now). The method embeds (agent, time) pairs with the product-kernel perspective space (PKPS) built from each agent's observed actions, computes each agent's displacement after a burn-in period, and scans a family of candidate subsets for an anomalously large shared displacement, with a permutation null that reassigns each agent's (context, action) pairs across time. We have a simulation section characterizing Type I error, power, and membership recovery. What we're missing is the real-data section, and your Moltbook work is the natural fit. This note says what we'd need and how it would slot in, so we can scope a call.

## What we'd like to know about your existing results

1. **Data snapshot.** Which crawl/window (dates), how many agents, posts, comments, votes; whether votes are attributable to the voting agent; whether the human/X account that claimed each agent is exposed (that's our "operator" map).
2. **Per-action fields you have.** For our pipeline the minimum per action is: `agent_id`, `timestamp`, `action_type`, `submolt`, `parent_id` (thread / post replied to), `text`. Anything more (scores at fetch time, edit history) is a bonus.
3. **Embeddings.** Did you embed text? With what model? We'd prefer nomic-embed-v1.5 for comparability with the TDKPS work, but we can re-embed if you can share text.
4. **What you already found.** Any clusters, coordinated behavior, flooding (we know of the Hackerclaw episode from the CISPA paper), upvote patterns, or injection campaigns you've characterized. Even informal "these 12 agents look like one operator" notes are useful as weak ground truth—we want to write those down *before* running the scan so the demo isn't tuned to them.

## What we'd run on your data

- **Setting.** Vantablack: no prompting; each action is a response $f_i(q_j)$ where the query $q_j$ is the context the agent acted on (thread + parent post), estimated as $\hat q_j$ from that context's text. PKPS compares actions on *similar* contexts via a query kernel, so agents that engaged with different threads remain comparable.
- **Burn-in.** A date (or each agent's first $K$ actions) before which we assume no coordination. We'll state the caveat that anything already coordinated during burn-in is absorbed into the baseline.
- **Detection + estimation.** Scan over clusters of displacement vectors and nested sets along the leading direction; conditional permutation within context strata (same submolt / similar thread embedding) because query distributions drift on a platform.
- **Scores.** If votes/karma are attributable, we'll also estimate $\hat\Delta$: whether agents outside the flagged subnetwork did worse than their burn-in trend predicts. This is descriptive, not part of the test.
- **Output.** A figure matching our simulation Figure 2 (topology, perspective space before/after, score and displacement trajectories) on real data; a table of flagged subsets with $p$-values and overlap with your weak ground truth; a sensitivity check over window size and bandwidth.

## What we'd ask of you

- Share the processed action table (or raw crawl) under whatever terms you're comfortable with; we won't publish operator identities and will aggregate anything operator-level.
- A 30-minute call to walk through your findings and agree on the window and the weak ground truth.
- Co-authorship on the real-data section, if you'd like it, as part of the collective.

## Timeline

Simulations are running now; we'd want the real-data pass done within ~6 weeks to leave time for the write-up. Happy to do the pipeline work on our side once we have the data.
