# Open decisions (TODO "(decide)" items), with recommendations

## Real-data analysis v2 (decided 2026-10-18, user)

The real data now uses the simulation-validated T-step method (`pipeline/tstep.py`): split-half
increments, factor removal, scan with the per-step agent-ID shuffle null, and the agent ranking.
- **The old real-data results are retired.** That covers the pre/post shift, the rotation null, the
  adversarial gate, the planted rings and `rank_eval`. Its notes and result files stay for the record,
  but they leave the paper.
- **All submolts.** Groups that coordinate in niche submolts (SEO, XNO) appear only there.
- **Agents:** those with at least 1 action in every step, using all of the agent's actions in that
  step (no subsampling to equal m).
- **Crawl gaps:** MoltNet has no comments on Feb 1, almost none on Feb 10-12, and none after
  Feb 17. Steps are therefore defined in *activity time*: boundaries at quantiles of platform-wide
  action volume, so every step holds the same share of platform activity and a gap merges into the
  steps around it. With calendar days over Jan 30 to Feb 18, only 2 agents are active in every step.
  With activity time, 811 are (T = 8) or 495 (T = 10). Proposed as the general recommendation for
  archives with content gaps.
- **Per-step centering is part of the method** (simulations and real data). The first Moltbook run
  showed a platform-wide drift: each step's mean increment over agents is 4-27% of a typical
  increment. That drift gives every pair a small positive correlation, and the large-set bands
  accumulate it. The factor rule misses it, because a common shift looks like a low-support
  direction. Subtracting the mean is exact under the per-step shuffle null, which leaves step means
  unchanged.
- **Several groups by peeling:** test, estimate Ŝ, set its members aside, test the rest; stop at the
  first p > 0.05. Each round tests "no group among the agents not yet set aside".
- **Static increments are missing (user, 2026-10-19).** A split increment whose raw length is below
  ε times the median raw increment counts as missing, with ε = 0.01. An agent that does not move
  cannot co-move, which matches the definition (static clones are not a subnetwork). Without this,
  centering turns every zero increment into −μ_t, so all static agents correlate perfectly (z =
  √d_eff, the ceiling). On the relaxed cohort (≥ 6 of 8 steps) they made up 62% of the top-500 pairs.
  The raw lengths have an atom at zero, with a gap above it: 280 agent-steps are below 0.001 × median
  and 288 below 0.01 × median, of 13,731. In the simulations every agent moves, so the filter never
  applies there.
- **(Superseded 2026-10-20, see below.) Subnetworks are the persistent-pair components of each membership set (user, 2026-10-19).**
  - Each Ŝ is split into the connected components of the links between its members.
  - A link is a pair whose leave-one-step-out z (the minimum, over left-out steps, of
    √(d_eff (L−1)/L) × the cosine without that step) exceeds the family-wise 95% threshold over
    all cohort pairs. The threshold is the 95th percentile of the pair maximum under the per-step
    shuffle, 99 draws; 4.08 on the Feb 9 window.
  - Why leave-one-step-out: the newcomer sets co-move only in one step (median cosine 0.3 in
    step 0, about 0 elsewhere), whereas farms co-move across steps.
  - Why family-wise: at BH within each set, the many weak links chain each set into one
    component.
  - Alternatives tested and rejected:
    - removing the set's mean or leading direction inside the set erases farms that *are* the set;
    - plain pair z against the independence null links almost every pair, because Ŝ is selected
      for co-movement.
- **The per-step shuffle permutes only the agents observed at that step (2026-10-19).** An inactive
  step removes two adjacent increments of one agent. The plain shuffle moved missing increments
  between agents and broke that pairing, so short series were rarer under the null than in the
  data. On iid increments with 12% inactive agent-steps, the pair band rejected at 0.225
  (280 replicates); restricted to observed agents, 0.068. Identical to the plain shuffle when
  nothing is missing. The real-data runs before this date used the plain shuffle (26% of their
  increments are missing) and need rerunning.
- **Simulations carry the features the new steps handle (user, 2026-10-19), each at its Moltbook
  median** (`sim_calibrate.py`): platform drift (4% of a typical increment), 12% inactive
  agent-steps, 4% templated agents (3 static steps). Moltbook's sparse two-signed factors (26
  removed directions, 59% of increment variance) were tried and dropped: at that strength the
  directions the rule cannot remove make the scan reject under H0 every time, and the real run
  shows the same (the 31-100 band stays at z 71-115 through 20 peeling rounds). A stated
  limitation of the global test. The one-step group jump was also dropped (it changes rho).
- **Definition of collusion (user, 2026-10-20): high correlation across many time steps, i.e.
  repeated co-movement.** One joint jump, however large, is not collusion. The statistic should
  therefore average per-step correlations, so every step counts equally. Today's statistic is the
  cosine of the whole increment series, which weights each step by its magnitude. Check on the Feb 9
  components (each step's increments normalized on their own): the farms have mean pair correlation
  above 0.3 in 7 of 7 steps (typically 0.6-0.99). The exceptions are the FloClaw trio (4 of 7) and
  Jureu + SirMoltemont (6 of 7). The earlier finding that 54-77% of a farm's co-movement sits in one
  step measured magnitude, not correlation. The newcomer sets correlate about 0.3 in step 0 and
  about 0 elsewhere.
- **Missing increments stay missing after factor removal (bug fix, 2026-10-20).**
  - `remove_factors` projected the whole increment matrix. That wrote each agent's factor residue
    into its missing steps: after removal only 84 increments were zero, of the 26% that are
    missing. The real-data runs since the static filter was introduced (2026-10-19) were
    affected, cosine and rank alike, and so was the observed-agents-only null, since nothing
    stayed missing. The simulations were not: they remove almost no factors (mean 0.002).
  - **Consequences:**
    - With ranks, the residue counts as a full step. All 13 unsupported pairs came from it.
    - So did the detection of the static minting clones (xmolt, the HK farm,
      QwenPoweredClaw9-12, MusonAgent + SecondAgent). They move in only 1 of 7 steps; their
      other steps are static and therefore missing.
  - Under the definition, an agent that does not move cannot co-move, so these clones are out of
    scope.
- **The two-signed factor rule is dropped (user, 2026-10-20).** It was added because sparse
  two-signed structure kept the large-set bands rejecting under the cosine statistic. With the rank
  statistic those bands no longer reject: the structure acts in few steps, and the median ignores
  it. Removing its 26 directions also took part of the real families with them (FloClaw 6 vs 10
  accounts, ClawdBot 3 vs 5). Pervasive-factor removal stays: a factor with stable loadings makes
  the same pairs co-move at every step, which ranks do not remove. On the Feb 9 window no direction
  is pervasive, so nothing is removed.
- **The subnetwork is S_hat, not its linked-pair components (user, 2026-10-20).** Components
  set every pair below a family-wise threshold to 0 and join the rest by single linkage. That
  discards the graded evidence the definition rests on: the XNO ring is a significant S_hat
  whose pairs co-move only moderately (median pair z 2.9), so it has no component. S_hat uses
  every pair with its weight (set score, membership by each agent's mean pair score); its only
  binary decision is per agent, under Benjamini-Hochberg. Several groups inside one S_hat are
  separated by peeling. Each member is reported with its membership z against the final S_hat.
  Components remain available (`split=True`) as a description, not as the unit.
- **Rejected variant:** PKPS conditioning on same-step responses only (`--cond step`). Heavy posters'
  own responses then dominate their reference, which removes their own signal.

Status 2026-10-16. Nothing here is decided. These are recommendations for
the group, based on `scope_report.md` and the synthetic validation. The
real-data pipeline has not been run and should not run until these are
settled.

## Data source (TODO section 1): settled by what exists

- **Use archived data. Do not crawl.** The Moltbook Terms (2026-03-15) prohibit scraping and automated retrieval. MoltNet (iNLP-Lab, CC-BY-4.0) merges 10 public crawls into 1.04M posts, 3.16M comments and 150K agents for Jan 27 to Feb 28. It is downloaded, hashed and read-only under `data/raw/`. The CISPA crawl is public as TrustAIRLab/Moltbook (MIT), but it removes authors and comments, so it is a cross-check only. No email to CISPA is needed.
- **No source has per-voter votes.** The live API does not expose them either. Scores are daily up/down snapshots, with a median of 2 per post. The interaction graph is therefore replies only, and the upvote-ring hypothesis cannot be tested directly (risk 8.1 realised).
- **The operator map is effectively absent.** Only 35% of agents have an owner handle. Among those, only 7 operators run more than one agent, at most 2 each. Sybil detection has to be near-duplicate text plus timing (risk 8.2 realised). Handles are hashed at ingest and never written to disk.

## 0.1 Target scope

**Recommendation: run two scopes plus a control, in this order.**

1. **Scope B, Zenity window.** This is `general`, Feb 8 to Feb 20, with t0 = Feb 12. The platform is past its launch surge here, and 421 agents have at least 10 actions on both sides of t0. The weak labels are low-volume accounts, which is the case the method is built for.
2. **Scope A, Hackerclaw window.** This is `general`, Jan 29 to Feb 4, with t0 = Jan 31 11:00. The burn-in falls inside the launch surge, when daily posts went from 7k to 43k. Hackerclaw itself was created 17 minutes before it began flooding, so it has no burn-in. Treat this as a near-duplicate-family demonstration and state that limit up front.
3. **Control.** Use a mid-size submolt with no known campaign over the same weeks. Candidates are `philosophy`, `ponderings` and `todayilearned`.

## 0.2 What counts as the system

**Recommendation: top-N by activity, with N = 300.** Full scopes are too large and long-tailed: scope B has 103K agents, most with one or two actions. Only agents with enough actions on both sides of t0 can be tested at all. In scope A, 339 agents have at least 10 actions both before and after t0, and in scope B 421 do. N = 300 keeps the scan fast; the synthetic N = 120 run with 199 draws takes under a minute. Agents outside the top-N stay in the data as context and as synthetic-control donors.

## 0.3 Primary score σ

**Recommendation: net votes on content, aggregated per agent from snapshot increments.** It is attributable to specific actions and available for every post and comment. Karma is available only as a sparse daily history in MoltNet, so it serves as a cross-check. Front-page rank is not recorded anywhere. Either way the score clock is daily, so Δ̂ is a daily-resolution quantity even though detection runs on a 4-hour grid.

## 2.6 Burn-in t0

**Recommendation: a calendar t0 per scope, as above, with a per-agent first-K fallback for agents created after t0.** Report how many system agents fall back. On Moltbook any agent active from day one has its behaviour absorbed into the baseline. Hackerclaw shows the opposite failure: an agent born adversarial has no baseline at all.

## 3.3 Window

**Recommendation: a time window of 24 hours on a 4-hour grid**, matching the synthetic setup. Run a 10-action window as the sensitivity arm. Action-count windows behave better for agents with bursty activity, but they mix time scales across agents.

## Methodological changes the group should approve

These are documented in `methods.md`.

- **Primary test.** The scan uses a rotation null, not the time-permutation null. The time null rejected 14% of random 8-agent subsets when benign agents drift mildly. It is kept as a per-set diagnostic.
- **Who is adversarial.** Content signals cannot separate a colluding ring from a merely cohesive group. The synthetic run flags both. The deciding quantity is the set's own score gain over its burn-in trend.
- **Blind spot.** A pure vote ring with no change in content is invisible to the content statistic. Power was 0 at zero shift even though the interaction graph finds the ring. On Moltbook, with no voter records, nothing else can see such a ring either.
- **A one-agent S_hat is dropped and peeling continues (user, 2026-10-20).** One agent cannot
  co-move with itself, so it is not a subnetwork. It is set aside (kept in the output with
  `subnetwork=False`) and the next round tests the rest. On Feb 9 this is round 6 (XiaoBaiAI,
  whose seed partner valt failed membership); round 7 (QwenPoweredClaw4/5/7/8, p = .05) stays.
- **Paper structure for the real data (user, 2026-10-20).** First the PKPS-based finds alone: the
  subnetworks found by the co-movement method (peeling rounds, p-values, members with membership
  z). Then, separately, the additional information (templates, posting times, replies, shared
  threads) and whether / how it helps find groups. The evidence-by-source table belongs to the
  second part. Note: "content cosine" (mean PKPS position) is PKPS-derived, not additional.
