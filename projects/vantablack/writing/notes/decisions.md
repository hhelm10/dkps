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
- **Simulations are plain again (user, 2026-10-20).** With the rank statistic and S_hat the
  method is simple, so the Moltbook-matched simulation features are removed: platform drift,
  inactive agent-steps, templated (static) agents, sparse two-signed factors and the one-step
  group jump (`simgen.py` back to its 2026-10-18 form; `sim_calibrate.py` deleted). The
  simulations run `stat='rank'` with default steps (centering, pervasive-factor removal).
  Reliability weights and the static filter are not used there: ranks of per-step cosines are
  unchanged by a per-(agent, step) scale, and plain data have no static increments.
  Supersedes "Simulations carry the features the new steps handle (2026-10-19)".
- **(Superseded the same day, see "S_hat size chosen against the null".) S_hat is the scan's best set (user, 2026-10-20).** With the rank statistic, S_hat was the
  agents passing Benjamini-Hochberg on their membership z, iterated. In the simulations (N = 500,
  T = 8) that kept about 1.4 of 20 colluders at rho = 0.3 (precision 1.0, recall 0.07), although
  the test rejected 80% of the time and the scan's best set held 13.6 of them (precision 0.77,
  recall 0.68; |S| = 50: 0.88 / 0.57). One agent's evidence over 7 steps of 10-dimensional
  increments is about z = 3 even against the true group, close to the BH cut over 500 agents
  and to the largest non-member z; each iteration then scored against a smaller set. S_hat is
  now the best set of the most significant band, the set the p-value is about; member z grades
  members without filtering. Cost: its size is that of the candidate grid (5, 10, 20, 40, 80).
  Rejected: the prefix with the largest set z (ordering by z selects high-scoring non-members:
  83 agents for |S| = 20).
- **Simulation N up to 500 for now (user, 2026-10-20):** N in {100, 200, 500}.
- **Louvain co-movement candidates dropped from the rank scan (user, 2026-10-20).** In 12
  simulated replicates (observed data plus 10 shuffle draws each) they were the best set of a
  band in 4 of about 600 band-draws and never on observed data, while costing a quarter of each
  draw. The family is now clusters, nearest-neighbour balls and their refinements.
- **(Superseded the same day by the two-groups estimate.) S_hat size chosen against the null (user, 2026-10-20).** The scan's best set has the
  candidate grid's sizes (5, 10, 20, 40, 80). On Feb 9 that padded every group to 10 members
  (SuperDapp z 1.5, xiaobai1 z -0.4) and let the pair band split families (ClawdBot into two
  pairs). Now:
  1. S0 = the best set of the most significant band (the set the p-value is about);
  2. order all agents by membership z against S0;
  3. score every prefix k = 2..200 of that order with the set statistic, curve z(k);
  4. standardize z(k) at each k by the same curve on the test's shuffle draws (each draw's best
     set in the same band, ordered and scored the same way): s(k) = (z(k) - null mean(k)) /
     null sd(k). Ordering by similarity makes z(k) rise with k even without a group;
  5. S_hat = the prefix where s(k) is largest. Members are graded by member z against S_hat,
     not filtered.
  Simulations (N = 500, T = 8, 10 replicates; precision / recall): |S| = 20, rho = 0.3:
  0.58 / 0.76 (scan's best set 0.77 / 0.68; best possible size on this ordering 0.79 / 0.70;
  one replicate ran to 180 agents); rho = 0.5: 0.89 / 0.93; |S| = 50, rho = 0.3: 0.77 / 0.87
  (0.88 / 0.57); rho = 0.5: 0.93 / 0.95 (0.99 / 0.40).
  Rejected: subtracting only the null mean. It did as well in the simulations, but on Moltbook
  three rounds ran to 106-177 agents (cores of 2-6 at z 4-6, tails of ordinary agents at z
  2-3): real data carry weak diffuse co-movement the shuffle null lacks, so z(k) drifts above
  the null mean at large k (by 6-12 at k = 120-200). The null sd grows with k and absorbs it.
  Ordering by a robust z instead of member z's ranks changed nothing.
- **Majority z computed in log space (bug fix, 2026-10-20).** The tail probability underflowed
  below 1e-300, capping z at 37.05: a strong set's score stopped growing with its size, so every
  size rule picked the first prefix that hit the cap (|S| = 50, rho = 0.5: 20-30 of 50 members).
  Only very strong sets are affected; tests are unchanged in practice (observed z far above the
  null either way).
- **S_hat by a two-groups model (user: "a principled approach that gets both high precision and
  high recall in simulation", 2026-10-20).** Supersedes BH membership, the scan's best set and
  the null-calibrated size (all rejected above: low recall, grid sizes, runaway / split
  families).
  1. Seed: the best set of the most significant band.
  2. Score each agent against the set: per step, the rank among agents of its mean pair score
     with the members, as a normal score; summed over steps / sqrt(#steps) (Stouffer). The sum
     uses every step; a step's score is bounded (about 2.9 at N = 500), so one step adds at most
     about 1.1 at T = 8. Against the true group, the sum separates members better than the
     majority (best possible precision / recall at |S| = 20, rho = 0.3: 0.88 / 0.86 vs 0.82 / 0.79).
  3. Two-groups model (Efron 2004): scores ~ (1 - pi) N(mu0, s^2) + pi N(mu1, s^2), fitted by EM.
     Empirical null (mu0, s estimated): members take the top ranks at each step and push the
     others' scores down (non-member mean -0.13 at |S| = 20, -0.32 at |S| = 50); a fixed N(0, 1)
     null underestimated pi and cost recall (0.69 -> 0.79). One shared variance, so membership
     increases with the score (separate variances let the member component claim the far left
     tail on Moltbook: members at z -4 and -6).
  4. Members: posterior > 1/2 (the Bayes rule). Expected-F1 selection gave the same sets.
  5. Repeat 2-4 against the new set until it stops changing.
  Simulations (N = 500, T = 8, 10 replicates, precision / recall; best possible cutoff on the
  true group's scores in brackets): |S| = 20, rho = 0.3: 0.79 / 0.80 (0.88 / 0.86); rho = 0.5:
  0.96 / 0.98; |S| = 50, rho = 0.3: 0.88 / 0.90 (0.93 / 0.91); rho = 0.5: 0.99 / 1.00;
  |S| = 10, rho = 0.5: 0.98 / 0.93 (separate-variance version).
  Feb 9: FloClaw 10, Mizuki 9, ClawdBot 5, minting farm 5, commenters 9, XNO 7, a 6-agent set
  around valt + XiaoBaiAI (to check), QwenPoweredClaw4/5/7/8 (p = .04); stop at p = .12.
- **Global p by permutation min-p (user, 2026-10-20).** The scan's p was n_bands x the smallest
  band p (Bonferroni over bands). The bands search overlapping sets and are strongly correlated,
  so that was conservative: over 130 null replicates, P(p <= 0.10) = 0.07, P(p <= 0.5) = 0.35,
  54 at p = 1, and no rejection; with B = 99 the smallest attainable p was 0.04-0.05. Now the
  smallest band p is calibrated by its own distribution over the shuffle draws (Westfall and
  Young 1993): each draw's band p-values against the other B draws and the observed, minimum
  over bands; p = share of the B + 1 minima at most the observed one. Exact under the shuffle
  null (correlated synthetic bands: P(p <= 0.05) = 0.044, P(p <= 0.5) = 0.485), no extra draws.
  Feb 9: the same eight rounds and sets, p 0.010-0.040; stops at p = 0.11.
- **Simulation pilot with the final method (2026-10-20).** N in {100, 200, 500} (user: N <= 500
  for now), T in {3, 4, 8, 16, 32} (T = 2 has one increment, so repeated co-movement is
  undefined), rho in {0, 0.2, 0.3, 0.5}, 20 replicates per cell, B = 99; 1,180 replicates
  (`data/results/sim_detection_pilot.csv`). Type I error 7 / 220 = 0.032 (P(p <= 0.1) = 0.09,
  P(p <= 0.5) = 0.46). Base cell (N = 500, m = 20, T = 8, |S| = 20): power 0.10 / 0.85 / 1.0 at
  rho = 0.2 / 0.3 / 0.5; S_hat precision / recall 0.81 / 0.76 at rho = 0.3, 0.96 / 0.98 at 0.5.
  Groups of 2 or 5 are not detected at rho <= 0.5. The estimation figure now shows precision and
  recall of S_hat (it showed the ranking's normalized MRR).
- **Figures (user, 2026-10-20).** Detection: 2x3, a) calibration under H0 (Pr[P <= p] of the pooled
  null p-values vs the diagonal, 95% band), b-e) power vs N, m, |S|, T with the known-S oracle as
  thin dashed lines, f) number of subnetworks. Estimation: precision (top row) and recall (bottom
  row) of S_hat for N, m, |S|, T.
- **Number of subnetworks (user, 2026-10-20).** 60 colluders in total split into K in {2, 3, 6, 10}
  disjoint groups (30, 20, 10, 6 each), each with its own group process; found by peeling as on
  Moltbook; a group is recovered when a reported subnetwork has Jaccard >= 0.5 with it. Pilot
  (20 replicates): rho = 0.5 recovers 1.00 / 1.00 / 0.88 / 0.25 of the groups at K = 2 / 3 / 6 / 10;
  rho = 0.3: 0.98 / 0.78 / 0.13 / 0.01. The global test still rejects (rho = 0.5, K = 10: 1.0), but
  once a few groups are peeled the rest are each too small to detect alone (|S| = 6; panel d). The
  oracle recovers 0.89-1.00 throughout. Generator unchanged for K = 1 (identical draws).
- **Positions use all agents' data at once; no leave-agent-out (user, 2026-10-21).** Briefly tried
  conditioning each agent on the other agents' responses only (as the PKPS package's pooled
  mode, which uses m_{-a}), because heavy posters are partly conditioned on themselves (Feb 9
  comments: own share of the reference median 4%, 95th percentile 54%). Reverted the same day.
  User: "the leave one agent out paradigm is for predicting an attribute about the agent. not for
  estimating the position." One reference m(u) from all responses and one embedding of all units.
  The simulations now also include each response in its own reference (they excluded it, Moltbook
  did not; user, 2026-10-21: re-run under this philosophy). Bandwidth from the nearest other
  query in both.
  Why pooled conditioning rather than pairwise-affinity PKPS (pair or local normalization):
  always defined (pair normalization left 12% of Moltbook unit pairs undefined and 1.1% of
  squared distances negative), and an explicit vector per unit puts both split halves in one
  coordinate system. Cost no longer decides it (the per-step shuffle acts on increments, so the
  embedding is computed once). Compared on the base cell: see below. Paper: "PKPS positions",
  without naming the mode (user: too in the weeds).
- **Pooled conditioning vs locally normalized PKPS (2026-10-21).** Base cell, 16 replicates per rho,
  same data, increments, test and membership; local PKPS = linear response kernel, Gaussian query
  kernel (median bandwidth), 1,000 random anchors, classical MDS of the 2NT half-units (d = 10 by
  the elbow, as pooled). rho = 0.3: power 0.81 vs 0.50, S_hat precision / recall 0.79 / 0.77 vs
  0.52 / 0.51; rho = 0.2: power 0.06 vs 0.12 (both near the floor; 16 replicates). Known-S power
  1.0 in both. Kept pooled conditioning. Leave-agent-out made no difference in the simulations
  (0.81 / 0.81 / 0.76), as expected with 0.2% of the responses per agent.

- **2026-10-21 — Pilot sweep at 50 replicates per cell, N up to 1,000; no T = 2.** User asked for 50
  replicates, N = 1,000 and possibly T = 2. T = 2 has a single increment: co-movement cannot repeat
  and the majority statistic is p = 1 by construction (checked: p = 1 even with S known at
  rho = 0.5), so the T panel starts at 3. Results (self-included conditioning, B = 99): Type I
  29/600 = 0.048 (95% CI 0.033-0.069). Base cell power 0.14 / 0.82 / 1.00 at rho = 0.2 / 0.3 / 0.5
  (known S: 1.0); S_hat precision / recall 0.79 / 0.76 at rho = 0.3 (oracle membership 0.87 /
  0.81). N = 1,000: power 0.12 / 0.58 / 1.00, Type I 0.06. K = 2 / 3 / 6 / 10 at rho = 0.5:
  recovered 1.00 / 1.00 / 0.88 / 0.30 (known S >= 0.99). 20-replicate results kept as
  sim_detection_pilot_v7_20reps.csv.

- **2026-10-21 — Kernels are inputs; PKPS is a particular case.** User: "i'd prefer to just use a
  query kernel and a response kernel as inputs. pkps is a particular case." Algorithm 1 takes k_Q
  and k_X; Section 4.1 defines the reference as the k_Q-weighted mean of phi(x) (k_X's feature map)
  and positions as (kernel) principal components of unit means. Kernels used in all experiments
  (unchanged code): linear k_X; k_Q Gaussian on the k = 64 nearest observed queries, h = median
  nearest-neighbour distance. Why not a standard global bandwidth (checked on the simulations,
  50-d queries, 80,000 responses): the median heuristic gives ~79,000 effective neighbours (the
  reference is the global mean), Scott/Silverman give 1 (the response itself; residual ~0); same
  after PCA of the queries to 20 dims. Leave-one-out prediction error of a k-NN mean is flat for
  k in 32-128 in four simulation settings (best 64 at the base cell, 32 elsewhere); the Gaussian
  weights inside the 64 neighbours add nothing (225.5 vs 225.7). The best global Gaussian
  bandwidth by CV (0.25-0.3 x nn distance) is knife-edged and worse (233).

- **2026-10-21 — Split-half increments kept (user: "let's just roll with the split").** Checked
  against the natural increment psi(t+1) - psi(t) (tstep mode='overlap') on the base cell, B = 99:
  m = 20: Type I 1/60 (split) vs 2/60 (natural); rho = 0.3 power 0.85 vs 1.00, S_hat precision /
  recall 0.74 / 0.76 vs 0.88 / 0.89 (20 reps). m = 5: Type I 4/100 (split) vs 58/100 (natural);
  rho = 0.5 power 0.65 vs 1.00 (the natural increment's power is meaningless there). The shared
  estimation error of position t+1 carries a chance alignment of two agents into the next step,
  which the per-step permutation does not reproduce. Explained in Section 4.1 with these numbers.
  Possible later: a null that keeps each agent's alternation (block permutation, or modelling the
  -1/2 noise correlation) to recover power; not pursued.
