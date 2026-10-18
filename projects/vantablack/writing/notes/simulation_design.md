# Simulation design (draft for discussion, 2026-10-17)

Fully synthetic. Content channel only. Moltbook motivates the structure and the scale; no Moltbook
data enters any simulation. The generator adapts the "Temporal Gaussian blobs" setting of
Bridgeford & Helm (arXiv 2512.05013, App. A.2) to Moltbook's structure:

- a small, unknown colluding group;
- one response per query;
- unpaired queries;
- platform drift.

Two experiments: detection (Type I error, power) and membership estimation.

## 1. Generator

Agents n = 1..N, two time steps t ∈ {1, 2} (before and after t0). Response embeddings in R^p.

**Queries (topics, unpaired).**
- K topics with centers c_k ∈ R^q.
- Agent n has topic preferences π_n ~ Dir(α 1_K). Small α means each agent concentrates on a few
  topics, so agents answer different queries.
- At each time step agent n answers exactly m queries. Each query is u = c_k + σ_u ξ, with
  k ~ π_n and ξ ~ N(0, I_q). Every query is answered once (R = 1).
- The query embedding is observed with error, û = u + σ_q ζ (vantablack).

**Responses.**

    x = O_k (μ^(t)_n + η_n + ν^(t)_n) + W u + ε,    ε ~ N(0, σ_ε² I_p)

- **O_k** is a random orthogonal matrix for each topic. Queries in the same topic rotate the latent
  perspective the same way, so the query kernel has something to borrow across nearby queries.
  Different topics are not comparable without conditioning.
- **W u** is a smooth query effect, W = G / √q with G Gaussian: what the query itself makes
  everyone say.
- **η_n** ~ N(0, σ_η² I) on the first p_s coordinates: a stable agent effect, as in the source paper.
- **μ^(t)_n** is the perspective profile on the p_s signal coordinates, as in the source paper:
  - front-loaded, α_s exp(−β j), for every agent at t = 1 and for agents outside S at t = 2;
  - for n ∈ S at t = 2, the norm-preserving shift toward back-loaded with effect size τ.

  This is the collusion: S changes *what* it says, in a shared way, not how much.
- **ν^(t)_n** is the nuisance, zero at t = 1. At t = 2 it is ν_n = λ_n c + F f_n, with:
  - c a common drift direction, and per-agent loadings λ_n ~ N(1, σ_λ²), so drift differs by agent;
  - F ∈ R^{p×r} with r shared factors and agent scores f_n ~ N(0, σ_f² I_r).

  This is the platform drift seen on Moltbook, which the source paper does not model.

**Under the null** nobody is in S (τ = 0), but the nuisance is on. Type I error is measured across
nuisance strengths, not only with the nuisance switched off.

## 2. Method under evaluation

The content pipeline as run on Moltbook:
1. PKPS pooled conditioning: each response minus the kernel-weighted mean of everyone's responses to
   nearby queries (Gaussian kernel on û, k nearest).
2. Per-agent, per-period mean residuals, then a joint embedding of the 2N (agent, period) units.
3. Displacements, then removal of diffuse factors.
4. Candidate family: clusters at many k, plus kNN co-movement communities.
5. Size-stratified rotation-null scan.

**Comparators.**
- The colleague's method (`colsim.py`: pairwise PKPS affinity, small k-means family,
  time-permutation null), at the N where it runs.
- An oracle: the O_k and W are known, so responses can be un-rotated. It gives a ceiling on power
  for detection and on recovery for membership.
- The sample-splitting point from the source paper (re-embedding inside the null inflates Type I
  error) is checked for our null.

## 3. Experiment 1: detection

**Question.** Does the scan reject when a colluding group is present, and only then?

**Metrics.**
- Type I error: the rejection rate at level 0.05 with τ = 0.
- Power: the rejection rate with τ > 0.

No Jaccard here; membership is Experiment 2.

**Factors, one at a time from the base.**

| Factor | Values | Base | Why |
|---|---|---|---|
| Effect size τ | 0, 0.1, …, 1 | 0.2 | Power curve |
| Responses per agent per step m | 5, 10, 20, 40 | 20 | Information per agent. Moltbook median is about 40 per side, after capping. |
| N | 500, 1000, 2000, 5000 | 2000 | Moltbook-sized populations |
| \|S\| | 5, 10, 25, 50 | 10 | Small minority, as on Moltbook |
| Topic concentration α | 0.1, 0.3, 1, 5 | 0.3 | How unpaired the queries are |
| Query noise σ_q | 0, 0.5, 1, 2 | 0.5 | Vantablack cost. 0 is the black-box reference. |
| Nuisance (σ_λ, σ_f, r) | none / mild / Moltbook-like / strong | mild | Type I error under drift. Run at τ = 0 and at the base τ. |

Check: if power depends on (τ, m, σ_ε) mainly through a signal-to-noise ratio like τ√m/σ_ε, the m
curves collapse when plotted against it. This connects to the theory and could shrink the grid.

## 4. Experiment 2: membership estimation

**Question.** When a group is present, how well does the method recover who is in it?

**The metric is still open.** Options:

| Option | Estimand | Metric | For | Against |
|---|---|---|---|---|
| A (recommended) | Colluding pairs | Pairwise precision-recall down the ranked groups, summarised by average precision, plus recall once k agents are covered (k = \|S\|, 2\|S\|, 5\|S\|) | No threshold; handles split or merged groups and several groups; shows imbalance; same metric as the real-data ranking | Less familiar than Jaccard |
| B | Agent set S | Agent-level average precision and precision at \|S\|, ranking agents by projection on the top group's direction | Simple | Cannot tell one group of 10 from two unrelated groups of 5 |
| C | Single returned set Ŝ | Jaccard, precision, recall | Familiar | Mixes ranking with threshold; assumes one group; not what we output |

AUC is not recommended. At |S|/N = 0.002 it can be 0.99 with dozens of innocent agents ranked
above the group.

**Factors.**
- τ × |S|.
- Two groups shifting in orthogonal directions: are they found separately or merged?
- N at fixed |S|.
- m.

All metrics are reported whether or not the test rejects.

## 5. Budget

The scan with B = 199 rotation draws costs minutes per replicate at N = 5000. Target replicates per
cell:
- 500 at τ = 0 (Type I error, standard error about 1%);
- 100 at τ > 0.

Proposal:
- run the full grid at N ≤ 2000;
- run only the base and nuisance cells at N = 5000;
- time a pilot first, then fix the grid.

## Open decisions

1. Membership estimand and metric (A, B or C).
2. Factor list and ranges (above), in particular the base values.
3. Whether to include a two-group design in Experiment 2.
4. Which comparators to include: the colleague's method, the oracle, or both.
