# Simulation set 1: detection (Type I error and power), design draft 2026-10-17 (v2)

Builds on `simulation_design.md`.

**Decided:**
- data fully synthetic;
- content channel only;
- single topic;
- equal responses per agent per step;
- Moltbook-sized N.

**v2 change:** collusion is *persistently correlated movement* over T > 2 time steps (user,
2026-10-17). The two-point shared shift is the special case T = 2. The number of time steps T is a
central factor.

## 1. Definition and hypotheses

- Agent n has a latent perspective p_n(t) at steps t = 1..T, with increments
  δ_n(t) = p_n(t) − p_n(t−1).
- **No collusion:** δ_n(t) = B_n f(t) + e_n(t).
  - f(t) are platform-wide factors and B_n each agent's loadings on them.
  - e_n(t) is the agent's own movement, independent across agents.
- **Colluding subnetwork S:** δ_i(t) = B_i f(t) + γ g_S(t) + e_i(t) for i ∈ S.
  - g_S(t) is a group process shared by the members at every step, with a new direction each step.
  - Collusion is persistent: g_S is active throughout. There is no burn-in and no onset.
- H0: no S exists; the factors may be present. H1: one S exists, with |S| ≪ N.
- Within S, the per-step correlation of two members' factor-free increments is

      ρ = γ²τ² / (γ²τ² + σ_e² + 2σ_x²/m_eff),

  where τ² = Var g_S, σ_e² is the agents' own movement and 2σ_x²/m_eff is the noise of estimating
  positions from m responses per step. Evidence accumulates over T − 1 increments.

### 1b. Telling groups from factors (assumption (b), user 2026-10-18; draft, not implemented)

The factor term and a collusion term are both low rank, so a sparse factor and a group are the same
kind of object, and something must separate them. The assumption:

- **(b) Collusion is one-signed.** In δ_n(t) = B_n f(t) + Σ_g γ a_gn g_g(t) + e_n(t), every group
  loading vector a_g is non-negative: the members share the group process with the same sign.
  Factor loadings B_n have no sign restriction.
- Every shared low-rank term is then one of three kinds:
  1. **Pervasive (any sign): a factor.** Its loadings cover a non-vanishing share of agents.
     - The equal-loading drift is the special case that per-step centering removes exactly.
     - The rest is removed by the existing rule: above the Marchenko–Pastur edge, with support ≥ N/3.
  2. **Sparse and two-signed: a factor (new).** Two subsets that move in opposite directions along
     one shared process are a factor, not two groups; that is the price of (b).
  3. **Sparse and one-signed: a group.** This includes a niche factor with one-signed loadings, which
     stays indistinguishable from collusion (cell N9, a stated limitation).
- **Estimation (rule).** For each singular direction above the edge, in order, with loading vector
  u (its overall sign is arbitrary):
  - count clear loadings on each side, c₊ = #{u_n > 2·floor} and c₋ = #{u_n < −2·floor}, using
    the same per-direction noise floor as the support rule;
  - for a pure group direction, the minority side holds only noise exceedances, so
    min(c₊, c₋) ~ Binomial(N, 0.0228);
  - classify the direction as two-signed, and remove it, when min(c₊, c₋) exceeds that binomial's
    1 − α quantile (α = 0.01);
  - stop at the first direction that is neither pervasive nor two-signed.
- **Known costs, to be measured.**
  - A group whose members overlap a sparse factor's support can be removed together with it.
  - Two genuine groups that anti-correlate are read as one factor.
  - With several sparse factors of similar strength, the singular vectors mix within their
    subspace. The rule then works on rotated loadings; that is fine as long as the subspace stays
    two-signed.

Open, and not part of the simulations: whether *static* similarity (clones that never change) is a
second kind of subnetwork. Under this definition clones count only when they change together, for
example through shared updates.

## 2. Generator (defaults)

| Piece | Model | Default |
|---|---|---|
| Agents | n = 1..N | N = 2000 |
| Time | T steps, m responses per agent per step | T = 8, m = 20 |
| Query space | One topic, as in `general`: queries in a continuous space R^q | q = 50 |
| Agent preference | Agent n draws queries around its own offset ρ_n ~ N(0, κ Σ_u); share κ/(1+κ) of query variance is agent-specific | Share 0.1 (Moltbook `general`: 0.09) |
| Queries | u = ρ_n + Σ_u^{1/2} ξ, all distinct (R = 1); observed û = u + σ_q ζ | σ_q = 0.5 |
| Response | x = O(u) L (η_n + p_n(t)) + W u + ε, in R^p | p = 100, latent perspective dimension d_ℓ = 10 |
| Lift | L: R^{d_ℓ} → R^p with orthonormal columns | |
| Query map | O(u): rotation varying smoothly with u, so nearby queries shape responses alike | Local overlap matched to `general`: 64th-nearest-neighbour distance ≈ 0.66 of the random-pair distance |
| Query effect | W u with W = G/√q | |
| Agent effect | η_n ~ N(0, σ_η² I), fixed over time | σ_η² = 0.5 |
| Own movement | e_n(t) ~ N(0, σ_e² I) | σ_e² chosen so agents visibly move |
| Group process | g_S(t) ~ N(0, τ² I), with strength τ | τ = 0 under H0 |
| Response noise | ε ~ N(0, σ_x² I) | σ_x² = 0.5 |
| Nuisance | f(t) ~ N(0, σ_f² I_r), loadings B_n | Section 4 |

Moltbook numbers that set the defaults (summary statistics only; no data enter the simulation):
- In `general`, agent identity explains 9% of the variance of the threads an agent answers.
- An agent's threads span 90% of `general`'s spread.
- The within-topic effective dimension is about 48.
- Nearest, 64th-nearest and random-pair thread distances are 0.37, 0.59 and 0.89.
- With daily steps, the window has about 18 steps and about 4 actions per agent-day on average.

## 3. Method under evaluation

1. **Condition** each response on its query (PKPS pooled conditioning on û), and estimate each
   agent's position at each step (mean residual, joint embedding of the N·T (agent, step) units).
2. **Residual increments.** Take increments δ̂_n(t). Remove the diffuse shared factors, estimated
   from the N × (T−1)·d increment matrix with the existing participation-ratio rule.
3. **Pair similarity:** the correlation of two agents' residual increment series, as the mean over
   t of the inner product of their standardized increments.
4. **Candidate family:** communities of the kNN graph on that similarity, plus clusters. The
   family is rebuilt on every null draw.
5. **Statistic:** Λ = max over the family of the size-standardized mean pairwise correlation.
6. **Null (decided 2026-10-17): per-step agent-ID shuffle.** At each step t, draw a new
   permutation π_t and give agent n the residual increment of agent π_t(n). This keeps each step's
   cross-section exactly (factors, anisotropy, heavy tails) and breaks only agent identity across
   steps, i.e. persistence. The family is rebuilt on every draw.
   - H0: after removing platform-wide factors, increments are independent across agents.
   - Niche groups that move together because of a shared niche stimulus are indistinguishable from
     colluders. This is accepted as a known property, not a failure.
   - **Known weak point:** estimation noise makes increments negatively autocorrelated
     (ν(t) − ν(t−1)), and independent autocorrelated series show spurious cross-correlation.
   - Fixed-set pilot, 400 replicates, standard error about 0.011:

     | Estimation noise / own movement | 1 or less | 4 |
     |---|---|---|
     | Per-step shuffle, T = 8 | 0.05 | 0.07 |
     | Per-step shuffle, T = 32 | 0.04–0.05 | 0.095 |
     | Non-overlapping increments, T = 32 | 0.04–0.05 | 0.04 |

     The block-2 shuffle holds at T = 32 (0.058) but is coarse at T = 8.
   - **Comparators:** block shuffle, non-overlapping increments, the rotation of each agent's
     whole trajectory, and the circular time shift.

At T = 2 the statistic reduces to the current cosine co-movement of single displacements, so the
existing pipeline is the base case.

### Implementation notes (2026-10-18, `scripts/pipeline/simgen.py`, `scripts/pipeline/tstep.py`)

- **Split-half increments.** δ̂(n, t) = ψ_A(n, t+1) − ψ_B(n, t), with A and B disjoint halves of each
  step's responses. Overlapping increments carry MA(1) estimation noise, which made observed pair
  correlations more spread out than the shuffle null (sd 0.097 vs 0.085). Split-half increments
  remove it (lag-1 autocorrelation 0.00; observed spread matches the shuffle).
- **Embedding dimension** by the first profile-likelihood elbow. It recovers d_ℓ = 10.
- **Factor removal.** Remove only directions above the Marchenko–Pastur noise edge whose loadings are
  diffuse, with participation ratio ≥ N/6 (Gaussian loadings give about N/3). The earlier
  threshold of 50 removed a strong colluding group as a "factor" in 4 of 10 runs.
- **Family.** Clusters, kNN-graph Louvain communities, and each agent's nearest-neighbour balls of
  size 5, 10, 20 and 40. Without the balls, the scan missed groups that a known-S test found
  10 of 10 times.
- **Mini check** (N = 500, T = 8, m = 20, no nuisance, B = 199):
  - τ = 0: Type I error 0.05 (40 runs);
  - τ = 0.3: power 0.2 (known S: 1.0);
  - τ = 0.6: power 1.0.

  About 17 s per run.

## 4. Type I error (τ = 0)

| Cell | Nuisance | Threatens |
|---|---|---|
| N0 | None | The clean level |
| N1 | One common factor, equal loadings | The colleague's setting |
| N2 | One common factor, loadings B_n ~ N(1, σ_B²) that differ by agent | Time-permutation null |
| N3 | r ∈ {3, 10} factors, agent-specific loadings | Factor removal |
| N4 | N2 + N3 at Moltbook-like strength | The realistic null |
| N5 | Anisotropic own movement and noise (eigenvalues decaying like j^(−1)) | Rotation null |
| N6 | Agent-specific noise levels | Standardization |
| N7 | Autocorrelated own movement, e_n(t) AR(1) | Shuffle nulls (spurious correlation) |
| N8 | High estimation noise relative to own movement (small m) | Per-step shuffle (MA(1) increments) |
| N9 | Niche factor loaded by a small set of agents | Shown, not fixed: indistinguishable from collusion by design |
| N10 | Sparse two-signed factors calibrated to Moltbook: r_s = 2, on 18% of agents, heavy-tailed loadings with random signs, at about 1.8× the MP edge | Rule 1b. Also crossed with a planted group (ρ ∈ {0.2, 0.3}), both disjoint from and overlapping the factor's support, to measure the power cost |

Each cell is crossed with T ∈ {2, 8, 32} and N ∈ {500, 2000, 5000} (N at T = 8 only).

**Replicates.** 500 per cell, giving a standard error of about 1% at 0.05.

## 5. Power (τ > 0)

The base uses the N4 nuisance.

**Headline: T.**
- T ∈ {2, 4, 8, 16, 32} at fixed m = 20.
- Also T at a fixed total of T·m = 160 responses per agent: more, noisier steps against fewer,
  cleaner ones.

**Other factors, one at a time,** each at 3 τ values (near 20/50/80% base power):
- |S| ∈ {5, 10, 25, 50};
- m;
- N;
- the agent query share;
- σ_q;
- nuisance strength.

**Replicates.** 200 per cell.

**Check.** If power depends mainly on ρ·√(T−1), the T and m curves collapse onto that axis.

**Ceiling.** An oracle that knows S and the true increments computes the members' mean pairwise
correlation.

## 6. Order of work

1. Write the generator (`scripts/sim_generate.py`) and the T-step statistic and nulls. The current
   scan is the T = 2 case; the new parts are the per-step positions, residual increments,
   correlation family and circular-shift null.
2. Pilot at the base cell:
   - timing;
   - 50 nulls at N0 and N4 under both nulls;
   - a coarse τ sweep at T ∈ {2, 8, 32}.
3. Fix the grid and budget. Type I error before power.
