# Methods as implemented, and where they depart from the TODO list

## Pipeline

1. **Context conditioning.** Each action embedding is ridge-regressed on its thread's context embedding, and the residual is kept. In simulation, submolt silhouette falls from 0.106 to -0.005, so residuals do not cluster by submolt. This check runs on every dataset.
2. **Windowed positions.** ψ̂_n(t) is the mean residual over a trailing 24 h window on a 4 h grid. k_n(t) is the number of actions in the window and sets the uncertainty disc, with radius σ/√k.
3. **Joint embedding.** Classical MDS on Euclidean distances between all (agent, t) windowed means is equivalent to PCA of the centred means. d comes from the second profile-likelihood elbow. Because the basis is linear, each action residual is projected once and every window is a sparse linear map, which makes resampling cheap.
4. **Statistic.** T(S) = z_E(S) + z_C(S).
   - z_E is the mean over members of each agent's energy distance between its own burn-in and post-burn-in windows. It is standardised within the agent's activity quartile.
   - z_C is the mean pairwise cosine of system-centred displacement vectors D_n, scaled by √(d·s(s−1)/2).
5. **Candidate family.**
   - Agglomerative clusters of D_n directions, at k = 5, 10 and 20.
   - Louvain communities of a kNN co-movement graph.
   - Louvain communities of the observed post-t0 interaction graph.
   - Near-duplicate components, linking agents that share at least 3 action pairs with cosine ≥ 0.97.
6. **Scan null (rotation).** Each agent's displacement is replaced by a uniformly random direction of the same length. This keeps how far and how noisily every agent moves and destroys alignment between agents. Families (a) and (b) are rebuilt on every draw, so the null distribution of Λ = max_S T(S) includes selection. Flagged sets are disjoint sets whose T exceeds the null of Λ at α = 0.05, up to 3 sets.
7. **Attribution.** For each flagged set we compute its own score gain over its members' burn-in trends, with a bootstrap over members. A positive interval marks the set adversarial; otherwise it is cohesive only.
8. **Δ̂.** Δ̂ is measured on agents in no flagged set and is positive when they were harmed. Two counterfactual estimators are used:
   - per-agent burn-in trend extrapolation;
   - synthetic control, with convex weights over donor agents outside the system, fit on burn-in.

   Intervals come from a percentile bootstrap.

## Departures from the TODO list, and why

| TODO item | Implemented | Reason |
|---|---|---|
| §4 time-permutation null (permute each agent's actions across t0 within matched contexts) | Kept as `time_null`, a per-set diagnostic. Not used for the scan | Synthetic check: 4% rejection on random subsets with no drift, 14% with mild benign drift. Any benign non-stationarity breaks time exchangeability for every agent |
| §4 T(S) = energy distance on pooled windows + co-movement | Per-agent energy, averaged and standardised. Co-movement as a z-score on system-centred D | Pooled energy mixes cross-agent alignment into the shift term and is not rotation invariant. Centring removes platform-wide shifts, such as a news event moving everyone |
| §4 Λ null via the whole pipeline on time-permuted data | Whole family-building step on rotation-null data | Same selection-aware idea, without the exchangeability assumption that drift breaks |
| §4 sample splitting (first attempt) | Dropped | Persistent per-agent drift makes clusters chosen on one half stay aligned on the other half. Splitting did not remove selection bias |
| §5 Δ̂ on [N]∖Ŝ | Δ̂ on agents in no flagged set | With two flagged sets, each one's complement contains the other's effect. The synthetic run showed a −29 bias for the cohesive group |

## Synthetic validation (`validate_synthetic.py`; 10 seeds per cell; 99 rotation draws)

| Check | Result |
|---|---|
| Negative control, no ring and no cohesive group: any flag | 1/20 runs (0.05) |
| Negative control: adversarial flag | 0/20 |
| Time-null rejection on random subsets, no drift | 0.04 |
| Time-null rejection on random subsets, drift sd 0.02/h | 0.14 |
| Power, ring size 8, shift 1 / 2 / 3 / 4 | 0.0 / 0.5 / 1.0 / 1.0 |
| Power, ring size 4, shift 1 / 2 / 3 / 4 | 0.0 / 0.1 / 0.4 / 0.6 |
| Power at shift 0 (vote ring plus bursts, no content change) | 0.0 for both sizes |

Here power means the ring is recovered with Jaccard ≥ 0.5 and attributed as adversarial.

**Headline demo** (`demo_synthetic.py`, seed 0, shift 3, 199 draws):
- The ring was recovered exactly, 8 of 8 members with no false members, at p_fw = 0.005.
- Estimated onset was 76 h against a true onset of 72 h.
- The ring's own gain was +506, with interval (+401, +675).
- The benign cohesive group was also flagged, with 7 of 8 members plus 4 others. Its own gain was −13, with interval (−21, −4), so it was correctly labelled cohesive only.

| Δ(T) on unflagged agents | Value | 95% interval |
|---|---|---|
| True, from the counterfactual simulation | +9.98 | |
| Extrapolation estimate | +7.98 | (+3.4, +12.7) |
| Synthetic-control estimate | +11.05 | (+1.7, +20.8) |

## Known gaps

- **Timing coordination is unused.** Synchronized bursts are not part of T(S). A co-activity statistic, such as excess same-thread co-occurrence within minutes, would cover the pure-vote-ring blind spot partly. On real data it is the only handle on rings, because no voter records exist.
- **Sizes are small.** All synthetic numbers come from 10 seeds per cell. The power curve needs about 50 seeds per cell before it goes in a paper.
- **The simulator is idealised.** Ring style shift is isotropic Gaussian. Real coordinated content, such as near-identical posts, is far more concentrated, and that helps detection.

# Platform-scale revision (2026-10-17)

Running on the whole platform showed four failures of the first design that the 120-agent simulation could not show. Each fix below is in the code and reapplied to the synthetic tests.

| Problem seen on 5,241 real agents | Fix |
|---|---|
| Mean-centring made every agent with near-zero displacement point along the negative mean direction. Low-movement agents looked perfectly aligned, with cosine about 0.87, in the data and in the null. This inflated the null threshold of the ring's size band to 101 | No centring. Shared drift is removed as diffuse factors of the uncentered displacements instead |
| Real displacements have low-rank shared structure: the top direction carries 20% of the variance, from platform trends | Remove leading displacement factors while their loadings are diffuse, with participation ratio at least the larger of 50 and 5% of agents. Cap the count at the first elbow. A colluding group loads on few agents, so it is never removed. The run removed 13 factors at d = 100. The synthetic 120-agent system removes 0 |
| Co-movement z grows with set size, so communities of 100 to 1,000 trend-following agents set the scan threshold | Size-stratified scan, with bands 3–10, 11–30, 31–100 and over 100. Λ is taken per band, with Bonferroni across occupied bands. Overlapping significant sets are ranked by mean pairwise co-movement, not by T |
| 235 real sets clear the co-movement scan. Agents in the same threads move together, so the independent-agents null is too permissive for real data | Intersection-union gate. A set is flagged only if it also shifted. Its summed per-agent energy z must beat 9,999 random sets of the same activity-quartile composition, Holm-corrected across the sets that pass the scan |
| d = 20 from the elbow rule explains 42% of variance. In 17 effective dimensions, chance alignment among 5,241 agents is strong | d = 100, explaining 73% of variance |

Two other changes:
- **A new candidate family, co-activity.** It contains Louvain communities of agents who acted in the same thread within 10 minutes of each other, more than 3 times after t0. It uses timing only.
- **A new Δ̂ estimator, difference-in-differences.** Each agent's deviation from its own burn-in trend is averaged over agents active where Ŝ acted, minus the same average over unexposed agents.

## Adversarial gate (final design)

Sets that pass the co-movement scan are flagged only if a combined test passes. Three per-set p-values are combined by Fisher's method, each against 9,999 random sets of the same activity-quartile composition:
- **Shift:** summed per-agent energy z.
- **Gain:** mean rank of members' score gain over their burn-in trend.
- **Co-activity:** mean pairwise rate of acting in the same thread within 10 minutes. This one uses 999 random sets.

The combined p-values are Holm-corrected across the scan-significant sets. These three are the one-pager's defining properties of an adversarial set, beyond co-movement. A shift-only gate missed 2 of 5 platform rings at shift 0.3: their shift p-values of about 0.012 did not survive Holm across 240 sets.

## Synthetic validation after the platform revision (10 seeds per cell, 99 draws)

| Check | Result |
|---|---|
| Negative control, any flag | 0/20 |
| Time-null diagnostic, random subsets, no drift / drift | 0.00 / 0.08 |
| Power, ring size 8, shift 1 / 2 / 3 / 4 | 0.0 / 0.8 / 1.0 / 1.0 |
| Power, ring size 4, shift 1 / 2 / 3 / 4 | 0.0 / 0.2 / 0.8 / 0.9 |
| Power at shift 0 (pure vote ring) | 0.0 |

Platform results are in `platform_results.md`.

## Real-data presentation: ranking instead of a cutoff (2026-10-17)

The real-data examples need only be coordinated, not adversarial. Candidate groups of at most 100 agents are ranked by the better of two ranks:
- **Content co-movement:** mean pairwise cosine of factor-adjusted displacements.
- **Timing co-activity:** z against activity-matched random sets.

Evaluation is the recall of independently known coordinated pairs down the ranking, so no threshold is needed. Ranking by z_C alone favours large weakly aligned groups, which is why the size limit and the size-free cosine are used. See `platform_results.md`.
