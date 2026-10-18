# Sync with the colluding-subnetworks paper draft (2026-10-17)

Source: colleague's zip (`colluding_subnetworks_paper/`): ICML-style draft (`main.tex`,
`sections/`), simulation code (`sims/colsim.py`, `run_grid.py`), a pilot
(R = 60, B = 100, base cell only), and `notes/colleague_note.md`, which asks for our
Moltbook results to fill the empty real-data section.

## Where the two pipelines agree

| Piece | Paper | This repo |
|---|---|---|
| Setting | Vantablack: the query is the context an action responds to | Same |
| Embedder | nomic-embed-text-v1.5 preferred | nomic-embed-text-v1.5, pinned revision, `clustering: ` prefix |
| Representation | PKPS over (agent, time) units, joint CMDS | Joint embedding of (agent, time); PKPS conditioning now running (`--content pkps`) |
| Burn-in | Fixed t0 | Fixed t0 = Feb 6 |
| Target | Shared displacement of an unknown subset | Same |
| Selection-aware null | Family rebuilt inside every null draw | Same (rotation null) |
| Size standardization | Per-size null moments | Per-size-band null maxima |
| Scores | Descriptive only, not used for detection | Same in the ranking presentation; Δ not identifiable on Moltbook |
| Figure 2 | Topology / before / after / score + displacement | `pipeline/figure2.py` draws the same four panels |

## Where they differ, and what the Moltbook data says

1. **Null distribution.** The paper permutes each agent's (query, response) pairs across
   time; for Moltbook it plans the conditional version within context strata. Under the
   paper's own H0 (δ = 0, any common c) pairs are not exchangeable when c ≠ 0, and on real
   data drift is agent-specific, not common. Our synthetic check (10 seeds): the
   time-permutation null rejected 0-4% of random subsets with no drift and 8-14% with mild
   per-agent drift (sd 0.02/h), depending on pipeline version. We use a
   rotation null over displacement directions instead and keep time-permutation as a
   diagnostic. Suggest: add heterogeneous drift (Δ_i = λ_i c, λ_i varying) to grid G and
   compare both nulls.
2. **Common component.** Model (eq. model) has one common c. Real displacements have
   low-rank shared structure: the top direction carries 20% of variance, 13 diffuse
   factors removed at d = 100. Median centering removes a constant shift, not factors with
   agent-specific loadings. Whitening by the trimmed covariance partly down-weights them.
   Suggest: Δ_i = Λ f_i + δ v 1{i ∈ S} + noise, with diffuse-factor removal.
3. **Candidate family at scale.** k-means with k = 2..4 plus nested sets along the leading
   direction. At N = 5,241 a 12-agent ring is not a 2-4-means cluster, and the leading
   direction is a platform trend, so the family would not contain it. Proposition 1's
   argument (S is a 2-means cluster once balls separate) holds as m grows with N fixed,
   not in the platform regime. Our family: clusters at many k, kNN co-movement
   communities, co-activity communities, interaction-graph communities, near-duplicate
   components.
4. **Statistic.** Paper: T(S) = |S| ‖mean whitened displacement‖² (Gaussian mean-shift).
   Ours: per-agent shift energy z + mean pairwise cosine z, plus a size-free cosine for
   ranking. Worth running both on Moltbook.
5. **Timing.** The paper is content-only by design. On Moltbook the planted ring was found
   by timing (same thread within minutes), the real bot families by content. Which thread
   an agent answers, and when, is itself behaviour: in the paper's terms the query choice.
   The paper treats query self-selection as a nuisance; for collusion it is a second
   signal channel.
6. **Internal inconsistencies in the draft.** method.tex: Gaussian response kernel and MCD;
   simulations: linear response kernel (README: "far more sensitive") and 25%-trimmed
   covariance. Eq. (affinity) includes j = l pairs; the simulations exclude them
   (U-statistic) and say the biased version "killed power entirely". The PKPS package on
   the `pkps` branch now defaults to local normalization because pair normalization can
   give negative D² (colsim.py clips at 0).

7. **The paper's estimator at Moltbook scale (prototype, 2026-10-17).** I computed the
   paper's affinity for the 5,241 system agents × {pre, post} = 10,482 units: pair
   normalization, U-statistic self term, linear response kernel on 100-dimensional PCA of the
   responses, Gaussian query kernel (comments: threads, truncated to 64 nearest; posts:
   submolts, untruncated). One full matrix took 6 minutes (sparse sums for threads, dense GPU
   for submolts). It is feasible, but:
   - 12% of unit pairs share no query-kernel mass, so their distance is undefined.
   - 1.1% of all squared distances are negative.
   - For each agent's own pre vs post distance, which is the quantity the method rests
     on, 15% are negative and 15% are undefined.
   - The permutation null recomputes the affinity on every draw: about 6 minutes per draw,
     about 20 hours for B = 200.

   The rotation null on displacements needs one affinity only. Local normalization
   (`pkps` branch default) is nonnegative by construction. Pooled mode (what the current
   runs use) gives an explicit vector per agent and is always defined.

## Answers to the colleague's questions

- **Snapshot.** MoltNet v2026-02-28 (iNLP-Lab, CC-BY) plus the TrustAIRLab/CISPA archive.
  Window Jan 31 - Feb 17 2026, burn-in to Feb 6. 143,173 agents active; 1.87M posts and
  comments in the window; 5,241 agents with at least 3 actions on each side of t0
  (2.54M of their actions, capped at 40 per agent-day to 419,465).
- **Votes.** Not attributable: only daily up/down snapshots per post or comment.
- **Operator map.** owner X handle exists but only 7 operators claim more than one system
  agent; stored as a salted hash only.
- **Per-action fields.** agent, timestamp, type, submolt, thread, parent, target agent,
  text. No edit history, no score at fetch.
- **Weak ground truth.** Pre-registered 2026-10-16 (`ground_truth_preregistration.md`):
  Hackerclaw flooding and the Zenity injection accounts. Neither is testable in this
  window (no burn-in or no post-t0 activity). Added later, by a mechanical rule never
  tuned on pipeline output: sibling-name families (FloClaw1-14, QwenPoweredClaw, xmolt,
  ...; 52 agents, 218 pairs). Planted ring: 12 real agents, semi-synthetic, exact
  counterfactual.
- **Found so far.** Ranked groups: sibling families, ClawdBotEighth-Eleventh, PortMonad
  trio, a five-agent SEO network, an XNO promotion network, the Mizuki chain-letter ring.
