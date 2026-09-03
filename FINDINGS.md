# Agentic-Traces DKPS: Experimental Findings Log

Working log of the agentic-traces extension (2026-08-10/11). Self-contained:
protocols, exact representation definitions, all headline numbers, and negative
results. Everything reproduces from the embedding caches + seeds below without
re-calling any API.

## 1. Data

**Small cohort** (`/home/ubuntu/20260729_traces.zip`, extract to `data/traces/`):
swe-agent on SWE-bench Verified, Langfuse span JSONL. 14 models x 12 instances
x 5 replicates = 839 traces (Nemotron-3-Nano missing replicate 5 of
sympy__sympy-14248). No correctness labels; `exit_status` is termination reason
only (7/14 models submit 100% -- saturated). Loader: `dkps.traces.langfuse`.

**Leaderboard corpus** (`data/leaderboard/verified/`, synced from
`s3://swe-bench-submissions/verified/<submission>/` with `--no-sign-request`;
note key prefix has NO `evaluation/`): 134 Verified submissions; 119 have
trajectories (mandatory since 2024-07); official per-instance resolved labels
harvested from the github.com/swe-bench/experiments git repo into
`data/leaderboard/verified_labels.json` (all 134). Loader:
`dkps.traces.leaderboard` (format-agnostic text render; handles dir-style
trajectories, `instance_` filename prefixes, .log/.yaml/.txt extensions).

**Panels** (fixed seeds; reproduce exactly):
- `q418` = sorted intersection of instance ids across the 107 gate-passing
  systems (>= 480 traj files + labels).
- `q150` = `np.random.default_rng(0).choice(q418, 150, replace=False)` sorted.
- `q20`  = `np.random.default_rng(1).choice(q150, 20, replace=False)` sorted.
- Head/tail embeddings exist for all 107 x q150 (+6 partial systems x q20/q150
  coverage); chunk embeddings and judge descriptions for 107(+6) x q20.

**Embedding caches** (gitignored, on this machine):
- `.dkps_cache_lb/<system>/<instance>.<cfg>.npz` where cfg =
  sha1(`openai/text-embedding-3-small|headtail8000`)[:8] holds `head`,`tail`
  (8K-token tiktoken slices); cfg for `|chunks4000` holds `chunks` (4K-char
  line-aligned chunks, full coverage). Embedder: OpenAI text-embedding-3-small
  (1536-d), token-bucket-paced (measured account limit: 5M TPM / 10K RPM).
- Judge descriptions: `data/judge/<judge-model>/<system>/<instance>.txt`
  (+`_emb.npz` for gpt-5.4-mini); prompt in `scripts/judge_describe.py`
  (describe behavior, not problem; report observed verification outcomes).
- Query vectors: `data/leaderboard/query_vecs_64.npz` = PCA-64 of embedded
  problem statements for q418.
- Small-cohort cache `.dkps_cache/` via `scripts/embed_traces.py`
  (nomic-embed-text-v1.5 local; NOTE this host's NVML is broken -- CUDA OOM
  asserts instead of recovering; embedder has small-batch long-text tier +
  halve-batch backoff for this).

## 2. Evaluation protocols

Target y = official resolve rate (|resolved|/500). Readouts:
- **kNN**: k=3, inverse-distance-weighted, on the DKPS distance matrix
  (`model_distance_matrix`: mean over replicates, flatten instances, pdist /
  sqrt(n_queries)).
- **ridge**: kernel-form ridge on flattened per-instance representations,
  lambda in {1,10,100}, best reported.

Reference protocols (who may serve as neighbors/training rows for a target):
- LOO: everyone else. Inflated by family leakage -- do not headline.
- **leave-one-LLM-out (STANDARD, user decision 2026-08-11)**: exclude systems
  sharing the target's underlying LLM (`model_display` tag from metadata.yaml).
  Same-scaffold references allowed (leave-family-out deemed too conservative;
  scaffold siblings cost only ~0.007 vs ~0.018 for LLM siblings).
- Ensemble alpha (blend with sample score) selected per target from its allowed
  references only (honest); grid alpha in linspace(0,1,21).

Context baselines (107 systems, LLM-out): predict-the-mean 0.136; date-3NN
0.114/rho .51; same-LLM-nearest-date 0.084 (requires LLM identity + sibling --
unavailable for genuinely new models). Oracle kNN floor (score-proximal
neighbors): 0.004 -- ceiling is not the constraint.

## 3. Best-known pipeline (as of 2026-08-11)

For a new system evaluated on m instances (references pre-cached):
1. Render trace to text (`load_leaderboard_trajectory`), take first & last 8K
   tokens, embed (text-embedding-3-small).
2. Judge-describe each trace (gpt-5.4-mini, behavior-only prompt), embed the
   description.
3. Per instance, subtract the reference pool's **median** vector (consensus
   centering), L2-normalize the residual; fuse [head+tail | judge] blocks at
   unit RMS each.
4. Distances via **PKPS** (product kernel: RBF over PCA-64 problem-statement
   embeddings x linear response kernel) so the target's m instances compare
   against references' full coverage. Paired DKPS when target coverage is full.
5. kNN(k=3) prediction over LLM-out references; blend with the target's sample
   score, alpha chosen on references.
6. Optional: evaluate on greedily selected informative tasks instead of random.

Numbers (q20 panel, 107 targets, LLM-out, 200 subsets):

| m | sample | pipeline (random tasks) | + selected tasks (held-out half) |
|---|--------|------------------------|----------------------------------|
| 1 | 0.438  | 0.095                  | 0.095                            |
| 3 | 0.208  | 0.088                  | 0.076                            |
| 5 | 0.157  | 0.082                  | 0.070                            |
| 10| 0.103  | 0.070                  | 0.057                            |
| 20| 0.061  | 0.050-0.053            | 0.057                            |

Layer ablations at m=1: naive paired geometry 0.107 -> judge-fused 0.105 ->
PKPS full-coverage refs 0.095. PKPS@m=1 ~= paired@m=20 for the geometry
component (0.098 vs 0.096). Selection halves the budget mid-range (m=10
selected ~= m=20 random); selection overfits its pool (in-pool 0.041-0.058 vs
held-out 0.057-0.070) -- always report held-out.

## 4. Findings (numbered for citation)

**F1. One trace nearly places a system.** Single-query model-distance matrices
correlate r~0.83 with full-panel distances; prediction error is nearly flat in
n_queries (small cohort AND 107-system corpus). Replicates do early work: 1
query x 1 rep 0.109 vs x5 reps 0.089 (small cohort, localization target).

**F2. Reference pool, not query budget, is the binding resource.** Error vs
n_models still declining at 13 (small cohort) and at ~100 (leaderboard).
Supervised channel weighting fails honestly at 13 refs (capacity exists:
in-sample 0.028 vs honest 0.083+) but ridge succeeds at ~97 refs.

**F3. Identity vs competence dissociation.** Action n-grams = best model
fingerprint (replicate-consistency 0.653); text/outcome surface = best
competence signal (Mantel r vs localization: whole 0.62, outcome 0.58, action
0.38). Small cohort.

**F4. Naive representations sit at the unsupervised ceiling.** On the honest
protocol, an equivalence class all lands ~0.078 ridge / ~0.095-0.10 kNN
(q150/q20 panels): head+tail embedding, chunk-mean, rubric sections (hard/soft,
any normalization), positional pyramid, MMD-RBF chunk distributions,
trajectory-dynamics features, displacement (tail-head), relational profiles,
per-instance PCs, global PCA (r>=16), hashed path footprint. Fusions of these
with the naive rep move nothing. Tail >> head (head barely beats date-only).

**F5. Consensus centering is the one geometry trick that matters.**
Per-instance median-centering + L2 of the residual direction: 0.086 -> 0.077
LOO (0.103 -> 0.095 LLM-out). Anchor must be global & robust: median > mean >
trimmed; local (kNN) anchors and PC-removal destroy signal monotonically --
the dominant modes of cross-system variation ARE the signal. Global PCA's top
components encode instance identity, not system behavior (r=2 collapses).

**F6. Symbolic footprint equivalence.** A 512-d hashed bag of file paths (no
embedding model) matches the 3072-d text embedding under both readouts.
Together with F4: what traces reveal is mostly where the system went and what
it produced, not the prose.

**F7. Gold-file localization is a label-free competence proxy.** Fraction of
gold-patch files touched (gold patches are public): spread 0.39-0.82 across
models, credible ranking, MAE 0.086 as a 1-dim/instance representation.
`dkps.traces.metrics`.

**F8. Ensemble with the sample score dominates at every budget.** Honest
alpha; see section 3 table. Even at full panel the geometric prior improves
the sample score (0.034-0.035 vs 0.039 on q150) -- shrinkage against sampling
noise. alpha* slides 0.09 -> 0.85 with m.

**F9. Judge-describe adds complementary signal, scaling with judge quality.**
Describe-then-embed (behavior-only prompt): gpt-4o-mini descriptions are worse
than naive standalone; gpt-5.4-mini's beat naive on kNN (0.097 vs 0.102), lose
on ridge (0.087 vs 0.078), and fused improve both (ridge 0.075; +2-4% through
the whole pipeline at every budget). Only source found whose value has a
quality gradient rather than flat equivalence. Frontier-judge rung untested.

**F10. PKPS earns its keep exactly in sparse coverage.** Target on m
instances, refs on full coverage: 10% geometry gain at m=1; converges to
paired as m grows; slight bias penalty at full coverage. On fully-paired data,
unpaired/PKPS ~= paired (parity, as theory predicts). Class name in code:
`UnpairedDKPS` (`dkps/unpaired_dkps.py`, from jataware/dkps@unpaired-20260401);
paper name: Product Kernel Perspective Space (PKPS).

**F11. Partial-coverage references are includable but not yet helpful.** +6
systems (14-19/20 coverage) via masked PKPS: slightly hurts geometry (coverage
mismatch biases their distances), ~wash for ensemble. Untried fix: the
`use_coverage=True` KDE adjustment in UnpairedDKPS; mismatch also shrinks on
larger panels.

**F12. Task selection transfers.** Greedy forward selection (objective:
ensemble MAE on half the systems) transfers to the held-out half: ~13-17%
better than random subsets at m=3-10; m=10 selected ~= m=20 random. Selection
order is django-heavy (densest instance population).

**F13. Family leakage inflates naive evaluations.** LOO -> LLM-out costs
~0.018 MAE (0.077 -> 0.095); scaffold siblings only ~0.007 more. Most of the
"performance" of metadata baselines is sibling score-copying; the strongest
(same-LLM nearest-date, 0.084) is unavailable for new models by construction.

**F14. Small-cohort results (localization target, pre-labels).** DKPS beats
direct eval below ~half the query budget; error-vs-queries flat (F1); channel
ablations (F3); exit_status is not a score.

**F15. Query-specific rubric extraction BEATS naive under both readouts (the
representation win).** Construction: per instance, LLM writes a 6-section
rubric from the problem statement (what understanding/localization/
reproduction/editing/verification/final_state specifically mean for THIS
issue; `data/judge/qspec_rubrics/`); a judge (gpt-5.4-mini) extracts 30-60
words per section from each trace against that rubric (JSON;
`data/judge/structured-qspec/`); sections embedded separately, per-
(instance,section) median-centered, L2, concatenated (6x1536). q20 panel,
LLM-out: knn 0.0768/0.804 vs naive 0.1022/0.613 (-25%); ridge 0.0758/0.821 vs
0.0779/0.780. Fused with naive: ridge 0.0708/0.837. Ensemble (PKPS + honest
alpha): new bests at every m>=3 (0.083/0.078/0.067/0.049 at m=3/5/10/20);
at m=1 generic reps transfer better (instance-calibrated contrasts don't
cross instances). Scale confirmation on q150 NOT yet run.

**F16. qspec audit & mechanism.** (a) Construction ranking, monotone in
information retained: descriptive qspec > verdict-style questions (0.083/
0.086) > fixed rubric (0.093/0.081) > free-form blob (0.095/0.083) -- forcing
YES/NO verdicts LOSES signal; the latent variable is graded behavioral
completeness, not a checklist. (b) Verbosity mediation, not leakage: system
mean extraction length correlates rho=.59 with score, but length-only is weak
(0.119/0.103), naive+length ~= naive, and residualizing length out of qspec
destroys it (over-control: length and semantics are surface forms of the same
behavioral facts). Keep the 6 length counts as a free auxiliary channel;
do not "correct" for them. (c) Sections individually redundant, collectively
robust; verification strongest for knn (0.091/0.74), editing for ridge
(0.082/0.80), understanding ~dead weight. (d) Rubric/extraction lengths well
controlled by prompt (27-57 / mean 39 words); parse failures ~5/2140.

**F17. Probe selection: CV-then-freeze, not CV-inside-search.** Greedy probe
selection (objective: ensemble MAE on half the systems; evaluated held-out)
improves the ensemble at every budget when hyperparameters are frozen first
(CV'd once, then fixed: sigma=med/16, k=5): m=5 0.074 -> 0.057 (-23%). Running
CV(sigma, k) inside every candidate evaluation halves the gains (m=5 0.070)
and destabilizes choices -- joint optimization over probe set x hyperparams
overfits a 53-system selection pool. Same degrees-of-freedom lesson as the
13-ref weight-learning failure. First frozen-greedy picks are interpretable:
medium-difficulty, high item-total-r instances from four repos, plus one
near-unsolvable instance (5% solve rate, item r=.03) picked for its trace
geometry -- selection exploits information item-response statistics cannot see.

**F18. Post-centering PCA is a real pipeline stage.** Global PCA on the
centered qubric/fusion tensors (fit on the stacked (systems x instances)
residuals) improves BOTH readouts at rank 64-128 (~5-6% flat-panel; budget
sweep: monotone gains growing with m, geometry 0.0745 -> 0.0690 and ensemble
0.0516 -> 0.0476 at m=20 -- pipeline under 0.05 for the first time). Works
here though it failed on raw embeddings (F4-era) because centering first
removes instance identity, so PCA extracts a shared cross-instance behavioral
basis instead of "which problem is this". Order matters: center -> PCA. Query-
side embeddings are also PCA'd (64-d) before the PKPS RBF kernel; sensitivity
check: kernel matrices stable for query dims 32 -> raw-1536 (r ~ .94-.96,
median-bandwidth self-adapts), materially distorted at 8-16d. Rank not yet in
the CV grid (chosen on-panel at 64; flag for the q150 confirmation).
ADDENDUM: PCA does NOT compound with greedy probe selection (greedy+PCA64
held-out: 0.083/0.064/0.057 at m=1/5/20 vs greedy full-d 0.078/0.057/0.053) --
selection and PCA spend the same slack; use PCA for random probes, full-d for
selected probes.

**F19. Similarity axes and the quotient ladder (intrinsic evaluation).**
Trace similarity decomposes into axes: task, provenance, lineage, format,
behavior, outcome. Trace-level nearest-neighbor metrics on q20 (chance rates:
task/provenance ~0.01, LLM 0.09, scaffold 0.06):

| axis            | raw   | centered | qubric-raw | qubric-cent |
|-----------------|-------|----------|------------|-------------|
| task NN         | 0.831 | 0.367    | 0.994      | 0.953       |
| provenance P@1  | 0.675 | 0.830    | 0.057      | 0.094       |
| same-LLM NN     | 0.215 | 0.201    | 0.128      | 0.117       |
| same-scaffold NN| 0.307 | 0.308    | 0.072      | 0.079       |
| outcome AUC     | 0.569 | 0.519    | 0.575      | 0.547       |

(a) The judge quotients the who-axes to ~chance; (b) it does NOT quotient the
task axis (rubric injects instance vocabulary; residuals stay instance-
conditioned, 0.95) -- the architecture handles task via within-instance
comparison + PKPS query kernel (also explains qubric's weak m=1-2 cross-
instance transfer); (c) centering alone AMPLIFIES provenance (0.68 -> 0.83);
(d) qubric's trace-level identity collapse is largely judge sampling noise:
10-instance aggregation recovers self-match 0.09 -> 0.52; (e) naive's
intrinsic scores are inflated by lineage leakage (sibling-NN 26%, 3x chance;
sibling-excluded score-gap advantage 0.014 -> 0.002). Intrinsic metrics must
be aggregation-aware and lineage-controlled.

**F20. Pillar definitions and independence evidence.** The seven radar
pillars: task fidelity (cross-instance instance retrieval), behavior fidelity
(within-instance Mantel vs outcome-free patch features: size+files), outcome
sensitivity (within-instance same-outcome AUC), provenance invariance at trace
AND aggregated (10-instance split-half) levels, LLM-lineage and scaffold
invariance (within-instance NN excl own system). Independence: (a) by
conditioning design (cross- vs within-instance); (b) double dissociations
across 6 OpenAI-embedded representations -- task/provenance occupy opposite
corners (qubric .95/.09 vs centered .37/.83); free-form judge holds LLM at
chance (.113) while provenance stays elevated (.343) and scaffold 8x chance
(.136); outcome/behavior dissociate (tail-only .58/-.006 vs judge reps
~.54/.025); (c) targeted checks: behavior spoke invariant to removing the
outcome-proxy feature (.026 -> .024); raw's LLM and scaffold NN hits only 28%
joint (10% for qubric) -- separate style vs format channels. Residual known
entanglement: trace-level provenance x task for instance-blocked reps; hence
dual-level provenance reporting. ADDENDUM (identity-beyond-type): among the 27 systems with a type-sibling
(same harness AND declared model family), raw embeddings retrieve the EXACT
submission at 0.82 vs 0.43 chance -- the Identity pillar is not reducible to
the Model-family x Harness conjunction; embeddings fingerprint individuals
(version/config/prompts). qubric reduces this to 0.62. Caveat: some sibling
groups match on 'Undisclosed'/'Multiple' model tags; clean disclosed groups
are small. Identity is also the only authorship pillar computable with zero
metadata. Final pillar names: Task, Behavior | Identity, Model family,
Harness. Data: figures/radar_all_data_v2.json;
exhibits: figures/radar_all.png, figures/independence_heatmap.png.

**F21. Cross-model judge matrix under a common local embedder (2026-08-19).**
All cached (construction x extraction-judge) cells re-embedded with
nomic-embed-text-v1.5 (the only apples-to-apples choice: nano judge texts were
never embedded with the OpenAI model) and scored with the standard protocol
(per-(instance,section) median centering, L2, concat; leave-one-LLM-out kNN
k=3 and ridge; 107 systems x q20). kNN MAE: 4o-mini blob .110; mini blob
.094 > generic .088 > verdict .086 > qubric .078 (construction ranking
reproduces under a local embedder -- F15/F16 are not an OpenAI-embedder
artifact); qubric nano .083 vs mini .078. Key reading: NANO x QUBRIC BEATS
MINI x BLOB AND MINI x GENERIC -- the construction contributes more than
extractor capability; judge degradation is graceful. Parse failures track
judge quality (mini 5, nano 25 of 2140). Rubric-writer axis has one cached
value (gpt-5.4-mini); filling it + structured extraction for 4o-mini/nano
blob cells queued with q150 (API-blocked). Data: figures/judge_matrix.json;
exhibit: figures/judge_matrix.png (Fig 3 right panel);
scripts: judge_matrix.py, fig_judge_matrix.py.

**F22. Open-weight judge migration, first rungs (2026-08-19).** q20 extraction
re-run with open judges on IDENTICAL cached mini-written rubrics (only the
extractor varies), scored in the F21 protocol (nomic embedder,
leave-one-LLM-out). kNN/ridge MAE: gpt-oss-20b .1134/.0999 (FAILS -- worse
than 4o-mini blob despite 0 parse failures); gpt-oss-120b .0864/.0916
(capability gradient confirmed, pipeline exonerated; still fails the
within-noise-of-mini bar); reference mini .0784/.0780, nano .0833/.0814.
Notable: closed NANO (tiny) beats open 120B on both readouts. Judge quality
manifests as description SPECIFICITY (file names, what was left unchanged),
not fluency or format compliance -- strict json_schema made malformed output
impossible yet 20b still failed on content. Operational: OpenRouter
price-routing + strict schema + trace-text cache; measured cost ~$1/judge/
q20 run; ~89 empty-content responses per run on ~60K-char traces at
effort=low, all recovered by escalation (8K tokens, effort=medium).
Scripts: judge_openrouter.py, judge_escalate.py.
ADDENDUM (deepseek-v3.1, 2026-08-19): kNN .0826/.793, ridge .0794/.792 --
CLEARS the bar (within .004/.001 of mini, inside panel resolution ~.01);
best open judge tested; ties nano kNN, beats nano rho. Full open ladder
monotone in capability: oss-20b .113 > oss-120b .086 > deepseek .083 ~= nano
.083 > mini .078. Costs/q20-run: oss ~$1, deepseek ~$11.5 (verbose
completions + higher pricing; input dominates -> q150 with deepseek ~$85).
DECISION: deepseek-v3.1 is the open judge for scale-up unless a cheaper
frontier-open (qwen3-235b, ~$4 test) matches it. Judge quality is NOT
price-monotone in the open tier (oss-120b cheapest AND worst-but-one).

**F23. Rubric-section ablation (2026-08-19, cache-only).** Drop-one from the
6-section qubric (mini judge, nomic embedder, F21 protocol): worst delta
+0.0077 kNN MAE (reproduction); verification drop -0.0024 (noise). Only-one:
every single section much worse than the union (best single: verification
.0913; worst: understanding .1218; all-6 .0784). Signal is distributed across
complementary behavioral facets -- the construction is not a one-section
trick. Data: figures/section_ablation.json.

**F24. CAPA-style correctness-only competitor (2026-08-19, cache-only).**
Chance-adjusted error-overlap (kappa) kNN over q20 correctness vectors,
leave-one-LLM-out, 40 draws: m=1 .369+-.049, m=5 .198+-.033, m=10 .126,
m=20 .072 (sample score alone: .437/.144/.105/.061). Trace-content geometry
reaches .095 at m=1 -- correctness-only methods need ~15-20 scored probes to
match what one TRACE provides; at m=20 the qubric ensemble (.048-.053) still
dominates. The resolution-floor argument (Yauney et al.) reproduced on our
corpus with the strongest correctness-only similarity. Data:
figures/capa_baseline.json. Caveat: quick-protocol numbers; recompute under
the exact Fig 3 pipeline before print.

**F25. Stability column + Identity collapsed to aggregated (2026-08-28).**
Heatmap now 6 properties x 6 representations. (c) Identity reported at the
aggregated (10-trace split-half) level only -- the harder test (qubric 0.52).
(a) Stability = agent-seed replicate retrieval P@1 on the small cohort
(14 x 12 x 5; chance 4/69=.058), all rows nomic-embedded; judge rows use
deepseek-v3.1 extractions of the compact render (12 deepseek-written
rubrics; ~$10, 1690 calls). Results: head-only .347 > centered .341 ~
free-form .337 > raw .324 > tail-only .291 > qubric .211. All 3.6-6x chance.
INTERPRETATION: replicate retrieval conflates behavioral consistency with
the identity fingerprint; fingerprint-rich rows are inflated (their Identity
column is .83-.93), qubric's .211 is behavior-only. Tail-only < head-only
gap = late-trace stochasticity (divergent endings across replicates).
PITFALL LOGGED: first computation rendered small-cohort traces WITHOUT tool
outputs (canonicalize_step defaults) -> every trace <=32K chars -> head ==
tail == whole, three rows identical (.350) and fingerprint understated;
re-rendered with include_tool_output_chars=1500 (median 43K chars).
Data: figures/stability_column.json; emb cache smallcohort_emb_nomic_v2.npz;
scripts: smallcohort_judge.py, stability_column.py, fig_heatmap.py.

**F26. Stability redefined + measured as pipeline test-retest (2026-08-29,
per HH).** Definition: the SAME trace re-processed should map to ~the same
place; stochastic stages are rubric creation + extraction; agent-run
variance is variance in the world, not the measurement (the F25 replicate
numbers are hereby relabeled BEHAVIORAL CONSISTENCY --
figures/behavioral_consistency.json). Measurements (deepseek judge, nomic,
q20 x 107, within-instance self-retrieval, chance 1/107=.009):
extraction retest (same mini rubrics, resampled extraction) trace P@1 .316
centered / .222 uncentered -- centering HELPS self-retrieval (removes shared
instance content, sharpens the residual); distance ratio d(self)/med
d(others) = .873; SYSTEM-level (20-instance aggregate) self-retrieval .804.
Pipeline retest (rubrics REWRITTEN by deepseek -- also a cross-writer test,
pass-1 rubrics were mini-written -- then extraction): trace P@1 .291 on 690
pairs (partial; 402-blocked), only .025 below extraction-only => rubric
perturbation is a minor noise source vs extraction sampling; first Model-1
sensitivity datum. Deterministic rows (raw/head/tail/centered) are exactly
1.0 by construction. PITFALL: computing pass-2 consensus centering over
rows that include blank (missing) extractions poisons per-instance medians
(P@1 .05 artifact -> .291 with valid-row centering). Cross-judge trace
agreement (.12-.28, F-earlier) is confirmed as a lower bound on same-judge
retest. Blocked on OpenRouter credits: 1450 pipeline pairs, free-form
retest row. Data: figures/stability_retest_full.json,
stability_retest_extraction.json, stability_column.json;
script: stability_retest.py.

**F27. Routing/selection v0 (2026-09-02, cache-only, PRELIMINARY).**
Per-instance system selection with no outcome knowledge on the target
instance: p(s solves q) = distance-weighted kNN over traces on OTHER
instances (leave-one-LLM-out wrt s), route to argmax. q20, nomic. Routed
resolve rate: qubric .700 > raw .650 > mean-system .593; best-single
(hindsight) .850, oracle .950. Clears HH's bar (beat raw) but by ONE
instance at q20 -- underpowered; rerun at q150 before claiming. Neither
variant beats best-single yet; framing = recovers gap from mean toward best
without knowing which system is best. Data: figures/routing.json;
script: routing.py.

**F28. Judge-input token pruning (2026-09-03, analysis + tooling; A/B
pending credits).** Composition of q20 judge inputs (300-trace sample,
~4.1M tokens): line-numbered file views 16.3%, harness state dumps 10.3%,
listings 5.1%, format-echo duplicates 1.6% -- ~33% identifiably
non-behavior (repo content the agent displayed, not what it did); more
hides in the 59% prose bucket as unnumbered code dumps. Pruner
(dkps/traces/prune.py; 6 format-agnostic rules; keeps prose, commands,
diffs, error lines, test verdict tails): 33.9% char reduction over all
2140 q20 texts (median 28%, p90 68%; sympy-14248 example 60K->19K with
agent edits kept verbatim). Traces contain no gold outcomes -- no true
cheating channel found; agent self-claims kept deliberately (rubric
separates claimed vs observed verification). Also a QUALITY lever:
prune-then-truncate fits more behavior in the 60K window than blind middle
truncation. Projected q150 input cost ~$58->$38 (deepseek). Validation A/B
queued (judge pruned vs unpruned q20, score in matrix protocol; ~$6
deepseek or ~$1 gpt-oss): --prune flag on judge_openrouter.py caches to
structured-qspec-<judge>-pruned. Cf. AgentDiet (arXiv:2509.23586): 40-60%
removable, consistent.
ADDENDUM (truncation prevalence, 2026-09-03): 81% of q20 judge inputs
(1742/2140) were truncated by the 40K+20K-char window; median raw render is
133K chars (judge saw ~45%), p90 666K (~9%), max 3.0M (~2%). ALL judged
results to date rest on partial-trace observation -- the window was
inherited from the head/tail embedding baseline, never decided or measured.
After pruning, 36/40 sampled FULL traces fit a 128K-token context ->
whole-trace judging is now feasible. PRIORITY A/B when credits return
(three arms, q20, same judge+rubrics, ~$25 deepseek): truncated (status
quo) vs pruned+60K-window vs pruned-full-context. If full-context wins,
current headline numbers UNDERSTATE qubric.

## 5. Negative results (do not re-run without new ideas)

- Supervised channel-weight learning at 13 refs: five schemes all <= uniform.
- Chunk-level product kernels (position/section side kernels, any content
  centering): reproduce pooled special cases, never beat them.
- Local/neighborhood centering, per-dim z-scoring, PC-removal: hurt (F5).
- Dynamics/displacement/relational/conformity reps: redundant with naive (F4).
- gpt-4o-mini as judge: below naive standalone (F9).
- Rubric approaches: hard-assignment + centering TIES naive (LOO 0.083 both,
  q20 full panel); soft pooling loses; 6-dim mass profile within 0.015 of the
  3072-d baseline (remarkable per-dim, not a win). 150-instance confirmation
  deliberately skipped (cost/benefit).

## 6. Caveats

- q20-panel differences under ~0.01 are within noise; q150 differences under
  ~0.005 likewise. Selection/ensemble tables use paired subsets where possible.
- Judge + chunk experiments live on q20 only; head/tail on q150.
- The 20-instance selection table's held-out half is a different population
  than the all-107 random baseline (worth ~0.004; see m=20 row).
- Sample scores at m use official per-instance labels ("run the harness on m
  instances"); DKPS-only numbers require no harness at all.

## 7. Repo map (this branch)

- `dkps/traces/`: schema, langfuse + leaderboard loaders, canonicalize,
  channels (incl. rubric/whole), rubric.py, embedder (local + OpenAI, paced),
  metrics (localization), assemble (build_dkps_input, distances).
- `dkps/unpaired_dkps.py`: PKPS implementation (verbatim from jataware fork).
- `scripts/`: embed_traces, analyze_traces, smoke_test (small cohort);
  predict_localization, learn_channel_weights (small cohort experiments);
  leaderboard_baselines, leaderboard_chunks, leaderboard_reprs, rubric_viability,
  chunk_pkps, leaderboard_unpaired, judge_describe (leaderboard).
- `figures/`: perspective spaces + error curves (small cohort, share-ready).
- One-off experiment code (centering variants, ensemble sweeps, selection,
  PKPS budget sweeps) ran inline; recipes fully specified above.
